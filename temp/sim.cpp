// ============================================================
//  Discrete Event Simulation of a Multi-Core Web Application
//  Assignment 2 - Performance Analysis
//
//  System Models:
//    - Multi-core server (thread-to-core affinity)
//    - Thread-per-task model with max thread limit (= num_cores)
//    - Round-robin scheduling with context-switch overhead
//    - Request timeouts (min + exponential variable component)
//    - Closed-loop users: request -> wait -> think -> repeat
//    - Think time: lognormal (mode well above zero)
//    - Service time: selectable (constant / uniform / exponential)
//
//  Metrics Collected (after warmup/transient is discarded):
//    - Average Response Time with 95% Confidence Intervals
//    - Goodput  (completions of non-timed-out requests / sec)
//    - Badput   (completions of already-timed-out requests / sec)
//    - Throughput = Goodput + Badput
//    - Drop Rate  (requests timed out in queue, never served)
//    - Average Core Utilization
//
//  Build:  g++ -O2 -std=c++17 -o web_sim web_des_sim.cpp
//  Run:    ./web_sim            (writes results.csv)
//          ./web_sim verbose    (also prints event trace)
// ============================================================

#include <iostream>
#include <fstream>
#include <queue>
#include <vector>
#include <map>
#include <string>
#include <random>
#include <cmath>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <functional>

// ============================================================
// Section 1: Random Number Generator utilities
// ============================================================
class RNG {
    std::mt19937 gen;
public:
    explicit RNG(unsigned seed = 42) : gen(seed) {}

    // Exponential with given mean
    double exponential(double mean) {
        std::exponential_distribution<double> d(1.0 / mean);
        return d(gen);
    }

    // Uniform [a, b)
    double uniform(double a, double b) {
        std::uniform_real_distribution<double> d(a, b);
        return d(gen);
    }

    // Lognormal: X = e^(mu + sigma * N(0,1))
    // Mode = exp(mu - sigma^2)   <- can be set away from zero easily
    double lognormal(double mu, double sigma) {
        std::lognormal_distribution<double> d(mu, sigma);
        return d(gen);
    }
};

// ============================================================
// Section 2: Enumerations
// ============================================================

// Types of events in the simulation
enum class EventType {
    USER_THINK_END,       // User finished thinking; now submits a request
    CTX_SWITCH_DONE,      // Context-switch overhead finished; service begins
    REQUEST_COMPLETE,     // Service finished on a core
    REQUEST_TIMEOUT,      // Request timed out while sitting in the wait queue
    SIM_END               // Sentinel – stop the simulation loop
};

// Service time distribution choices
enum class DistType { CONSTANT, UNIFORM, EXPONENTIAL };

// ============================================================
// Section 3: Core data structures
// ============================================================

// Represents a single pending or completed request
struct Request {
    int    id;               // Unique request ID
    int    user_id;          // Which user issued this request
    double arrive_time;      // Wall-clock time request reached server
    double timeout_at;       // Absolute time when request expires
    double service_dur;      // Sampled service duration (const once sampled)
    double service_start;    // Time service actually began (-1 = not yet)
    int    assigned_core;    // Which core is handling it (-1 = in queue)
    bool   timed_out;        // Set true when timeout event fires
    bool   completed;        // Set true on completion

    Request() = default;
    Request(int id, int uid, double arr, double deadline, double svc)
        : id(id), user_id(uid), arrive_time(arr), timeout_at(deadline),
          service_dur(svc), service_start(-1.0), assigned_core(-1),
          timed_out(false), completed(false) {}
};

// An entry in the global event queue
struct Event {
    double    time;
    EventType type;
    int       request_id;  // -1 if not relevant
    int       user_id;     // -1 if not relevant
    int       core_id;     // -1 if not relevant

    Event(double t, EventType et, int rid = -1, int uid = -1, int cid = -1)
        : time(t), type(et), request_id(rid), user_id(uid), core_id(cid) {}

    // Min-heap: smallest time has highest priority
    bool operator>(const Event& o) const { return time > o.time; }
};

// Represents one CPU core
struct Core {
    int    id;
    bool   busy         = false;
    int    current_req  = -1;    // ID of request being served
    double busy_since   = 0.0;
};

// ============================================================
// Section 4: Simulation parameters
// ============================================================
struct SimParams {
    int       num_cores         = 4;
    int       num_users         = 10;
    double    sim_duration      = 8000.0;  // total wall-clock simulation time (s)
    double    warmup_duration   = 800.0;   // transient to discard (Welch-style)

    // Service time
    DistType  service_dist      = DistType::EXPONENTIAL;
    double    service_mean      = 0.031;   // mean service time (seconds)
    double    service_min       = 0.02;   // uniform lower bound
    double    service_max       = 0.25;   // uniform upper bound
    double    service_const     = 0.10;   // constant service time

    // Think time: lognormal (mode = exp(mu - sigma^2))
    // With mu=1.0, sigma=0.5 -> mode ≈ 2.1s  (well above zero)
    double    think_mu          = 1.2613;
    double    think_sigma       = 0.5;

    // Timeout = timeout_min + Exp(timeout_exp_mean)
    double    timeout_min       = 5;   // minimum timeout (s)
    double    timeout_exp_mean  = 1.50;   // mean of variable part (s)

    // Context-switch overhead: Exp(ctx_sw_mean)
    double    ctx_sw_mean       = 0.002;  // 2 ms mean

    // Round-robin cursor (mutable during run)
    int       rr_cursor         = 0;

    unsigned  seed              = 42;
    bool      verbose           = false;
};

// ============================================================
// Section 5: Per-run statistics
// ============================================================
struct Stats {
    // Response time of every request finished AFTER warmup
    std::vector<double> response_times;

    int    num_submitted  = 0;   // total requests submitted after warmup
    int    num_goodput    = 0;   // completed before timeout
    int    num_badput     = 0;   // completed but already timed out
    int    num_dropped    = 0;   // timed out in queue, never served
    double total_rt       = 0.0; // sum of all response times (good+bad)
    double stat_duration  = 0.0; // sim_duration - warmup_duration

    // Per-core accumulated busy time (after warmup only)
    std::vector<double> core_busy;

    void reset(int ncores) {
        response_times.clear();
        num_submitted = num_goodput = num_badput = num_dropped = 0;
        total_rt = stat_duration = 0.0;
        core_busy.assign(ncores, 0.0);
    }

    // Derived metrics
    double avgRT()        const {
        int n = num_goodput + num_badput;
        return n ? total_rt / n : 0.0;
    }
    double goodput()      const { return stat_duration ? num_goodput  / stat_duration : 0.0; }
    double badput()       const { return stat_duration ? num_badput   / stat_duration : 0.0; }
    double throughput()   const { return goodput() + badput(); }
    double dropRate()     const { return num_submitted ? (double)num_dropped / num_submitted : 0.0; }
    double coreUtil(int n) const {
        double tot = 0;
        for (double t : core_busy) tot += t;
        return (n > 0 && stat_duration > 0) ? tot / (n * stat_duration) : 0.0;
    }
};

// ============================================================
// Section 6: The Simulator class
// ============================================================
class Simulator {
    SimParams  P;
    RNG        rng;

    // Event queue (min-heap on time)
    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> evq;

    // Active requests (keyed by request ID)
    std::map<int, Request> live;

    // Cores
    std::vector<Core> cores;

    // Wait queue: IDs of requests waiting for a free core
    std::queue<int> wait_queue;

    double current_time    = 0.0;
    int    next_req_id     = 0;

public:
    Stats  stats;

    explicit Simulator(SimParams p, unsigned seed_offset = 0)
        : P(p), rng(p.seed + seed_offset)
    {
        cores.resize(p.num_cores);
        for (int i = 0; i < p.num_cores; i++) cores[i].id = i;
    }

    // ---- Variate generators ----------------------------------------

    double genService() {
        switch (P.service_dist) {
            case DistType::CONSTANT:    return P.service_const;
            case DistType::UNIFORM:     return rng.uniform(P.service_min, P.service_max);
            case DistType::EXPONENTIAL: // fall-through default
            default:                    return rng.exponential(P.service_mean);
        }
    }

    double genThink() {
        // Lognormal with mode ≈ exp(mu - sigma^2) ≈ 2.1 s by default
        return rng.lognormal(P.think_mu, P.think_sigma);
    }

    double genTimeout() {
        // Minimum plus exponential variable component
        return P.timeout_min + rng.exponential(P.timeout_exp_mean);
    }

    double genCtxSw() {
        return rng.exponential(P.ctx_sw_mean);
    }

    // ---- Event scheduling ------------------------------------------

    void sched(double t, EventType et, int rid = -1, int uid = -1, int cid = -1) {
        evq.push(Event(t, et, rid, uid, cid));
    }

    // ---- Round-robin core selection --------------------------------

    // Returns index of the next idle core, advancing the RR cursor.
    // Returns -1 if all cores are busy.
    int pickCore() {
        for (int i = 0; i < P.num_cores; i++) {
            int idx = (P.rr_cursor + i) % P.num_cores;
            if (!cores[idx].busy) {
                P.rr_cursor = (idx + 1) % P.num_cores;
                return idx;
            }
        }
        return -1;
    }

    // ---- Helper: assign a request to a specific core after ctx-sw --

    void assignToCore(int rid, int cid) {
        // Context switch overhead first, then actual service starts
        double ctx = genCtxSw();
        sched(current_time + ctx, EventType::CTX_SWITCH_DONE, rid, -1, cid);
        // Mark core logically taken so no other request is assigned to it
        cores[cid].busy = true;
        cores[cid].current_req = rid;
        cores[cid].busy_since  = current_time;  // ctx switch time is overhead

        auto& req = live.at(rid);
        req.assigned_core = cid;

        if (P.verbose)
            std::cout << "  [assign] req=" << rid << " -> core=" << cid
                      << " (ctx-sw=" << std::fixed << std::setprecision(4) << ctx << "s)\n";
    }

    // ---- Event handlers --------------------------------------------

    // USER_THINK_END: user submits a new request
    void onThinkEnd(const Event& e) {
        int uid    = e.user_id;
        double to  = genTimeout();
        double svc = genService();

        int rid = next_req_id++;
        live.emplace(rid, Request(rid, uid, current_time,
                                  current_time + to, svc));

        bool after_warmup = (current_time >= P.warmup_duration);
        if (after_warmup) stats.num_submitted++;

        if (P.verbose)
            std::cout << std::fixed << std::setprecision(4)
                      << "[" << current_time << "] THINK_END  user=" << uid
                      << " req=" << rid
                      << " svc=" << svc
                      << " timeout_at=" << current_time + to << "\n";

        int cid = pickCore();
        if (cid >= 0) {
            // A core is free -> assign immediately (with ctx-sw)
            assignToCore(rid, cid);
        } else {
            // All cores busy -> enqueue; schedule timeout event
            wait_queue.push(rid);
            sched(live.at(rid).timeout_at, EventType::REQUEST_TIMEOUT, rid, uid, -1);

            if (P.verbose)
                std::cout << "  [queued] queue_size=" << wait_queue.size() << "\n";
        }
    }

    // CTX_SWITCH_DONE: overhead done, service actually starts
    void onCtxSwDone(const Event& e) {
        int rid = e.request_id;
        int cid = e.core_id;

        // Request may have been removed (timed out during ctx-sw)
        auto it = live.find(rid);
        if (it == live.end()) {
            // Request already gone; free the core and try queue
            cores[cid].busy = false;
            cores[cid].current_req = -1;
            drainQueue(cid);
            return;
        }

        Request& req = it->second;

        // Check if timeout already fired during context switch
        if (current_time >= req.timeout_at) {
            if (P.verbose)
                std::cout << std::fixed << std::setprecision(4)
                          << "[" << current_time << "] CTX_SW_DONE req=" << rid
                          << " EXPIRED during ctx-sw -> dropped\n";

            bool after_warmup = (req.arrive_time >= P.warmup_duration ||
                                 current_time    >= P.warmup_duration);
            // Count as drop (timed out before actual service)
            if (after_warmup) stats.num_dropped++;

            // User re-enters think phase
            sched(current_time + genThink(), EventType::USER_THINK_END,
                  -1, req.user_id, -1);
            live.erase(it);

            cores[cid].busy = false;
            cores[cid].current_req = -1;
            drainQueue(cid);
            return;
        }

        // Service begins
        req.service_start = current_time;

        if (P.verbose)
            std::cout << std::fixed << std::setprecision(4)
                      << "[" << current_time << "] CTX_SW_DONE req=" << rid
                      << " core=" << cid
                      << " -> service for " << req.service_dur << "s\n";

        sched(current_time + req.service_dur,
              EventType::REQUEST_COMPLETE, rid, req.user_id, cid);
    }

    // REQUEST_COMPLETE: service finished on a core
    void onComplete(const Event& e) {
        int rid = e.request_id;
        int cid = e.core_id;

        auto it = live.find(rid);
        if (it == live.end()) return;  // already cleaned up

        Request& req = it->second;
        bool after_warmup = (req.arrive_time >= P.warmup_duration ||
                             current_time    >= P.warmup_duration);

        // Accumulate core busy time (from when core was marked busy)
        if (after_warmup) {
            double busy_start = std::max(cores[cid].busy_since, P.warmup_duration);
            stats.core_busy[cid] += current_time - busy_start;
        }

        double rt = current_time - req.arrive_time;
        bool   late = (current_time > req.timeout_at);  // badput

        if (P.verbose)
            std::cout << std::fixed << std::setprecision(4)
                      << "[" << current_time << "] COMPLETE   req=" << rid
                      << " core=" << cid
                      << (late ? " [BADPUT]" : " [GOODPUT]")
                      << " RT=" << rt << "s\n";

        if (after_warmup) {
            stats.total_rt += rt;
            stats.response_times.push_back(rt);
            if (late) stats.num_badput++;
            else      stats.num_goodput++;
        }

        // User starts thinking again
        sched(current_time + genThink(), EventType::USER_THINK_END,
              -1, req.user_id, -1);

        live.erase(it);

        // Free core and serve next queued request
        cores[cid].busy = false;
        cores[cid].current_req = -1;
        drainQueue(cid);
    }

    // REQUEST_TIMEOUT: request timed out while in the wait queue
    void onTimeout(const Event& e) {
        int rid = e.request_id;

        auto it = live.find(rid);
        if (it == live.end()) return;  // already completed or being served

        Request& req = it->second;

        // Only act if still in queue (not yet assigned to a core)
        // If already assigned (ctx-sw in progress), onCtxSwDone handles it.
        if (req.assigned_core >= 0) return;

        if (P.verbose)
            std::cout << std::fixed << std::setprecision(4)
                      << "[" << current_time << "] TIMEOUT    req=" << rid
                      << " user=" << req.user_id << "\n";

        bool after_warmup = (req.arrive_time >= P.warmup_duration ||
                             current_time    >= P.warmup_duration);
        if (after_warmup) stats.num_dropped++;

        // User re-enters think phase
        sched(current_time + genThink(), EventType::USER_THINK_END,
              -1, req.user_id, -1);

        live.erase(it);
        // Note: the ID is still sitting in wait_queue but drainQueue will skip it
    }

    // Try to dequeue a waiting request onto the now-free core `cid`
    void drainQueue(int cid) {
        while (!wait_queue.empty()) {
            int rid = wait_queue.front();
            wait_queue.pop();

            auto it = live.find(rid);
            if (it == live.end()) continue;  // already timed out or handled

            Request& req = it->second;
            if (req.timed_out) continue;

            // Check if request already expired while waiting
            if (current_time >= req.timeout_at) {
                bool after_warmup = (req.arrive_time >= P.warmup_duration ||
                                     current_time    >= P.warmup_duration);
                if (after_warmup) stats.num_dropped++;
                sched(current_time + genThink(), EventType::USER_THINK_END,
                      -1, req.user_id, -1);
                live.erase(it);
                continue;
            }

            // Valid request: assign to this core
            assignToCore(rid, cid);
            return;
        }
        // Queue empty: core remains idle
    }

    // ---- Main simulation loop --------------------------------------

    void run() {
        stats.reset(P.num_cores);
        current_time = 0.0;
        next_req_id  = 0;

        // Bootstrap: all users start with a think phase
        for (int u = 0; u < P.num_users; u++) {
            sched(genThink(), EventType::USER_THINK_END, -1, u, -1);
        }
        // Sentinel to stop simulation
        sched(P.sim_duration, EventType::SIM_END);

        while (!evq.empty()) {
            Event e = evq.top(); evq.pop();
            current_time = e.time;

            if (e.type == EventType::SIM_END) {
                // Flush remaining core busy times
                for (int i = 0; i < P.num_cores; i++) {
                    if (cores[i].busy) {
                        double busy_start = std::max(cores[i].busy_since,
                                                     P.warmup_duration);
                        stats.core_busy[i] += current_time - busy_start;
                    }
                }
                stats.stat_duration = P.sim_duration - P.warmup_duration;
                if (P.verbose)
                    std::cout << "[" << current_time << "] SIM_END\n";
                break;
            }

            switch (e.type) {
                case EventType::USER_THINK_END:    onThinkEnd(e);   break;
                case EventType::CTX_SWITCH_DONE:   onCtxSwDone(e);  break;
                case EventType::REQUEST_COMPLETE:  onComplete(e);   break;
                case EventType::REQUEST_TIMEOUT:   onTimeout(e);    break;
                default: break;
            }
        }
    }
};

// ============================================================
// Section 7: Statistical analysis helpers
// ============================================================

// Welch's t confidence interval (95%) for a vector of sample means
struct CI {
    double mean, half;
    double lo() const { return mean - half; }
    double hi() const { return mean + half; }
};

// t critical values for 95% two-tailed CI
static double tCrit(int df) {
    if (df <= 0)  return 12.706;
    if (df == 1)  return 12.706;
    if (df == 2)  return 4.303;
    if (df == 3)  return 3.182;
    if (df == 4)  return 2.776;
    if (df == 5)  return 2.571;
    if (df <= 9)  return 2.306;
    if (df <= 14) return 2.145;
    if (df <= 19) return 2.093;
    if (df <= 29) return 2.045;
    return 1.960;
}

CI computeCI(const std::vector<double>& xs) {
    int n = (int)xs.size();
    if (n == 0) return {0, 0};
    double mu = std::accumulate(xs.begin(), xs.end(), 0.0) / n;
    double var = 0;
    for (double x : xs) var += (x - mu) * (x - mu);
    if (n > 1) var /= (n - 1);
    double se = std::sqrt(var / n);
    return {mu, tCrit(n - 1) * se};
}

// ============================================================
// Section 8: One data point = multiple replications
// ============================================================
struct DataPoint {
    int    users;
    CI     rt_ci;        // response time with CI
    double goodput, badput, throughput, drop_rate, core_util;
};

DataPoint runExperiment(SimParams base, int num_users,
                        int num_reps = 10) {
    base.num_users = num_users;

    std::vector<double> rts, gps, bps, tps, drs, cus;

    for (int r = 0; r < num_reps; r++) {
        Simulator sim(base, r * 7919u);  // different seed per replica
        sim.run();
        const Stats& s = sim.stats;
        rts.push_back(s.avgRT());
        gps.push_back(s.goodput());
        bps.push_back(s.badput());
        tps.push_back(s.throughput());
        drs.push_back(s.dropRate());
        cus.push_back(s.coreUtil(base.num_cores));
    }

    auto avg = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };

    DataPoint dp;
    dp.users     = num_users;
    dp.rt_ci     = computeCI(rts);
    dp.goodput   = avg(gps);
    dp.badput    = avg(bps);
    dp.throughput= avg(tps);
    dp.drop_rate = avg(drs);
    dp.core_util = avg(cus);
    return dp;
}

// ============================================================
// Section 9: Main – experiments and output
// ============================================================
int main(int argc, char* argv[]) {
    bool verbose = (argc > 1 && std::string(argv[1]) == "verbose");

    // ------ Base configuration ----------------------------------
    SimParams base;
    base.num_cores        = 4;
    base.sim_duration     = 8000.0;    // 8 000 s per replica (after warmup)
    base.warmup_duration  = 800.0;     // first 800 s discarded (Welch)
    base.service_dist     = DistType::EXPONENTIAL;
    base.service_mean     = 0.10;      // 100 ms mean service time
    base.think_mu         = 1.0;       // lognormal mean-of-log
    base.think_sigma      = 0.5;       // -> mode ≈ 2.1 s think time
    base.timeout_min      = 0.50;      // 500 ms minimum timeout
    base.timeout_exp_mean = 1.50;      // + Exp(1.5) variable component
    base.ctx_sw_mean      = 0.002;     // 2 ms context-switch overhead
    base.seed             = 42;
    base.verbose          = verbose;

    const int NUM_REPS = 10;           // independent replications per point
    const std::vector<int> USER_COUNTS =
        {20, 30, 40, 50, 60, 70, 80, 100, 120, 140, 150, 160};

    // ------ Print header ----------------------------------------
    std::cout << "================================================\n";
    std::cout << " Web App DES  |  Cores=" << base.num_cores
              << "  Reps=" << NUM_REPS << "\n";
    std::cout << " Service dist : EXPONENTIAL  mean=" << base.service_mean << "s\n";
    std::cout << " Think time   : lognormal(mu=" << base.think_mu
              << ", sig=" << base.think_sigma
              << ")  mode≈2.1s\n";
    std::cout << " Timeout      : " << base.timeout_min
              << " + Exp(" << base.timeout_exp_mean << ") s\n";
    std::cout << " Ctx-switch   : Exp(" << base.ctx_sw_mean << ") s\n";
    std::cout << " Warmup       : " << base.warmup_duration << " s discarded\n";
    std::cout << "================================================\n\n";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::setw(6) << "Users"
              << std::setw(10) << "AvgRT(s)"
              << std::setw(14) << "95%CI"
              << std::setw(10) << "Goodput"
              << std::setw(9)  << "Badput"
              << std::setw(11) << "Throughput"
              << std::setw(10) << "DropRate%"
              << std::setw(10) << "CoreUtil%"
              << "\n";
    std::cout << std::string(90, '-') << "\n";

    // ------ CSV for plotting ------------------------------------
    std::ofstream csv("results.csv");
    csv << "users,avg_rt,rt_lo,rt_hi,goodput,badput,throughput,drop_rate_pct,core_util_pct\n";

    std::vector<DataPoint> all_points;
    for (int N : USER_COUNTS) {
        DataPoint dp = runExperiment(base, N, NUM_REPS);
        all_points.push_back(dp);

        std::cout << std::setw(6)  << dp.users
                  << std::setw(10) << dp.rt_ci.mean
                  << "  [" << std::setw(6) << dp.rt_ci.lo()
                  << "," << std::setw(6) << dp.rt_ci.hi() << "]"
                  << std::setw(10) << dp.goodput
                  << std::setw(9)  << dp.badput
                  << std::setw(11) << dp.throughput
                  << std::setw(9)  << dp.drop_rate * 100.0
                  << std::setw(10) << dp.core_util * 100.0
                  << "\n";

        csv << dp.users       << ","
            << dp.rt_ci.mean  << ","
            << dp.rt_ci.lo()  << ","
            << dp.rt_ci.hi()  << ","
            << dp.goodput     << ","
            << dp.badput      << ","
            << dp.throughput  << ","
            << dp.drop_rate * 100.0 << ","
            << dp.core_util  * 100.0 << "\n";
    }

    csv.close();
    std::cout << "\nResults written to results.csv\n";

    // ------ What-If: vary number of cores -----------------------
    std::cout << "\n================================================\n";
    std::cout << " What-If: Effect of Core Count (N=20 users)\n";
    std::cout << "================================================\n";
    std::ofstream csv2("results_cores.csv");
    csv2 << "cores,avg_rt,rt_lo,rt_hi,goodput,drop_rate_pct,core_util_pct\n";

    std::cout << std::setw(8) << "Cores"
              << std::setw(10) << "AvgRT(s)"
              << std::setw(14) << "95%CI"
              << std::setw(10) << "Goodput"
              << std::setw(10) << "DropRate%"
              << std::setw(10) << "CoreUtil%"
              << "\n";
    std::cout << std::string(70, '-') << "\n";

    for (int nc : {1, 2, 3, 4, 6, 8}) {
        SimParams p2 = base;
        p2.num_cores = nc;
        p2.rr_cursor = 0;
        DataPoint dp = runExperiment(p2, 20, NUM_REPS);

        std::cout << std::setw(8)  << nc
                  << std::setw(10) << dp.rt_ci.mean
                  << "  [" << std::setw(6) << dp.rt_ci.lo()
                  << "," << std::setw(6) << dp.rt_ci.hi() << "]"
                  << std::setw(10) << dp.goodput
                  << std::setw(9)  << dp.drop_rate * 100.0
                  << std::setw(10) << dp.core_util * 100.0
                  << "\n";

        csv2 << nc << "," << dp.rt_ci.mean << "," << dp.rt_ci.lo()
             << "," << dp.rt_ci.hi() << "," << dp.goodput
             << "," << dp.drop_rate * 100.0 << "," << dp.core_util * 100.0 << "\n";
    }
    csv2.close();
    std::cout << "Results written to results_cores.csv\n";

    // ------ What-If: vary service distribution ------------------
    std::cout << "\n================================================\n";
    std::cout << " What-If: Effect of Service Distribution (N=15, 4 cores)\n";
    std::cout << "================================================\n";

    struct DistCase { const char* name; DistType dt; };
    std::vector<DistCase> dist_cases = {
        {"Constant(0.10)",   DistType::CONSTANT},
        {"Uniform(0.02,0.25)",DistType::UNIFORM},
        {"Exp(mean=0.10)",   DistType::EXPONENTIAL}
    };

    for (auto& dc : dist_cases) {
        SimParams p3 = base;
        p3.service_dist  = dc.dt;
        p3.rr_cursor     = 0;
        DataPoint dp = runExperiment(p3, 15, NUM_REPS);

        std::cout << std::left << std::setw(26) << dc.name
                  << "  RT=" << std::right << std::setw(7) << dp.rt_ci.mean
                  << "  Goodput=" << std::setw(7) << dp.goodput
                  << "  Drop=" << std::setw(6) << dp.drop_rate * 100.0 << "%\n";
    }

    // ------ Verbose trace demo ----------------------------------
    if (verbose) {
        std::cout << "\n================================================\n";
        std::cout << " Short Trace Demo (5 users, 3 cores, 30s sim)\n";
        std::cout << "================================================\n";
        SimParams pv = base;
        pv.num_users      = 5;
        pv.num_cores      = 3;
        pv.sim_duration   = 30.0;
        pv.warmup_duration = 0.0;
        pv.verbose        = true;
        Simulator demo(pv, 999);
        demo.run();
        std::cout << "\n--- Trace Stats ---\n";
        const Stats& s = demo.stats;
        std::cout << "  Submitted : " << s.num_submitted << "\n";
        std::cout << "  Goodput   : " << s.num_goodput   << "\n";
        std::cout << "  Badput    : " << s.num_badput    << "\n";
        std::cout << "  Dropped   : " << s.num_dropped   << "\n";
        std::cout << "  AvgRT     : " << s.avgRT()       << " s\n";
    }

    std::cout << "\n================================================\n";
    std::cout << " Simulation complete.\n";
    std::cout << " Plot results.csv and results_cores.csv in Python/Excel.\n";
    std::cout << "================================================\n";

    return 0;
}