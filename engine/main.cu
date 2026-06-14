#include "cuda_check.h"
#include "layout.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <vector>

using namespace psearch;

struct FloatDB {
    int n = 0, d = 0;
    std::vector<float> data;
};

static FloatDB load_fp32_db(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "[fatal] cannot open %s\n", path.c_str()); std::exit(1); }
    FloatDB out;
    f.read(reinterpret_cast<char*>(&out.n), 4);
    f.read(reinterpret_cast<char*>(&out.d), 4);
    out.data.resize(static_cast<size_t>(out.n) * out.d);
    f.read(reinterpret_cast<char*>(out.data.data()),
           static_cast<std::streamsize>(out.data.size() * sizeof(float)));
    if (!f) { std::fprintf(stderr, "[fatal] short read on %s\n", path.c_str()); std::exit(1); }
    return out;
}

struct PQDB {
    int n_db = 0, sub = 0, sub_dim = 0, n_cent = 0;
    std::vector<float>   codebook;
    std::vector<uint8_t> codes;
};

static PQDB load_pq_db(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "[fatal] cannot open %s\n", path.c_str()); std::exit(1); }
    PQDB out;
    f.read(reinterpret_cast<char*>(&out.n_db),    4);
    f.read(reinterpret_cast<char*>(&out.sub),     4);
    f.read(reinterpret_cast<char*>(&out.sub_dim), 4);
    f.read(reinterpret_cast<char*>(&out.n_cent),  4);
    out.codebook.resize(static_cast<size_t>(out.sub) * out.n_cent * out.sub_dim);
    out.codes.resize(static_cast<size_t>(out.n_db) * out.sub);
    f.read(reinterpret_cast<char*>(out.codebook.data()),
           static_cast<std::streamsize>(out.codebook.size() * sizeof(float)));
    f.read(reinterpret_cast<char*>(out.codes.data()),
           static_cast<std::streamsize>(out.codes.size() * sizeof(uint8_t)));
    if (!f) { std::fprintf(stderr, "[fatal] short read on %s\n", path.c_str()); std::exit(1); }
    return out;
}

struct Args {
    std::string db_path;
    std::string pq_path;
    std::string queries_path;
    std::string out_path;
    std::string version = "v1";
    int  n_warmup = 3;
    int  n_runs   = 10;
    int  default_queries = 100;
    bool bench  = false;
    bool hello  = false;
};

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "[fatal] %s needs a value\n", s.c_str()); std::exit(2); }
            return std::string(argv[++i]);
        };
        if      (s == "--hello")    a.hello = true;
        else if (s == "--bench")    a.bench = true;
        else if (s == "--db")       a.db_path = next();
        else if (s == "--pq")       a.pq_path = next();
        else if (s == "--queries")  a.queries_path = next();
        else if (s == "--out")      a.out_path = next();
        else if (s == "--version")  a.version = next();
        else if (s == "--warmup")   a.n_warmup = std::atoi(next().c_str());
        else if (s == "--runs")     a.n_runs   = std::atoi(next().c_str());
        else if (s == "--n-queries") a.default_queries = std::atoi(next().c_str());
        else { std::fprintf(stderr, "[fatal] unknown arg: %s\n", s.c_str()); std::exit(2); }
    }
    return a;
}

static int hello() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("Hello from CUDA on %s (sm_%d%d, %zu MB)\n",
                prop.name, prop.major, prop.minor, prop.totalGlobalMem >> 20);
    return 0;
}

static void run_version(const std::string& version,
                        const DBView& dv, const DBPQView* pv,
                        const QueryBatch& qb,
                        TopKResult* d_results,
                        cudaStream_t stream) {
    if (version == "v1") {
        launch_v1_naive(dv, qb, d_results, stream);
    } else if (version == "v2") {
        launch_v2_tiled(dv, qb, d_results, stream);
    } else if (version == "v3") {
        if (!pv) { std::fprintf(stderr, "[fatal] v3 needs --pq <path>\n"); std::exit(2); }
        launch_v3_pq(*pv, qb, d_results, stream);
    } else {
        std::fprintf(stderr, "[fatal] unknown version: %s\n", version.c_str()); std::exit(2);
    }
}

int main(int argc, char** argv) {
    Args args = parse(argc, argv);
    if (args.hello) return hello();
    if (args.db_path.empty()) {
        std::fprintf(stderr, "[fatal] missing --db (or use --hello)\n");
        return 2;
    }

    std::printf("[load] db=%s\n", args.db_path.c_str());
    FloatDB db = load_fp32_db(args.db_path);
    if (db.d != kDim) {
        std::fprintf(stderr, "[fatal] db dim %d != kDim %d\n", db.d, kDim);
        return 1;
    }
    std::printf("       n=%d dim=%d (%.2f MB)\n", db.n, db.d,
                static_cast<double>(db.data.size() * sizeof(float)) / 1e6);

    PQDB pq;
    bool has_pq = !args.pq_path.empty();
    if (has_pq) {
        std::printf("[load] pq=%s\n", args.pq_path.c_str());
        pq = load_pq_db(args.pq_path);
        if (pq.sub != kPQSub || pq.n_cent != kPQCent || pq.sub_dim != kPQDim) {
            std::fprintf(stderr, "[fatal] PQ shape mismatch\n");
            return 1;
        }
    }

    FloatDB queries;
    if (!args.queries_path.empty()) {
        queries = load_fp32_db(args.queries_path);
        if (queries.d != kDim) {
            std::fprintf(stderr, "[fatal] queries dim %d != kDim %d\n", queries.d, kDim);
            return 1;
        }
    } else {
        std::printf("[load] no --queries given, sampling %d from the db\n",
                    args.default_queries);
        const int n_q = std::min(args.default_queries, db.n);
        queries.n = n_q;
        queries.d = kDim;
        queries.data.resize(static_cast<size_t>(n_q) * kDim);
        const int step = std::max(1, db.n / n_q);
        for (int i = 0; i < n_q; ++i) {
            const int src = (i * step) % db.n;
            std::memcpy(queries.data.data() + static_cast<size_t>(i) * kDim,
                        db.data.data() + static_cast<size_t>(src) * kDim,
                        kDim * sizeof(float));
        }
    }
    std::printf("       n_queries=%d\n", queries.n);

    float* d_db = nullptr;
    float* d_q  = nullptr;
    TopKResult* d_results = nullptr;
    uint8_t* d_codes = nullptr;
    float*   d_codebook = nullptr;

    CUDA_CHECK(cudaMalloc(&d_db, db.data.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_db, db.data.data(), db.data.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_q, queries.data.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_q, queries.data.data(), queries.data.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_results, queries.n * sizeof(TopKResult)));

    if (has_pq) {
        CUDA_CHECK(cudaMalloc(&d_codes, pq.codes.size()));
        CUDA_CHECK(cudaMemcpy(d_codes, pq.codes.data(), pq.codes.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_codebook, pq.codebook.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_codebook, pq.codebook.data(),
                              pq.codebook.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    DBView dv{ d_db, db.n };
    QueryBatch qb{ d_q, queries.n };
    DBPQView pv{ d_codes, d_codebook, pq.n_db };

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (int i = 0; i < args.n_warmup; ++i) {
        run_version(args.version, dv, has_pq ? &pv : nullptr, qb, d_results, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (args.bench) {
        cudaEvent_t evt_start, evt_stop;
        CUDA_CHECK(cudaEventCreate(&evt_start));
        CUDA_CHECK(cudaEventCreate(&evt_stop));

        std::vector<double> times_ms;
        times_ms.reserve(args.n_runs);
        for (int r = 0; r < args.n_runs; ++r) {
            CUDA_CHECK(cudaEventRecord(evt_start, stream));
            run_version(args.version, dv, has_pq ? &pv : nullptr, qb, d_results, stream);
            CUDA_CHECK(cudaEventRecord(evt_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(evt_stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, evt_start, evt_stop));
            times_ms.push_back(static_cast<double>(ms));
        }
        std::sort(times_ms.begin(), times_ms.end());
        const double median = times_ms[times_ms.size() / 2];
        const double p99    = times_ms[(times_ms.size() * 99) / 100];
        const double qps    = 1000.0 * queries.n / median;

        std::printf("[bench] version=%s  n_q=%d  n_db=%d\n",
                    args.version.c_str(), queries.n, db.n);
        std::printf("        median = %.3f ms   p99 = %.3f ms\n", median, p99);
        std::printf("        QPS    = %.1f   (q/s, batch of %d)\n", qps, queries.n);
        std::printf("        per-query lat = %.3f ms\n", median / queries.n);

        CUDA_CHECK(cudaEventDestroy(evt_start));
        CUDA_CHECK(cudaEventDestroy(evt_stop));
    } else {
        run_version(args.version, dv, has_pq ? &pv : nullptr, qb, d_results, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::printf("[run] single run complete.\n");
    }

    if (!args.out_path.empty()) {
        std::vector<TopKResult> h_results(queries.n);
        CUDA_CHECK(cudaMemcpy(h_results.data(), d_results,
                              queries.n * sizeof(TopKResult), cudaMemcpyDeviceToHost));
        std::ofstream f(args.out_path, std::ios::binary);
        const int n_q = queries.n;
        const int k = kK;
        f.write(reinterpret_cast<const char*>(&n_q), 4);
        f.write(reinterpret_cast<const char*>(&k),   4);
        f.write(reinterpret_cast<const char*>(h_results.data()),
                static_cast<std::streamsize>(queries.n * sizeof(TopKResult)));
        std::printf("[out] wrote %s\n", args.out_path.c_str());
    }

    cudaFree(d_db); cudaFree(d_q); cudaFree(d_results);
    if (has_pq) { cudaFree(d_codes); cudaFree(d_codebook); }
    cudaStreamDestroy(stream);
    return 0;
}
