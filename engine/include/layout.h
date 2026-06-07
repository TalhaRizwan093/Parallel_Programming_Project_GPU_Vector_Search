#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace psearch {

static constexpr int kDim     = 384;
static constexpr int kK       = 10;
static constexpr int kPQSub   = 32;
static constexpr int kPQCent  = 256;
static constexpr int kPQDim   = kDim / kPQSub;

static_assert(kDim % kPQSub == 0, "kDim must be divisible by kPQSub");

struct DBView {
    const float* vectors;
    int          n_vectors;
};

struct DBPQView {
    const uint8_t* codes;
    const float*   codebook;
    int            n_vectors;
};

struct QueryBatch {
    const float* vectors;
    int          n_queries;
};

struct TopKResult {
    int32_t indices[kK];
    float   distances[kK];
};

void launch_v1_naive(const DBView& db, const QueryBatch& queries,
                     TopKResult* results, cudaStream_t stream = 0);

void launch_v2_tiled(const DBView& db, const QueryBatch& queries,
                     TopKResult* results, cudaStream_t stream = 0);

void launch_v3_pq(const DBPQView& db, const QueryBatch& queries,
                  TopKResult* results, cudaStream_t stream = 0);

}
