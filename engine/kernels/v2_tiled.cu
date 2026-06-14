#include "cuda_check.h"
#include "layout.h"
#include <cfloat>
#include <cuda_runtime.h>

namespace psearch {

static constexpr int kThreads2 = 256;
static constexpr int kWarpSize = 32;
static constexpr int kWarpsPerBlock = kThreads2 / kWarpSize;
static_assert(kDim % kWarpSize == 0, "kDim must be a multiple of warp size");

template <int K>
__device__ __forceinline__ void topk_insert(
    float (&dists)[K], int (&idx)[K], float new_d, int new_i) {
    if (new_d >= dists[K - 1]) return;
    int pos = K - 1;
    #pragma unroll
    for (int p = K - 1; p > 0; --p) {
        if (pos > 0 && dists[pos - 1] > new_d) {
            dists[pos] = dists[pos - 1];
            idx[pos]   = idx[pos - 1];
            --pos;
        }
    }
    dists[pos] = new_d;
    idx[pos]   = new_i;
}

__device__ __forceinline__ float warp_sum(float x) {

    for (int off = 16; off > 0; off >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, off);
    }
    return x;
}

__global__ void knn_v2_tiled(
    const float* __restrict__ db,
    int                       n_db,
    const float* __restrict__ queries,
    int                       n_q,
    TopKResult* __restrict__  results) {
    const int qid = blockIdx.x;
    if (qid >= n_q) return;

    const int tid  = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int wid  = tid >> 5;

    extern __shared__ unsigned char smem_raw[];
    float* s_q     = reinterpret_cast<float*>(smem_raw);
    float* s_topkd = s_q + kDim;
    int*   s_topki = reinterpret_cast<int*>(s_topkd + kWarpsPerBlock * kK);

    for (int i = tid; i < kDim; i += kThreads2) {
        s_q[i] = queries[qid * kDim + i];
    }
    __syncthreads();

    float top_d[kK];
    int   top_i[kK];
    #pragma unroll
    for (int i = 0; i < kK; ++i) { top_d[i] = FLT_MAX; top_i[i] = -1; }

    constexpr int kDimChunks = kDim / kWarpSize;

    for (int v = wid; v < n_db; v += kWarpsPerBlock) {
        const float* dv = db + v * kDim;
        float partial = 0.0f;
        #pragma unroll
        for (int c = 0; c < kDimChunks; ++c) {
            const int idx = c * kWarpSize + lane;
            float diff = s_q[idx] - dv[idx];
            partial += diff * diff;
        }

        float dist = warp_sum(partial);

        if (lane == 0) {
            topk_insert<kK>(top_d, top_i, dist, v);
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < kK; ++i) {
            s_topkd[wid * kK + i] = top_d[i];
            s_topki[wid * kK + i] = top_i[i];
        }
    }
    __syncthreads();

    if (tid == 0) {
        float final_d[kK];
        int   final_i[kK];
        #pragma unroll
        for (int i = 0; i < kK; ++i) { final_d[i] = FLT_MAX; final_i[i] = -1; }
        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            #pragma unroll
            for (int i = 0; i < kK; ++i) {
                topk_insert<kK>(final_d, final_i, s_topkd[w * kK + i], s_topki[w * kK + i]);
            }
        }
        #pragma unroll
        for (int i = 0; i < kK; ++i) {
            results[qid].distances[i] = final_d[i];
            results[qid].indices[i]   = final_i[i];
        }
    }
}

void launch_v2_tiled(const DBView& db, const QueryBatch& queries,
                     TopKResult* results, cudaStream_t stream) {
    const dim3 grid(queries.n_queries);
    const dim3 block(kThreads2);

    const size_t smem_bytes =
        kDim * sizeof(float)
      + kWarpsPerBlock * kK * (sizeof(float) + sizeof(int));
    knn_v2_tiled<<<grid, block, smem_bytes, stream>>>(
        db.vectors, db.n_vectors,
        queries.vectors, queries.n_queries,
        results);
    CUDA_LAUNCH_CHECK("knn_v2_tiled");
}

}
