#include "cuda_check.h"
#include "layout.h"
#include <cfloat>
#include <cuda_runtime.h>

namespace psearch {

static constexpr int kChunk   = 16384;
static constexpr int kThreads3 = 256;
static_assert(kChunk >= kThreads3, "Each thread must have at least 1 code");

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

__global__ void build_lut(
    const float* __restrict__ queries,
    int                       n_q,
    const float* __restrict__ codebook,
    float* __restrict__       lut) {
    const int qid = blockIdx.x;
    if (qid >= n_q) return;
    const float* q = queries + qid * kDim;
    float* lut_q = lut + qid * (kPQSub * kPQCent);
    for (int cid = threadIdx.x; cid < kPQCent; cid += blockDim.x) {
        for (int s = 0; s < kPQSub; ++s) {
            const float* qsub = q + s * kPQDim;
            const float* csub = codebook + s * kPQCent * kPQDim + cid * kPQDim;
            float d = 0.0f;
            #pragma unroll
            for (int k = 0; k < kPQDim; ++k) {
                float diff = qsub[k] - csub[k];
                d += diff * diff;
            }
            lut_q[s * kPQCent + cid] = d;
        }
    }
}

__global__ void knn_v3_pq_fat(
    const uint8_t* __restrict__ codes,
    int                         n_db,
    const float* __restrict__   lut,
    int                         n_q,
    TopKResult* __restrict__    block_topk
) {
    const int qid      = blockIdx.x;
    const int chunk_id = blockIdx.y;
    if (qid >= n_q) return;
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* s_lut = smem;
    const float* lut_q = lut + qid * (kPQSub * kPQCent);

    for (int i = tid; i < kPQSub * kPQCent; i += kThreads3) {
        s_lut[i] = lut_q[i];
    }
    __syncthreads();

    const int c_start = chunk_id * kChunk;
    const int c_end   = min(c_start + kChunk, n_db);

    float my_d[kK];
    int   my_i[kK];
    #pragma unroll
    for (int i = 0; i < kK; ++i) { my_d[i] = FLT_MAX; my_i[i] = -1; }

    for (int c = c_start + tid; c < c_end; c += kThreads3) {
        const uint8_t* code = codes + c * kPQSub;
        float dist = 0.0f;
        #pragma unroll
        for (int m = 0; m < kPQSub; ++m) {
            dist += s_lut[m * kPQCent + code[m]];
        }
        topk_insert<kK>(my_d, my_i, dist, c);
    }

    __syncthreads();
    float* r_d = smem;
    int*   r_i = reinterpret_cast<int*>(smem + kThreads3 * kK);

    #pragma unroll
    for (int i = 0; i < kK; ++i) {
        r_d[tid * kK + i] = my_d[i];
        r_i[tid * kK + i] = my_i[i];
    }
    __syncthreads();

    if (tid == 0) {
        float top_d[kK];
        int   top_i[kK];
        #pragma unroll
        for (int i = 0; i < kK; ++i) { top_d[i] = FLT_MAX; top_i[i] = -1; }
        for (int t = 0; t < kThreads3; ++t) {
            #pragma unroll
            for (int i = 0; i < kK; ++i) {
                topk_insert<kK>(top_d, top_i, r_d[t * kK + i], r_i[t * kK + i]);
            }
        }
        TopKResult* out = block_topk + qid * gridDim.y + chunk_id;
        #pragma unroll
        for (int i = 0; i < kK; ++i) {
            out->distances[i] = top_d[i];
            out->indices[i]   = top_i[i];
        }
    }
}

__global__ void reduce_topk_pq(
    const TopKResult* __restrict__ block_topk,
    int               n_chunks,
    int               n_q,
    TopKResult* __restrict__ results) {
    const int qid = blockIdx.x;
    if (qid >= n_q || threadIdx.x != 0) return;

    float top_d[kK];
    int   top_i[kK];
    #pragma unroll
    for (int i = 0; i < kK; ++i) { top_d[i] = FLT_MAX; top_i[i] = -1; }

    const TopKResult* arr = block_topk + qid * n_chunks;
    for (int c = 0; c < n_chunks; ++c) {
        #pragma unroll
        for (int i = 0; i < kK; ++i) {
            topk_insert<kK>(top_d, top_i, arr[c].distances[i], arr[c].indices[i]);
        }
    }
    #pragma unroll
    for (int i = 0; i < kK; ++i) {
        results[qid].distances[i] = top_d[i];
        results[qid].indices[i]   = top_i[i];
    }
}

void launch_v3_pq(const DBPQView& db, const QueryBatch& queries,
                  TopKResult* results, cudaStream_t stream) {
    const int n_q  = queries.n_queries;
    const int n_db = db.n_vectors;
    const int n_chunks = (n_db + kChunk - 1) / kChunk;

    float* lut = nullptr;
    CUDA_CHECK(cudaMallocAsync(&lut,
                    static_cast<size_t>(n_q) * kPQSub * kPQCent * sizeof(float),
                    stream));

    TopKResult* block_topk = nullptr;
    CUDA_CHECK(cudaMallocAsync(&block_topk,
                    static_cast<size_t>(n_q) * n_chunks * sizeof(TopKResult),
                    stream));

    build_lut<<<n_q, kPQCent, 0, stream>>>(
        queries.vectors, n_q, db.codebook, lut);
    CUDA_LAUNCH_CHECK("build_lut");

    const size_t smem_lut    = static_cast<size_t>(kPQSub * kPQCent) * sizeof(float);
    const size_t smem_reduce = static_cast<size_t>(kThreads3 * kK)
                             * (sizeof(float) + sizeof(int));
    const size_t smem_bytes  = smem_lut > smem_reduce ? smem_lut : smem_reduce;
    dim3 grid_score(n_q, n_chunks);
    knn_v3_pq_fat<<<grid_score, kThreads3, smem_bytes, stream>>>(
        db.codes, n_db, lut, n_q, block_topk);
    CUDA_LAUNCH_CHECK("knn_v3_pq_fat");

    reduce_topk_pq<<<n_q, 32, 0, stream>>>(block_topk, n_chunks, n_q, results);
    CUDA_LAUNCH_CHECK("reduce_topk_pq");

    CUDA_CHECK(cudaFreeAsync(lut, stream));
    CUDA_CHECK(cudaFreeAsync(block_topk, stream));
}

}
