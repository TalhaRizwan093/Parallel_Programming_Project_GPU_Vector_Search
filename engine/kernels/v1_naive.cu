#include "cuda_check.h"
#include "layout.h"
#include <cfloat>
#include <cuda_runtime.h>

namespace psearch {

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

__global__ void knn_v1_naive(
    const float* __restrict__ db,
    int                       n_db,
    const float* __restrict__ queries,
    int                       n_q,
    TopKResult* __restrict__  results) {
    const int qid = blockIdx.x;
    if (qid >= n_q) return;

    const float* q = queries + qid * kDim;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    float my_d[kK];
    int   my_i[kK];
    #pragma unroll
    for (int i = 0; i < kK; ++i) { my_d[i] = FLT_MAX; my_i[i] = -1; }

    static_assert(kDim % 4 == 0, "kDim must be divisible by 4 for float4 loads");
    const float4* q4 = reinterpret_cast<const float4*>(q);

    for (int dbi = tid; dbi < n_db; dbi += nth) {
        const float4* v4 = reinterpret_cast<const float4*>(db + dbi * kDim);
        float dist = 0.0f;
        #pragma unroll
        for (int k = 0; k < kDim / 4; ++k) {
            float4 a = q4[k];
            float4 b = v4[k];
            float dx = a.x - b.x;
            float dy = a.y - b.y;
            float dz = a.z - b.z;
            float dw = a.w - b.w;
            dist += dx*dx + dy*dy + dz*dz + dw*dw;
        }
        topk_insert<kK>(my_d, my_i, dist, dbi);
    }

    extern __shared__ unsigned char smem_raw[];
    float* s_d = reinterpret_cast<float*>(smem_raw);
    int*   s_i = reinterpret_cast<int*>(s_d + nth * kK);

    #pragma unroll
    for (int i = 0; i < kK; ++i) {
        s_d[tid * kK + i] = my_d[i];
        s_i[tid * kK + i] = my_i[i];
    }
    __syncthreads();

    if (tid == 0) {
        float final_d[kK];
        int   final_i[kK];
        #pragma unroll
        for (int i = 0; i < kK; ++i) { final_d[i] = FLT_MAX; final_i[i] = -1; }

        for (int t = 0; t < nth; ++t) {
            #pragma unroll
            for (int i = 0; i < kK; ++i) {
                topk_insert<kK>(final_d, final_i, s_d[t * kK + i], s_i[t * kK + i]);
            }
        }
        #pragma unroll
        for (int i = 0; i < kK; ++i) {
            results[qid].distances[i] = final_d[i];
            results[qid].indices[i]   = final_i[i];
        }
    }
}

void launch_v1_naive(const DBView& db, const QueryBatch& queries,
                     TopKResult* results, cudaStream_t stream) {
    const int kThreads = 256;
    const dim3 grid(queries.n_queries);
    const dim3 block(kThreads);
    const size_t smem = kThreads * kK * (sizeof(float) + sizeof(int));
    knn_v1_naive<<<grid, block, smem, stream>>>(
        db.vectors, db.n_vectors,
        queries.vectors, queries.n_queries,
        results);
    CUDA_LAUNCH_CHECK("knn_v1_naive");
}

}
