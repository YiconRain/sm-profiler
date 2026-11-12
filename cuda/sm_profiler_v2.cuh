/* Version 2 CUDA recorder. One lane owns each stream; no recorder barriers. */
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace sm_profiler_v2 {
constexpr uint32_t schema_version = 2;
struct Record {
  uint64_t start_ns, end_ns, payload0, payload1;
  uint32_t phase, site, block, group, sm, kind, reserved0, reserved1;
};
static_assert(sizeof(Record) == 64);
struct Context {
  Record* records = nullptr;
  uint32_t* attempted = nullptr;
  uint32_t blocks = 0, groups = 0, capacity = 0;
};
__device__ __forceinline__ uint64_t timestamp() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
  return value;
}
struct Range {
  Record* record;
  __device__ __forceinline__ Range(Context context, uint32_t phase, uint32_t site,
                                  uint64_t payload0, uint64_t payload1, bool selected,
                                  uint32_t kind = 0) : record(nullptr) {
    if (!selected || !context.records || !context.attempted || !context.capacity || kind > 1) return;
    size_t linear_block = (size_t(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    if (linear_block >= context.blocks) return;
    uint32_t block = static_cast<uint32_t>(linear_block);
    uint32_t thread = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    uint32_t group = thread / 32;
    if (block >= context.blocks || group >= context.groups) return;
    size_t track = size_t(block) * context.groups + group;
    uint32_t index = context.attempted[track];
    // Saturate so an exhausted counter cannot wrap and overwrite earlier records.
    if (index == UINT32_MAX) return;
    context.attempted[track] = index + 1;
    if (index >= context.capacity) return;
    uint32_t sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    record = context.records + size_t(track) * context.capacity + index;
    *record = {0, 0, payload0, payload1, phase, site, block, group, sm, kind, 0, 0};
    record->start_ns = timestamp();
    if (kind) { record->end_ns = record->start_ns; record = nullptr; }
  }
  Range(Range const&) = delete;
  Range& operator=(Range const&) = delete;
  __device__ __forceinline__ void finish() {
    if (record) { record->end_ns = timestamp(); record = nullptr; }
  }
  __device__ __forceinline__ ~Range() { finish(); }
};
}  // namespace sm_profiler_v2
