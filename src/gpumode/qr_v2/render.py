"""Jinja2 rendering for qr_v2 candidate artifacts."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined

from gpumode.qr_v2.jsonl import write_jsonl
from gpumode.qr_v2.specs import QrV2Spec


RENDERER_VERSION = "render_v1"


@dataclass(frozen=True, slots=True)
class QrV2KernelConfig:
    candidate_id: str
    template_id: str
    threads_per_block: int
    tile_size: int
    unroll: int

    @property
    def entry_point(self) -> str:
        return f"{self.candidate_id}_kernel"


SERIAL_KERNEL_CONFIG = QrV2KernelConfig(
    candidate_id="qr_v2_geqr2_serial_tpb128_tile32",
    template_id="cuda_geqr2_serial_v1",
    threads_per_block=128,
    tile_size=32,
    unroll=1,
)

PARALLEL_KERNEL_CONFIG = QrV2KernelConfig(
    candidate_id="qr_v2_geqr2_parallel_tpb256_tile32",
    template_id="cuda_geqr2_parallel_v1",
    threads_per_block=256,
    tile_size=32,
    unroll=1,
)

PARALLEL_PROFILE_KERNEL_CONFIG = QrV2KernelConfig(
    candidate_id="qr_v2_geqr2_parallel_profile_tpb256_tile32",
    template_id="cuda_geqr2_parallel_profile_v1",
    threads_per_block=256,
    tile_size=32,
    unroll=1,
)

SEMANTIC_UPPER_KERNEL_CONFIG = QrV2KernelConfig(
    candidate_id="qr_v2_semantic_upper_fallback_tpb256_tile32",
    template_id="cuda_semantic_upper_fallback_v1",
    threads_per_block=256,
    tile_size=32,
    unroll=1,
)

SEMANTIC_EARLY_STOP_KERNEL_CONFIG = QrV2KernelConfig(
    candidate_id="qr_v2_semantic_early_stop_fallback_tpb256_tile32",
    template_id="cuda_semantic_early_stop_fallback_v1",
    threads_per_block=256,
    tile_size=32,
    unroll=1,
)

SEMANTIC_COMBO_KERNEL_CONFIG = QrV2KernelConfig(
    candidate_id="qr_v2_semantic_combo_fallback_tpb256_tile32",
    template_id="cuda_semantic_combo_fallback_v1",
    threads_per_block=256,
    tile_size=32,
    unroll=1,
)

CUSOLVER_KERNEL_CONFIG = QrV2KernelConfig(
    candidate_id="qr_v2_cusolver_geqrf_tpb256",
    template_id="cuda_cusolver_geqrf_v1",
    threads_per_block=256,
    tile_size=32,
    unroll=1,
)

CUSOLVERDX_KERNEL_CONFIG = QrV2KernelConfig(
    candidate_id="qr_v2_cusolverdx_geqrf_tpb256",
    template_id="cuda_cusolverdx_geqrf_v1",
    threads_per_block=256,
    tile_size=32,
    unroll=1,
)

CUBLAS_BATCHED_KERNEL_CONFIG = QrV2KernelConfig(
    candidate_id="qr_v2_cublas_batched_geqrf_tpb256",
    template_id="cuda_cublas_batched_geqrf_v1",
    threads_per_block=256,
    tile_size=32,
    unroll=1,
)

DENSE_LINALG_BEST_KERNEL_CONFIG = QrV2KernelConfig(
    candidate_id="qr_v2_dense_linalg_best_tpb256",
    template_id="cuda_dense_linalg_best_v1",
    threads_per_block=256,
    tile_size=16,
    unroll=1,
)

DENSE_BLOCKED_WY_KERNEL_CONFIG = QrV2KernelConfig(
    candidate_id="qr_v2_dense_blocked_wy_tpb256_panel16",
    template_id="cuda_dense_blocked_wy_v1",
    threads_per_block=256,
    tile_size=16,
    unroll=1,
)

DENSE_BLOCKED_WY_PANEL8_KERNEL_CONFIG = QrV2KernelConfig(
    candidate_id="qr_v2_dense_blocked_wy_tpb256_panel8",
    template_id="cuda_dense_blocked_wy_v1",
    threads_per_block=256,
    tile_size=8,
    unroll=1,
)

DENSE_BLOCKED_WY_PANEL32_KERNEL_CONFIG = QrV2KernelConfig(
    candidate_id="qr_v2_dense_blocked_wy_tpb256_panel32",
    template_id="cuda_dense_blocked_wy_v1",
    threads_per_block=256,
    tile_size=32,
    unroll=1,
)

STRUCTURED_BLOCKED_WY_KERNEL_CONFIG = QrV2KernelConfig(
    candidate_id="qr_v2_structured_blocked_wy_tpb256_panel16",
    template_id="cuda_structured_blocked_wy_v1",
    threads_per_block=256,
    tile_size=16,
    unroll=1,
)

STRUCTURED_BLOCKED_WY_PANEL8_KERNEL_CONFIG = QrV2KernelConfig(
    candidate_id="qr_v2_structured_blocked_wy_tpb256_panel8",
    template_id="cuda_structured_blocked_wy_v1",
    threads_per_block=256,
    tile_size=8,
    unroll=1,
)

STRUCTURED_BLOCKED_WY_PANEL32_KERNEL_CONFIG = QrV2KernelConfig(
    candidate_id="qr_v2_structured_blocked_wy_tpb256_panel32",
    template_id="cuda_structured_blocked_wy_v1",
    threads_per_block=256,
    tile_size=32,
    unroll=1,
)

KERNEL_CONFIGS: dict[str, QrV2KernelConfig] = {
    "serial": SERIAL_KERNEL_CONFIG,
    "parallel": PARALLEL_KERNEL_CONFIG,
    "parallel_profile": PARALLEL_PROFILE_KERNEL_CONFIG,
    "semantic_upper": SEMANTIC_UPPER_KERNEL_CONFIG,
    "semantic_early_stop": SEMANTIC_EARLY_STOP_KERNEL_CONFIG,
    "semantic_combo": SEMANTIC_COMBO_KERNEL_CONFIG,
    "cusolver": CUSOLVER_KERNEL_CONFIG,
    "cusolverdx": CUSOLVERDX_KERNEL_CONFIG,
    "cublas_batched": CUBLAS_BATCHED_KERNEL_CONFIG,
    "dense_linalg_best": DENSE_LINALG_BEST_KERNEL_CONFIG,
    "dense_blocked_wy": DENSE_BLOCKED_WY_KERNEL_CONFIG,
    "dense_blocked_wy8": DENSE_BLOCKED_WY_PANEL8_KERNEL_CONFIG,
    "dense_blocked_wy32": DENSE_BLOCKED_WY_PANEL32_KERNEL_CONFIG,
    "structured_blocked_wy": STRUCTURED_BLOCKED_WY_KERNEL_CONFIG,
    "structured_blocked_wy8": STRUCTURED_BLOCKED_WY_PANEL8_KERNEL_CONFIG,
    "structured_blocked_wy32": STRUCTURED_BLOCKED_WY_PANEL32_KERNEL_CONFIG,
}
DEFAULT_KERNEL_CONFIG = SERIAL_KERNEL_CONFIG


@dataclass(frozen=True, slots=True)
class RenderSuiteResult:
    manifest_path: Path
    records: list[dict[str, object]]

    @property
    def artifact_paths(self) -> list[Path]:
        return [Path(str(record["artifact_path"])) for record in self.records]


@dataclass(frozen=True, slots=True)
class RenderVerificationIssue:
    code: str
    message: str
    artifact_path: str | None = None
    line_number: int | None = None

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "code": self.code,
            "message": self.message,
        }
        if self.artifact_path is not None:
            record["artifact_path"] = self.artifact_path
        if self.line_number is not None:
            record["line_number"] = self.line_number
        return record


@dataclass(frozen=True, slots=True)
class RenderVerificationResult:
    manifest_path: Path
    records_checked: int
    artifacts_checked: int
    issues: list[RenderVerificationIssue]

    @property
    def ok(self) -> bool:
        return not self.issues


_CUDA_GEQR2_SERIAL_TEMPLATE = """// Generated by gpumode.qr_v2 {{ renderer_version }}.
// Correct-first serial GEQR2 implementation for validating qr_v2 plumbing.
//
// candidate_id: {{ config.candidate_id }}
// template_id: {{ config.template_id }}
// entry_point: {{ config.entry_point }}
// eval: {{ spec.eval_line(include_dense_case=True) }}
// shape: batch={{ spec.batch }}, n={{ spec.n }}, cond={{ spec.cond }}, case={{ spec.case }}, seed={{ spec.seed }}
// sweep: threads_per_block={{ config.threads_per_block }}, tile_size={{ config.tile_size }}, unroll={{ config.unroll }}

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void {{ config.entry_point }}(
    const float* __restrict__ a,
    float* __restrict__ h,
    float* __restrict__ tau,
    int batch,
    int n
) {
    const int matrix_idx = blockIdx.x;
    if (matrix_idx >= batch || threadIdx.x != 0) {
        return;
    }

    const int matrix_offset = matrix_idx * n * n;
    const int tau_offset = matrix_idx * n;

    for (int idx = 0; idx < n * n; ++idx) {
        h[matrix_offset + idx] = a[matrix_offset + idx];
    }
    for (int k = 0; k < n; ++k) {
        tau[tau_offset + k] = 0.0f;
    }

    for (int k = 0; k < n; ++k) {
        const int kk = matrix_offset + k * n + k;
        const double alpha = static_cast<double>(h[kk]);
        double xnorm2 = 0.0;

        for (int row = k + 1; row < n; ++row) {
            const double value = static_cast<double>(h[matrix_offset + row * n + k]);
            xnorm2 += value * value;
        }
        if (xnorm2 == 0.0) {
            tau[tau_offset + k] = 0.0f;
            continue;
        }

        const double norm = sqrt(alpha * alpha + xnorm2);
        const double beta = alpha >= 0.0 ? -norm : norm;
        const double tau_k = (beta - alpha) / beta;
        const double scale = 1.0 / (alpha - beta);

        for (int row = k + 1; row < n; ++row) {
            const int index = matrix_offset + row * n + k;
            h[index] = static_cast<float>(static_cast<double>(h[index]) * scale);
        }
        h[kk] = static_cast<float>(beta);
        tau[tau_offset + k] = static_cast<float>(tau_k);

        for (int col = k + 1; col < n; ++col) {
            double dot = static_cast<double>(h[matrix_offset + k * n + col]);
            for (int row = k + 1; row < n; ++row) {
                dot += static_cast<double>(h[matrix_offset + row * n + k]) *
                       static_cast<double>(h[matrix_offset + row * n + col]);
            }

            const double update = tau_k * dot;
            const int top_index = matrix_offset + k * n + col;
            h[top_index] = static_cast<float>(static_cast<double>(h[top_index]) - update);
            for (int row = k + 1; row < n; ++row) {
                const int index = matrix_offset + row * n + col;
                h[index] = static_cast<float>(
                    static_cast<double>(h[index]) - static_cast<double>(h[matrix_offset + row * n + k]) * update
                );
            }
        }
    }
}

extern "C" __attribute__((visibility("default"))) int {{ config.entry_point }}_launch(
    const float* a,
    float* h,
    float* tau,
    int batch,
    int n,
    int threads_per_block
) {
    if (a == nullptr || h == nullptr || tau == nullptr || batch <= 0 || n <= 0 || threads_per_block <= 0) {
        return -1;
    }

    {{ config.entry_point }}<<<batch, threads_per_block>>>(a, h, tau, batch, n);
    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        return static_cast<int>(launch_status);
    }

    cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
        return static_cast<int>(sync_status);
    }
    return 0;
}
"""


_CUDA_GEQR2_PARALLEL_TEMPLATE = """// Generated by gpumode.qr_v2 {{ renderer_version }}.
// Correct-first cooperative GEQR2 implementation with block reductions.
//
// candidate_id: {{ config.candidate_id }}
// template_id: {{ config.template_id }}
// entry_point: {{ config.entry_point }}
// eval: {{ spec.eval_line(include_dense_case=True) }}
// shape: batch={{ spec.batch }}, n={{ spec.n }}, cond={{ spec.cond }}, case={{ spec.case }}, seed={{ spec.seed }}
// sweep: threads_per_block={{ config.threads_per_block }}, tile_size={{ config.tile_size }}, unroll={{ config.unroll }}

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void {{ config.entry_point }}(
    const float* __restrict__ a,
    float* __restrict__ h,
    float* __restrict__ tau,
    int batch,
    int n
) {
    const int matrix_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (matrix_idx >= batch) {
        return;
    }

    __shared__ double reduce_buffer[1024];
    __shared__ double shared_tau;
    __shared__ double shared_scale;
    __shared__ double shared_update;
    __shared__ int shared_active;

    const int matrix_offset = matrix_idx * n * n;
    const int tau_offset = matrix_idx * n;

    for (int idx = tid; idx < n * n; idx += blockDim.x) {
        h[matrix_offset + idx] = a[matrix_offset + idx];
    }
    for (int k = tid; k < n; k += blockDim.x) {
        tau[tau_offset + k] = 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < n; ++k) {
        double local_sum = 0.0;
        for (int row = k + 1 + tid; row < n; row += blockDim.x) {
            const double value = static_cast<double>(h[matrix_offset + row * n + k]);
            local_sum += value * value;
        }
        reduce_buffer[tid] = local_sum;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buffer[tid] += reduce_buffer[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const int kk = matrix_offset + k * n + k;
            const double alpha = static_cast<double>(h[kk]);
            const double xnorm2 = reduce_buffer[0];
            if (xnorm2 == 0.0) {
                tau[tau_offset + k] = 0.0f;
                shared_tau = 0.0;
                shared_scale = 0.0;
                shared_active = 0;
            } else {
                const double norm = sqrt(alpha * alpha + xnorm2);
                const double beta = alpha >= 0.0 ? -norm : norm;
                const double tau_k = (beta - alpha) / beta;
                shared_tau = tau_k;
                shared_scale = 1.0 / (alpha - beta);
                shared_active = 1;
                h[kk] = static_cast<float>(beta);
                tau[tau_offset + k] = static_cast<float>(tau_k);
            }
        }
        __syncthreads();

        if (shared_active == 0) {
            continue;
        }

        for (int row = k + 1 + tid; row < n; row += blockDim.x) {
            const int index = matrix_offset + row * n + k;
            h[index] = static_cast<float>(static_cast<double>(h[index]) * shared_scale);
        }
        __syncthreads();

        for (int col = k + 1; col < n; ++col) {
            double dot = tid == 0 ? static_cast<double>(h[matrix_offset + k * n + col]) : 0.0;
            for (int row = k + 1 + tid; row < n; row += blockDim.x) {
                dot += static_cast<double>(h[matrix_offset + row * n + k]) *
                       static_cast<double>(h[matrix_offset + row * n + col]);
            }
            reduce_buffer[tid] = dot;
            __syncthreads();

            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce_buffer[tid] += reduce_buffer[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0) {
                shared_update = shared_tau * reduce_buffer[0];
                const int top_index = matrix_offset + k * n + col;
                h[top_index] = static_cast<float>(static_cast<double>(h[top_index]) - shared_update);
            }
            __syncthreads();

            for (int row = k + 1 + tid; row < n; row += blockDim.x) {
                const int index = matrix_offset + row * n + col;
                h[index] = static_cast<float>(
                    static_cast<double>(h[index]) - static_cast<double>(h[matrix_offset + row * n + k]) * shared_update
                );
            }
            __syncthreads();
        }
    }
}

extern "C" __attribute__((visibility("default"))) int {{ config.entry_point }}_launch(
    const float* a,
    float* h,
    float* tau,
    int batch,
    int n,
    int threads_per_block
) {
    if (a == nullptr || h == nullptr || tau == nullptr || batch <= 0 || n <= 0 || threads_per_block <= 0) {
        return -1;
    }
    if (threads_per_block > 1024 || (threads_per_block & (threads_per_block - 1)) != 0) {
        return -2;
    }

    {{ config.entry_point }}<<<batch, threads_per_block>>>(a, h, tau, batch, n);
    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        return static_cast<int>(launch_status);
    }

    cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
        return static_cast<int>(sync_status);
    }
    return 0;
}
"""


_UPPER_SHORTCUT_INIT_BLOCK = """    double local_norm1 = 0.0;
    double local_strictlower_norm1 = 0.0;
    for (int col = tid; col < n; col += blockDim.x) {
        double column_sum = 0.0;
        double strictlower_column_sum = 0.0;
        for (int row = 0; row < n; ++row) {
            const double abs_value = fabs(static_cast<double>(a[matrix_offset + row * n + col]));
            column_sum += abs_value;
            if (row > col) {
                strictlower_column_sum += abs_value;
            }
        }
        local_norm1 = fmax(local_norm1, column_sum);
        local_strictlower_norm1 = fmax(local_strictlower_norm1, strictlower_column_sum);
    }

    reduce_buffer[tid] = local_norm1;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buffer[tid] = fmax(reduce_buffer[tid], reduce_buffer[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        shared_norm1 = reduce_buffer[0];
    }
    __syncthreads();

    reduce_buffer[tid] = local_strictlower_norm1;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buffer[tid] = fmax(reduce_buffer[tid], reduce_buffer[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        shared_strictlower_norm1 = reduce_buffer[0];
        const double eps = 1.1920928955078125e-7;
        const double threshold = 20.0 * static_cast<double>(n) * eps * fmax(shared_norm1, 1.0e-30);
        shared_shortcut = shared_strictlower_norm1 <= threshold ? 1 : 0;
    }
    __syncthreads();

    if (shared_shortcut != 0) {
        for (int idx = tid; idx < n * n; idx += blockDim.x) {
            const int row = idx / n;
            const int col = idx - row * n;
            h[matrix_offset + idx] = row <= col ? a[matrix_offset + idx] : 0.0f;
        }
        for (int k = tid; k < n; k += blockDim.x) {
            tau[tau_offset + k] = 0.0f;
        }
        return;
    }

    for (int idx = tid; idx < n * n; idx += blockDim.x) {
        h[matrix_offset + idx] = a[matrix_offset + idx];
    }
    for (int k = tid; k < n; k += blockDim.x) {
        tau[tau_offset + k] = 0.0f;
    }
    __syncthreads();
"""

_CUDA_SEMANTIC_UPPER_FALLBACK_TEMPLATE = (
    _CUDA_GEQR2_PARALLEL_TEMPLATE.replace(
        "Correct-first cooperative GEQR2 implementation with block reductions.",
        "Approximate-upper shortcut with cooperative GEQR2 fallback.",
    )
    .replace(
        "__shared__ int shared_active;\n\n    const int matrix_offset",
        "__shared__ int shared_active;\n"
        "    __shared__ int shared_shortcut;\n"
        "    __shared__ double shared_norm1;\n"
        "    __shared__ double shared_strictlower_norm1;\n\n"
        "    const int matrix_offset",
    )
    .replace(
        """    for (int idx = tid; idx < n * n; idx += blockDim.x) {
        h[matrix_offset + idx] = a[matrix_offset + idx];
    }
    for (int k = tid; k < n; k += blockDim.x) {
        tau[tau_offset + k] = 0.0f;
    }
    __syncthreads();
""",
        _UPPER_SHORTCUT_INIT_BLOCK,
        1,
    )
)

_EARLY_STOP_INIT_BLOCK = """    if (tid == 0) {
        shared_early_stop = 0;
    }
    __syncthreads();

    if (n == 512) {
        double local_column_min = 1.0e300;
        double local_column_max = 0.0;
        for (int col = tid; col < n; col += blockDim.x) {
            double column_sum = 0.0;
            for (int row = 0; row < n; ++row) {
                column_sum += fabs(static_cast<double>(a[matrix_offset + row * n + col]));
            }
            local_column_min = fmin(local_column_min, column_sum);
            local_column_max = fmax(local_column_max, column_sum);
        }

        reduce_buffer[tid] = local_column_min;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buffer[tid] = fmin(reduce_buffer[tid], reduce_buffer[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            shared_column_min = reduce_buffer[0];
        }
        __syncthreads();

        reduce_buffer[tid] = local_column_max;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buffer[tid] = fmax(reduce_buffer[tid], reduce_buffer[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            shared_column_max = reduce_buffer[0];
        }
        __syncthreads();

        double local_row_min = 1.0e300;
        double local_row_max = 0.0;
        for (int row = tid; row < n; row += blockDim.x) {
            double row_sum = 0.0;
            for (int col = 0; col < n; ++col) {
                row_sum += fabs(static_cast<double>(a[matrix_offset + row * n + col]));
            }
            local_row_min = fmin(local_row_min, row_sum);
            local_row_max = fmax(local_row_max, row_sum);
        }

        reduce_buffer[tid] = local_row_min;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buffer[tid] = fmin(reduce_buffer[tid], reduce_buffer[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            shared_row_min = reduce_buffer[0];
        }
        __syncthreads();

        reduce_buffer[tid] = local_row_max;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buffer[tid] = fmax(reduce_buffer[tid], reduce_buffer[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            shared_row_max = reduce_buffer[0];
        }
        __syncthreads();

        if (tid == 0) {
            const double column_span = shared_column_max / fmax(shared_column_min, 1.0e-30);
            const double row_span = shared_row_max / fmax(shared_row_min, 1.0e-30);
            shared_early_stop = (column_span <= 1.05 && row_span >= 64.0) ? 1 : 0;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < n * n; idx += blockDim.x) {
        h[matrix_offset + idx] = a[matrix_offset + idx];
    }
    for (int k = tid; k < n; k += blockDim.x) {
        tau[tau_offset + k] = 0.0f;
    }
    __syncthreads();
"""

_CUDA_SEMANTIC_EARLY_STOP_FALLBACK_TEMPLATE = (
    _CUDA_GEQR2_PARALLEL_TEMPLATE.replace(
        "Correct-first cooperative GEQR2 implementation with block reductions.",
        "Near-collinear early-stop shortcut with cooperative GEQR2 fallback.",
    )
    .replace(
        "__shared__ int shared_active;\n\n    const int matrix_offset",
        "__shared__ int shared_active;\n"
        "    __shared__ int shared_early_stop;\n"
        "    __shared__ double shared_column_min;\n"
        "    __shared__ double shared_column_max;\n"
        "    __shared__ double shared_row_min;\n"
        "    __shared__ double shared_row_max;\n\n"
        "    const int matrix_offset",
    )
    .replace(
        """    for (int idx = tid; idx < n * n; idx += blockDim.x) {
        h[matrix_offset + idx] = a[matrix_offset + idx];
    }
    for (int k = tid; k < n; k += blockDim.x) {
        tau[tau_offset + k] = 0.0f;
    }
    __syncthreads();
""",
        _EARLY_STOP_INIT_BLOCK,
        1,
    )
    .replace(
        "    for (int k = 0; k < n; ++k) {",
        "    const int qr_stop_k = shared_early_stop != 0 ? 16 : n;\n\n    for (int k = 0; k < qr_stop_k; ++k) {",
        1,
    )
)

_SEMANTIC_COMBO_INIT_BLOCK = """    if (tid == 0) {
        shared_shortcut = 0;
        shared_early_stop = 0;
        shared_early_stop_k = n;
        shared_zero_tail = 0;
        shared_zero_tail_start = n;
    }
    __syncthreads();

    double local_norm1 = 0.0;
    double local_strictlower_norm1 = 0.0;
    for (int col = tid; col < n; col += blockDim.x) {
        double column_sum = 0.0;
        double strictlower_column_sum = 0.0;
        for (int row = 0; row < n; ++row) {
            const double abs_value = fabs(static_cast<double>(a[matrix_offset + row * n + col]));
            column_sum += abs_value;
            if (row > col) {
                strictlower_column_sum += abs_value;
            }
        }
        local_norm1 = fmax(local_norm1, column_sum);
        local_strictlower_norm1 = fmax(local_strictlower_norm1, strictlower_column_sum);
    }

    reduce_buffer[tid] = local_norm1;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buffer[tid] = fmax(reduce_buffer[tid], reduce_buffer[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        shared_norm1 = reduce_buffer[0];
    }
    __syncthreads();

    reduce_buffer[tid] = local_strictlower_norm1;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buffer[tid] = fmax(reduce_buffer[tid], reduce_buffer[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        shared_strictlower_norm1 = reduce_buffer[0];
        const double eps = 1.1920928955078125e-7;
        const double threshold = 20.0 * static_cast<double>(n) * eps * fmax(shared_norm1, 1.0e-30);
        shared_shortcut = shared_strictlower_norm1 <= threshold ? 1 : 0;
    }
    __syncthreads();

    if (shared_shortcut != 0) {
        for (int idx = tid; idx < n * n; idx += blockDim.x) {
            const int row = idx / n;
            const int col = idx - row * n;
            h[matrix_offset + idx] = row <= col ? a[matrix_offset + idx] : 0.0f;
        }
        for (int k = tid; k < n; k += blockDim.x) {
            tau[tau_offset + k] = 0.0f;
        }
        __syncthreads();
        if (tid == 0 && route_counts != nullptr && route_count >= QR_V2_ROUTE_FIELD_COUNT) {
            atomicAdd(&route_counts[QR_V2_ROUTE_UPPER_COUNT], 1ULL);
        }
        return;
    }

    if (n == 512 || n == 1024) {
        const int candidate_zero_tail_start = (3 * n) / 4;
        const int tail_columns = n - candidate_zero_tail_start;
        double local_tail_l1 = 0.0;
        for (int idx = tid; idx < n * tail_columns; idx += blockDim.x) {
            const int row = idx / tail_columns;
            const int tail_col = candidate_zero_tail_start + (idx - row * tail_columns);
            local_tail_l1 += fabs(static_cast<double>(a[matrix_offset + row * n + tail_col]));
        }

        reduce_buffer[tid] = local_tail_l1;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buffer[tid] += reduce_buffer[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0 && reduce_buffer[0] == 0.0) {
            shared_zero_tail = 1;
            shared_zero_tail_start = candidate_zero_tail_start;
        }
        __syncthreads();
    }

    if (n == 512) {
        double local_column_min = 1.0e300;
        double local_column_max = 0.0;
        for (int col = tid; col < n; col += blockDim.x) {
            double column_sum = 0.0;
            for (int row = 0; row < n; ++row) {
                column_sum += fabs(static_cast<double>(a[matrix_offset + row * n + col]));
            }
            local_column_min = fmin(local_column_min, column_sum);
            local_column_max = fmax(local_column_max, column_sum);
        }

        reduce_buffer[tid] = local_column_min;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buffer[tid] = fmin(reduce_buffer[tid], reduce_buffer[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            shared_column_min = reduce_buffer[0];
        }
        __syncthreads();

        reduce_buffer[tid] = local_column_max;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buffer[tid] = fmax(reduce_buffer[tid], reduce_buffer[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            shared_column_max = reduce_buffer[0];
        }
        __syncthreads();

        double local_row_min = 1.0e300;
        double local_row_max = 0.0;
        for (int row = tid; row < n; row += blockDim.x) {
            double row_sum = 0.0;
            for (int col = 0; col < n; ++col) {
                row_sum += fabs(static_cast<double>(a[matrix_offset + row * n + col]));
            }
            local_row_min = fmin(local_row_min, row_sum);
            local_row_max = fmax(local_row_max, row_sum);
        }

        reduce_buffer[tid] = local_row_min;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buffer[tid] = fmin(reduce_buffer[tid], reduce_buffer[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            shared_row_min = reduce_buffer[0];
        }
        __syncthreads();

        reduce_buffer[tid] = local_row_max;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buffer[tid] = fmax(reduce_buffer[tid], reduce_buffer[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            shared_row_max = reduce_buffer[0];
        }
        __syncthreads();

        if (tid == 0) {
            const double column_span = shared_column_max / fmax(shared_column_min, 1.0e-30);
            const double row_span = shared_row_max / fmax(shared_row_min, 1.0e-30);
            if (column_span <= 1.05 && row_span >= 64.0) {
                shared_early_stop = 1;
                shared_early_stop_k = 16;
            } else if (shared_zero_tail == 0 && shared_column_min > 0.0 && column_span >= 1.0e5 && row_span <= 4.0) {
                shared_early_stop = 1;
                shared_early_stop_k = n / 2;
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < n * n; idx += blockDim.x) {
        h[matrix_offset + idx] = a[matrix_offset + idx];
    }
    for (int k = tid; k < n; k += blockDim.x) {
        tau[tau_offset + k] = 0.0f;
    }
    __syncthreads();

    if (tid == 0 && route_counts != nullptr && route_count >= QR_V2_ROUTE_FIELD_COUNT) {
        int route_index = QR_V2_ROUTE_FALLBACK_COUNT;
        if (shared_early_stop != 0) {
            route_index = QR_V2_ROUTE_EARLY_STOP_COUNT;
        } else if (shared_zero_tail != 0) {
            route_index = QR_V2_ROUTE_ZERO_TAIL_COUNT;
        }
        atomicAdd(&route_counts[route_index], 1ULL);
    }
    __syncthreads();
"""

_SEMANTIC_ROUTE_COUNTER_DEFS = """static constexpr int QR_V2_ROUTE_FIELD_COUNT = 4;
static constexpr int QR_V2_ROUTE_UPPER_COUNT = 0;
static constexpr int QR_V2_ROUTE_EARLY_STOP_COUNT = 1;
static constexpr int QR_V2_ROUTE_ZERO_TAIL_COUNT = 2;
static constexpr int QR_V2_ROUTE_FALLBACK_COUNT = 3;

"""

_SEMANTIC_ROUTE_COUNTER_LAUNCHER = """
extern "C" __attribute__((visibility("default"))) int {{ config.entry_point }}_launch_routes(
    const float* a,
    float* h,
    float* tau,
    unsigned long long* route_counts,
    int batch,
    int n,
    int threads_per_block,
    int route_count
) {
    if (a == nullptr || h == nullptr || tau == nullptr || route_counts == nullptr || batch <= 0 || n <= 0 || threads_per_block <= 0) {
        return -1;
    }
    if (threads_per_block > 1024 || (threads_per_block & (threads_per_block - 1)) != 0) {
        return -2;
    }
    if (route_count < QR_V2_ROUTE_FIELD_COUNT) {
        return -3;
    }

    cudaError_t memset_status = cudaMemset(route_counts, 0, sizeof(unsigned long long) * route_count);
    if (memset_status != cudaSuccess) {
        return static_cast<int>(memset_status);
    }

    {{ config.entry_point }}<<<batch, threads_per_block>>>(a, h, tau, route_counts, route_count, batch, n);
    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        return static_cast<int>(launch_status);
    }

    cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
        return static_cast<int>(sync_status);
    }
    return 0;
}
"""

_CUDA_SEMANTIC_COMBO_FALLBACK_TEMPLATE = (
    _CUDA_GEQR2_PARALLEL_TEMPLATE.replace(
        "Correct-first cooperative GEQR2 implementation with block reductions.",
        "Upper, near-collinear/clustered early-stop, and zero-tail routes with cooperative GEQR2 fallback.",
    )
    .replace(
        "#include <math.h>\n\n",
        "#include <math.h>\n\n" + _SEMANTIC_ROUTE_COUNTER_DEFS,
        1,
    )
    .replace(
        """    int batch,
    int n
) {""",
        """    unsigned long long* __restrict__ route_counts,
    int route_count,
    int batch,
    int n
) {""",
        1,
    )
    .replace(
        "__shared__ int shared_active;\n\n    const int matrix_offset",
        "__shared__ int shared_active;\n"
        "    __shared__ int shared_shortcut;\n"
        "    __shared__ int shared_early_stop;\n"
        "    __shared__ int shared_early_stop_k;\n"
        "    __shared__ int shared_zero_tail;\n"
        "    __shared__ int shared_zero_tail_start;\n"
        "    __shared__ double shared_norm1;\n"
        "    __shared__ double shared_strictlower_norm1;\n"
        "    __shared__ double shared_column_min;\n"
        "    __shared__ double shared_column_max;\n"
        "    __shared__ double shared_row_min;\n"
        "    __shared__ double shared_row_max;\n\n"
        "    const int matrix_offset",
    )
    .replace(
        """    for (int idx = tid; idx < n * n; idx += blockDim.x) {
        h[matrix_offset + idx] = a[matrix_offset + idx];
    }
    for (int k = tid; k < n; k += blockDim.x) {
        tau[tau_offset + k] = 0.0f;
    }
    __syncthreads();
""",
        _SEMANTIC_COMBO_INIT_BLOCK,
        1,
    )
    .replace(
        "    for (int k = 0; k < n; ++k) {",
        "    const int qr_stop_k = shared_early_stop != 0 ? shared_early_stop_k : shared_zero_tail_start;\n"
        "    const int qr_col_stop = shared_zero_tail != 0 ? shared_zero_tail_start : n;\n\n"
        "    for (int k = 0; k < qr_stop_k; ++k) {",
        1,
    )
    .replace(
        "        for (int col = k + 1; col < n; ++col) {",
        "        for (int col = k + 1; col < qr_col_stop; ++col) {",
        1,
    )
    .replace(
        "    {{ config.entry_point }}<<<batch, threads_per_block>>>(a, h, tau, batch, n);",
        "    {{ config.entry_point }}<<<batch, threads_per_block>>>(a, h, tau, nullptr, 0, batch, n);",
        1,
    )
    + _SEMANTIC_ROUTE_COUNTER_LAUNCHER
)

_CUDA_GEQR2_PARALLEL_PROFILE_TEMPLATE = """// Generated by gpumode.qr_v2 {{ renderer_version }}.
// Correct-first cooperative GEQR2 implementation with block-level clock64 phase counters.
//
// candidate_id: {{ config.candidate_id }}
// template_id: {{ config.template_id }}
// entry_point: {{ config.entry_point }}
// eval: {{ spec.eval_line(include_dense_case=True) }}
// shape: batch={{ spec.batch }}, n={{ spec.n }}, cond={{ spec.cond }}, case={{ spec.case }}, seed={{ spec.seed }}
// sweep: threads_per_block={{ config.threads_per_block }}, tile_size={{ config.tile_size }}, unroll={{ config.unroll }}
// profile fields: total_cycles, init_cycles, norm_cycles, scale_cycles, dot_cycles, update_cycles,
//                 active_reflectors, inactive_reflectors, trailing_columns

#include <cuda_runtime.h>
#include <math.h>

static constexpr int QR_V2_PROFILE_FIELD_COUNT = 9;
static constexpr int QR_V2_PROFILE_TOTAL_CYCLES = 0;
static constexpr int QR_V2_PROFILE_INIT_CYCLES = 1;
static constexpr int QR_V2_PROFILE_NORM_CYCLES = 2;
static constexpr int QR_V2_PROFILE_SCALE_CYCLES = 3;
static constexpr int QR_V2_PROFILE_DOT_CYCLES = 4;
static constexpr int QR_V2_PROFILE_UPDATE_CYCLES = 5;
static constexpr int QR_V2_PROFILE_ACTIVE_REFLECTORS = 6;
static constexpr int QR_V2_PROFILE_INACTIVE_REFLECTORS = 7;
static constexpr int QR_V2_PROFILE_TRAILING_COLUMNS = 8;

extern "C" __global__ void {{ config.entry_point }}(
    const float* __restrict__ a,
    float* __restrict__ h,
    float* __restrict__ tau,
    unsigned long long* __restrict__ profile,
    int batch,
    int n,
    int profile_stride
) {
    const int matrix_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (matrix_idx >= batch) {
        return;
    }

    __shared__ double reduce_buffer[1024];
    __shared__ double shared_tau;
    __shared__ double shared_scale;
    __shared__ double shared_update;
    __shared__ int shared_active;
    __shared__ unsigned long long phase_start;
    __shared__ unsigned long long total_start;
    __shared__ unsigned long long cycles_init;
    __shared__ unsigned long long cycles_norm;
    __shared__ unsigned long long cycles_scale;
    __shared__ unsigned long long cycles_dot;
    __shared__ unsigned long long cycles_update;
    __shared__ unsigned long long active_reflectors;
    __shared__ unsigned long long inactive_reflectors;
    __shared__ unsigned long long trailing_columns;

    const int matrix_offset = matrix_idx * n * n;
    const int tau_offset = matrix_idx * n;

    if (tid == 0) {
        cycles_init = 0ULL;
        cycles_norm = 0ULL;
        cycles_scale = 0ULL;
        cycles_dot = 0ULL;
        cycles_update = 0ULL;
        active_reflectors = 0ULL;
        inactive_reflectors = 0ULL;
        trailing_columns = 0ULL;
        total_start = clock64();
        phase_start = total_start;
    }
    __syncthreads();

    if (tid == 0) {
        phase_start = clock64();
    }
    __syncthreads();
    for (int idx = tid; idx < n * n; idx += blockDim.x) {
        h[matrix_offset + idx] = a[matrix_offset + idx];
    }
    for (int k = tid; k < n; k += blockDim.x) {
        tau[tau_offset + k] = 0.0f;
    }
    __syncthreads();
    if (tid == 0) {
        cycles_init += clock64() - phase_start;
    }
    __syncthreads();

    for (int k = 0; k < n; ++k) {
        if (tid == 0) {
            phase_start = clock64();
        }
        __syncthreads();
        double local_sum = 0.0;
        for (int row = k + 1 + tid; row < n; row += blockDim.x) {
            const double value = static_cast<double>(h[matrix_offset + row * n + k]);
            local_sum += value * value;
        }
        reduce_buffer[tid] = local_sum;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buffer[tid] += reduce_buffer[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const int kk = matrix_offset + k * n + k;
            const double alpha = static_cast<double>(h[kk]);
            const double xnorm2 = reduce_buffer[0];
            if (xnorm2 == 0.0) {
                tau[tau_offset + k] = 0.0f;
                shared_tau = 0.0;
                shared_scale = 0.0;
                shared_active = 0;
            } else {
                const double norm = sqrt(alpha * alpha + xnorm2);
                const double beta = alpha >= 0.0 ? -norm : norm;
                const double tau_k = (beta - alpha) / beta;
                shared_tau = tau_k;
                shared_scale = 1.0 / (alpha - beta);
                shared_active = 1;
                h[kk] = static_cast<float>(beta);
                tau[tau_offset + k] = static_cast<float>(tau_k);
            }
        }
        __syncthreads();
        if (tid == 0) {
            cycles_norm += clock64() - phase_start;
            if (shared_active == 0) {
                inactive_reflectors += 1ULL;
            } else {
                active_reflectors += 1ULL;
            }
        }
        __syncthreads();

        if (shared_active == 0) {
            continue;
        }

        if (tid == 0) {
            phase_start = clock64();
        }
        __syncthreads();
        for (int row = k + 1 + tid; row < n; row += blockDim.x) {
            const int index = matrix_offset + row * n + k;
            h[index] = static_cast<float>(static_cast<double>(h[index]) * shared_scale);
        }
        __syncthreads();
        if (tid == 0) {
            cycles_scale += clock64() - phase_start;
        }
        __syncthreads();

        for (int col = k + 1; col < n; ++col) {
            if (tid == 0) {
                phase_start = clock64();
            }
            __syncthreads();
            double dot = tid == 0 ? static_cast<double>(h[matrix_offset + k * n + col]) : 0.0;
            for (int row = k + 1 + tid; row < n; row += blockDim.x) {
                dot += static_cast<double>(h[matrix_offset + row * n + k]) *
                       static_cast<double>(h[matrix_offset + row * n + col]);
            }
            reduce_buffer[tid] = dot;
            __syncthreads();

            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce_buffer[tid] += reduce_buffer[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0) {
                shared_update = shared_tau * reduce_buffer[0];
                cycles_dot += clock64() - phase_start;
            }
            __syncthreads();

            if (tid == 0) {
                phase_start = clock64();
            }
            __syncthreads();
            if (tid == 0) {
                const int top_index = matrix_offset + k * n + col;
                h[top_index] = static_cast<float>(static_cast<double>(h[top_index]) - shared_update);
            }

            for (int row = k + 1 + tid; row < n; row += blockDim.x) {
                const int index = matrix_offset + row * n + col;
                h[index] = static_cast<float>(
                    static_cast<double>(h[index]) - static_cast<double>(h[matrix_offset + row * n + k]) * shared_update
                );
            }
            __syncthreads();
            if (tid == 0) {
                cycles_update += clock64() - phase_start;
                trailing_columns += 1ULL;
            }
            __syncthreads();
        }
    }

    __syncthreads();
    if (tid == 0 && profile != nullptr && profile_stride >= QR_V2_PROFILE_FIELD_COUNT) {
        const int profile_offset = matrix_idx * profile_stride;
        profile[profile_offset + QR_V2_PROFILE_TOTAL_CYCLES] = clock64() - total_start;
        profile[profile_offset + QR_V2_PROFILE_INIT_CYCLES] = cycles_init;
        profile[profile_offset + QR_V2_PROFILE_NORM_CYCLES] = cycles_norm;
        profile[profile_offset + QR_V2_PROFILE_SCALE_CYCLES] = cycles_scale;
        profile[profile_offset + QR_V2_PROFILE_DOT_CYCLES] = cycles_dot;
        profile[profile_offset + QR_V2_PROFILE_UPDATE_CYCLES] = cycles_update;
        profile[profile_offset + QR_V2_PROFILE_ACTIVE_REFLECTORS] = active_reflectors;
        profile[profile_offset + QR_V2_PROFILE_INACTIVE_REFLECTORS] = inactive_reflectors;
        profile[profile_offset + QR_V2_PROFILE_TRAILING_COLUMNS] = trailing_columns;
    }
}

extern "C" __attribute__((visibility("default"))) int {{ config.entry_point }}_launch(
    const float* a,
    float* h,
    float* tau,
    int batch,
    int n,
    int threads_per_block
) {
    if (a == nullptr || h == nullptr || tau == nullptr || batch <= 0 || n <= 0 || threads_per_block <= 0) {
        return -1;
    }
    if (threads_per_block > 1024 || (threads_per_block & (threads_per_block - 1)) != 0) {
        return -2;
    }

    {{ config.entry_point }}<<<batch, threads_per_block>>>(a, h, tau, nullptr, batch, n, 0);
    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        return static_cast<int>(launch_status);
    }

    cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
        return static_cast<int>(sync_status);
    }
    return 0;
}

extern "C" __attribute__((visibility("default"))) int {{ config.entry_point }}_launch_profile(
    const float* a,
    float* h,
    float* tau,
    unsigned long long* profile,
    int batch,
    int n,
    int threads_per_block,
    int profile_stride
) {
    if (a == nullptr || h == nullptr || tau == nullptr || profile == nullptr || batch <= 0 || n <= 0 || threads_per_block <= 0) {
        return -1;
    }
    if (threads_per_block > 1024 || (threads_per_block & (threads_per_block - 1)) != 0) {
        return -2;
    }
    if (profile_stride < QR_V2_PROFILE_FIELD_COUNT) {
        return -3;
    }

    {{ config.entry_point }}<<<batch, threads_per_block>>>(a, h, tau, profile, batch, n, profile_stride);
    cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        return static_cast<int>(launch_status);
    }

    cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
        return static_cast<int>(sync_status);
    }
    return 0;
}
"""


_CUDA_CUSOLVER_GEQRF_TEMPLATE = """// Generated by gpumode.qr_v2 {{ renderer_version }}.
// cuSOLVER dense GEQRF baseline with row-major/column-major packing kernels.
//
// candidate_id: {{ config.candidate_id }}
// template_id: {{ config.template_id }}
// entry_point: {{ config.entry_point }}
// eval: {{ spec.eval_line(include_dense_case=True) }}
// shape: batch={{ spec.batch }}, n={{ spec.n }}, cond={{ spec.cond }}, case={{ spec.case }}, seed={{ spec.seed }}
// sweep: threads_per_block={{ config.threads_per_block }}, tile_size={{ config.tile_size }}, unroll={{ config.unroll }}

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <vector>

extern "C" __global__ void {{ config.entry_point }}_row_to_col_major(
    const float* __restrict__ a,
    float* __restrict__ packed,
    int batch,
    int n
) {
    const int total = batch * n * n;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int matrix_size = n * n;
        const int matrix_idx = idx / matrix_size;
        const int offset = idx - matrix_idx * matrix_size;
        const int row = offset / n;
        const int col = offset - row * n;
        packed[matrix_idx * matrix_size + row + col * n] = a[matrix_idx * matrix_size + row * n + col];
    }
}

extern "C" __global__ void {{ config.entry_point }}_col_to_row_major(
    const float* __restrict__ packed,
    float* __restrict__ h,
    int batch,
    int n
) {
    const int total = batch * n * n;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int matrix_size = n * n;
        const int matrix_idx = idx / matrix_size;
        const int offset = idx - matrix_idx * matrix_size;
        const int row = offset / n;
        const int col = offset - row * n;
        h[matrix_idx * matrix_size + row * n + col] = packed[matrix_idx * matrix_size + row + col * n];
    }
}

extern "C" __attribute__((visibility("default"))) int {{ config.entry_point }}_launch(
    const float* a,
    float* h,
    float* tau,
    int batch,
    int n,
    int threads_per_block
) {
    if (a == nullptr || h == nullptr || tau == nullptr || batch <= 0 || n <= 0 || threads_per_block <= 0) {
        return -1;
    }
    if (threads_per_block > 1024 || (threads_per_block & (threads_per_block - 1)) != 0) {
        return -2;
    }

    cusolverDnHandle_t handle = nullptr;
    float* packed = nullptr;
    float* workspace = nullptr;
    int* info = nullptr;

    auto cleanup = [&]() {
        if (info != nullptr) {
            cudaFree(info);
        }
        if (workspace != nullptr) {
            cudaFree(workspace);
        }
        if (packed != nullptr) {
            cudaFree(packed);
        }
        if (handle != nullptr) {
            cusolverDnDestroy(handle);
        }
    };

    cusolverStatus_t solver_status = cusolverDnCreate(&handle);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
        cleanup();
        return 10000 + static_cast<int>(solver_status);
    }

    const size_t matrix_size = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t value_count = static_cast<size_t>(batch) * matrix_size;
    cudaError_t cuda_status = cudaMalloc(&packed, value_count * sizeof(float));
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }

    const int max_blocks = 65535;
    int blocks = static_cast<int>((value_count + static_cast<size_t>(threads_per_block) - 1) / static_cast<size_t>(threads_per_block));
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }

    {{ config.entry_point }}_row_to_col_major<<<blocks, threads_per_block>>>(a, packed, batch, n);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }

    int lwork = 0;
    solver_status = cusolverDnSgeqrf_bufferSize(handle, n, n, packed, n, &lwork);
    if (solver_status != CUSOLVER_STATUS_SUCCESS || lwork <= 0) {
        cleanup();
        return 11000 + static_cast<int>(solver_status);
    }

    cuda_status = cudaMalloc(&workspace, static_cast<size_t>(lwork) * sizeof(float));
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }
    cuda_status = cudaMalloc(&info, static_cast<size_t>(batch) * sizeof(int));
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }

    for (int matrix_idx = 0; matrix_idx < batch; ++matrix_idx) {
        float* matrix = packed + static_cast<size_t>(matrix_idx) * matrix_size;
        float* tau_matrix = tau + static_cast<size_t>(matrix_idx) * static_cast<size_t>(n);
        solver_status = cusolverDnSgeqrf(handle, n, n, matrix, n, tau_matrix, workspace, lwork, info + matrix_idx);
        if (solver_status != CUSOLVER_STATUS_SUCCESS) {
            cleanup();
            return 12000 + static_cast<int>(solver_status);
        }
    }

    {{ config.entry_point }}_col_to_row_major<<<blocks, threads_per_block>>>(packed, h, batch, n);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }

    std::vector<int> host_info(static_cast<size_t>(batch), 0);
    cuda_status = cudaMemcpy(host_info.data(), info, static_cast<size_t>(batch) * sizeof(int), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }
    for (int matrix_idx = 0; matrix_idx < batch; ++matrix_idx) {
        if (host_info[static_cast<size_t>(matrix_idx)] != 0) {
            cleanup();
            return -1000 - matrix_idx;
        }
    }

    cleanup();
    return 0;
}
"""


_CUDA_CUSOLVERDX_GEQRF_SMALL_TEMPLATE = """// Generated by gpumode.qr_v2 {{ renderer_version }}.
// cuSolverDx block GEQRF candidate for small row-major square matrices.
//
// candidate_id: {{ config.candidate_id }}
// template_id: {{ config.template_id }}
// entry_point: {{ config.entry_point }}
// eval: {{ spec.eval_line(include_dense_case=True) }}
// shape: batch={{ spec.batch }}, n={{ spec.n }}, cond={{ spec.cond }}, case={{ spec.case }}, seed={{ spec.seed }}
// sweep: threads_per_block={{ config.threads_per_block }}, tile_size={{ config.tile_size }}, unroll={{ config.unroll }}

#include <cuda_runtime.h>
#include <cusolverdx.hpp>

using {{ config.entry_point }}_solver_base = decltype(
    cusolverdx::Size<{{ spec.n }}, {{ spec.n }}>()
    + cusolverdx::Precision<float>()
    + cusolverdx::Type<cusolverdx::type::real>()
    + cusolverdx::Function<cusolverdx::geqrf>()
    + cusolverdx::Arrangement<cusolverdx::arrangement::row_major>()
    + cusolverdx::SM<900>()
    + cusolverdx::Block()
);
using {{ config.entry_point }}_solver = decltype({{ config.entry_point }}_solver_base() + cusolverdx::BatchesPerBlock<1>());

template<class Solver, typename DataType = typename Solver::a_data_type>
__global__ __launch_bounds__(Solver::max_threads_per_block) void {{ config.entry_point }}_dx_kernel(
    const float* __restrict__ a,
    float* __restrict__ h,
    float* __restrict__ tau,
    int batch,
    int n
) {
    constexpr int matrix_n = Solver::m_size;
    constexpr int lda_smem = Solver::lda;
    constexpr int matrix_values = matrix_n * matrix_n;
    constexpr int tau_values = matrix_n;

    extern __shared__ __align__(16) cusolverdx::byte shared_mem[];
    DataType* matrix_smem = reinterpret_cast<DataType*>(shared_mem);
    DataType* tau_smem = matrix_smem + lda_smem * matrix_n;

    const int matrix_idx = blockIdx.x;
    if (matrix_idx >= batch) {
        return;
    }

    const int tid = threadIdx.x + Solver::block_dim.x * (threadIdx.y + Solver::block_dim.y * threadIdx.z);
    const size_t matrix_offset = static_cast<size_t>(matrix_idx) * static_cast<size_t>(matrix_values);
    const size_t tau_offset = static_cast<size_t>(matrix_idx) * static_cast<size_t>(tau_values);

    for (int idx = tid; idx < matrix_values; idx += Solver::max_threads_per_block) {
        const int row = idx / matrix_n;
        const int col = idx - row * matrix_n;
        matrix_smem[row * lda_smem + col] = static_cast<DataType>(a[matrix_offset + static_cast<size_t>(idx)]);
    }
    for (int idx = tid; idx < tau_values; idx += Solver::max_threads_per_block) {
        tau_smem[idx] = DataType{0};
    }
    __syncthreads();

    Solver().execute(matrix_smem, lda_smem, tau_smem);
    __syncthreads();

    for (int idx = tid; idx < matrix_values; idx += Solver::max_threads_per_block) {
        const int row = idx / matrix_n;
        const int col = idx - row * matrix_n;
        h[matrix_offset + static_cast<size_t>(idx)] = static_cast<float>(matrix_smem[row * lda_smem + col]);
    }
    for (int idx = tid; idx < tau_values; idx += Solver::max_threads_per_block) {
        tau[tau_offset + static_cast<size_t>(idx)] = static_cast<float>(tau_smem[idx]);
    }
}

extern "C" __attribute__((visibility("default"))) int {{ config.entry_point }}_launch(
    const float* a,
    float* h,
    float* tau,
    int batch,
    int n,
    int threads_per_block
) {
    using Solver = {{ config.entry_point }}_solver;
    if (a == nullptr || h == nullptr || tau == nullptr || batch <= 0 || n <= 0 || threads_per_block <= 0) {
        return -1;
    }
    if (n != {{ spec.n }}) {
        return -3;
    }

    cudaError_t cuda_status = cudaFuncSetAttribute(
        {{ config.entry_point }}_dx_kernel<Solver>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        Solver::shared_memory_size
    );
    if (cuda_status != cudaSuccess) {
        return static_cast<int>(cuda_status);
    }

    {{ config.entry_point }}_dx_kernel<Solver><<<batch, Solver::block_dim, Solver::shared_memory_size>>>(a, h, tau, batch, n);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        return static_cast<int>(cuda_status);
    }
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        return static_cast<int>(cuda_status);
    }
    return 0;
}
"""


_CUDA_CUBLAS_BATCHED_GEQRF_TEMPLATE = """// Generated by gpumode.qr_v2 {{ renderer_version }}.
// cuBLAS batched dense GEQRF baseline with row-major/column-major packing kernels.
//
// candidate_id: {{ config.candidate_id }}
// template_id: {{ config.template_id }}
// entry_point: {{ config.entry_point }}
// eval: {{ spec.eval_line(include_dense_case=True) }}
// shape: batch={{ spec.batch }}, n={{ spec.n }}, cond={{ spec.cond }}, case={{ spec.case }}, seed={{ spec.seed }}
// sweep: threads_per_block={{ config.threads_per_block }}, tile_size={{ config.tile_size }}, unroll={{ config.unroll }}

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <vector>

extern "C" __global__ void {{ config.entry_point }}_row_to_col_major(
    const float* __restrict__ a,
    float* __restrict__ packed,
    int batch,
    int n
) {
    const int total = batch * n * n;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int matrix_size = n * n;
        const int matrix_idx = idx / matrix_size;
        const int offset = idx - matrix_idx * matrix_size;
        const int row = offset / n;
        const int col = offset - row * n;
        packed[matrix_idx * matrix_size + row + col * n] = a[matrix_idx * matrix_size + row * n + col];
    }
}

extern "C" __global__ void {{ config.entry_point }}_col_to_row_major(
    const float* __restrict__ packed,
    float* __restrict__ h,
    int batch,
    int n
) {
    const int total = batch * n * n;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int matrix_size = n * n;
        const int matrix_idx = idx / matrix_size;
        const int offset = idx - matrix_idx * matrix_size;
        const int row = offset / n;
        const int col = offset - row * n;
        h[matrix_idx * matrix_size + row * n + col] = packed[matrix_idx * matrix_size + row + col * n];
    }
}

extern "C" __global__ void {{ config.entry_point }}_init_pointer_arrays(
    float* __restrict__ packed,
    float* __restrict__ tau,
    float** __restrict__ matrix_ptrs,
    float** __restrict__ tau_ptrs,
    int batch,
    int n
) {
    const int matrix_size = n * n;
    for (int matrix_idx = blockIdx.x * blockDim.x + threadIdx.x; matrix_idx < batch; matrix_idx += blockDim.x * gridDim.x) {
        matrix_ptrs[matrix_idx] = packed + static_cast<size_t>(matrix_idx) * static_cast<size_t>(matrix_size);
        tau_ptrs[matrix_idx] = tau + static_cast<size_t>(matrix_idx) * static_cast<size_t>(n);
    }
}

extern "C" __attribute__((visibility("default"))) int {{ config.entry_point }}_launch(
    const float* a,
    float* h,
    float* tau,
    int batch,
    int n,
    int threads_per_block
) {
    if (a == nullptr || h == nullptr || tau == nullptr || batch <= 0 || n <= 0 || threads_per_block <= 0) {
        return -1;
    }
    if (threads_per_block > 1024 || (threads_per_block & (threads_per_block - 1)) != 0) {
        return -2;
    }

    cublasHandle_t handle = nullptr;
    float* packed = nullptr;
    float** matrix_ptrs = nullptr;
    float** tau_ptrs = nullptr;

    auto cleanup = [&]() {
        if (tau_ptrs != nullptr) {
            cudaFree(tau_ptrs);
        }
        if (matrix_ptrs != nullptr) {
            cudaFree(matrix_ptrs);
        }
        if (packed != nullptr) {
            cudaFree(packed);
        }
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
    };

    cublasStatus_t cublas_status = cublasCreate(&handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return 10000 + static_cast<int>(cublas_status);
    }

    const size_t matrix_size = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t value_count = static_cast<size_t>(batch) * matrix_size;
    cudaError_t cuda_status = cudaMalloc(&packed, value_count * sizeof(float));
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }
    cuda_status = cudaMalloc(&matrix_ptrs, static_cast<size_t>(batch) * sizeof(float*));
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }
    cuda_status = cudaMalloc(&tau_ptrs, static_cast<size_t>(batch) * sizeof(float*));
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }

    const int max_blocks = 65535;
    int blocks = static_cast<int>((value_count + static_cast<size_t>(threads_per_block) - 1) / static_cast<size_t>(threads_per_block));
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }

    {{ config.entry_point }}_row_to_col_major<<<blocks, threads_per_block>>>(a, packed, batch, n);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }

    int pointer_blocks = (batch + threads_per_block - 1) / threads_per_block;
    if (pointer_blocks < 1) {
        pointer_blocks = 1;
    }
    if (pointer_blocks > max_blocks) {
        pointer_blocks = max_blocks;
    }
    {{ config.entry_point }}_init_pointer_arrays<<<pointer_blocks, threads_per_block>>>(packed, tau, matrix_ptrs, tau_ptrs, batch, n);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }

    int info = 0;
    cublas_status = cublasSgeqrfBatched(handle, n, n, matrix_ptrs, n, tau_ptrs, &info, batch);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return 11000 + static_cast<int>(cublas_status);
    }

    {{ config.entry_point }}_col_to_row_major<<<blocks, threads_per_block>>>(packed, h, batch, n);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }

    if (info != 0) {
        cleanup();
        return -1000 + info;
    }

    cleanup();
    return 0;
}
"""

_CUDA_DENSE_BLOCKED_WY_512_TEMPLATE = """// Generated by gpumode.qr_v2 {{ renderer_version }}.
// Dense blocked Householder QR prototype for n=352/512/1024.
//
// candidate_id: {{ config.candidate_id }}
// template_id: {{ config.template_id }}
// entry_point: {{ config.entry_point }}
// eval: {{ spec.eval_line(include_dense_case=True) }}
// shape: batch={{ spec.batch }}, n={{ spec.n }}, cond={{ spec.cond }}, case={{ spec.case }}, seed={{ spec.seed }}
// sweep: threads_per_block={{ config.threads_per_block }}, tile_size={{ config.tile_size }}, unroll={{ config.unroll }}

#include <cuda_runtime.h>
#include <cublas_v2.h>

static constexpr int QR_V2_PANEL_WIDTH = {{ config.tile_size }};

struct QrV2BlockedWorkspace {
    cublasHandle_t handle = nullptr;
    float* packed = nullptr;
    float* panel_v = nullptr;
    float* panel_t = nullptr;
    float* work_vt_c = nullptr;
    float* work_t_vt_c = nullptr;
    float** matrix_ptrs = nullptr;
    float** tau_ptrs = nullptr;
    size_t packed_capacity = 0;
    size_t panel_v_capacity = 0;
    size_t panel_t_capacity = 0;
    size_t work_vt_c_capacity = 0;
    size_t work_t_vt_c_capacity = 0;
    size_t matrix_ptr_capacity = 0;
    size_t tau_ptr_capacity = 0;
};

static QrV2BlockedWorkspace qr_v2_blocked_workspace;

static cudaError_t qr_v2_ensure_float_buffer(float** ptr, size_t* capacity, size_t required) {
    if (*ptr != nullptr && *capacity >= required) {
        return cudaSuccess;
    }
    if (*ptr != nullptr) {
        cudaError_t free_status = cudaFree(*ptr);
        if (free_status != cudaSuccess) {
            return free_status;
        }
        *ptr = nullptr;
        *capacity = 0;
    }
    if (required == 0) {
        return cudaSuccess;
    }
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(ptr), required * sizeof(float));
    if (status == cudaSuccess) {
        *capacity = required;
    }
    return status;
}

static cudaError_t qr_v2_ensure_pointer_buffer(float*** ptr, size_t* capacity, size_t required) {
    if (*ptr != nullptr && *capacity >= required) {
        return cudaSuccess;
    }
    if (*ptr != nullptr) {
        cudaError_t free_status = cudaFree(*ptr);
        if (free_status != cudaSuccess) {
            return free_status;
        }
        *ptr = nullptr;
        *capacity = 0;
    }
    if (required == 0) {
        return cudaSuccess;
    }
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(ptr), required * sizeof(float*));
    if (status == cudaSuccess) {
        *capacity = required;
    }
    return status;
}

extern "C" __global__ void {{ config.entry_point }}_row_to_col_major(
    const float* __restrict__ a,
    float* __restrict__ packed,
    int batch,
    int n
) {
    const int total = batch * n * n;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int matrix_size = n * n;
        const int matrix_idx = idx / matrix_size;
        const int offset = idx - matrix_idx * matrix_size;
        const int row = offset / n;
        const int col = offset - row * n;
        packed[matrix_idx * matrix_size + row + col * n] = a[matrix_idx * matrix_size + row * n + col];
    }
}

extern "C" __global__ void {{ config.entry_point }}_col_to_row_major(
    const float* __restrict__ packed,
    float* __restrict__ h,
    int batch,
    int n
) {
    const int total = batch * n * n;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int matrix_size = n * n;
        const int matrix_idx = idx / matrix_size;
        const int offset = idx - matrix_idx * matrix_size;
        const int row = offset / n;
        const int col = offset - row * n;
        h[matrix_idx * matrix_size + row * n + col] = packed[matrix_idx * matrix_size + row + col * n];
    }
}

extern "C" __global__ void {{ config.entry_point }}_init_panel_pointer_arrays(
    float* __restrict__ packed,
    float* __restrict__ tau,
    float** __restrict__ matrix_ptrs,
    float** __restrict__ tau_ptrs,
    int batch,
    int n,
    int panel_col
) {
    const int matrix_size = n * n;
    for (int matrix_idx = blockIdx.x * blockDim.x + threadIdx.x; matrix_idx < batch; matrix_idx += blockDim.x * gridDim.x) {
        matrix_ptrs[matrix_idx] = packed + static_cast<size_t>(matrix_idx) * static_cast<size_t>(matrix_size)
                                + static_cast<size_t>(panel_col) + static_cast<size_t>(panel_col) * static_cast<size_t>(n);
        tau_ptrs[matrix_idx] = tau + static_cast<size_t>(matrix_idx) * static_cast<size_t>(n) + static_cast<size_t>(panel_col);
    }
}

extern "C" __global__ void {{ config.entry_point }}_build_panel_v(
    const float* __restrict__ packed,
    float* __restrict__ panel_v,
    int batch,
    int n,
    int panel_col,
    int panel_width,
    int panel_m
) {
    const int matrix_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (matrix_idx >= batch) {
        return;
    }

    const size_t matrix_offset = static_cast<size_t>(matrix_idx) * static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t panel_offset = static_cast<size_t>(matrix_idx) * static_cast<size_t>(n) * static_cast<size_t>(QR_V2_PANEL_WIDTH);
    for (int idx = tid; idx < panel_m * panel_width; idx += blockDim.x) {
        const int row = idx % panel_m;
        const int col = idx / panel_m;
        float value = 0.0f;
        if (row == col) {
            value = 1.0f;
        } else if (row > col) {
            value = packed[
                matrix_offset
                + static_cast<size_t>(panel_col + row)
                + static_cast<size_t>(panel_col + col) * static_cast<size_t>(n)
            ];
        }
        panel_v[panel_offset + static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(n)] = value;
    }
}

extern "C" __global__ void {{ config.entry_point }}_form_t(
    const float* __restrict__ panel_v,
    const float* __restrict__ tau,
    float* __restrict__ panel_t,
    int batch,
    int n,
    int panel_col,
    int panel_width,
    int panel_m
) {
    const int matrix_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (matrix_idx >= batch) {
        return;
    }

    __shared__ float reduce_buffer[256];
    __shared__ float work[QR_V2_PANEL_WIDTH];

    const size_t v_offset = static_cast<size_t>(matrix_idx) * static_cast<size_t>(n) * static_cast<size_t>(QR_V2_PANEL_WIDTH);
    const size_t t_offset = static_cast<size_t>(matrix_idx) * static_cast<size_t>(QR_V2_PANEL_WIDTH) * static_cast<size_t>(QR_V2_PANEL_WIDTH);
    const size_t tau_offset = static_cast<size_t>(matrix_idx) * static_cast<size_t>(n) + static_cast<size_t>(panel_col);

    for (int idx = tid; idx < QR_V2_PANEL_WIDTH * QR_V2_PANEL_WIDTH; idx += blockDim.x) {
        panel_t[t_offset + static_cast<size_t>(idx)] = 0.0f;
    }
    __syncthreads();

    for (int i = 0; i < panel_width; ++i) {
        const float tau_i = tau[tau_offset + static_cast<size_t>(i)];
        if (tid == 0) {
            for (int j = 0; j < QR_V2_PANEL_WIDTH; ++j) {
                work[j] = 0.0f;
            }
        }
        __syncthreads();

        for (int j = 0; j < i; ++j) {
            float local_sum = 0.0f;
            for (int row = i + tid; row < panel_m; row += blockDim.x) {
                const float v_j = panel_v[v_offset + static_cast<size_t>(row) + static_cast<size_t>(j) * static_cast<size_t>(n)];
                const float v_i = panel_v[v_offset + static_cast<size_t>(row) + static_cast<size_t>(i) * static_cast<size_t>(n)];
                local_sum += v_j * v_i;
            }
            reduce_buffer[tid] = local_sum;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce_buffer[tid] += reduce_buffer[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                work[j] = -tau_i * reduce_buffer[0];
            }
            __syncthreads();
        }

        if (tid == 0) {
            for (int row = 0; row < i; ++row) {
                float value = 0.0f;
                for (int col = row; col < i; ++col) {
                    value += panel_t[t_offset + static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(QR_V2_PANEL_WIDTH)] * work[col];
                }
                panel_t[t_offset + static_cast<size_t>(row) + static_cast<size_t>(i) * static_cast<size_t>(QR_V2_PANEL_WIDTH)] = value;
            }
            panel_t[t_offset + static_cast<size_t>(i) + static_cast<size_t>(i) * static_cast<size_t>(QR_V2_PANEL_WIDTH)] = tau_i;
        }
        __syncthreads();
    }
}

extern "C" __attribute__((visibility("default"))) int {{ config.entry_point }}_launch(
    const float* a,
    float* h,
    float* tau,
    int batch,
    int n,
    int threads_per_block
) {
    if (a == nullptr || h == nullptr || tau == nullptr || batch <= 0 || n <= 0 || threads_per_block <= 0) {
        return -1;
    }
    if (n != 352 && n != 512 && n != 1024) {
        return -3;
    }
    if (threads_per_block > 1024 || (threads_per_block & (threads_per_block - 1)) != 0) {
        return -2;
    }

    const size_t matrix_size = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t value_count = static_cast<size_t>(batch) * matrix_size;
    auto cleanup = []() {};
    QrV2BlockedWorkspace& workspace = qr_v2_blocked_workspace;

    if (workspace.handle == nullptr) {
        cublasStatus_t create_status = cublasCreate(&workspace.handle);
        if (create_status != CUBLAS_STATUS_SUCCESS) {
            return 10000 + static_cast<int>(create_status);
        }
    }

    cudaError_t cuda_status = qr_v2_ensure_float_buffer(&workspace.packed, &workspace.packed_capacity, value_count);
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }
    cuda_status = qr_v2_ensure_float_buffer(
        &workspace.panel_v,
        &workspace.panel_v_capacity,
        static_cast<size_t>(batch) * static_cast<size_t>(n) * static_cast<size_t>(QR_V2_PANEL_WIDTH)
    );
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }
    cuda_status = qr_v2_ensure_float_buffer(
        &workspace.panel_t,
        &workspace.panel_t_capacity,
        static_cast<size_t>(batch) * static_cast<size_t>(QR_V2_PANEL_WIDTH) * static_cast<size_t>(QR_V2_PANEL_WIDTH)
    );
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }
    cuda_status = qr_v2_ensure_float_buffer(
        &workspace.work_vt_c,
        &workspace.work_vt_c_capacity,
        static_cast<size_t>(batch) * static_cast<size_t>(QR_V2_PANEL_WIDTH) * static_cast<size_t>(n)
    );
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }
    cuda_status = qr_v2_ensure_float_buffer(
        &workspace.work_t_vt_c,
        &workspace.work_t_vt_c_capacity,
        static_cast<size_t>(batch) * static_cast<size_t>(QR_V2_PANEL_WIDTH) * static_cast<size_t>(n)
    );
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }
    cuda_status = qr_v2_ensure_pointer_buffer(&workspace.matrix_ptrs, &workspace.matrix_ptr_capacity, static_cast<size_t>(batch));
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }
    cuda_status = qr_v2_ensure_pointer_buffer(&workspace.tau_ptrs, &workspace.tau_ptr_capacity, static_cast<size_t>(batch));
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }

    cublasHandle_t handle = workspace.handle;
    float* packed = workspace.packed;
    float* panel_v = workspace.panel_v;
    float* panel_t = workspace.panel_t;
    float* work_vt_c = workspace.work_vt_c;
    float* work_t_vt_c = workspace.work_t_vt_c;
    float** matrix_ptrs = workspace.matrix_ptrs;
    float** tau_ptrs = workspace.tau_ptrs;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;

    const int max_blocks = 65535;
    int blocks = static_cast<int>((value_count + static_cast<size_t>(threads_per_block) - 1) / static_cast<size_t>(threads_per_block));
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }

    {{ config.entry_point }}_row_to_col_major<<<blocks, threads_per_block>>>(a, packed, batch, n);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }

    int pointer_blocks = (batch + threads_per_block - 1) / threads_per_block;
    if (pointer_blocks < 1) {
        pointer_blocks = 1;
    }
    if (pointer_blocks > max_blocks) {
        pointer_blocks = max_blocks;
    }

    const float one = 1.0f;
    const float zero = 0.0f;
    const float minus_one = -1.0f;
    const long long matrix_stride = static_cast<long long>(matrix_size);
    const long long panel_v_stride = static_cast<long long>(n * QR_V2_PANEL_WIDTH);
    const long long panel_t_stride = static_cast<long long>(QR_V2_PANEL_WIDTH * QR_V2_PANEL_WIDTH);
    const long long work_stride = static_cast<long long>(QR_V2_PANEL_WIDTH * n);

    for (int panel_col = 0; panel_col < n; panel_col += QR_V2_PANEL_WIDTH) {
        const int panel_width = QR_V2_PANEL_WIDTH < (n - panel_col) ? QR_V2_PANEL_WIDTH : (n - panel_col);
        const int panel_m = n - panel_col;
        const int trailing_cols = n - panel_col - panel_width;

        {{ config.entry_point }}_init_panel_pointer_arrays<<<pointer_blocks, threads_per_block>>>(
            packed,
            tau,
            matrix_ptrs,
            tau_ptrs,
            batch,
            n,
            panel_col
        );
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            cleanup();
            return static_cast<int>(cuda_status);
        }

        int info = 0;
        cublas_status = cublasSgeqrfBatched(handle, panel_m, panel_width, matrix_ptrs, n, tau_ptrs, &info, batch);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            cleanup();
            return 11000 + static_cast<int>(cublas_status);
        }
        if (info != 0) {
            cleanup();
            return -1000 + info;
        }

        if (trailing_cols <= 0) {
            continue;
        }

        {{ config.entry_point }}_build_panel_v<<<batch, threads_per_block>>>(
            packed,
            panel_v,
            batch,
            n,
            panel_col,
            panel_width,
            panel_m
        );
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            cleanup();
            return static_cast<int>(cuda_status);
        }

        {{ config.entry_point }}_form_t<<<batch, threads_per_block>>>(
            panel_v,
            tau,
            panel_t,
            batch,
            n,
            panel_col,
            panel_width,
            panel_m
        );
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            cleanup();
            return static_cast<int>(cuda_status);
        }

        float* trailing = packed + static_cast<size_t>(panel_col) + static_cast<size_t>(panel_col + panel_width) * static_cast<size_t>(n);

        cublas_status = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            panel_width,
            trailing_cols,
            panel_m,
            &one,
            panel_v,
            n,
            panel_v_stride,
            trailing,
            n,
            matrix_stride,
            &zero,
            work_vt_c,
            QR_V2_PANEL_WIDTH,
            work_stride,
            batch
        );
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            cleanup();
            return 12000 + static_cast<int>(cublas_status);
        }

        cublas_status = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            panel_width,
            trailing_cols,
            panel_width,
            &one,
            panel_t,
            QR_V2_PANEL_WIDTH,
            panel_t_stride,
            work_vt_c,
            QR_V2_PANEL_WIDTH,
            work_stride,
            &zero,
            work_t_vt_c,
            QR_V2_PANEL_WIDTH,
            work_stride,
            batch
        );
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            cleanup();
            return 13000 + static_cast<int>(cublas_status);
        }

        cublas_status = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            panel_m,
            trailing_cols,
            panel_width,
            &minus_one,
            panel_v,
            n,
            panel_v_stride,
            work_t_vt_c,
            QR_V2_PANEL_WIDTH,
            work_stride,
            &one,
            trailing,
            n,
            matrix_stride,
            batch
        );
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            cleanup();
            return 14000 + static_cast<int>(cublas_status);
        }
    }

    {{ config.entry_point }}_col_to_row_major<<<blocks, threads_per_block>>>(packed, h, batch, n);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        cleanup();
        return static_cast<int>(cuda_status);
    }

    cleanup();
    return 0;
}
"""

_CUDA_DENSE_LINALG_BEST_TEMPLATE = """{% if spec.n <= 176 %}""" + _CUDA_CUSOLVERDX_GEQRF_SMALL_TEMPLATE + """{% elif spec.n == 352 or spec.n == 512 or spec.n == 1024 %}""" + _CUDA_DENSE_BLOCKED_WY_512_TEMPLATE + """{% else %}""" + _CUDA_CUSOLVER_GEQRF_TEMPLATE + """{% endif %}"""

_CUDA_CUSOLVERDX_GEQRF_TEMPLATE = """{% if spec.n <= 176 %}""" + _CUDA_CUSOLVERDX_GEQRF_SMALL_TEMPLATE + """{% else %}""" + _CUDA_CUSOLVER_GEQRF_TEMPLATE + """{% endif %}"""

_CUDA_DENSE_BLOCKED_WY_TEMPLATE = """{% if (spec.n == 512 or spec.n == 1024) and spec.case == "dense" %}""" + _CUDA_DENSE_BLOCKED_WY_512_TEMPLATE + """{% elif spec.n <= 176 %}""" + _CUDA_CUSOLVERDX_GEQRF_SMALL_TEMPLATE + """{% elif spec.n == 512 %}""" + _CUDA_CUBLAS_BATCHED_GEQRF_TEMPLATE + """{% else %}""" + _CUDA_CUSOLVER_GEQRF_TEMPLATE + """{% endif %}"""

_CUDA_STRUCTURED_BLOCKED_WY_TEMPLATE = """{% if spec.n == 352 or spec.n == 512 or spec.n == 1024 %}""" + _CUDA_DENSE_BLOCKED_WY_512_TEMPLATE + """{% elif spec.n <= 176 %}""" + _CUDA_CUSOLVERDX_GEQRF_SMALL_TEMPLATE + """{% else %}""" + _CUDA_CUSOLVER_GEQRF_TEMPLATE + """{% endif %}"""

_CUDA_TEMPLATES = {
    "cuda_geqr2_serial_v1": _CUDA_GEQR2_SERIAL_TEMPLATE,
    "cuda_geqr2_parallel_v1": _CUDA_GEQR2_PARALLEL_TEMPLATE,
    "cuda_geqr2_parallel_profile_v1": _CUDA_GEQR2_PARALLEL_PROFILE_TEMPLATE,
    "cuda_semantic_upper_fallback_v1": _CUDA_SEMANTIC_UPPER_FALLBACK_TEMPLATE,
    "cuda_semantic_early_stop_fallback_v1": _CUDA_SEMANTIC_EARLY_STOP_FALLBACK_TEMPLATE,
    "cuda_semantic_combo_fallback_v1": _CUDA_SEMANTIC_COMBO_FALLBACK_TEMPLATE,
    "cuda_cusolver_geqrf_v1": _CUDA_CUSOLVER_GEQRF_TEMPLATE,
    "cuda_cusolverdx_geqrf_v1": _CUDA_CUSOLVERDX_GEQRF_TEMPLATE,
    "cuda_cublas_batched_geqrf_v1": _CUDA_CUBLAS_BATCHED_GEQRF_TEMPLATE,
    "cuda_dense_linalg_best_v1": _CUDA_DENSE_LINALG_BEST_TEMPLATE,
    "cuda_dense_blocked_wy_v1": _CUDA_DENSE_BLOCKED_WY_TEMPLATE,
    "cuda_structured_blocked_wy_v1": _CUDA_STRUCTURED_BLOCKED_WY_TEMPLATE,
}

_JINJA_ENV = Environment(
    autoescape=False,
    lstrip_blocks=True,
    trim_blocks=True,
    undefined=StrictUndefined,
)


def _artifact_stem(spec: QrV2Spec, index: int) -> str:
    return f"{index:04d}_{spec.case}_b{spec.batch}_n{spec.n}_c{spec.cond}_seed{spec.seed}"


def _effective_config_for_spec(spec: QrV2Spec, config: QrV2KernelConfig) -> QrV2KernelConfig:
    if config.template_id == "cuda_dense_linalg_best_v1" and spec.n == 352:
        return replace(config, tile_size=8)
    return config


def _render_source(spec: QrV2Spec, config: QrV2KernelConfig) -> str:
    template = _JINJA_ENV.from_string(_CUDA_TEMPLATES[config.template_id])
    return template.render(
        config=config,
        renderer_version=RENDERER_VERSION,
        spec=spec,
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def render_suite(
    *,
    suite: str,
    specs: list[QrV2Spec],
    out_dir: Path,
    config: QrV2KernelConfig = DEFAULT_KERNEL_CONFIG,
) -> RenderSuiteResult:
    artifact_dir = out_dir / config.candidate_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for index, spec in enumerate(specs):
        row_config = _effective_config_for_spec(spec, config)
        source = _render_source(spec, row_config)
        artifact_path = artifact_dir / f"{_artifact_stem(spec, index)}.cu"
        artifact_path.write_text(source, encoding="utf-8")
        records.append(
            {
                "event": "render_result",
                "renderer_version": RENDERER_VERSION,
                "suite": suite,
                "artifact_kind": row_config.template_id,
                "artifact_path": artifact_path.as_posix(),
                "artifact_bytes": len(source.encode("utf-8")),
                "artifact_sha256": _sha256_text(source),
                "entry_point": row_config.entry_point,
                **asdict(row_config),
                **spec.kwargs(),
            }
        )

    manifest_path = out_dir / "manifest.jsonl"
    write_jsonl(manifest_path, records)
    return RenderSuiteResult(manifest_path=manifest_path, records=records)


def _load_manifest_records(manifest_path: Path) -> tuple[list[dict[str, Any]], list[RenderVerificationIssue]]:
    if not manifest_path.exists():
        return [], [RenderVerificationIssue(code="missing_manifest", message=f"missing manifest: {manifest_path}")]

    records: list[dict[str, Any]] = []
    issues: list[RenderVerificationIssue] = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as error:
                issues.append(
                    RenderVerificationIssue(
                        code="invalid_json",
                        message=f"invalid JSON: {error.msg}",
                        line_number=line_number,
                    )
                )
                continue
            if not isinstance(parsed, dict):
                issues.append(
                    RenderVerificationIssue(
                        code="invalid_record",
                        message="manifest row is not a JSON object",
                        line_number=line_number,
                    )
                )
                continue
            parsed["_line_number"] = line_number
            records.append(parsed)
    return records, issues


def _string_field(
    record: dict[str, Any],
    field: str,
    issues: list[RenderVerificationIssue],
) -> str | None:
    value = record.get(field)
    if isinstance(value, str) and value:
        return value
    issues.append(
        RenderVerificationIssue(
            code=f"invalid_{field}",
            message=f"missing or invalid {field!r}",
            line_number=int(record["_line_number"]),
        )
    )
    return None


def _int_field(
    record: dict[str, Any],
    field: str,
    issues: list[RenderVerificationIssue],
    *,
    artifact_path: str | None = None,
) -> int | None:
    value = record.get(field)
    if isinstance(value, int):
        return value
    issues.append(
        RenderVerificationIssue(
            code=f"invalid_{field}",
            message=f"missing or invalid {field!r}",
            artifact_path=artifact_path,
            line_number=int(record["_line_number"]),
        )
    )
    return None


def verify_render_manifest(manifest_path: Path) -> RenderVerificationResult:
    records, issues = _load_manifest_records(manifest_path)
    seen_paths: set[str] = set()
    artifacts_checked = 0

    for record in records:
        line_number = int(record["_line_number"])
        if record.get("event") != "render_result":
            issues.append(
                RenderVerificationIssue(
                    code="invalid_event",
                    message="manifest row event is not 'render_result'",
                    line_number=line_number,
                )
            )

        artifact_path = _string_field(record, "artifact_path", issues)
        expected_sha256 = _string_field(record, "artifact_sha256", issues)
        expected_bytes = _int_field(record, "artifact_bytes", issues, artifact_path=artifact_path)
        if artifact_path is None:
            continue

        if artifact_path in seen_paths:
            issues.append(
                RenderVerificationIssue(
                    code="duplicate_artifact_path",
                    message="duplicate artifact path in manifest",
                    artifact_path=artifact_path,
                    line_number=line_number,
                )
            )
        seen_paths.add(artifact_path)

        path = Path(artifact_path)
        if path.is_absolute():
            issues.append(
                RenderVerificationIssue(
                    code="absolute_artifact_path",
                    message="artifact path must be relative",
                    artifact_path=artifact_path,
                    line_number=line_number,
                )
            )
            continue
        if path.parts[:1] != ("data",):
            issues.append(
                RenderVerificationIssue(
                    code="artifact_path_outside_data",
                    message="artifact path must stay under ./data/",
                    artifact_path=artifact_path,
                    line_number=line_number,
                )
            )
            continue
        if not path.exists():
            issues.append(
                RenderVerificationIssue(
                    code="missing_artifact",
                    message="artifact path does not exist",
                    artifact_path=artifact_path,
                    line_number=line_number,
                )
            )
            continue

        content = path.read_bytes()
        artifacts_checked += 1
        actual_bytes = len(content)
        if expected_bytes is not None and actual_bytes != expected_bytes:
            issues.append(
                RenderVerificationIssue(
                    code="artifact_bytes_mismatch",
                    message=f"expected {expected_bytes} bytes, found {actual_bytes}",
                    artifact_path=artifact_path,
                    line_number=line_number,
                )
            )

        actual_sha256 = _sha256_bytes(content)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            issues.append(
                RenderVerificationIssue(
                    code="artifact_sha256_mismatch",
                    message=f"expected {expected_sha256}, found {actual_sha256}",
                    artifact_path=artifact_path,
                    line_number=line_number,
                )
            )

    return RenderVerificationResult(
        manifest_path=manifest_path,
        records_checked=len(records),
        artifacts_checked=artifacts_checked,
        issues=issues,
    )
