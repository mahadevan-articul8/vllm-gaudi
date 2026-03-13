# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU override for the GatedDeltaNet (GDN) attention backend.

Registers HPUGDNAttentionBackend as the GDN_ATTN backend so that
Qwen3NextGatedDeltaNet layers use HPU-compatible metadata construction
instead of the default Triton-backed path.

The metadata builder itself (GDNAttentionMetadataBuilder) only uses
PyTorch and Python to construct index tensors — no Triton calls — so
inheriting from the base class is sufficient for Phase 2/3.
"""

from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import (
    MambaAttentionBackendEnum,
    register_backend,
)


@register_backend(MambaAttentionBackendEnum.GDN_ATTN, is_mamba=True)
class HPUGDNAttentionBackend(GDNAttentionBackend):
    """HPU-specific GDN attention backend.

    Overrides the default GDNAttentionBackend to signal HPU usage.
    The metadata builder is inherited unchanged because it only uses
    pure-PyTorch tensor operations (no Triton kernels).

    The actual recurrent computation is performed by HPUQwen3NextGatedDeltaNet
    which calls the PyTorch GDN ops in hpu_gdn.py instead of the FLA Triton
    kernels.
    """

    @staticmethod
    def get_name() -> str:
        return "HPU_GDN_ATTN"

    @staticmethod
    def get_builder_cls():
        return HPUGDNAttentionMetadataBuilder


class HPUGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    """GDN metadata builder for HPU.

    Inherits the base class build() logic which constructs cu_seqlens,
    state_indices_tensor, and causal_conv1d metadata tensors using standard
    PyTorch ops.  All tensors land on the target device (HPU).
    """
    pass


__all__ = ["HPUGDNAttentionBackend", "HPUGDNAttentionMetadataBuilder"]
