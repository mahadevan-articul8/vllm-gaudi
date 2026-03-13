# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU adaptations for Qwen3-Next / Qwen3.5 models.

Key changes vs upstream Qwen3NextGatedDeltaNet._forward_core():
  1. causal_conv1d_fn / causal_conv1d_update
       → hpu_causal_conv1d_fn / hpu_causal_conv1d_update   (already in Gaudi)
  2. fused_gdn_gating (Triton kernel)
       → hpu_fused_gdn_gating                              (new, hpu_gdn.py)
  3. chunk_gated_delta_rule (FLA Triton)
       → hpu_chunk_gated_delta_rule                        (new, hpu_gdn.py)
  4. fused_recurrent_gated_delta_rule (FLA Triton)
       → hpu_fused_recurrent_gated_delta_rule              (new, hpu_gdn.py)

The MoE block (Qwen3NextSparseMoeBlock) gets the same 3-D reshape guard
that qwen3_moe.py applies to Qwen3MoeSparseMoeBlock.

HpuQwen3NextForCausalLM wraps Qwen3NextForCausalLM and replaces the
sub-modules after __init__ so that no upstream code needs patching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalFeatureSpec

from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextForCausalLM,
    Qwen3NextGatedDeltaNet,
    Qwen3NextSparseMoeBlock,
    fused_gdn_gating,                # upstream Triton version (imported for reference)
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5DecoderLayer,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_gaudi.ops.causal_conv1d_pytorch import (
    hpu_causal_conv1d_fn,
    hpu_causal_conv1d_update,
)
from vllm_gaudi.ops.hpu_gdn import (
    hpu_chunk_gated_delta_rule,
    hpu_fused_gdn_gating,
    hpu_fused_recurrent_gated_delta_rule,
)
from vllm_gaudi.ops.gdn_diagnostics import (
    GDN_BYPASS,
    GDN_CPU,
    diag_begin,
    diag_enabled,
    diag_end,
    diag_stage,
    install_layer_trace_hooks,
)


# ---------------------------------------------------------------------------
# GDN layer override
# ---------------------------------------------------------------------------

class HPUQwen3NextGatedDeltaNet(Qwen3NextGatedDeltaNet):
    """Qwen3NextGatedDeltaNet with Triton kernels replaced by HPU-safe ops."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """Override to handle batched 3-D (B, T, H) input from HPU model runner.

        The upstream forward expects 2-D (T, H) tensors because it uses
        ``hidden_states.size(0)`` as ``num_tokens`` and ``rearrange``
        patterns like ``"l p d -> l (p d)"`` that require exactly 3 dims.

        The HPU V1 model runner passes:
          - Prefill: (B=1, T, H) — squeeze(0) would work, but reshape is safer
          - Decode B=1: (1, 1, H) — identical
          - Decode B>1: (B, 1, H) — squeeze(0) is a no-op (B≠1), breaking split

        Use reshape(-1, H) to always flatten to (tokens, H) regardless of batch.
        """
        if hidden_states.dim() == 3:
            H = hidden_states.size(-1)
            hidden_states = hidden_states.reshape(-1, H)   # (B, T, H) → (B*T, H)
            out_H = output.size(-1)
            output = output.reshape(-1, out_H)             # (B, T, H) → (B*T, H)
        result = super().forward(hidden_states, output)
        return result

    def _find_mamba_group_idx(self) -> int:
        """Return the kv_cache_groups index for the MambaSpec group (GDN layers).

        The HPU model runner creates one group per KV cache type.  For a
        hybrid model there is typically one FullAttentionSpec group and one
        MambaSpec group.  We look up which group index has MambaSpec so we
        can extract the right row from the 2-D state_indices_tensor.
        """
        from vllm.config import get_current_vllm_config
        from vllm.v1.kv_cache_interface import MambaSpec
        from vllm.forward_context import get_forward_context
        import logging
        _logger = logging.getLogger(__name__)
        try:
            # kv_cache_config lives on the HPU model runner, not vllm_config.
            # We access it via the kv_cache on the first GDN layer: each layer's
            # kv_cache is a list indexed by virtual_engine, and kv_cache_config
            # is stored on the model runner.  Instead, inspect the kv_cache
            # groups via the forward context's attn_metadata source.
            #
            # Fallback: use vllm_config route (works on some vLLM builds).
            vllm_config = get_current_vllm_config()
            # Try cache_config path (different from compilation_config).
            kv_cache_config = getattr(vllm_config, 'kv_cache_config', None)
            if kv_cache_config is None:
                # Some builds store it nested differently.
                kv_cache_config = getattr(
                    vllm_config.compilation_config, 'kv_cache_config', None)
            if kv_cache_config is not None:
                for idx, group in enumerate(kv_cache_config.kv_cache_groups):
                    if isinstance(group.kv_cache_spec, MambaSpec):
                        _logger.info(
                            "HPU GDN: mamba_group_idx=%d (from kv_cache_config)", idx)
                        return idx
        except Exception as e:
            _logger.warning("HPU GDN: _find_mamba_group_idx failed: %s", e)

        # Hard fallback: probe by checking which kv_cache slot has the GDN
        # state shape.  The GDN kv_cache has 2 elements: conv_state and
        # ssm_state.  FullAttention kv_cache has 2 elements too (k, v) but
        # with different shapes.  We can't probe here without the actual
        # tensors, so log a warning and return 0.
        _logger.warning(
            "HPU GDN: could not resolve mamba_group_idx, defaulting to 0. "
            "If outputs are garbled, verify kv_cache_groups ordering.")
        return 0

    def _build_gdn_metadata(self, hpu_meta, mixed_qkv: torch.Tensor):
        """Construct GDNAttentionMetadata from HPUAttentionMetadataV1.

        The HPU model runner does not build a GDNAttentionMetadata dict; it
        passes an HPUAttentionMetadataV1 with Mamba-related fields already
        populated (state_indices_tensor, has_initial_states_p,
        query_start_loc_p, …).  This method translates those fields into the
        GDNAttentionMetadata that our _forward_core expects.

        LAZY-MODE SAFE: Uses CPU copies of metadata (query_start_loc_p_cpu,
        state_indices_tensor_cpu, has_initial_states_p_cpu) for .item() calls.
        The HPU tensors are still passed through for GPU-side computation.
        """
        state_idx = hpu_meta.state_indices_tensor  # [num_groups, padded_bs] or None
        # Read CPU copy from side-channel (not from metadata — CPU tensors
        # must stay out of the lazy graph / TrimmedAttentionMetadata).
        from vllm_gaudi.v1.attention.backends.hpu_attn import _gdn_cpu_metadata
        state_idx_cpu = _gdn_cpu_metadata.get('state_indices_tensor_cpu')  # CPU copy
        if state_idx is not None and state_idx.dim() == 2:
            # Lazily resolve the Mamba group index and cache it.
            if not hasattr(self, '_mamba_group_idx'):
                self._mamba_group_idx = self._find_mamba_group_idx()
            state_idx = state_idx[self._mamba_group_idx]  # → [padded_bs]
            if state_idx_cpu is not None and state_idx_cpu.dim() == 2:
                state_idx_cpu = state_idx_cpu[self._mamba_group_idx]

        # query_start_loc_p is [num_seqs+1] with cumulative token counts.
        qsl = hpu_meta.query_start_loc_p  # HPU tensor for GPU computation
        qsl_cpu = _gdn_cpu_metadata.get('query_start_loc_p_cpu')  # CPU copy for .item() calls

        if hpu_meta.is_prompt:
            # Prefill: potentially multiple sequences.
            # Use CPU tensors for scalar extraction (lazy-mode safe).
            if qsl_cpu is not None:
                n_prefills = qsl_cpu.shape[0] - 1
                n_actual = int(qsl_cpu[-1].item())
            elif qsl is not None:
                n_prefills = qsl.shape[0] - 1
                n_actual = int(qsl[-1].item())
            else:
                n_prefills = 1
                n_actual = mixed_qkv.shape[0]

            has_init = hpu_meta.has_initial_states_p  # [num_seqs] bool (HPU)

            return GDNAttentionMetadata(
                num_prefills=n_prefills,
                num_prefill_tokens=n_actual,
                num_decodes=0,
                num_decode_tokens=0,
                num_spec_decodes=0,
                num_spec_decode_tokens=0,
                num_actual_tokens=n_actual,
                has_initial_state=has_init,
                non_spec_query_start_loc=qsl,
                non_spec_state_indices_tensor=state_idx,
            )
        else:
            # Decode: each active sequence contributes exactly one token.
            if qsl_cpu is not None:
                n_actual = int(qsl_cpu[-1].item())
                n_decodes = qsl_cpu.shape[0] - 1
            elif qsl is not None:
                n_actual = int(qsl[-1].item())
                n_decodes = qsl.shape[0] - 1
            else:
                n_actual = mixed_qkv.shape[0]
                n_decodes = n_actual

            return GDNAttentionMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=n_decodes,
                num_decode_tokens=n_actual,
                num_spec_decodes=0,
                num_spec_decode_tokens=0,
                num_actual_tokens=n_actual,
                has_initial_state=None,
                non_spec_query_start_loc=qsl,
                non_spec_state_indices_tensor=state_idx,
            )

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """Core GDN attention — HPU version.

        Identical flow to upstream _forward_core but with:
          - causal_conv1d_fn/update  → hpu variants
          - fused_gdn_gating         → hpu_fused_gdn_gating
          - chunk_gated_delta_rule   → hpu_chunk_gated_delta_rule
          - fused_recurrent_gated_delta_rule → hpu version
        """
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Profile run — nothing to do
            return

        _diag = diag_enabled(self.prefix)

        # Read CPU metadata from side-channel (lazy-mode safe — not in lazy graph)
        from vllm_gaudi.v1.attention.backends.hpu_attn import _gdn_cpu_metadata
        _qsl_cpu = _gdn_cpu_metadata.get('query_start_loc_p_cpu')
        _state_idx_cpu = _gdn_cpu_metadata.get('state_indices_tensor_cpu')
        if _state_idx_cpu is not None and _state_idx_cpu.dim() == 2:
            if not hasattr(self, '_mamba_group_idx'):
                self._mamba_group_idx = self._find_mamba_group_idx()
            _state_idx_cpu = _state_idx_cpu[self._mamba_group_idx]

        if isinstance(attn_metadata, dict):
            # Upstream GPU path: attn_metadata is dict[layer_prefix, GDNAttentionMetadata]
            attn_metadata = attn_metadata[self.prefix]
            assert isinstance(attn_metadata, GDNAttentionMetadata)
        else:
            # HPU model runner path: attn_metadata is HPUAttentionMetadataV1
            attn_metadata = self._build_gdn_metadata(attn_metadata, mixed_qkv)

        has_initial_state = attn_metadata.has_initial_state
        # HPU model runner sends has_initial_states_p as int32 (via
        # async_h2d_copy with dtype=torch.int32).  Bitwise NOT (~) on int32
        # produces wrong indices (e.g. ~0 = -1), so convert to bool here.
        if has_initial_state is not None and has_initial_state.dtype != torch.bool:
            has_initial_state = has_initial_state.bool()
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor

        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        if _diag:
            is_prefill = attn_metadata.num_prefills > 0
            diag_begin(self.prefix, is_prefill, {
                "num_prefills": attn_metadata.num_prefills,
                "num_decodes": attn_metadata.num_decodes,
                "num_actual_tokens": num_actual_tokens,
                "has_initial_state": has_initial_state,
                "non_spec_query_start_loc": non_spec_query_start_loc,
                "non_spec_state_indices_tensor": non_spec_state_indices_tensor,
                "spec_sequence_masks": spec_sequence_masks,
                "conv_state_shape": list(conv_state.shape),
                "ssm_state_shape": list(ssm_state.shape),
            })
            diag_stage(self.prefix, "0-INPUTS",
                       mixed_qkv=mixed_qkv, b=b, a=a)

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        if _diag:
            diag_stage(self.prefix, "0-INPUTS-SLICED",
                       mixed_qkv=mixed_qkv, b=b, a=a)

        # ----------------------------------------------------------------
        # 1. causal_conv1d
        # ----------------------------------------------------------------
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            if (attn_metadata.num_prefills == 0
                    and attn_metadata.num_decodes == 0):
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(
                    0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1 speculative tokens
        if spec_sequence_masks is not None:
            mixed_qkv_spec = hpu_causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][
                    :attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # 1.2 normal tokens
        if attn_metadata.num_prefills > 0:
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            mixed_qkv_non_spec = hpu_causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                activation=self.activation,
                metadata=attn_metadata,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec = hpu_causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[
                    :attn_metadata.num_actual_tokens
                ],
                query_start_loc=non_spec_query_start_loc,
                validate_data=False,
            )
        else:
            mixed_qkv_non_spec = None

        if _diag:
            diag_stage(self.prefix, "1-CONV1D",
                       mixed_qkv_non_spec=mixed_qkv_non_spec,
                       mixed_qkv_spec=mixed_qkv_spec)

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(
            mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec)

        if _diag:
            diag_stage(self.prefix, "1-QKV-SPLIT",
                       query_non_spec=query_non_spec,
                       key_non_spec=key_non_spec,
                       value_non_spec=value_non_spec)

        # ----------------------------------------------------------------
        # 2. Gating  (replaces Triton fused_gdn_gating_kernel)
        # ----------------------------------------------------------------
        g, beta = hpu_fused_gdn_gating(self.A_log, a, b, self.dt_bias)

        if _diag:
            diag_stage(self.prefix, "2-GATING",
                       g=g, beta=beta,
                       A_log=self.A_log, dt_bias=self.dt_bias)

        if spec_sequence_masks is not None:
            if (attn_metadata.num_prefills == 0
                    and attn_metadata.num_decodes == 0):
                g_spec = g
                beta_spec = beta
                g_non_spec = None
                beta_non_spec = None
            else:
                g_spec = g.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                g_non_spec = g.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
        else:
            g_spec = None
            beta_spec = None
            g_non_spec = g
            beta_non_spec = beta

        # ----------------------------------------------------------------
        # 3. Recurrent attention
        # ----------------------------------------------------------------

        # 3.1 speculative tokens
        if spec_sequence_masks is not None:
            core_attn_out_spec, _ = hpu_fused_recurrent_gated_delta_rule(
                q=query_spec,
                k=key_spec,
                v=value_spec,
                g=g_spec,
                beta=beta_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[
                    :attn_metadata.num_spec_decodes + 1
                ],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
                scale=self.head_k_dim ** -0.5,
                # No CPU copies for spec path (not the common path)
            )
        else:
            core_attn_out_spec = None

        # 3.2 normal tokens (prefill)
        if attn_metadata.num_prefills > 0:
            initial_state = ssm_state[
                non_spec_state_indices_tensor].contiguous()

            if _diag:
                diag_stage(self.prefix, "3-PRE-RECURRENT-PREFILL",
                           initial_state_before_zero=initial_state,
                           has_initial_state=has_initial_state,
                           ssm_state_slot=ssm_state[non_spec_state_indices_tensor])

            initial_state[~has_initial_state, ...] = 0

            if _diag:
                diag_stage(self.prefix, "3-INITIAL-STATE-AFTER-ZERO",
                           initial_state=initial_state)

            core_attn_out_non_spec, last_state = hpu_chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=non_spec_query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
                scale=self.head_k_dim ** -0.5,
                cu_seqlens_cpu=_qsl_cpu,
            )

            if _diag:
                diag_stage(self.prefix, "3-RECURRENT-PREFILL-OUTPUT",
                           core_attn_out_non_spec=core_attn_out_non_spec,
                           last_state=last_state)

            # Write updated states back to the cache
            ssm_state[non_spec_state_indices_tensor] = last_state.to(
                ssm_state.dtype)
            # Commit the state write to HPU before any subsequent read (lazy mode)
            try:
                import habana_frameworks.torch as _htorch
                _htorch.core.mark_step()
            except ImportError:
                pass

        # 3.3 normal tokens (decode)
        elif attn_metadata.num_decodes > 0:
            if _diag:
                diag_stage(self.prefix, "3-PRE-RECURRENT-DECODE",
                           ssm_state_shape=ssm_state.shape,
                           non_spec_state_indices=non_spec_state_indices_tensor,
                           non_spec_qsl=non_spec_query_start_loc)

            core_attn_out_non_spec, _ = hpu_fused_recurrent_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc[
                    :attn_metadata.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
                scale=self.head_k_dim ** -0.5,
                cu_seqlens_cpu=_qsl_cpu[:attn_metadata.num_decodes + 1] if _qsl_cpu is not None else None,
                ssm_state_indices_cpu=_state_idx_cpu,
            )

            if _diag:
                diag_stage(self.prefix, "3-RECURRENT-DECODE-OUTPUT",
                           core_attn_out_non_spec=core_attn_out_non_spec)
        else:
            core_attn_out_non_spec = None

        # ----------------------------------------------------------------
        # 4. Merge outputs
        # ----------------------------------------------------------------
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

        if _diag:
            diag_end(self.prefix, core_attn_out[:num_actual_tokens])


# ---------------------------------------------------------------------------
# Qwen3.5-specific GDN override (flat Q/K/V/Z format)
# ---------------------------------------------------------------------------

class HPUQwen3_5GatedDeltaNet(HPUQwen3NextGatedDeltaNet):
    """GDN for Qwen3.5 models (qwen3_5_moe / qwen3_5 model_type).

    Qwen3.5 checkpoint weights are stored in FLAT format:
      in_proj_qkv  [key_dim*2 + value_dim, hidden]  →  [q, k, v]
      in_proj_z    [value_dim, hidden]               →  [z]

    Qwen3NextGatedDeltaNet.fix_query_key_value_ordering() assumes a
    GROUPED-PER-K-HEAD layout and reshapes the projection output as
    (tokens, num_k_heads, features_per_group) — this is WRONG for the
    flat layout, producing garbled Q/K/V/Z and garbage text output.

    Fix: bypass fix_query_key_value_ordering entirely and do a simple
    flat split, matching the upstream Qwen3_5GatedDeltaNet.forward().
    The _forward_core() (HPU ops) is unchanged because it receives the
    same mixed_qkv / b / a shapes either way.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        # Handle 3-D (B, T, H) input from HPU model runner.
        if hidden_states.dim() == 3:
            H = hidden_states.size(-1)
            hidden_states = hidden_states.reshape(-1, H)
            out_H = output.size(-1)
            output = output.reshape(-1, out_H)

        _diag = diag_enabled(self.prefix)
        from einops import rearrange as _rearrange
        num_tokens = hidden_states.size(0)

        if _diag:
            diag_stage(self.prefix, "FWD-ENTRY",
                       hidden_states=hidden_states,
                       key_dim=self.key_dim,
                       value_dim=self.value_dim,
                       tp_size=self.tp_size,
                       num_v_heads=self.num_v_heads,
                       head_v_dim=self.head_v_dim,
                       head_k_dim=self.head_k_dim)

        # Part 1: Input projection — flat split, no fix_query_key_value_ordering.
        # in_proj_qkvz weight = cat([in_proj_qkv, in_proj_z], dim=0), so the
        # linear output columns are [q, k, v, z] in contiguous flat order.
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size   = self.value_dim // self.tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z.reshape(z.size(0), -1, self.head_v_dim)

        # in_proj_ba weight = cat([in_proj_b, in_proj_a], dim=0).
        ba, _ = self.in_proj_ba(hidden_states)
        b, a = ba.chunk(2, dim=-1)
        b = b.contiguous()
        a = a.contiguous()

        if _diag:
            diag_stage(self.prefix, "FWD-PROJECTIONS",
                       mixed_qkvz=mixed_qkvz,
                       mixed_qkv=mixed_qkv, z=z,
                       b=b, a=a,
                       qkv_size=qkv_size, z_size=z_size)

        # Part 2: Core GDN attention (dispatches to HPUQwen3NextGatedDeltaNet._forward_core).
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if not GDN_BYPASS:
            torch.ops.vllm.gdn_attention_core(mixed_qkv, b, a, core_attn_out, self.prefix)

        if _diag:
            diag_stage(self.prefix, "FWD-AFTER-CORE",
                       core_attn_out=core_attn_out)

        # Part 3: Output projection (same as upstream Qwen3_5GatedDeltaNet.forward).
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)

        if _diag:
            diag_stage(self.prefix, "FWD-AFTER-NORM",
                       core_attn_out_normed=core_attn_out)

        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = _rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

        if _diag:
            diag_stage(self.prefix, "FWD-OUTPUT",
                       output=output[:num_tokens])


# ---------------------------------------------------------------------------
# MoE block override  (same 3-D reshape guard as qwen3_moe.py)
# ---------------------------------------------------------------------------

class HPUQwen3NextAttention(Qwen3NextAttention):
    """Qwen3NextAttention with HPU shape handling.

    The HPU model runner passes (B, T, H) 3-D tensors.  The base forward
    captures ``orig_shape = q_gate.shape[:-1]`` which preserves the T
    dimension, keeping gate as 3-D (B, T, H).  Then ``attn_output * gate``
    broadcasts to 3-D and ``o_proj`` returns 3-D ``(s0, s2, H)`` where s2 is
    a symbolic dim that dynamo cannot prove equals 1 — causing the expand
    error when assigning into the pre-allocated ``output`` buffer.

    Fix: flatten to 2-D ``(B*T, H)`` before delegating.  Since output is a
    contiguous tensor allocated by ``torch.empty_like(hidden_states)``, the
    reshaped 2-D view shares storage — writes propagate back to the caller.
    """

    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        import os as _os_fa, sys as _sys_fa
        _fa_diag = _os_fa.environ.get("FA_ATTN_DIAG", "0") == "1"
        if _fa_diag:
            from vllm.forward_context import get_forward_context
            _fctx = get_forward_context()
            _meta = _fctx.attn_metadata if _fctx else None
            _seq_lens = None
            if _meta is not None and hasattr(_meta, 'seq_lens_tensor') and _meta.seq_lens_tensor is not None:
                _seq_lens = _meta.seq_lens_tensor.cpu().tolist()
            T = hidden_states.shape[-2] if hidden_states.dim() == 3 else hidden_states.shape[0]
            pos_vals = positions.cpu().tolist() if positions.numel() <= 6 else positions.reshape(-1)[:6].cpu().tolist()
            _layer_idx = getattr(self, '_hpu_fa_idx', '?')
            print(f"[FA_FWD] layer={_layer_idx} T={T} pos_shape={list(positions.shape)} pos={pos_vals} seq_lens={_seq_lens} hs_norm={hidden_states.float().norm().item():.4f}", flush=True, file=_sys_fa.stderr)
        if hidden_states.dim() == 3:
            H = hidden_states.size(-1)
            hidden_states = hidden_states.reshape(-1, H)   # (B, T, H) → (B*T, H)
            out_H = output.size(-1)
            output = output.reshape(-1, out_H)             # (B, T, H) → (B*T, H)
        result = super().forward(
            positions=positions, output=output, hidden_states=hidden_states
        )
        if _fa_diag:
            print(f"[FA_OUT] out_norm={output.float().norm().item():.4f}", flush=True, file=_sys_fa.stderr)
        return result


class HPUQwen3NextSparseMoeBlock(Qwen3NextSparseMoeBlock):
    """Qwen3NextSparseMoeBlock with HPU-friendly tensor handling."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        hs = hidden_states.reshape(-1, hidden_dim)
        return super().forward(hs).reshape(*orig_shape[:-1], hidden_dim)


# ---------------------------------------------------------------------------
# In-place upgrade helpers
# ---------------------------------------------------------------------------

def upgrade_qwen3_next_blocks_inplace(model: nn.Module) -> None:
    """Walk the model and replace GDN + MoE blocks with HPU variants."""
    # Handle VL wrapper: model.language_model.model.layers
    lm_model = getattr(model, "model", None)
    if lm_model is None:
        # Try VL path: model.language_model.model
        lm = getattr(model, "language_model", None)
        if lm is not None:
            lm_model = getattr(lm, "model", None)
    layers = getattr(lm_model, "layers", None)
    if layers is None:
        return

    # Detect Qwen3.5 vs Qwen3-Next by checking if GDN layers are
    # Qwen3_5GatedDeltaNet (from qwen3_5.py) or Qwen3NextGatedDeltaNet.
    # This is more reliable than checking model_type since the config
    # may be nested differently across VL/text-only models.
    cfg = getattr(model, "config", None)
    hf_model_type = getattr(cfg, "model_type", "") or ""
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        hf_model_type = getattr(text_cfg, "model_type", hf_model_type) or hf_model_type
    is_qwen3_5 = hf_model_type.startswith("qwen3_5")
    gdn_class = HPUQwen3_5GatedDeltaNet if is_qwen3_5 else HPUQwen3NextGatedDeltaNet
    # Base class to match: Qwen3_5GatedDeltaNet inherits from Qwen3NextGatedDeltaNet
    gdn_base = Qwen3NextGatedDeltaNet

    import logging
    _log = logging.getLogger(__name__)
    _log.info("HPU Qwen3Next upgrade: model_type=%r → GDN class=%s",
              hf_model_type, gdn_class.__name__)

    gdn_replaced = 0
    fa_replaced = 0
    moe_replaced = 0

    for layer in layers:
        # Replace GDN (linear attention) block.
        for gdn_attr in ("linear_attn", "self_attn"):
            gdn_block = getattr(layer, gdn_attr, None)
            if (gdn_block is not None
                    and isinstance(gdn_block, gdn_base)
                    and not isinstance(gdn_block, gdn_class)):
                gdn_block.__class__ = gdn_class
                gdn_replaced += 1
                break

        # Replace full-attention block (stored as layer.self_attn on FA layers).
        # GDN layers also have self_attn but it is a Qwen3NextGatedDeltaNet,
        # so the isinstance check is exclusive.
        fa_block = getattr(layer, "self_attn", None)
        if (fa_block is not None
                and isinstance(fa_block, Qwen3NextAttention)
                and not isinstance(fa_block, HPUQwen3NextAttention)):
            fa_block.__class__ = HPUQwen3NextAttention
            fa_block._hpu_fa_idx = fa_replaced  # 0-based FA layer index for diagnostics
            fa_replaced += 1

        # Replace MoE block
        mlp = getattr(layer, "mlp", None)
        if (mlp is not None
                and isinstance(mlp, Qwen3NextSparseMoeBlock)
                and not isinstance(mlp, HPUQwen3NextSparseMoeBlock)):
            mlp.__class__ = HPUQwen3NextSparseMoeBlock
            moe_replaced += 1

    if gdn_replaced or fa_replaced or moe_replaced:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            "HPU Qwen3Next: replaced %d GDN, %d FA, %d MoE layers.",
            gdn_replaced, fa_replaced, moe_replaced,
        )

    # Install per-layer trace hooks if GDN_LAYER_TRACE=1
    install_layer_trace_hooks(model)


# ---------------------------------------------------------------------------
# Top-level model class registered with ModelRegistry
# ---------------------------------------------------------------------------

class HpuQwen3NextForCausalLM(Qwen3NextForCausalLM):
    """Qwen3NextForCausalLM with HPU-optimised sub-modules."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        upgrade_qwen3_next_blocks_inplace(self)


class _Qwen35MRoPEMixin:
    """Mixin providing text-only mRoPE positions for Qwen3.5 hybrid models.

    Qwen3.5 has ``mrope_section`` in its ``rope_parameters``, which
    triggers ``uses_mrope=True`` in vLLM.  For text-only inference
    (no vision tokens) all three mRoPE sections share the same
    sequential positions.
    """

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: "list[MultiModalFeatureSpec]",
    ) -> "tuple[torch.Tensor, int]":
        n = len(input_tokens)
        pos = np.broadcast_to(np.arange(n), (3, n))
        return torch.from_numpy(pos.copy()), 0


class HpuQwen3_5MoeForCausalLM(
    _Qwen35MRoPEMixin, Qwen3_5MoeForCausalLM
):
    """Qwen3_5MoeForCausalLM with HPU-optimised sub-modules."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        upgrade_qwen3_next_blocks_inplace(self)


class HpuQwen3_5MoeForConditionalGeneration(
    _Qwen35MRoPEMixin, Qwen3_5MoeForConditionalGeneration
):
    """Qwen3_5MoeForConditionalGeneration with HPU-optimised sub-modules.

    The Qwen3.5-35B-A3B checkpoint stores text weights under
    ``model.language_model.*`` with stacked expert weights. The
    base class ``Qwen3_5MoeForConditionalGeneration`` uses
    ``AutoWeightsLoader`` which traverses to ``language_model.model``
    and calls ``Qwen3_5Model.load_weights`` — that method already
    handles GDN stacked params (in_proj_qkv→qkvz, in_proj_b→ba)
    and expert mapping via ``get_expert_mapping()``.

    This HPU override only needs to:
    1. Strip the ``model.language_model.`` prefix
    2. Unstack the fused expert tensors (gate_up_proj, down_proj)
    3. Skip vision weights
    """

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        upgrade_qwen3_next_blocks_inplace(self)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        return super().load_weights(self._remap_weights(weights))

    @staticmethod
    def _remap_weights(
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Transform Qwen3.5 VL checkpoint keys for text-only loading.

        Checkpoint layout:
          lm_head.weight                              (top-level)
          model.language_model.{embed_tokens,layers,norm}.*
          model.language_model.layers.N.mlp.experts.{gate_up_proj,down_proj}  (stacked)
          model.visual.*                              (skipped)

        Module hierarchy:
          HpuQwen3_5MoeForConditionalGeneration
            └─ language_model  (Qwen3_5MoeForCausalLM)
                  ├─ lm_head
                  └─ model  (Qwen3_5Model: embed_tokens, layers, norm)

        Remapping:
          model.language_model.X  →  language_model.model.X
          lm_head.weight          →  language_model.lm_head.weight
          model.visual.*          →  (skipped)
        """
        _LM_PFX = "model.language_model."
        _VIS_PFX = "model.visual."

        for name, tensor in weights:
            # Skip vision encoder — not used by text-only inference
            if name.startswith(_VIS_PFX):
                continue

            # Top-level lm_head → language_model.lm_head
            if name.startswith("lm_head."):
                name = "language_model." + name
            # model.language_model.X → language_model.model.X
            # (embed_tokens, layers, norm are inside Qwen3_5Model = language_model.model)
            elif name.startswith(_LM_PFX):
                name = "language_model.model." + name[len(_LM_PFX):]

            # Stacked expert weights → per-expert entries
            if name.endswith(".mlp.experts.gate_up_proj"):
                base = name[: -len(".gate_up_proj")]
                n_experts = tensor.shape[0]
                half = tensor.shape[1] // 2
                for i in range(n_experts):
                    yield base + f".{i}.gate_proj.weight", tensor[i, :half]
                    yield base + f".{i}.up_proj.weight", tensor[i, half:]
                continue

            if name.endswith(".mlp.experts.down_proj"):
                base = name[: -len(".down_proj")]
                n_experts = tensor.shape[0]
                for i in range(n_experts):
                    yield base + f".{i}.down_proj.weight", tensor[i]
                continue

            yield name, tensor
