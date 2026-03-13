# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch reference implementations for GatedDeltaNet (GDN) operations.

These replace the Triton-only FLA kernels in:
  vllm.model_executor.layers.fla.ops.fused_recurrent
  vllm.model_executor.layers.fla.ops.chunk

On HPU, Triton has no active driver and is disabled at import time.
These pure-PyTorch implementations are graph-capturable by the HPU lazy
execution engine (matmul + elementwise → auto-fused).

Enabled via the existing env selector pattern used for Mamba ops.
The Qwen3NextGatedDeltaNet layer checks this at _forward_core time.

Tensor shape conventions (continuous-batching, head-first after rearrange):
  q, k : [1, T, H,  K]   H  = num_key_heads   (16 for Qwen3.5-35B-A3B)
  v    : [1, T, HV, V]   HV = num_value_heads  (32 for Qwen3.5-35B-A3B)
  g    : [1, T, HV]      decay gate  (float32)
  beta : [1, T, HV]      update gate
  state: [N_slots, HV, K, V]  recurrent hidden state
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import habana_frameworks.torch as htorch
    _htorch_available = True
except ImportError:
    _htorch_available = False

# Flush the HPU lazy queue every N token steps inside _gdn_recurrent_segment.
# Without this, all T iterations queue up before the outer mark_step() fires,
# creating a T-op unrolled recipe that takes minutes to compile for T=2048.
# With this, SynapseAI compiles ONE small recipe on step 0 and replays it for
# steps 1..T-1 from cache — first-run compilation drops from minutes to seconds.
_GDN_MARK_STEP_INTERVAL = 1  # flush every step; recipe reuse makes this fast


# ---------------------------------------------------------------------------
# Helper: expand key/query heads to match value-head count (like GQA)
# ---------------------------------------------------------------------------

def _expand_kq_to_hv(
    x: torch.Tensor,
    num_v_heads: int,
) -> torch.Tensor:
    """Expand [B, T, H, D] to [B, T, HV, D] by repeating each key head.

    With H=16 key heads and HV=32 value heads each key head covers 2 value
    heads (repeat_interleave factor = HV // H = 2).
    """
    h = x.shape[2]
    if h == num_v_heads:
        return x
    assert num_v_heads % h == 0, f"HV={num_v_heads} must be divisible by H={h}"
    return x.repeat_interleave(num_v_heads // h, dim=2)


# ---------------------------------------------------------------------------
# 1. Gating computation (replaces fused_gdn_gating_kernel Triton op)
# ---------------------------------------------------------------------------

def hpu_fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token decay (g) and update gate (beta_out).

    Args:
        A_log   : [H]       log of A parameter, one per value head
        a       : [T, H]    dt-like input (pre-softplus)
        b       : [T, H]    gate input (pre-sigmoid)
        dt_bias : [H]       bias added before softplus
        beta    : softplus beta parameter (default 1.0)
        threshold: threshold above which softplus == identity (default 20.0)

    Returns:
        g        : [1, T, H]  float32 decay gate
        beta_out : [1, T, H]  update gate (same dtype as b)
    """
    # x = a + dt_bias  (broadcast H over T)
    x = a.float() + dt_bias.float()                    # [T, H]

    # softplus(a + dt_bias)
    sp_x = F.softplus(x)                               # [T, H]

    # g = -exp(A_log) * softplus(a + dt_bias)
    g = -A_log.float().exp() * sp_x                    # [T, H]

    # beta_out = sigmoid(b)
    beta_out = torch.sigmoid(b)                        # [T, H]

    return g.unsqueeze(0), beta_out.unsqueeze(0)       # [1,T,H], [1,T,H]


# ---------------------------------------------------------------------------
# 2. Core recurrent update over one sequence segment
# ---------------------------------------------------------------------------

def _gdn_recurrent_segment(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    use_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run GatedDeltaNet recurrent update over T tokens for a single sequence.

    All tensors are pre-sliced to this sequence's token range.

    Args:
        q    : [T, HV, K]   query  (already expanded to HV heads)
        k    : [T, HV, K]   key    (already expanded to HV heads)
        v    : [T, HV, V]   value
        g    : [T, HV]      decay gate  (float32, negative)
        beta : [T, HV]      update gate
        h    : [HV, K, V]   initial hidden state (modified in-place)
        scale: query scale factor
        use_l2norm: whether to L2-normalise q and k

    Returns:
        o : [T, HV, V]  output
        h : [HV, K, V]  final hidden state (same tensor as input h)
    """
    T = q.shape[0]
    device = q.device
    work_dtype = torch.float32

    # Upcast for numerical stability
    q = q.to(work_dtype)
    k = k.to(work_dtype)
    v = v.to(work_dtype)
    g = g.to(work_dtype)
    beta = beta.to(work_dtype)
    h = h.to(work_dtype)

    if use_l2norm:
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)     # [T, HV, K]
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)     # [T, HV, K]

    q = q * scale

    outputs = torch.zeros(T, *v.shape[1:], dtype=work_dtype, device=device)

    for t in range(T):
        # Decay: h *= exp(g[t])  — g is negative so this shrinks h
        # g[t]: [HV], h: [HV, K, V]
        h = h * g[t].exp().unsqueeze(-1).unsqueeze(-1)  # [HV, K, V]

        # Delta error: e = v[t] - k[t]^T @ h
        # k[t]: [HV, K],  h: [HV, K, V]  →  kh: [HV, V]
        kh = torch.einsum('hk,hkv->hv', k[t], h)
        e = v[t] - kh                                  # [HV, V]

        # State update: h += outer(k[t], beta[t] * e)
        # beta[t]: [HV],  e: [HV, V]  →  beta*e: [HV, V]
        # k[t]: [HV, K]               →  outer: [HV, K, V]
        h = h + torch.einsum('hk,hv->hkv', k[t], beta[t].unsqueeze(-1) * e)

        # Output: o[t] = q[t] @ h
        # q[t]: [HV, K],  h: [HV, K, V]  →  o: [HV, V]
        outputs[t] = torch.einsum('hk,hkv->hv', q[t], h)

        # Flush the HPU lazy queue every _GDN_MARK_STEP_INTERVAL steps.
        # This breaks the T-step unrolled lazy graph into small reusable recipes:
        # step 0 compiles a ~5-op recipe; steps 1..T-1 replay it from cache.
        if _htorch_available and device.type == 'hpu' and (t + 1) % _GDN_MARK_STEP_INTERVAL == 0:
            htorch.core.mark_step()

    return outputs, h


# ---------------------------------------------------------------------------
# 3. fused_recurrent_gated_delta_rule  (decode path — one token per seq)
# ---------------------------------------------------------------------------

def hpu_fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    scale: float = 1.0,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens_cpu: torch.Tensor | None = None,
    ssm_state_indices_cpu: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch GDN recurrent update.

    Replaces fused_recurrent_gated_delta_rule from FLA (Triton).

    For decode: T = N_decode (one token per sequence).
    For prefill: T = sum of all sequence lengths; cu_seqlens delimits them.

    Args:
        q, k  : [1, T, H,  K]
        v     : [1, T, HV, V]
        g     : [1, T, HV]      decay (float32, from hpu_fused_gdn_gating)
        beta  : [1, T, HV]      update gate
        initial_state : [N_slots, HV, K, V]  state cache (indexed by ssm_state_indices)
        inplace_final_state: write updated state back into initial_state
        cu_seqlens : [N_seqs+1]  cumulative token offsets; None = single sequence
        ssm_state_indices : [N_seqs] or [N_seqs, num_spec]  state slot per sequence
        num_accepted_tokens: for speculative decoding (not yet supported)
        scale  : query scaling factor
        use_qk_l2norm_in_kernel: apply L2 norm to q and k before computation
        cu_seqlens_cpu: CPU copy of cu_seqlens (avoids .cpu() in lazy mode)
        ssm_state_indices_cpu: CPU copy of ssm_state_indices

    Returns:
        o            : [1, T, HV, V]  output activations
        final_state  : same as initial_state (modified in-place if requested)
    """
    # Strip the batch=1 leading dimension
    q = q.squeeze(0)    # [T, H,  K]
    k = k.squeeze(0)    # [T, H,  K]
    v = v.squeeze(0)    # [T, HV, V]
    g = g.squeeze(0)    # [T, HV]
    beta = beta.squeeze(0)  # [T, HV]

    T_total = q.shape[0]
    HV = v.shape[1]
    orig_dtype = v.dtype

    # Expand q, k from H heads to HV heads (GQA-style repeat)
    q_exp = _expand_kq_to_hv(q.unsqueeze(0), HV).squeeze(0)  # [T, HV, K]
    k_exp = _expand_kq_to_hv(k.unsqueeze(0), HV).squeeze(0)  # [T, HV, K]

    # Determine sequence boundaries using CPU tensors (lazy-mode safe)
    _cu_cpu = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    _idx_cpu = ssm_state_indices_cpu if ssm_state_indices_cpu is not None else ssm_state_indices

    if _cu_cpu is None:
        # Single sequence covering all T tokens
        seq_starts = [0]
        seq_ends = [T_total]
        if _idx_cpu is None:
            state_slots = [0]
        else:
            state_slots = [int(_idx_cpu[0].item())]
    else:
        cu = _cu_cpu.cpu().tolist() if _cu_cpu.device.type != 'cpu' else _cu_cpu.tolist()
        seq_starts = cu[:-1]
        seq_ends = cu[1:]
        N_seqs = len(seq_starts)
        if _idx_cpu is None:
            state_slots = list(range(N_seqs))
        elif _idx_cpu.ndim == 1:
            state_slots = _idx_cpu.cpu().tolist() if _idx_cpu.device.type != 'cpu' else _idx_cpu.tolist()
        else:
            # 2D: [N_seqs, num_spec] — use column 0 (last accepted token's slot)
            if num_accepted_tokens is not None:
                _nat = num_accepted_tokens.cpu() if num_accepted_tokens.device.type != 'cpu' else num_accepted_tokens
                idxs = (_nat - 1).clamp(min=0).tolist()
                state_slots = [
                    int(_idx_cpu[i, idxs[i]].item())
                    for i in range(N_seqs)
                ]
            else:
                col0 = _idx_cpu[:, 0]
                state_slots = col0.cpu().tolist() if col0.device.type != 'cpu' else col0.tolist()

    all_outputs = torch.zeros(T_total, HV, v.shape[-1],
                              dtype=torch.float32, device=v.device)

    for i, (s, e, slot) in enumerate(zip(seq_starts, seq_ends, state_slots)):
        if e <= s:
            continue
        slot = int(slot)
        h = initial_state[slot].clone()               # [HV, K, V]

        o_seg, h_new = _gdn_recurrent_segment(
            q=q_exp[s:e],
            k=k_exp[s:e],
            v=v[s:e],
            g=g[s:e],
            beta=beta[s:e],
            h=h,
            scale=scale,
            use_l2norm=use_qk_l2norm_in_kernel,
        )
        all_outputs[s:e] = o_seg

        # CPU reference check — only when GDN_CPU_CMP_DIAG=1 (expensive: forces HPU sync)
        import os as _os_gdn, sys
        _gdn_cmp_diag = _os_gdn.environ.get("GDN_CPU_CMP_DIAG", "0") == "1"
        if _gdn_cmp_diag:
            print(f"[GDN_DBG] T_total={T_total} dev={v.device.type} slot={slot}", flush=True, file=sys.stderr)
        if _gdn_cmp_diag and T_total == 1:
            # Force sync to get actual HPU output
            if _htorch_available:
                htorch.core.mark_step()
            hpu_out_val = o_seg.float().cpu()
            # Run same computation on CPU with same inputs
            cpu_q = q_exp[s:e].float().cpu()
            cpu_k = k_exp[s:e].float().cpu()
            cpu_v = v[s:e].float().cpu()
            cpu_g = g[s:e].float().cpu()
            cpu_beta = beta[s:e].float().cpu()
            cpu_h_init = initial_state[slot].float().cpu()  # pre-update state
            cpu_o_seg, cpu_h_new = _gdn_recurrent_segment(
                q=cpu_q, k=cpu_k, v=cpu_v, g=cpu_g, beta=cpu_beta,
                h=cpu_h_init, scale=scale, use_l2norm=use_qk_l2norm_in_kernel,
            )
            max_diff = (hpu_out_val - cpu_o_seg.float()).abs().max().item()
            rel_diff = max_diff / (cpu_o_seg.float().norm().item() + 1e-8)
            print(f"[GDN_CPU_CMP] dev={v.device.type} slot={slot} max_diff={max_diff:.6f} rel={rel_diff:.4f} hpu_norm={hpu_out_val.norm():.4f} cpu_norm={cpu_o_seg.float().norm():.4f}", flush=True, file=sys.stderr)

        if inplace_final_state:
            if _gdn_cmp_diag:
                _norm_before = initial_state[slot].float().norm().item()
            initial_state[slot].copy_(h_new.to(initial_state.dtype))
            # Force HPU lazy graph to commit the state write before next read
            if _htorch_available:
                htorch.core.mark_step()
            if _gdn_cmp_diag:
                _norm_after = initial_state[slot].float().norm().item()
                import sys
                print(f"[GDN_DECODE] slot={slot} norm_before={_norm_before:.4f} norm_after={_norm_after:.4f} h_new_norm={h_new.float().norm().item():.4f}", flush=True, file=sys.stderr)

    # Restore batch dim and original dtype
    output = all_outputs.to(orig_dtype).unsqueeze(0)  # [1, T, HV, V]
    return output, initial_state


# ---------------------------------------------------------------------------
# 4. chunk_gated_delta_rule  (prefill path)
# ---------------------------------------------------------------------------

def hpu_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    scale: float = 1.0,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch GDN chunked prefill.

    Phase 2a: Delegates to the recurrent implementation.  Correct but O(T)
    sequential.  A true chunk algorithm (Phase 2b) will process chunk_size=64
    tokens in parallel using matmul, but requires more complex state carry-over.

    Args:
        q, k  : [1, T, H,  K]  (head_first=False layout, as used in _forward_core)
        v     : [1, T, HV, V]
        g     : [1, T, HV]
        beta  : [1, T, HV]
        initial_state : [N_seqs, HV, K, V]  one entry per sequence in this batch
        output_final_state: return the updated state
        cu_seqlens : [N_seqs+1] cumulative token offsets
        head_first : must be False (the calling code passes head_first=False)
        scale  : query scaling factor
        use_qk_l2norm_in_kernel: L2 normalise q and k

    Returns:
        o           : [1, T, HV, V]
        final_state : [N_seqs, HV, K, V]  (only meaningful if output_final_state)
    """
    assert not head_first, "head_first=True layout not supported in HPU GDN fallback"

    # For prefill, the initial_state is already pre-sliced to [N_seqs, HV, K, V]
    # by the caller:
    #   initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
    # We write back into a copy and the caller copies it back to ssm_state.

    q = q.squeeze(0)        # [T, H,  K]
    k = k.squeeze(0)        # [T, H,  K]
    v = v.squeeze(0)        # [T, HV, V]
    g = g.squeeze(0)        # [T, HV]
    beta = beta.squeeze(0)  # [T, HV]

    T_total = q.shape[0]
    HV = v.shape[1]
    V = v.shape[2]
    orig_dtype = v.dtype

    q_exp = _expand_kq_to_hv(q.unsqueeze(0), HV).squeeze(0)   # [T, HV, K]
    k_exp = _expand_kq_to_hv(k.unsqueeze(0), HV).squeeze(0)   # [T, HV, K]

    _cu_cpu = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    if _cu_cpu is None:
        # Single sequence
        cu = [0, T_total]
    else:
        cu = _cu_cpu.cpu().tolist() if _cu_cpu.device.type != 'cpu' else _cu_cpu.tolist()

    N_seqs = len(cu) - 1
    all_outputs = torch.zeros(T_total, HV, V, dtype=torch.float32, device=v.device)
    final_states = initial_state.clone().float()               # [N_seqs, HV, K, V]

    for i in range(N_seqs):
        s, e = cu[i], cu[i + 1]
        if e <= s:
            continue

        o_seg, h_new = _gdn_recurrent_segment(
            q=q_exp[s:e],
            k=k_exp[s:e],
            v=v[s:e],
            g=g[s:e],
            beta=beta[s:e],
            h=final_states[i],
            scale=scale,
            use_l2norm=use_qk_l2norm_in_kernel,
        )
        all_outputs[s:e] = o_seg
        final_states[i] = h_new

    output = all_outputs.to(orig_dtype).unsqueeze(0)           # [1, T, HV, V]
    final_states = final_states.to(initial_state.dtype)

    return output, final_states
