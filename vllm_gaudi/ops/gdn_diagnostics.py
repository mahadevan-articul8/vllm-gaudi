# SPDX-License-Identifier: Apache-2.0
"""Diagnostics for GDN forward pass — temporary instrumentation.

Usage: set GDN_DIAG=1 env var to enable.  Prints tensor stats at each
stage of _forward_core for the first N calls per layer.

    GDN_DIAG=1 GDN_DIAG_CALLS=3 python test_qwen35_port.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import torch

_ENABLED = os.environ.get("GDN_DIAG", "0") == "1"
_MAX_CALLS = int(os.environ.get("GDN_DIAG_CALLS", "3"))
_LAYER_FILTER = os.environ.get("GDN_DIAG_LAYER", "")  # e.g. "0" or "0,1,2" or "" for all
_LAYER_TRACE = os.environ.get("GDN_LAYER_TRACE", "0") == "1"
_LAYER_TRACE_CALLS = int(os.environ.get("GDN_LAYER_TRACE_CALLS", "2"))
GDN_BYPASS = os.environ.get("GDN_BYPASS", "0") == "1"  # Skip GDN core, return zeros
GDN_CPU = os.environ.get("GDN_CPU", "0") == "1"  # Run GDN core on CPU to test HPU correctness
_trace_call_count = 0
_call_counts: dict[str, int] = defaultdict(int)


def _ts(t: torch.Tensor | None, name: str = "") -> str:
    """Tensor summary: shape, dtype, min/max/mean/std, any NaN/Inf."""
    if t is None:
        return f"{name}=None"
    s = t.shape
    d = t.dtype
    try:
        tf = t.float()
        mn = tf.min().item()
        mx = tf.max().item()
        me = tf.mean().item()
        sd = tf.std().item() if tf.numel() > 1 else 0.0
        has_nan = torch.isnan(tf).any().item()
        has_inf = torch.isinf(tf).any().item()
    except Exception as e:
        return f"{name}: shape={list(s)} dtype={d} (stats failed: {e})"
    flags = ""
    if has_nan:
        flags += " NAN!"
    if has_inf:
        flags += " INF!"
    return (f"{name}: shape={list(s)} dtype={d} "
            f"min={mn:.6g} max={mx:.6g} mean={me:.6g} std={sd:.6g}{flags}")


def _layer_allowed(prefix: str) -> bool:
    """Check if this layer is in the filter (e.g. 'layers.0.' matches filter '0')."""
    if not _LAYER_FILTER:
        return True
    allowed = {s.strip() for s in _LAYER_FILTER.split(",")}
    # Extract layer number from prefix like 'language_model.model.layers.0.linear_attn'
    for part in prefix.split("."):
        if part.isdigit() and part in allowed:
            return True
    return False


def diag_enabled(prefix: str) -> bool:
    """Check if diagnostics should fire for the NEXT call to diag_begin."""
    if not _ENABLED:
        return False
    if not _layer_allowed(prefix):
        return False
    return _call_counts[prefix] < _MAX_CALLS


def diag_begin(prefix: str, is_prompt: bool, metadata_dict: dict):
    """Call at the start of _forward_core.  Increments counter AFTER this call
    so that all diag_stage/diag_end calls within the same invocation still fire.
    The counter is checked in diag_enabled() which gates the NEXT invocation."""
    # Note: counter is incremented in diag_end, not here, so that all stages
    # within a single forward pass see the same count.
    n = _call_counts[prefix] + 1
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"[GDN DIAG] {prefix} call #{n} "
          f"{'PREFILL' if is_prompt else 'DECODE'}", file=sys.stderr)
    for k, v in metadata_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"  meta.{k}: {_ts(v, k)}", file=sys.stderr)
        else:
            print(f"  meta.{k}: {v}", file=sys.stderr)
    sys.stderr.flush()


def diag_stage(prefix: str, stage: str, **tensors):
    """Call after each processing stage with named tensors.

    Note: callers gate this with `if _diag:` so no counter check needed here.
    """
    if not _ENABLED:
        return
    print(f"  [{stage}]", file=sys.stderr)
    for name, t in tensors.items():
        if isinstance(t, torch.Tensor):
            print(f"    {_ts(t, name)}", file=sys.stderr)
        else:
            print(f"    {name}={t}", file=sys.stderr)
    sys.stderr.flush()


def diag_end(prefix: str, core_attn_out: torch.Tensor):
    """Call at the end of _forward_core.  Also increments the call counter."""
    if not _ENABLED:
        return
    _call_counts[prefix] += 1
    print(f"  [FINAL OUTPUT] {_ts(core_attn_out, 'core_attn_out')}", file=sys.stderr)
    # Print first few values to spot obvious patterns
    flat = core_attn_out.reshape(-1)[:20]
    print(f"    first 20 values: {flat.tolist()}", file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Per-layer hidden-state trace (GDN_LAYER_TRACE=1)
# ---------------------------------------------------------------------------

def install_layer_trace_hooks(model: torch.nn.Module) -> None:
    """Attach post-forward hooks to every decoder layer.

    Each hook prints one line per layer to stderr with hidden_states stats.
    Controlled by GDN_LAYER_TRACE=1 and GDN_LAYER_TRACE_CALLS=N env vars.
    """
    if not _LAYER_TRACE:
        return

    # Navigate to inner model layers
    lm = getattr(model, "language_model", model)
    inner = getattr(lm, "model", lm)
    layers = getattr(inner, "layers", None)
    if layers is None:
        print("[LAYER TRACE] Could not find model layers — skipping", file=sys.stderr)
        return

    global _trace_call_count

    def _make_hook(layer_idx: int, layer_type: str):
        def _hook(module, args, output):
            global _trace_call_count
            # output is (hidden_states, residual) tuple
            if isinstance(output, tuple):
                hs = output[0]
                res = output[1]
            else:
                hs = output
                res = None

            # Only trace first N full forward passes (all layers per pass)
            # We detect "pass start" when layer_idx == 0
            if layer_idx == 0:
                _trace_call_count += 1
                if _trace_call_count <= _LAYER_TRACE_CALLS:
                    print(f"\n{'~'*70}", file=sys.stderr)
                    print(f"[LAYER TRACE] forward pass #{_trace_call_count}", file=sys.stderr)

            if _trace_call_count > _LAYER_TRACE_CALLS:
                return

            hs_info = _ts(hs, "hs")
            res_info = _ts(res, "res") if res is not None else "res=None"

            # Check for corruption markers
            flags = ""
            try:
                tf = hs.float()
                if torch.isnan(tf).any().item():
                    flags += " **NAN**"
                if torch.isinf(tf).any().item():
                    flags += " **INF**"
                absmax = tf.abs().max().item()
                if absmax > 1e4:
                    flags += f" **LARGE({absmax:.1f})**"
            except Exception:
                flags += " (stats failed)"

            # Print residual stats too (the accumulated signal)
            res_flags = ""
            if res is not None:
                try:
                    rf = res.float()
                    if torch.isnan(rf).any().item():
                        res_flags += " **NAN**"
                    if torch.isinf(rf).any().item():
                        res_flags += " **INF**"
                except Exception:
                    res_flags += " (stats failed)"

            print(f"  L{layer_idx:02d} [{layer_type:3s}] {hs_info}{flags}", file=sys.stderr)
            print(f"       {'':>10} {res_info}{res_flags}", file=sys.stderr)
            sys.stderr.flush()

        return _hook

    def _norm_hook(module, args, output):
        """Hook on final norm to see the normalized output before lm_head."""
        global _trace_call_count
        if _trace_call_count > _LAYER_TRACE_CALLS:
            return
        if isinstance(output, tuple):
            normed = output[0]
        else:
            normed = output
        print(f"  [FINAL NORM] {_ts(normed, 'normed')}", file=sys.stderr)
        # Print first-position values for decode (single token)
        if normed.dim() >= 2:
            first_tok = normed.reshape(-1, normed.shape[-1])[0]
            print(f"    first_tok[:10]: {first_tok[:10].tolist()}", file=sys.stderr)
        sys.stderr.flush()

    def _lm_head_hook(module, args, output):
        """Hook on lm_head to see logit distribution."""
        global _trace_call_count
        if _trace_call_count > _LAYER_TRACE_CALLS:
            return
        logits = output[0] if isinstance(output, tuple) else output
        print(f"  [LM HEAD] {_ts(logits, 'logits')}", file=sys.stderr)
        # Show top-5 predictions for first token
        if logits.dim() >= 2:
            first_logits = logits.reshape(-1, logits.shape[-1])[0]
        else:
            first_logits = logits
        topk = first_logits.topk(10)
        print(f"    top-10 token_ids: {topk.indices.tolist()}", file=sys.stderr)
        print(f"    top-10 values:    {[f'{v:.3f}' for v in topk.values.tolist()]}", file=sys.stderr)
        sys.stderr.flush()

    num_hooked = 0
    for i, layer in enumerate(layers):
        lt = getattr(layer, "layer_type", "???")
        short = "GDN" if lt == "linear_attention" else "FA " if lt == "full_attention" else lt[:3]
        layer.register_forward_hook(_make_hook(i, short))
        num_hooked += 1

    # Hook the final norm
    norm = getattr(inner, "norm", None)
    if norm is not None:
        norm.register_forward_hook(_norm_hook)
        print("[LAYER TRACE] Installed hook on final norm", file=sys.stderr)

    # Hook the lm_head
    lm_head = getattr(lm, "lm_head", None)
    if lm_head is not None:
        lm_head.register_forward_hook(_lm_head_hook)
        print("[LAYER TRACE] Installed hook on lm_head", file=sys.stderr)

    print(f"[LAYER TRACE] Installed hooks on {num_hooked} layers "
          f"(max {_LAYER_TRACE_CALLS} passes)", file=sys.stderr)
    sys.stderr.flush()
