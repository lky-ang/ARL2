# Copyright 2026 Kunyang Li, Mubarak Shah, Yuzhang Shang (ARL²)
# Licensed under the Apache License, Version 2.0
"""
HybridBlockAttention V2c: Intra-block softmax + Inter-block GDN (block-level query).

V2c changes from V2b:
  1. Block-level query: FLA kernel only does state update (output discarded),
     query is q[block] · state (all tokens in a block see same state) → spatial consistency
  2. Inference: update_state=False skips FLA kernel entirely (no wasted compute)
  3. Removed ShortConv on Q/K/V — HALO reports ShortConv fails to converge on
     1.7B models; our V2b analysis confirms v_conv norm collapse (||W|| down 22-36%
     over 3000 steps). Saves 18K params/layer and removes instability source.

Retained from V2b:
  - phi_q, phi_k, phi_v (feature maps for GDN Q/K/V)
  - RoPE on GDN Q/K (HypeNet: +18.2 NIAH)
  - Headwise sigmoid gate (no RMSNorm — avoids gradient compression)
  - gate_proj: headwise sigmoid gate R^{d×H} (replaces inter_gate + old elementwise gate)
  - FLA chunk_gated_delta_rule for efficient state update with gradients

Data flow:
  Intra: softmax with RoPE (unchanged)
  Inter GDN:
    Q/K: phi → RoPE → L2Norm
    V:   phi_v
    Query: q[block] · State (block-level, all tokens see same state)
    Update: FLA kernel (token-level, output discarded, only final_state kept)
    Output: intra + inter * sigmoid(W_h @ x)  [headwise gate, 18K params]
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules.l2norm import l2_norm
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

from wan.modules.model import WanRMSNorm, rope_apply
from wan.modules.causal_model import causal_rope_apply, CausalWanSelfAttention


class HybridBlockAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_frame_per_block: int = 1,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_frame_per_block = num_frame_per_block
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.scale = self.head_dim ** -0.5

        # === Shared projections (copied from teacher, frozen) ===
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # === Per-token gate projections for GDN state update ===
        self.a_proj = nn.Linear(dim, num_heads, bias=False)
        self.b_proj = nn.Linear(dim, num_heads, bias=False)
        A = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        # === Per-head feature maps for GDN Q/K/V ===
        self.phi_q = nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads, bias=False)
        self.phi_k = nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads, bias=False)
        self.phi_v = nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads, bias=False)

        # === Data-dependent headwise sigmoid gate (replaces inter_gate + gate_proj) ===
        # Per-token per-head: sigmoid(W_h @ x) ∈ R^{B,L,H}
        # Qwen (NeurIPS 2025): headwise ≈ elementwise, head-specific is what matters
        # 18K params vs old 2.36M (130x reduction)
        self.gate_proj = nn.Linear(dim, num_heads, bias=False)
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=2 ** -2.5)

        # === GDN-path LoRA (Stage 2) ===
        self.lora_rank = 0
        self.latest_diagnostics = None

        self._init_weights()
        self._init_phi_identity()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.a_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.b_proj.weight, gain=2 ** -2.5)

    def _init_phi_identity(self):
        H, D = self.num_heads, self.head_dim
        with torch.no_grad():
            for phi in (self.phi_q, self.phi_k, self.phi_v):
                phi.weight.zero_()
                for h in range(H):
                    phi.weight[h * D:(h + 1) * D, :, 0] = torch.eye(D)

    def _apply_phi(self, phi, x):
        B, L, H, D = x.shape
        out = phi(x.flatten(2).transpose(1, 2)).transpose(1, 2).contiguous()
        return out.view(B, L, H, D)

    def _apply_output_gate(self, inter_out, x):
        """Headwise sigmoid gate (no RMSNorm — gate handles amplitude implicitly).
        gate: sigmoid(W_h @ x) ∈ R^{B,L,H,1} — per-token per-head scalar.
        """
        B, L, H, D = inter_out.shape
        g = self.gate_proj(x).sigmoid().view(B, L, H, 1)
        return inter_out * g, g

    def _update_diagnostics(self, intra_out, inter_out, gated_inter, gate):
        with torch.no_grad():
            intra_rms = intra_out.float().pow(2).mean().sqrt()
            inter_raw_rms = inter_out.float().pow(2).mean().sqrt()
            gated_inter_rms = gated_inter.float().pow(2).mean().sqrt()
            gate_f = gate.float()
            self.latest_diagnostics = {
                "inter_raw_rms": inter_raw_rms.item(),
                "intra_rms": intra_rms.item(),
                "gated_inter_rms": gated_inter_rms.item(),
                "gated_inter_over_intra": (
                    gated_inter_rms / intra_rms.clamp_min(self.eps)
                ).item(),
                "gate_mean": gate_f.mean().item(),
                "gate_sat_frac": (
                    ((gate_f < 0.05) | (gate_f > 0.95)).float().mean().item()
                ),
            }

    def enable_gdn_lora(self, rank=16, alpha=32.0):
        dim = self.dim
        self.lora_rank = rank
        self.lora_alpha = alpha
        self.lora_scale = alpha / rank
        self.lora_q_A = nn.Parameter(torch.zeros(dim, rank))
        self.lora_q_B = nn.Parameter(torch.zeros(rank, dim))
        nn.init.kaiming_uniform_(self.lora_q_A, a=5 ** 0.5)
        self.lora_k_A = nn.Parameter(torch.zeros(dim, rank))
        self.lora_k_B = nn.Parameter(torch.zeros(rank, dim))
        nn.init.kaiming_uniform_(self.lora_k_A, a=5 ** 0.5)
        self.lora_v_A = nn.Parameter(torch.zeros(dim, rank))
        self.lora_v_B = nn.Parameter(torch.zeros(rank, dim))
        nn.init.kaiming_uniform_(self.lora_v_A, a=5 ** 0.5)
        return [self.lora_q_A, self.lora_q_B,
                self.lora_k_A, self.lora_k_B,
                self.lora_v_A, self.lora_v_B]

    def _lora_delta(self, x, A, B):
        return ((x @ A) @ B) * self.lora_scale

    @classmethod
    def from_teacher(cls, teacher_attn, num_frame_per_block=1):
        hybrid = cls(
            dim=teacher_attn.dim, num_heads=teacher_attn.num_heads,
            num_frame_per_block=num_frame_per_block,
            local_attn_size=teacher_attn.local_attn_size,
            sink_size=teacher_attn.sink_size,
            qk_norm=teacher_attn.qk_norm, eps=teacher_attn.eps)
        for name in ('q', 'k', 'v', 'o', 'norm_q', 'norm_k'):
            getattr(hybrid, name).load_state_dict(
                getattr(teacher_attn, name).state_dict())
        return hybrid

    def _intra_block_attention(self, roped_q, roped_k, v, num_blocks, block_seqlen):
        B = roped_q.shape[0]
        H, D = self.num_heads, self.head_dim
        q_b = roped_q[:, :num_blocks * block_seqlen].reshape(
            B * num_blocks, block_seqlen, H, D).transpose(1, 2)
        k_b = roped_k[:, :num_blocks * block_seqlen].reshape(
            B * num_blocks, block_seqlen, H, D).transpose(1, 2)
        v_b = v[:, :num_blocks * block_seqlen].reshape(
            B * num_blocks, block_seqlen, H, D).transpose(1, 2)
        out = F.scaled_dot_product_attention(q_b, k_b, v_b)
        return out.transpose(1, 2).reshape(B, num_blocks * block_seqlen, H, D)

    # ── Block-level query + FLA kernel state update ──

    def _inter_block_gdn(self, x, q_l2, k_l2, v_gdn, num_blocks, block_seqlen,
                         initial_state=None):
        """Block-level query + token-level state update via FLA kernel.

        Key design (from V2 docs):
          - All tokens in a block query the SAME pre-update state → spatial consistency
          - FLA kernel updates state efficiently (output discarded, only final_state kept)
          - Mathematically identical to inference behavior
        """
        B, H, D = q_l2.shape[0], self.num_heads, self.head_dim
        state = initial_state if initial_state is not None else \
            torch.zeros(B, H, D, D, device=q_l2.device, dtype=torch.float32)

        # Precompute per-token gates for full sequence
        g_all = -self.A_log.float().exp() * F.softplus(
            self.a_proj(x).float() + self.dt_bias)
        beta_all = self.b_proj(x).sigmoid().float() * 2

        inter_outs = []
        for i in range(num_blocks):
            s, e = i * block_seqlen, (i + 1) * block_seqlen

            # Query: ALL tokens in block query the SAME state (block-level)
            inter_raw = torch.einsum(
                'bthd,bhdv->bthv', q_l2[:, s:e], state.type_as(q_l2)) * self.scale
            inter_outs.append(inter_raw)

            # Update: FLA kernel for efficient token-level state update
            # Output is DISCARDED — only final_state is kept for next block
            _, state = chunk_gated_delta_rule(
                q=q_l2[:, s:e],
                k=k_l2[:, s:e],
                v=v_gdn[:, s:e],
                g=g_all[:, s:e],
                beta=beta_all[:, s:e],
                scale=self.scale,
                initial_state=state,
                output_final_state=True,
            )

        return torch.cat(inter_outs, dim=1), state

    def _inter_block_gdn_tf(self, x, q_l2, k_l2, v_gdn, num_blocks, block_seqlen):
        """Block-level query for teacher forcing (clean/noisy split)."""
        B, H, D = q_l2.shape[0], self.num_heads, self.head_dim
        L_half = num_blocks * block_seqlen

        q_clean, q_noisy = q_l2[:, :L_half], q_l2[:, L_half:]
        k_clean = k_l2[:, :L_half]
        v_clean = v_gdn[:, :L_half]

        x_clean = x[:, :L_half]
        g_clean = -self.A_log.float().exp() * F.softplus(
            self.a_proj(x_clean).float() + self.dt_bias)
        beta_clean = self.b_proj(x_clean).sigmoid().float() * 2

        state = torch.zeros(B, H, D, D, device=q_l2.device, dtype=torch.float32)
        clean_outs, noisy_outs = [], []

        for i in range(num_blocks):
            s, e = i * block_seqlen, (i + 1) * block_seqlen
            sc = state.type_as(q_clean)
            # Block-level query: both clean and noisy tokens see same state
            clean_outs.append(torch.einsum('bthd,bhdv->bthv', q_clean[:, s:e], sc) * self.scale)
            noisy_outs.append(torch.einsum('bthd,bhdv->bthv', q_noisy[:, s:e], sc) * self.scale)
            # State updated from clean tokens only
            _, state = chunk_gated_delta_rule(
                q=q_clean[:, s:e],
                k=k_clean[:, s:e],
                v=v_clean[:, s:e],
                g=g_clean[:, s:e],
                beta=beta_clean[:, s:e],
                scale=self.scale,
                initial_state=state,
                output_final_state=True,
            )

        return torch.cat([torch.cat(clean_outs, 1), torch.cat(noisy_outs, 1)], 1)

    # ── GDN Q/K/V preparation ──

    def _prepare_gdn_qkv(self, q_gdn_base, k_gdn_base, v, grid_sizes, freqs,
                          is_inference=False, start_frame=0):
        """phi → RoPE → L2Norm (Q/K), phi_v (V)."""
        q_phi = self._apply_phi(self.phi_q, q_gdn_base)
        if is_inference:
            q_roped = causal_rope_apply(q_phi, grid_sizes, freqs, start_frame=start_frame)
        else:
            q_roped = rope_apply(q_phi, grid_sizes, freqs)
        q_gdn = l2_norm(q_roped.type_as(q_phi))

        k_phi = self._apply_phi(self.phi_k, k_gdn_base)
        if is_inference:
            k_roped = causal_rope_apply(k_phi, grid_sizes, freqs, start_frame=start_frame)
        else:
            k_roped = rope_apply(k_phi, grid_sizes, freqs)
        k_gdn = l2_norm(k_roped.type_as(k_phi))

        v_gdn = self._apply_phi(self.phi_v, v)

        return q_gdn, k_gdn, v_gdn

    # ── Forward paths ──

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask,
                kv_cache=None, current_start=0, cache_start=None):
        B, L, _ = x.shape
        H, D = self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        q_raw, k_raw, v_raw = self.q(x), self.k(x), self.v(x)
        q = self.norm_q(q_raw).view(B, L, H, D)
        k = self.norm_k(k_raw).view(B, L, H, D)
        v = v_raw.view(B, L, H, D)

        if self.lora_rank > 0:
            q_gdn_base = self.norm_q(q_raw + self._lora_delta(x, self.lora_q_A, self.lora_q_B)).view(B, L, H, D)
            k_gdn_base = self.norm_k(k_raw + self._lora_delta(x, self.lora_k_A, self.lora_k_B)).view(B, L, H, D)
            v_for_gdn = (v_raw + self._lora_delta(x, self.lora_v_A, self.lora_v_B)).view(B, L, H, D)
        else:
            q_gdn_base, k_gdn_base = q, k
            v_for_gdn = v

        if kv_cache is not None:
            return self._forward_inference(x, q, k, v, q_gdn_base, k_gdn_base, v_for_gdn,
                                           grid_sizes, freqs, kv_cache, current_start, cache_start)

        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        num_frames = grid_sizes[0][0].item()
        block_seqlen = frame_seqlen * self.num_frame_per_block
        num_blocks = num_frames // self.num_frame_per_block
        is_tf = (L == seq_lens[0].item() * 2)

        if is_tf:
            return self._forward_train_tf(x, q, k, v, q_gdn_base, k_gdn_base, v_for_gdn,
                                          grid_sizes, freqs, num_blocks, block_seqlen, frame_seqlen)
        else:
            return self._forward_train(x, q, k, v, q_gdn_base, k_gdn_base, v_for_gdn,
                                       grid_sizes, freqs, num_blocks, block_seqlen, frame_seqlen)

    def _forward_train(self, x, q, k, v, q_gdn_base, k_gdn_base, v_for_gdn,
                       grid_sizes, freqs, num_blocks, block_seqlen, frame_seqlen):
        B, L, H, D = q.shape

        roped_q = rope_apply(q, grid_sizes, freqs).type_as(v)
        roped_k = rope_apply(k, grid_sizes, freqs).type_as(v)
        intra_out = self._intra_block_attention(roped_q, roped_k, v, num_blocks, block_seqlen)

        q_gdn, k_gdn, v_gdn = self._prepare_gdn_qkv(
            q_gdn_base, k_gdn_base, v_for_gdn, grid_sizes, freqs)

        inter_out, _ = self._inter_block_gdn(x, q_gdn, k_gdn, v_gdn,
                                              num_blocks, block_seqlen)

        o_inter, gate = self._apply_output_gate(inter_out, x)
        self._update_diagnostics(intra_out, inter_out, o_inter, gate)
        output = intra_out + o_inter.type_as(intra_out)

        return self.o(output.flatten(2))

    def _forward_train_tf(self, x, q, k, v, q_gdn_base, k_gdn_base, v_for_gdn,
                          grid_sizes, freqs, num_blocks, block_seqlen, frame_seqlen):
        B, L, H, D = q.shape
        L_half = num_blocks * block_seqlen

        q_c, q_n = q[:, :L_half], q[:, L_half:]
        k_c, k_n = k[:, :L_half], k[:, L_half:]
        v_c, v_n = v[:, :L_half], v[:, L_half:]

        rq_c = rope_apply(q_c, grid_sizes, freqs).type_as(v)
        rk_c = rope_apply(k_c, grid_sizes, freqs).type_as(v)
        rq_n = rope_apply(q_n, grid_sizes, freqs).type_as(v)
        rk_n = rope_apply(k_n, grid_sizes, freqs).type_as(v)

        intra_out = torch.cat([
            self._intra_block_attention(rq_c, rk_c, v_c, num_blocks, block_seqlen),
            self._intra_block_attention(rq_n, rk_n, v_n, num_blocks, block_seqlen),
        ], dim=1)

        q_gdn, k_gdn, v_gdn = self._prepare_gdn_qkv(
            q_gdn_base, k_gdn_base, v_for_gdn, grid_sizes, freqs)

        inter_out = self._inter_block_gdn_tf(x, q_gdn, k_gdn, v_gdn,
                                              num_blocks, block_seqlen)

        o_inter, gate = self._apply_output_gate(inter_out, x)
        self._update_diagnostics(intra_out, inter_out, o_inter, gate)
        output = intra_out + o_inter.type_as(intra_out)

        return self.o(output.flatten(2))

    def _forward_inference(self, x, q, k, v, q_gdn_base, k_gdn_base, v_for_gdn,
                           grid_sizes, freqs, kv_cache, current_start, cache_start):
        B, L, H, D = q.shape
        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        csf = current_start // frame_seqlen

        roped_q = causal_rope_apply(q, grid_sizes, freqs, start_frame=csf).type_as(v)
        roped_k = causal_rope_apply(k, grid_sizes, freqs, start_frame=csf).type_as(v)
        intra_out = self._intra_block_attention(roped_q, roped_k, v, 1, L)

        # q_gdn is always needed to query the pre-update state.
        q_phi = self._apply_phi(self.phi_q, q_gdn_base)
        q_roped = causal_rope_apply(q_phi, grid_sizes, freqs, start_frame=csf)
        q_gdn = l2_norm(q_roped.type_as(q_phi))

        state = kv_cache.get("gdn_state", None)
        if state is None:
            state = torch.zeros(B, H, D, D, device=x.device, dtype=torch.float32)

        # Block-level query: all tokens in current block query same state
        inter_out = torch.einsum(
            'bthd,bhdv->bthv', q_gdn, state.type_as(q_gdn)) * self.scale

        # State update: only when update_state=True (skip during denoising).
        # k_gdn/v_gdn/g/beta are only consumed by the FLA kernel, so defer them
        # — avoids 2 phi Conv1d + 1 rope + 1 l2norm + a_proj/b_proj per denoising
        # pass (4 of 5 passes per video block).
        if kv_cache.get("update_state", True):
            k_phi = self._apply_phi(self.phi_k, k_gdn_base)
            k_roped = causal_rope_apply(k_phi, grid_sizes, freqs, start_frame=csf)
            k_gdn = l2_norm(k_roped.type_as(k_phi))
            v_gdn = self._apply_phi(self.phi_v, v_for_gdn)

            g = -self.A_log.float().exp() * F.softplus(
                self.a_proj(x).float() + self.dt_bias)
            beta = self.b_proj(x).sigmoid().float() * 2
            _, new_state = fused_recurrent_gated_delta_rule(
                q=q_gdn, k=k_gdn, v=v_gdn, g=g, beta=beta,
                scale=self.scale,
                initial_state=state,
                output_final_state=True,
            )
            kv_cache["gdn_state"] = new_state
            current_end = current_start + L
            if "global_end_index" not in kv_cache:
                kv_cache["global_end_index"] = torch.tensor(current_end, device=x.device)
            else:
                kv_cache["global_end_index"].fill_(current_end)

        # Diagnostics are only consumed by WanLayerAligner during training;
        # skip the per-layer fp32 materialization + .item() CUDA syncs at inference.
        o_inter, _ = self._apply_output_gate(inter_out, x)
        output = intra_out + o_inter.type_as(intra_out)
        return self.o(output.flatten(2))
