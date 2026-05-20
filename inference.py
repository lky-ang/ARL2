# Copyright 2026 Kunyang Li, Mubarak Shah, Yuzhang Shang (ARL²)
# Licensed under the Apache License, Version 2.0
# Modified from Causal Forcing (https://github.com/thu-ml/Causal-Forcing)
import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
import imageio.v2 as imageio

def write_video(output_path, frames, fps=16):
    """imageio-ffmpeg wrapper matching torchvision.io.write_video's signature.

    frames: Tensor[T, H, W, C] (uint8 or float 0-255).
    """
    import numpy as np
    arr = frames
    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264",
                                quality=8, macro_block_size=1)
    for f in arr:
        writer.append_data(f)
    writer.close()
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import json

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21, help="Number of overlap frames between sliding windows")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--metrics_dir", type=str, default=None,
                    help="If set, write per-video timing & GPU-memory JSON into this folder.")
parser.add_argument("--fname_len", type=int, default=100,
                    help="Truncate prompt to this many chars when building output filename. "
                         "Set larger (e.g. 200) to avoid VBench prompt collisions.")
parser.add_argument("--hybrid_layers", type=int, nargs="+", default=None,
                    help="Layer indices to replace with HybridBlockAttention (Stage 2 hybrid model). "
                         "If set, inject before loading checkpoint.")
parser.add_argument("--lora_rank", type=int, default=16,
                    help="LoRA rank for hybrid injection (only used with --hybrid_layers).")
parser.add_argument("--lora_alpha", type=float, default=32.0,
                    help="LoRA alpha for hybrid injection.")
parser.add_argument("--ffn_lora_all_layers", action="store_true",
                    help="If set, apply FFN LoRA on ALL 30 blocks (v2c-all); "
                         "default is FFN LoRA only on hybrid layers (v2c).")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1

set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

# Inject hybrid layers + LoRA BEFORE loading checkpoint so the state_dict
# keys (which include HybridBlockAttention and FFN LoRA params) match up.
if args.hybrid_layers:
    from wan.modules.hybrid_attention import HybridBlockAttention
    from wan.modules.causal_model import CausalWanSelfAttention
    from utils.lora import apply_lora_to_ffn

    _model = pipeline.generator.model
    _nfpb = getattr(config, "num_frame_per_block", 3)
    for _idx in args.hybrid_layers:
        _block = _model.blocks[_idx]
        if isinstance(_block.self_attn, CausalWanSelfAttention):
            _new = HybridBlockAttention.from_teacher(_block.self_attn, num_frame_per_block=_nfpb)
            _new.enable_gdn_lora(rank=args.lora_rank, alpha=args.lora_alpha)
            _block.self_attn = _new
        # FFN LoRA: only on hybrid layers unless --ffn_lora_all_layers
        if not args.ffn_lora_all_layers:
            apply_lora_to_ffn(_block, rank=args.lora_rank, alpha=args.lora_alpha)

    if args.ffn_lora_all_layers:
        for _block in _model.blocks:
            apply_lora_to_ffn(_block, rank=args.lora_rank, alpha=args.lora_alpha)

    print(f"[hybrid] injected HybridBlockAttention + FFN LoRA at layers {args.hybrid_layers} "
          f"(lora_rank={args.lora_rank}, alpha={args.lora_alpha})")

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    key = 'generator_ema' if args.use_ema else 'generator'
    gen_sd = state_dict[key]

    # Strip common FSDP / checkpoint wrapper prefixes
    def _strip_prefixes(sd):
        out = {}
        for k, v in sd.items():
            nk = k
            for pref in ("model._fsdp_wrapped_module.",
                         "_fsdp_wrapped_module.",
                         "_checkpoint_wrapped_module."):
                nk = nk.replace(pref, "")
            if "model." not in nk.split(".", 1)[0]:
                # ensure leading "model." prefix preserved if stripped
                pass
            out[nk] = v
        return out

    try:
        pipeline.generator.load_state_dict(gen_sd)
    except RuntimeError:
        fixed = _strip_prefixes(gen_sd)
        missing, unexpected = pipeline.generator.load_state_dict(fixed, strict=False)
        if missing or unexpected:
            print(f"[ckpt] load_state_dict strict=False: "
                  f"{len(missing)} missing, {len(unexpected)} unexpected keys")
            if len(missing) <= 10:
                for m in missing: print(f"  missing: {m}")
            if len(unexpected) <= 10:
                for u in unexpected: print(f"  unexpected: {u}")

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


# ------------------------------------------------------------------
# Per-video profiling (optional)
#
# When --metrics_dir is set we record, for each generated video:
#   * wall-clock total generation time (GPU-synchronised)
#   * total & per-call elapsed time of every attention forward, split into
#     "self_attn" and "cross_attn" (classified by class name)
#   * GPU memory: allocated/reserved before, peak, after, and peak delta
# ------------------------------------------------------------------
class AttnProfiler:
    """Per-call CUDA-event timing hooks for attention modules.

    Uses pre/post forward hooks to bracket each forward() with cuda events.
    .reset() clears state, .finalize() synchronises and aggregates.
    """
    def __init__(self):
        self._completed = []        # list of (qualname, class, start_evt, end_evt)
        self._pending = {}          # qualname -> stack of (start_evt, end_evt)
        self._hook_handles = []
        self._enabled = False

    @staticmethod
    def _is_attention(name, module):
        cls = type(module).__name__
        return cls in (
            "CausalWanSelfAttention",
            "WanSelfAttention",
            "WanT2VCrossAttention",
            "WanI2VCrossAttention",
            "WanGanCrossAttention",
            "HybridBlockAttention",
        )

    def attach(self, model):
        for name, module in model.named_modules():
            if not self._is_attention(name, module):
                continue
            cls = type(module).__name__

            def _pre(mod, inputs, qn=name, cl=cls):
                if not self._enabled:
                    return
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                self._pending.setdefault(qn, []).append((cl, s, e))

            def _post(mod, inputs, output, qn=name):
                if not self._enabled:
                    return
                stack = self._pending.get(qn)
                if not stack:
                    return
                cl, s, e = stack.pop()
                e.record()
                self._completed.append((qn, cl, s, e))

            self._hook_handles.append(module.register_forward_pre_hook(_pre))
            self._hook_handles.append(module.register_forward_hook(_post))

    def start(self):
        self._completed.clear()
        self._pending.clear()
        self._enabled = True

    def stop(self):
        self._enabled = False

    def finalize(self):
        """Sync CUDA, compute per-module + by-type totals in seconds."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        per_module = {}         # qualname -> {"total_s", "n_calls", "class"}
        by_type = {"self_attn": {"total_s": 0.0, "n_calls": 0},
                   "cross_attn": {"total_s": 0.0, "n_calls": 0},
                   "other": {"total_s": 0.0, "n_calls": 0}}
        for qn, cl, s, e in self._completed:
            ms = s.elapsed_time(e)       # ms
            sec = ms / 1000.0
            slot = per_module.setdefault(qn, {"total_s": 0.0, "n_calls": 0, "class": cl})
            slot["total_s"] += sec
            slot["n_calls"] += 1
            if "Cross" in cl:
                bucket = "cross_attn"
            elif "Self" in cl or "HybridBlock" in cl:
                bucket = "self_attn"
            else:
                bucket = "other"
            by_type[bucket]["total_s"] += sec
            by_type[bucket]["n_calls"] += 1
        total_attn_time_s = sum(v["total_s"] for v in by_type.values())
        return {
            "total_attn_time_s": total_attn_time_s,
            "by_type": by_type,
            "per_module": per_module,
        }

    def detach(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


profiler = None
if args.metrics_dir:
    if local_rank == 0:
        os.makedirs(args.metrics_dir, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    profiler = AttnProfiler()
    profiler.attach(pipeline.generator)
    print(f"[metrics] attached {len(profiler._hook_handles)//2} attention hooks; "
          f"writing per-video JSON to {args.metrics_dir}")


def _mem_mb(fn):
    try:
        return fn(device) / (1024 * 1024)
    except Exception:
        return None


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames
    
    
    if args.i2v:
        assert config.num_frame_per_block == 1, "Current I2V only supports the frame-wise model."
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        output_path = os.path.join(args.output_folder, f'{prompt[:args.fname_len]}.mp4')
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        prompts = [prompt] 
        sampled_noise = torch.randn(
            [1, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        output_path = os.path.join(args.output_folder, f'{prompt[:args.fname_len]}.mp4')
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] 
        else:
            prompts = [prompt] 

        initial_latent = None
        sampled_noise = torch.randn(
            [1, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    # ----- profiling: reset memory & start attention timers -----
    timing_out = None
    if profiler is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        mem_before_alloc_mb = _mem_mb(torch.cuda.memory_allocated)
        mem_before_reserved_mb = _mem_mb(torch.cuda.memory_reserved)
        profiler.start()
        import time as _time
        gen_t0 = _time.perf_counter()
        timing_out = {}

    # Generate 81 frames
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        timing_out=timing_out,
    )

    # ----- profiling: stop timers, collect memory -----
    metrics_payload = None
    if profiler is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        gen_wall_s = _time.perf_counter() - gen_t0
        profiler.stop()
        attn_stats = profiler.finalize()
        mem_peak_alloc_mb = _mem_mb(torch.cuda.max_memory_allocated)
        mem_peak_reserved_mb = _mem_mb(torch.cuda.max_memory_reserved)
        mem_after_alloc_mb = _mem_mb(torch.cuda.memory_allocated)
        mem_after_reserved_mb = _mem_mb(torch.cuda.memory_reserved)
        # Throughput uses generation_s (denoise_loop + vae_decode only, excludes text_encode)
        generation_s = (timing_out or {}).get("generation_s")
        n_video_frames = args.num_output_frames * 4 - 3
        throughput_fps = (n_video_frames / generation_s) if generation_s else None
        metrics_payload = {
            "idx": idx,
            "prompt": prompt,
            "video_path": os.path.abspath(output_path),
            "num_output_frames_latent": args.num_output_frames,
            "num_video_frames": n_video_frames,
            "resolution": [480, 832],
            "fps": 16,
            "total_generation_time_s": gen_wall_s,
            "text_encode_s": (timing_out or {}).get("text_encode_s"),
            "denoise_loop_s": (timing_out or {}).get("denoise_loop_s"),
            "vae_decode_s": (timing_out or {}).get("vae_decode_s"),
            "generation_s": generation_s,
            "throughput_fps": throughput_fps,
            "attention_timing": attn_stats,
            "gpu_memory_MB": {
                "before_allocated": mem_before_alloc_mb,
                "before_reserved": mem_before_reserved_mb,
                "peak_allocated": mem_peak_alloc_mb,
                "peak_reserved": mem_peak_reserved_mb,
                "after_allocated": mem_after_alloc_mb,
                "after_reserved": mem_after_reserved_mb,
                "peak_alloc_delta": (
                    None if mem_peak_alloc_mb is None or mem_before_alloc_mb is None
                    else mem_peak_alloc_mb - mem_before_alloc_mb
                ),
                "peak_reserved_delta": (
                    None if mem_peak_reserved_mb is None or mem_before_reserved_mb is None
                    else mem_peak_reserved_mb - mem_before_reserved_mb
                ),
            },
        }

    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    clean_latent = latents[0].cpu()
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    output_path = os.path.join(args.output_folder, f'{prompt[:args.fname_len]}.mp4')
    write_video(output_path, video[0], fps=16)

    # ----- profiling: write per-video metrics file -----
    if metrics_payload is not None and args.metrics_dir:
        safe_stem = prompt[:args.fname_len].replace("/", "_")
        metrics_file = os.path.join(args.metrics_dir, f"{safe_stem}.json")
        with open(metrics_file, "w", encoding="utf-8") as mf:
            json.dump(metrics_payload, mf, indent=2, ensure_ascii=False)
        # Also append one line to a consolidated JSONL for easy scanning
        jsonl_path = os.path.join(args.metrics_dir, "all_metrics.jsonl")
        with open(jsonl_path, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(metrics_payload, ensure_ascii=False) + "\n")

       
