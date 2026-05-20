<div align="center">

# ARL²
### Attend Locally, Remember Linearly: Linear Attention as Cross-Frame Memory for Autoregressive Video Diffusion

<p align="center">
  <div>
    <a href="https://scholar.google.com/citations?user=NcqZtt8AAAAJ" target="_blank">Kunyang Li</a>,
    <a href="https://scholar.google.com/citations?user=p8gsO3gAAAAJ" target="_blank">Mubarak Shah</a>,
    <a href="https://scholar.google.com/citations?user=6ZPL5E0AAAAJ" target="_blank">Yuzhang Shang</a>
  </div>
  <div>
    University of Central Florida
  </div>
</p>

<h3 align="center">
  <a href="https://arxiv.org/abs/2605.16579">Paper</a> |
  <a href="https://lky-ang.github.io/ARL2/">Website</a> |
  <a href="#">Models (Coming Soon)</a>
</h3>

</div>

-----

ARL² replaces quadratic cross-frame softmax attention in autoregressive video diffusion with a **fixed-size recurrent linear state**, achieving **2.26x wall-clock speedup** and **54% memory reduction** while maintaining comparable quality and improved temporal consistency. Our hybrid attention decomposes self-attention into an intra-frame softmax branch (spatial detail) and an inter-frame gated recurrent linear branch (temporal memory).

-----

## Key Features

- **Hybrid Attention**: Intra-block softmax + inter-block Gated Delta Network (GDN) with block-level query
- **Clean-state update**: Recurrent state updated only post-denoising to prevent noise corruption
- **Scalable**: Constant memory for cross-frame attention regardless of video length
- **Two-stage training**: Per-layer distillation (Stage 1) followed by end-to-end teacher distillation (Stage 2)
- **Compatible**: Built on [Causal Forcing](https://github.com/thu-ml/Causal-Forcing) + [Wan 2.1](https://github.com/Wan-Video/Wan2.1) backbone

## Quick Start

### Installation
```bash
conda create -n arl2 python=3.10 -y
conda activate arl2
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install flash-attn --no-build-isolation
pip install triton fla
python setup.py develop
```

### Download Checkpoints

Base models (Wan 2.1 + Causal Forcing):
```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download zhuhz22/Causal-Forcing chunkwise/causal_forcing.pt --local-dir checkpoints
```

ARL² hybrid attention checkpoints (coming soon):
```bash
# huggingface-cli download lky-ang/ARL2 ... --local-dir checkpoints
```

### Inference

#### T2V with Hybrid Attention (ARL²)

```bash
python inference.py \
  --config_path configs/causal_forcing_dmd_chunkwise.yaml \
  --output_folder output/chunkwise \
  --checkpoint_path checkpoints/chunkwise/causal_forcing.pt \
  --data_path prompts/test.txt \
  --hybrid_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
```

#### T2V with Base Causal Forcing (without hybrid attention)

```bash
python inference.py \
  --config_path configs/causal_forcing_dmd_chunkwise.yaml \
  --output_folder output/chunkwise_base \
  --checkpoint_path checkpoints/chunkwise/causal_forcing.pt \
  --data_path prompts/test.txt
```

## Architecture

ARL² introduces a dual-branch hybrid attention mechanism:

| Branch | Scope | Mechanism | Complexity |
|---|---|---|---|
| Intra-frame | Within each denoising block | Softmax attention + RoPE | O(n²) per block |
| Inter-frame | Across frames | Gated Delta Network (GDN) with fixed-size recurrent state | O(1) memory |

Key design choices:
- **Block-level query**: All tokens in a block see the same recurrent state, ensuring spatial consistency
- **Clean-state update**: State is updated only after denoising completes, preventing noise from corrupting the temporal memory
- **Headwise sigmoid gate**: Learned per-head gate balances intra vs. inter contributions

## Training (Coming Soon)

Training code will be released in a future update. The training pipeline consists of:
- **Stage 1**: Per-layer distillation — align hybrid attention outputs to original softmax attention via MSE
- **Stage 2**: End-to-end teacher distillation — frozen teacher (original model) supervises the hybrid student

## Acknowledgements

This codebase is built on top of:
- [Causal Forcing](https://github.com/thu-ml/Causal-Forcing) (Zhu et al.) — autoregressive diffusion distillation framework
- [Wan 2.1](https://github.com/Wan-Video/Wan2.1) (Alibaba) — base video diffusion model
- [CausVid](https://github.com/tianweiy/CausVid) / [Self Forcing](https://github.com/guandehe/Self-Forcing) — distillation infrastructure
- [FLA](https://github.com/sustcsonglin/flash-linear-attention) — efficient linear attention kernels

## Citation

If you find this work useful, please cite:
```bibtex
@article{li2026attend,
  title={Attend Locally, Remember Linearly: Linear Attention as Cross-Frame Memory for Autoregressive Video Diffusion},
  author={Li, Kunyang and Shah, Mubarak and Shang, Yuzhang},
  journal={arXiv preprint arXiv:2605.16579},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Portions of this codebase are derived from [Causal Forcing](https://github.com/thu-ml/Causal-Forcing) (Apache 2.0) and [Wan 2.1](https://github.com/Wan-Video/Wan2.1) (Apache 2.0). See [NOTICE](NOTICE) for third-party attributions.
