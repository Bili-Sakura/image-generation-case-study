# README

This repo hold codes for case study of existing open-sourced diffusion models' capability in text-to-image generation and image editing.

## Model List

### Open-Source Text-to-Image Models:

- [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [zai-org/CogView3-Plus-3B](https://huggingface.co/zai-org/CogView3-Plus-3B)
- [PixArt-alpha/PixArt-XL-2-512x512](https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512)
- [PixArt-alpha/PixArt-Sigma-XL-2-512-MS](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-512-MS)
- [Alpha-VLLM/Lumina-Next-SFT-diffusers](https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT-diffusers)
- [Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers)
- [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
- [stabilityai/stable-diffusion-3-5-large](https://huggingface.co/stabilityai/stable-diffusion-3-5-large)
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [Efficient-Large-Model/Sana_600M_512px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_600M_512px_diffusers)
- [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
- [thu-ml/unidiffuser-v1](https://huggingface.co/thu-ml/unidiffuser-v1)
- [stabilityai/stable-cascade](https://huggingface.co/stabilityai/stable-cascade)
- [zai-org/CogView4-6B](https://huggingface.co/zai-org/CogView4-6B)
- [kandinsky-community/kandinsky-3](https://huggingface.co/kandinsky-community/kandinsky-3)
- [HiDream-ai/HiDream-I1-Dev](https://huggingface.co/HiDream-ai/HiDream-I1-Dev)
- [Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers](https://huggingface.co/Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers)
- [Alpha-VLLM/Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0)

## Usage：Gradio Web UI (Recommended)

Launch the interactive web interface by choosing a mode:

**Developer Mode (Recommended for regular use):**

```bash
python run.py --dev
```

**Benchmark Mode (For comprehensive evaluation):**

```bash
python run.py --bench
```

## Citation

If you find this repository useful, please cite it as:

```bibtex
@misc{bili_sakura_image_generation_case_study,
  author       = {Bili-Sakura},
  title        = {Image Generation Case Study},
  year         = {2025},
  howpublished = {\url{https://github.com/Bili-Sakura/image-generation-case-study}},
  note         = {Accessed: 2025-10-08}
}
```
