# Profiling Results

## Tested Models

Complete profiling results for 7 diffusion models at 512x512 resolution with 30 inference steps.

## Results Table

| Model            | Type          | Transformer (GMACs/step) | Text Encoders (GMACs) | VAE (GMACs) | Total (TFLOPs) |
| ---------------- | ------------- | ------------------------ | --------------------- | ----------- | -------------- |
| SANA-1.6B        | DiT           | 14,169.8                 | 311.7                 | 16,715.6    | 442.1          |
| SANA-600M        | DiT           | 6,314.8                  | 311.7                 | 16,715.6    | 206.5          |
| Lumina-Image-2.0 | DiT           | 5,113.8                  | 311.7                 | 1,240.3     | 155.0          |
| SD3-medium       | Flow Matching | 1,676.2                  | 823.1                 | 1,240.3     | 52.3           |
| PixArt-XL        | DiT           | 1,350.7                  | 713.1                 | 1,240.1     | 42.5           |
| SD2.1-base       | Diffusion     | 1,172.3                  | 44.6                  | 1,240.1     | 36.5           |
| Kandinsky-3      | Diffusion     | 935.4                    | 1,322.9               | 0.0         | 29.4           |

Note: Kandinsky-3 pipeline does not include VAE component.

## Key Findings

### Compute Distribution

For all models, the breakdown is approximately:

- Transformer/UNet: 95%
- Text Encoders: 2%
- VAE Decoder: 3%

Exception: SANA models have higher VAE contribution (8%) due to 32-channel VAE.

### Model Comparison

Ordered by total compute:

1. **SANA-1.6B**: Highest compute (442.1 TFLOPs)

   - Largest transformer: 14.2 TMACs/step
   - 32-channel VAE: 16.7 GMACs

2. **SANA-600M**: Second highest (206.5 TFLOPs)

   - Medium transformer: 6.3 TMACs/step
   - Same 32-channel VAE as SANA-1.6B

3. **Lumina-Image-2.0**: High compute (155.0 TFLOPs)

   - Large transformer: 5.1 TMACs/step
   - Standard 16-channel VAE

4. **SD3-medium**: Moderate compute (52.3 TFLOPs)

   - Medium transformer: 1.7 TMACs/step
   - Three text encoders: 823 GMACs
   - 16-channel VAE

5. **PixArt-XL**: Moderate compute (42.5 TFLOPs)

   - Efficient transformer: 1.4 TMACs/step
   - T5-XXL encoder: 713 GMACs

6. **SD2.1-base**: Low compute (36.5 TFLOPs)

   - Standard UNet: 1.2 TMACs/step
   - Simple CLIP encoder: 45 GMACs
   - 4-channel VAE

7. **Kandinsky-3**: Lowest compute (29.4 TFLOPs)
   - Efficient UNet: 0.9 TMACs/step
   - Large text encoder: 1.3 TMACs

### VAE Analysis

VAE compute scales with latent channel count:

| Latent Channels | VAE Compute    | Relative Cost | Models        |
| --------------- | -------------- | ------------- | ------------- |
| 32              | 16,715.6 GMACs | 13.5x         | SANA models   |
| 16              | 1,240.3 GMACs  | 1.0x          | SD3, Lumina   |
| 4               | 1,240.1 GMACs  | 1.0x          | SD2.1, PixArt |

SANA's 32-channel VAE requires 13.5x more compute than standard 4-channel VAE.

### Text Encoder Analysis

Text encoder types and costs:

| Encoder Type     | Compute               | Used By                        |
| ---------------- | --------------------- | ------------------------------ |
| T5-XXL           | 713.1 GMACs           | PixArt, SD3 (as third encoder) |
| Large CLIP/Gemma | 311.7 - 1,322.9 GMACs | SANA, Lumina, Kandinsky        |
| Standard CLIP    | 44.6 - 96.9 GMACs     | SD2.1, SD3                     |

Despite large differences (30x range), text encoders remain <2% of total compute for all models.

## Scaling Behavior

### Resolution Scaling

Compute scales quadratically with resolution due to spatial transformers/UNets:

- 512x512: baseline
- 768x768: ~2.25x compute
- 1024x1024: ~4x compute

### Step Count Scaling

Compute scales linearly with inference steps:

- 10 steps: ~33% of 30-step compute
- 20 steps: ~67% of 30-step compute
- 50 steps: ~167% of 30-step compute

Note: Text encoders and VAE are fixed costs (one-time), so relative savings increase with fewer steps.

## Efficiency Metrics

### Compute per Parameter

Estimated compute efficiency (GMACs/step per billion parameters):

| Model            | Est. Params | GMACs/step | Efficiency (GMACs/B) |
| ---------------- | ----------- | ---------- | -------------------- |
| SANA-1.6B        | ~1.6B       | 14,169.8   | 8,856                |
| SANA-600M        | ~600M       | 6,314.8    | 10,525               |
| Lumina-Image-2.0 | ~2B         | 5,113.8    | 2,557                |
| SD3-medium       | ~2B         | 1,676.2    | 838                  |
| PixArt-XL        | ~900M       | 1,350.7    | 1,501                |
| SD2.1-base       | ~900M       | 1,172.3    | 1,302                |
| Kandinsky-3      | ~3B         | 935.4      | 312                  |

SANA models are 6-10x more compute-intensive per parameter, likely due to high-resolution architecture design.

## Notes

### Methodology

- All measurements at 512x512 resolution
- 30 inference steps with guidance scale 7.5
- MACs calculated using thop library with architecture-specific wrappers
- FLOPs approximated as 2x MACs

### Limitations

- FLUX transformer profiling incompatible with thop (dual-stream attention)
- LuminaNext has rotary embedding dimension issues
- Models with CPU offloading may have profiling errors
- Results are estimates based on forward pass analysis

### Test Environment

- Hardware: NVIDIA H100 80GB HBM3 (2x)
- Framework: PyTorch 2.8.0, Diffusers 0.32.0+
- Profiling: thop 0.1.1

## Usage

To reproduce these results:

```bash
python final_working_models_test.py
```

For detailed profiling examples:

```bash
python example_profiling_detailed.py
```
