# Comprehensive DiT Architecture Comparison

## Executive Summary

This document provides a detailed comparison of all major Diffusion Transformer (DiT) architectures, focusing on how they handle text tokens, noised latents, and timestep conditioning.

## Complete Architecture Comparison Table

| Model                  | Release | Total Params | Architecture Type | Text Processing       | Timestep Method         | Block Type                     | Text-Image Fusion               | Code Reference                 |
| ---------------------- | ------- | ------------ | ----------------- | --------------------- | ----------------------- | ------------------------------ | ------------------------------- | ------------------------------ |
| **PixArt-α**           | 2023/10 | 5.46B        | Cross-Attention   | Separate (cross-attn) | Time only (adaLN)       | `BasicTransformerBlock`        | Cross-attention: Q_img × K_text | `pixart_transformer_2d.py`     |
| **PixArt-Σ**           | 2024/04 | 5.46B        | Cross-Attention   | Separate (cross-attn) | Time only (adaLN)       | `BasicTransformerBlock`        | Cross-attention: Q_img × K_text | `pixart_transformer_2d.py`     |
| **Lumina-T2I**         | 2024/04 | TBD          | Cross-Attention   | Separate (cross-attn) | Time only (adaLN)       | `BasicTransformerBlock`        | Cross-attention: Q_img × K_text | Similar to PixArt              |
| **Lumina-Next-T2I**    | 2024/05 | 4.34B        | Cross-Attention   | Separate (cross-attn) | Time only (adaLN)       | `BasicTransformerBlock`        | Cross-attention: Q_img × K_text | Similar to PixArt              |
| **Stable Diffusion 3** | 2024/06 | 7.69B        | Double-Stream     | Parallel streams      | **Time + Text** (fused) | `JointTransformerBlock`        | Joint attention (both streams)  | `transformer_sd3.py`           |
| **Flux.1-Dev**         | 2024/08 | 16.87B       | **Hybrid**        | Mixed approach        | **Time + Text** (fused) | Double + Single Stream         | Early: joint attn, Late: concat | `transformer_flux.py`          |
| **CogView3-Plus**      | 2024/10 | 8.02B        | Double-Stream     | Parallel streams      | Time + Size (fused)     | `CogView3PlusTransformerBlock` | Joint attention                 | `transformer_cogview3plus.py`  |
| **Hunyuan-DiT**        | 2024/12 | 3.61B        | Cross-Attention   | Separate (cross-attn) | **Time + Text + Style** | `HunyuanDiTBlock`              | Cross-attention: Q_img × K_text | `hunyuan_transformer_2d.py`    |
| **SANA**               | 2025/01 | 3.52B        | Cross-Attention   | Separate (cross-attn) | Time + Guidance         | `SanaTransformerBlock`         | Cross-attention: Q_img × K_text | `sana_transformer.py`          |
| **Lumina-Image 2.0**   | 2025/01 | TBD          | Cross-Attention   | Separate (cross-attn) | Time only (adaLN)       | Similar to Lumina-Next         | Cross-attention                 | Similar to Lumina              |
| **SANA 1.5**           | 2025/03 | TBD          | Cross-Attention   | Separate (cross-attn) | Time + Guidance         | `SanaTransformerBlock`         | Cross-attention                 | `sana_transformer.py`          |
| **HiDream-I1-Dev**     | 2025/04 | TBD          | Double-Stream     | Parallel streams      | Time only               | `HiDreamImageTransformerBlock` | Joint attention                 | `transformer_hidream_image.py` |
| **CogView4-6B**        | 2025/05 | 6.00B        | Double-Stream     | Concat then split     | Time + Size (fused)     | `CogView4TransformerBlock`     | Concat → Attention → Split      | `transformer_cogview4.py`      |
| **Qwen-Image**         | 2025/08 | 28.85B       | Double-Stream     | Parallel streams      | Time only               | `QwenImageTransformerBlock`    | Joint attention                 | `transformer_qwenimage.py`     |
| **OmniGen**            | N/A     | N/A          | **Single-Stream** | **Token concat**      | **Time as token**       | `OmniGenBlock`                 | **Concat [text, time, img]**    | `transformer_omnigen.py`       |

---

## Detailed Architecture Breakdown

### 1. Cross-Attention Architecture Family

**Models**: PixArt-α/Σ, Lumina-T2I/Next, Hunyuan-DiT, SANA

**Block Structure**:

```
Input: hidden_states (image), encoder_hidden_states (text), timestep

1. Self-Attention Layer:
   norm_hidden_states = norm1(hidden_states)
   attn_output = self_attention(norm_hidden_states)  # Image attends to itself
   hidden_states = hidden_states + attn_output

2. Cross-Attention Layer:
   norm_hidden_states = norm2(hidden_states)
   attn_output = cross_attention(
       Q: norm_hidden_states,           # Query from image
       K,V: encoder_hidden_states       # Key,Value from text
   )
   hidden_states = hidden_states + attn_output

3. Feed-Forward:
   hidden_states = hidden_states + ff(norm3(hidden_states))

Output: hidden_states (modified image)
```

**Timestep Handling**:

- **PixArt/Lumina/SANA**: `AdaLayerNormSingle` - timestep → 6 modulation params (shift, scale, gate)
- **Hunyuan-DiT**: `HunyuanCombinedTimestepTextSizeStyleEmbedding` - fuses timestep + pooled text + size + style

**Key Characteristics**:
✅ **Pros**:

- Clean separation of modalities
- Explicit control over text conditioning
- Well-established architecture

❌ **Cons**:

- Two attention operations per block (more compute)
- Text processing is one-way (image queries text)

**Code Evidence** (PixArt):

```python
# pixart_transformer_2d.py lines 390-395
hidden_states = block(
    hidden_states,
    attention_mask=attention_mask,
    encoder_hidden_states=encoder_hidden_states,  # Text separate
    encoder_attention_mask=encoder_attention_mask,
    timestep=timestep,
)
```

---

### 2. Double-Stream Architecture Family

**Models**: SD3, CogView3-Plus, CogView4, Qwen-Image, HiDream

**Block Structure**:

```
Input: hidden_states (image), encoder_hidden_states (text), temb

1. Normalize Both Streams with adaLN:
   norm_img = norm1(hidden_states, temb)
   norm_txt = norm1_context(encoder_hidden_states, temb)

2. Joint Attention:
   # Compute Q,K,V for both streams
   Q_img, K_img, V_img = project_image(norm_img)
   Q_txt, K_txt, V_txt = project_text(norm_txt)

   # Option A: Concatenate and attend (CogView4, FLUX single-stream)
   hidden = concat([norm_txt, norm_img], dim=1)
   attn_output = attention(hidden)
   txt_out, img_out = split(attn_output)

   # Option B: Separate projections, joint compute (SD3, FLUX double-stream)
   img_out = attention(Q_img, concat([K_txt, K_img]), concat([V_txt, V_img]))
   txt_out = attention(Q_txt, concat([K_txt, K_img]), concat([V_txt, V_img]))

3. Add Residuals with Gates:
   hidden_states = hidden_states + gate_msa * img_out
   encoder_hidden_states = encoder_hidden_states + c_gate_msa * txt_out

4. Feed-Forward (Separate for each stream):
   hidden_states = hidden_states + ff(norm2(hidden_states))
   encoder_hidden_states = encoder_hidden_states + ff_context(norm2_context(encoder_hidden_states))

Output: hidden_states, encoder_hidden_states (both modified)
```

**Timestep Handling**:

- **SD3/FLUX**: `CombinedTimestepTextProjEmbeddings` - **timestep + pooled_text fused**
- **CogView3/4**: `CogView3CombinedTimestepSizeEmbeddings` - timestep + size conditions
- **Qwen/HiDream**: Timestep only

**Key Characteristics**:
✅ **Pros**:

- Bidirectional text-image interaction
- More efficient than two separate attentions
- Text stream gets updated (can benefit from image info)

❌ **Cons**:

- More complex attention mechanism
- Requires careful synchronization of streams

**Code Evidence** (SD3):

```python
# transformer_sd3.py lines 365-367
temb = self.time_text_embed(timestep, pooled_projections)  # FUSED!
encoder_hidden_states = self.context_embedder(encoder_hidden_states)

# attention.py lines 690-694 (JointTransformerBlock)
attn_output, context_attn_output = self.attn(
    hidden_states=norm_hidden_states,
    encoder_hidden_states=norm_encoder_hidden_states,  # Both streams to attention
)
```

**Code Evidence** (CogView4 - concat variant):

```python
# transformer_cogview4.py lines 139-140
# Concatenate text and image
hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

# lines 142-145
query = attn.to_q(hidden_states)  # Single Q,K,V over concatenated
key = attn.to_k(hidden_states)
value = attn.to_v(hidden_states)

# lines 188-190
# Split back after attention
encoder_hidden_states, hidden_states = hidden_states.split(
    [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
)
```

---

### 3. Hybrid Architecture (FLUX)

**FLUX uses BOTH approaches in sequence**:

**Double-Stream Blocks** (early layers):

```python
# transformer_flux.py FluxTransformerBlock
# Processes text and image in parallel with joint attention
encoder_hidden_states, hidden_states = block(
    hidden_states, encoder_hidden_states, temb
)
```

**Single-Stream Blocks** (later layers):

```python
# transformer_flux.py FluxSingleTransformerBlock lines 386, 405
# Concatenate streams
hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

# Process together
attn_output = self.attn(hidden_states)

# Split back
encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
```

**Rationale**:

- Early layers: Keep modalities separate for specialized processing
- Later layers: Merge for unified semantic understanding

---

### 4. Single-Stream Architecture (OmniGen)

**Block Structure**:

```
Input: SINGLE concatenated sequence [text_tokens, time_token, image_tokens]

1. Self-Attention over entire sequence:
   norm_hidden_states = norm(hidden_states)  # Single norm for all
   attn_output = self_attention(
       hidden_states=norm_hidden_states,
       encoder_hidden_states=norm_hidden_states,  # Self-attention
       attention_mask=attention_mask,             # Controls visibility
       image_rotary_emb=image_rotary_emb         # Position encoding
   )

2. Feed-Forward:
   hidden_states = hidden_states + ff(norm2(hidden_states))

Output: hidden_states (entire concatenated sequence)
```

**Sequence Construction**:

```python
# transformer_omnigen.py lines 427-435
# Create time token
time_token = self.time_token(timestep_proj).unsqueeze(1)  # Shape: (B, 1, D)

# Get text + input image tokens
condition_tokens = self._get_multimodal_embeddings(...)  # Text + input images

# Concatenate ALL into single sequence
hidden_states = torch.cat([condition_tokens, time_token, hidden_states], dim=1)
#                          ^^^^^^^^^^^^^^^^  ^^^^^^^^^^  ^^^^^^^^^^^^^
#                          Text tokens       TIME TOKEN  Output image tokens
```

**Key Characteristics**:
✅ **Pros**:

- Unified processing (everything is a token)
- Simpler architecture (no special attention types)
- Flexible: attention decides how to use each modality
- Naturally multimodal (can easily add more token types)

❌ **Cons**:

- Longer sequence lengths
- Requires careful attention masking
- Less explicit control over cross-modal interactions

---

## Timestep Conditioning Strategies

### Strategy 1: Time Only (AdaLN Modulation)

**Used by**: PixArt, Lumina, SANA, Qwen, HiDream

**Mechanism**:

```python
# Timestep → embedding → modulation parameters
timestep_emb = timestep_embedder(timestep)  # (B, D)
mod_params = linear(silu(timestep_emb))     # (B, 6*D)
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_params.chunk(6)

# Apply to activations
norm_hidden_states = layer_norm(hidden_states)
norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
```

**Characteristics**:

- Clean separation: text via attention, time via modulation
- Standard diffusion conditioning

### Strategy 2: Time + Text Fusion

**Used by**: SD3, FLUX, Hunyuan-DiT

**Mechanism**:

```python
# embeddings.py lines 1585-1591 (CombinedTimestepTextProjEmbeddings)
timesteps_emb = self.timestep_embedder(timesteps_proj)
pooled_text_emb = self.text_embedder(pooled_projection)
conditioning = timesteps_emb + pooled_text_emb  # ADD together!

# This fused embedding modulates BOTH streams
norm_img = norm1(hidden_states, emb=conditioning)
norm_txt = norm1_context(encoder_hidden_states, emb=conditioning)
```

**Rationale**:

- Both timestep and pooled text are "global" conditioning signals
- Efficient: single modulation for both
- Pooled text = global semantic, Timestep = noise level

### Strategy 3: Time as Token

**Used by**: OmniGen

**Mechanism**:

```python
# Timestep becomes a token in the sequence
time_token = time_token_embedder(timestep).unsqueeze(1)  # (B, 1, D)
sequence = concat([text_tokens, time_token, image_tokens])

# Processed via standard self-attention
attn_output = self_attention(sequence)
```

**Characteristics**:

- Most flexible: attention learns how to use timestep
- Natural for unified architectures
- Also maintains separate `temb` for output layer modulation

---

## Attention Mask Strategies

### Cross-Attention (No Masking Needed)

- Image tokens can only attend to text via dedicated cross-attention
- Text is encoder-only, doesn't attend to image

### Double-Stream (Causal/Bidirectional)

- **Bidirectional within each stream**: tokens can see all tokens in same stream
- **Cross-stream**: configurable via attention mask
- Most models allow bidirectional cross-stream attention

### Single-Stream (Complex Masking)

- **OmniGen**: Attention mask controls which tokens can see which
- Typically:
  - Text can see text
  - Image can see text + previous image
  - Time can see everything (or just text)

---

## Performance Implications

### Computational Cost (per block)

| Architecture        | Attention Ops | Complexity    | Notes                               |
| ------------------- | ------------- | ------------- | ----------------------------------- |
| **Cross-Attention** | 2             | `O(N² + N·M)` | N=image tokens, M=text tokens       |
| **Double-Stream**   | 1 (joint)     | `O((N+M)²)`   | More efficient than cross-attn      |
| **Single-Stream**   | 1             | `O((N+M+1)²)` | +1 for time token, longest sequence |

### Memory Footprint

- **Cross-Attention**: Moderate (separate KV caches)
- **Double-Stream**: Higher (joint KV cache, but shared)
- **Single-Stream**: Highest (longest sequence)

### Expressiveness

- **Cross-Attention**: Good (explicit text-to-image)
- **Double-Stream**: Better (bidirectional interaction)
- **Single-Stream**: Best (fully flexible attention)

---

## Architecture Decision Tree

```
Need text conditioning?
├─ YES
│  ├─ Want bidirectional text-image interaction?
│  │  ├─ YES → Double-Stream (SD3, CogView4, Qwen)
│  │  │  ├─ Large model? → Hybrid (FLUX)
│  │  │  └─ Simpler implementation? → CogView4 style (concat→attn→split)
│  │  └─ NO → Cross-Attention (PixArt, SANA, Hunyuan)
│  │     └─ Benefits: Simpler, well-understood, good results
│  └─ Need ultimate flexibility/multimodal?
│     └─ YES → Single-Stream (OmniGen)
│        └─ Everything is a token, unified processing
└─ NO (class-conditional, unconditional)
   └─ Use U-ViT or simple DiT
```

---

## Summary of Key Insights

### 1. **No Conflict Between Concatenation and Cross-Attention**

Different architectures use fundamentally different paradigms:

- Cross-attention: **NO concatenation**, separate modality streams
- Double-stream: **NO concatenation** (except CogView4 variant), parallel processing
- Single-stream: **YES concatenation**, unified token sequence

### 2. **Timestep Strategies Vary**

- **Modulation-only**: Clean, works well
- **Fused with text**: Efficient for double-stream
- **As token**: Most flexible, unified with single-stream

### 3. **Evolution Trend**

- **Early (2023)**: Cross-attention dominates (PixArt)
- **Mid (2024)**: Double-stream emerges (SD3)
- **Latest (2025)**: Mixed approaches (FLUX hybrid, OmniGen unified, CogView4 concat-variant)

### 4. **Trade-offs**

| Aspect                     | Cross-Attn | Double-Stream | Single-Stream |
| -------------------------- | ---------- | ------------- | ------------- |
| **Simplicity**             | ⭐⭐⭐     | ⭐⭐          | ⭐⭐⭐        |
| **Efficiency**             | ⭐⭐       | ⭐⭐⭐        | ⭐⭐          |
| **Flexibility**            | ⭐⭐       | ⭐⭐⭐        | ⭐⭐⭐⭐      |
| **Text-Image Interaction** | ⭐⭐       | ⭐⭐⭐⭐      | ⭐⭐⭐⭐      |
| **Maturity**               | ⭐⭐⭐⭐   | ⭐⭐⭐        | ⭐⭐          |

---

## Conclusion

The figure you showed is **architecturally accurate**! It correctly distinguishes:

1. ✅ **Cross-Attention Models** (PixArt, Lumina): Separate text/image processing
2. ✅ **Double-Stream Models** (SD3, FLUX): Parallel streams with joint attention
3. ✅ **Single-Stream Models** (OmniGen): Concatenated token sequence
4. ✅ **Timestep Labels**: Correctly shows "Time", "Time+Text", and token concatenation

Each architecture makes different trade-offs between simplicity, efficiency, and expressiveness. Modern models are exploring hybrid approaches (FLUX) and unified token-based methods (OmniGen) for maximum flexibility.
