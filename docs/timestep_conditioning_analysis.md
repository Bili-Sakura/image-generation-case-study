# Timestep Conditioning in DiT Architectures

## Summary

Your figure's timestep labels are **CORRECT**! Different architectures handle timestep conditioning in fundamentally different ways:

- **PixArt/Lumina**: Timestep only (via adaLN modulation)
- **SD3/FLUX**: Timestep + Text fused (combined embedding)
- **OmniGen**: Timestep as token (concatenated in sequence)

## Detailed Analysis

---

### 1. PixArt / Lumina-Next: Timestep-Only Conditioning

**Implementation**: `AdaLayerNormSingle` in `normalization.py`

```python
# transformer_2d.py lines 178-179
self.adaln_single = AdaLayerNormSingle(
    self.inner_dim, use_additional_conditions=self.use_additional_conditions
)

# Forward pass (line 368-370)
timestep, embedded_timestep = self.adaln_single(
    timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
)
```

**AdaLayerNormSingle implementation** (lines 236-267 in `normalization.py`):

```python
def forward(self, timestep, added_cond_kwargs, batch_size, hidden_dtype):
    # Embed timestep with size conditions (resolution, aspect ratio)
    embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size)

    # Project to modulation parameters
    return self.linear(self.silu(embedded_timestep)), embedded_timestep
    # Returns (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
```

**How it's used in BasicTransformerBlock** (lines 976-981 in `attention.py`):

```python
# Extract 6 modulation parameters from timestep
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
).chunk(6, dim=1)

# Apply to activations
norm_hidden_states = self.norm1(hidden_states)
norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
```

**Key Points**:

- ✅ Timestep is processed independently
- ✅ Text conditioning comes via cross-attention (separate path)
- ✅ AdaLN modulates feature activations with timestep-derived parameters
- ✅ Figure label: "Time" ✓

---

### 2. Stable Diffusion 3: Time + Text Fusion

**Implementation**: `CombinedTimestepTextProjEmbeddings` in `embeddings.py`

```python
# transformer_sd3.py lines 151-153
self.time_text_embed = CombinedTimestepTextProjEmbeddings(
    embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
)

# Forward pass (line 366)
temb = self.time_text_embed(timestep, pooled_projections)
#                                     ^^^^^^^^^^^^^^^^^^
#                                     Pooled text embeddings!
```

**CombinedTimestepTextProjEmbeddings** (lines 1577-1593 in `embeddings.py`):

```python
def forward(self, timestep, pooled_projection):
    # Process timestep
    timesteps_proj = self.time_proj(timestep)
    timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)

    # Process pooled text
    pooled_projections = self.text_embedder(pooled_projection)

    # FUSE: Add them together!
    conditioning = timesteps_emb + pooled_projections  # (N, D)
    return conditioning
```

**How it's used in JointTransformerBlock** (lines 676-687 in `attention.py`):

```python
# Use fused time+text embedding for BOTH streams
norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
    hidden_states, emb=temb  # temb contains time+text!
)

norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
    encoder_hidden_states, emb=temb  # Same temb for text stream!
)
```

**Key Points**:

- ✅ Timestep + pooled text are **ADDED** together
- ✅ Creates unified conditioning signal `temb`
- ✅ Same `temb` modulates both image and text streams
- ✅ Sequential (caption) text still goes through separate stream
- ✅ Figure label: "Time + Text" ✓

---

### 3. FLUX: Time + Text Fusion (Same as SD3)

**Implementation**: Similar to SD3

```python
# transformer_flux.py lines 699-703
temb = (
    self.time_text_embed(timestep, pooled_projections)
    if guidance is None
    else self.time_text_embed(timestep, guidance, pooled_projections)
)
```

**FluxTransformerBlock usage** (lines 445-449):

```python
# Both streams modulated by same time+text embedding
norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
    hidden_states, emb=temb
)

norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
    encoder_hidden_states, emb=temb  # Same temb!
)
```

**Key Points**:

- ✅ Identical to SD3 approach
- ✅ Can optionally include guidance scale in fusion
- ✅ Figure label: "Time + Text" ✓

---

### 4. OmniGen: Timestep as Token

**Implementation**: Timestep becomes part of the sequence!

```python
# transformer_omnigen.py lines 363-365
self.time_proj = Timesteps(time_step_dim, flip_sin_to_cos, downscale_freq_shift)
self.time_token = TimestepEmbedding(time_step_dim, hidden_size, timestep_activation_fn)
self.t_embedder = TimestepEmbedding(time_step_dim, hidden_size, timestep_activation_fn)
```

**Two ways timestep is used**:

**A. As a token in the sequence** (lines 427-435):

```python
# 1. Embed timestep as a token
timestep_proj = self.time_proj(timestep).type_as(hidden_states)
time_token = self.time_token(timestep_proj).unsqueeze(1)  # Shape: (B, 1, D)

# 2. Get text and image tokens
condition_tokens = self._get_multimodal_embeddings(...)  # Text + input images
hidden_states = self.patch_embedding(hidden_states, ...)  # Output image patches

# 3. CONCATENATE: [text, time, image]
hidden_states = torch.cat([condition_tokens, time_token, hidden_states], dim=1)
#                          ^^^^^^^^^^^^^^^^  ^^^^^^^^^^  ^^^^^^^^^^^^^
#                          Text tokens       TIME TOKEN  Image tokens
```

**B. As adaLN conditioning for output** (lines 429, 462):

```python
# Also create embedding for output modulation
temb = self.t_embedder(timestep_proj)

# Use in output layer
hidden_states = self.norm_out(hidden_states, temb=temb)
```

**OmniGenBlock processing** (lines 264-281):

```python
def forward(self, hidden_states, attention_mask, image_rotary_emb):
    # Standard self-attention over entire sequence (including time token!)
    norm_hidden_states = self.input_layernorm(hidden_states)
    attn_output = self.self_attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_hidden_states,  # Self-attention
        attention_mask=attention_mask,
        image_rotary_emb=image_rotary_emb,
    )
    # Time token is just another token in the sequence
```

**Key Points**:

- ✅ Timestep becomes a **token** in the concatenated sequence
- ✅ Processed via self-attention along with text and image
- ✅ Position: `[text_tokens, TIME, image_tokens]`
- ✅ Also used for output layer modulation (dual role)
- ✅ Figure label: Shows time with ⊙ (concatenation) ✓

---

## Comparison Table

| Model             | Timestep Method  | Text Fusion?                | Label in Figure | Code Reference                       |
| ----------------- | ---------------- | --------------------------- | --------------- | ------------------------------------ |
| **PixArt/Lumina** | AdaLN modulation | ❌ No (separate cross-attn) | "Time"          | `AdaLayerNormSingle`                 |
| **SD3**           | AdaLN modulation | ✅ Yes (time+pooled_text)   | "Time + Text"   | `CombinedTimestepTextProjEmbeddings` |
| **FLUX**          | AdaLN modulation | ✅ Yes (time+pooled_text)   | "Time + Text"   | `CombinedTimestepTextProjEmbeddings` |
| **OmniGen**       | Token + AdaLN    | ❌ No (time as token)       | "Time" (⊙)      | `time_token` + concatenation         |

---

## Why Different Approaches?

### AdaLN Modulation (PixArt, SD3, FLUX)

**Mechanism**: Timestep → linear projection → modulation parameters (shift, scale, gate)

```python
# Generic adaLN pattern
modulation_params = linear(silu(timestep_embedding))
normalized = layer_norm(features) * (1 + scale) + shift
output = gate * attention_or_ffn(normalized)
```

**Advantages**:

- Efficient: affects all features with few parameters
- Interpretable: explicit scaling and shifting
- Standard in diffusion models

### Token-Based (OmniGen)

**Mechanism**: Timestep → embedding → concatenate as token

```python
time_token = timestep_embedder(timestep)  # Shape: (B, 1, D)
sequence = cat([text, time_token, image], dim=1)
# Process via standard self-attention
```

**Advantages**:

- Unified architecture: same attention for all modalities
- Flexible: attention decides how to use timestep
- Simpler: no special modulation layers

### Time + Text Fusion (SD3, FLUX)

**Why fuse?**

- Pooled text captures global semantic information
- Timestep indicates noise level
- Both are "global" conditioning signals
- Efficient: single modulation for both

```python
temb = timestep_emb + pooled_text_emb
# Both modulate the same way
```

---

## Validation of Figure

Your figure correctly shows:

1. ✅ **PixArt/Lumina**: "Time" label

   - Separate input, no fusion with text
   - Text via cross-attention

2. ✅ **SD3**: "Time + Text" label

   - Explicit fusion via addition
   - Code: `conditioning = timesteps_emb + pooled_projections`

3. ✅ **OmniGen**: "Time" with ⊙ (concatenation)

   - Time as token in sequence
   - Code: `torch.cat([condition_tokens, time_token, hidden_states])`

4. ✅ **FLUX**: "Time + Text" label
   - Same fusion as SD3
   - Double-stream blocks use fused embedding

## Conclusion

The figure **accurately represents** timestep conditioning across all architectures! The key distinctions are:

- **"Time"** alone = timestep modulation without text fusion
- **"Time + Text"** = fused timestep+text embedding
- **⊙ with Time** = timestep as concatenated token

Each approach has different trade-offs for efficiency, flexibility, and architectural simplicity.
