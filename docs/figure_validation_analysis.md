# DiT Architecture Data Flow Validation

## Summary

**The figure is CORRECT!** There is no conflict between concatenation and cross-attention/adaLN approaches. Different DiT architectures use fundamentally different strategies to handle text and image modalities.

## Key Insight

The confusion arises from mixing up different architectural paradigms:

- **Cross-Attention Models**: Keep modalities separate, use cross-attention to attend from image to text
- **Double-Stream Models**: Process modalities in parallel streams with joint attention
- **Single-Stream Models**: Concatenate all tokens and process together with causal/self-attention

## Detailed Architecture Analysis

### 1. PixArt / Lumina-Next: Cross-Attention Block

**Implementation**: `BasicTransformerBlock` in `diffusers/models/attention.py`

```python
# Lines 946-1035 in attention.py
def forward(self, hidden_states, encoder_hidden_states, ...):
    # 1. Self-Attention on image tokens
    attn_output = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=None,  # No cross-attention here
        attention_mask=attention_mask,
    )

    # 2. Cross-Attention: image attends to text
    attn_output = self.attn2(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,  # Text tokens as context
        attention_mask=encoder_attention_mask,
    )
```

**Data Flow**:

- Text tokens → `encoder_hidden_states` (separate stream)
- Noised latents → `hidden_states` (main stream)
- NO concatenation - modalities kept separate
- Image queries attend to text keys/values via cross-attention

**Figure Validation**: ✅ CORRECT - Shows separate inputs with cross-attention mechanism

---

### 2. Stable Diffusion 3: Double-Stream Block

**Implementation**: `JointTransformerBlock` in `diffusers/models/attention.py`

```python
# Lines 667-734 in attention.py
def forward(self, hidden_states, encoder_hidden_states, temb, ...):
    # Normalize both streams with adaLN
    norm_hidden_states = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, emb=temb)

    # Joint attention - processes both streams together
    attn_output, context_attn_output = self.attn(
        hidden_states=norm_hidden_states,  # Image stream
        encoder_hidden_states=norm_encoder_hidden_states,  # Text stream
        **joint_attention_kwargs,
    )

    # Separate processing for each stream
    hidden_states = hidden_states + attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output

    # Separate feed-forwards
    ff_output = self.ff(norm_hidden_states)
    context_ff_output = self.ff_context(norm_encoder_hidden_states)
```

**Key Detail in JointAttnProcessor2_0** (`attention_processor.py`):

```python
# The attention mechanism processes both but keeps them separate:
# - Uses added_kv_proj_dim for text stream
# - Regular projections for image stream
# - Returns separate outputs for both streams
```

**Data Flow**:

- Two parallel streams (text and image)
- Joint attention mechanism processes both simultaneously
- Separate normalization and feed-forward for each
- NOT concatenation, but parallel processing
- The ⊙ symbol represents the joint attention operation

**Figure Validation**: ✅ CORRECT - Shows two streams with ⊙ indicating joint processing

---

### 3. OmniGen: Single-Stream Causal Block

**Implementation**: `OmniGenTransformer2DModel` in `transformers/transformer_omnigen.py`

```python
# Lines 408-469 in transformer_omnigen.py
def forward(self, hidden_states, timestep, input_ids, input_img_latents, ...):
    # 1. Embed output image patches
    hidden_states = self.patch_embedding(hidden_states, is_input_image=False)

    # 2. Create time token
    time_token = self.time_token(timestep_proj).unsqueeze(1)

    # 3. Get multimodal embeddings (text + input images)
    condition_tokens = self._get_multimodal_embeddings(
        input_ids, input_img_latents, input_image_sizes
    )

    # 4. CONCATENATE everything into single sequence!
    hidden_states = torch.cat([condition_tokens, time_token, hidden_states], dim=1)
    #                          ^^^^^^^^^^^^^^^^  ^^^^^^^^^^  ^^^^^^^^^^^^^
    #                          Text tokens       Time        Output image

    # 5. Process through causal self-attention blocks
    for block in self.layers:
        hidden_states = block(
            hidden_states,  # Single concatenated sequence
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb
        )
```

**OmniGenBlock** (lines 238-281):

```python
def forward(self, hidden_states, attention_mask, image_rotary_emb):
    # Single self-attention over concatenated sequence
    norm_hidden_states = self.input_layernorm(hidden_states)
    attn_output = self.self_attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_hidden_states,  # Same as input!
        attention_mask=attention_mask,
    )
```

**Data Flow**:

- Text tokens embedded via `embed_tokens`
- Time embedded as single token
- Image patches embedded via `patch_embedding`
- **ALL CONCATENATED** into single sequence: `[text, time, noised_image]`
- Single causal self-attention processes entire sequence
- Attention mask controls which tokens can see which

**Figure Validation**: ✅ CORRECT - Shows concatenation (⊙ symbols) of all modalities into single stream

---

### 4. FLUX: Hybrid Architecture

**Implementation**: `FluxTransformerBlock` and `FluxSingleTransformerBlock` in `transformers/transformer_flux.py`

#### Double-Stream Block (lines 410-470):

```python
class FluxTransformerBlock(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, temb, ...):
        # Normalize both streams separately
        norm_hidden_states = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, emb=temb)

        # Joint attention
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )

        # Returns: (hidden_states, encoder_hidden_states)
        # Separate FFNs for each stream
```

#### Single-Stream Block (lines 356-406):

```python
class FluxSingleTransformerBlock(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, temb, ...):
        text_seq_len = encoder_hidden_states.shape[1]

        # CONCATENATE text and image!
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        #                          ^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^
        #                          Text tokens           Image tokens

        # Single attention over concatenated sequence
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # Split back after processing
        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
```

**Data Flow**:

- FLUX uses BOTH architectures in sequence:
  - Early layers: Double-stream blocks (parallel processing)
  - Later layers: Single-stream blocks (concatenated processing)
- This hybrid approach combines benefits of both

**Figure Validation**: ✅ CORRECT - Shows both double-stream and single-stream blocks

---

## Answer to Your Main Question

> "How do they do concatenation which seems to be conflicted with cross-attention/adaLN?"

**There is NO conflict!** These are three distinct architectural paradigms:

### 1. **Cross-Attention Approach** (PixArt, Lumina)

- **Architecture**: Self-attention + Cross-attention
- **Modalities**: Kept separate
- **Conditioning**: Image queries attend to text keys/values
- **Pros**: Clean separation, interpretable
- **Cons**: Two attention operations per block

### 2. **Double-Stream Approach** (SD3, FLUX double-stream)

- **Architecture**: Joint attention with separate streams
- **Modalities**: Parallel processing
- **Conditioning**: Via adaLN (timestep modulation)
- **Mechanism**: Attention sees both streams but maintains separation via separate projections
- **Pros**: Efficient, can model interactions
- **Cons**: More complex attention mechanism

### 3. **Single-Stream Concatenation** (OmniGen, FLUX single-stream)

- **Architecture**: Pure self-attention on concatenated sequence
- **Modalities**: Concatenated into single sequence
- **Conditioning**: Implicit through attention mask and position
- **Mechanism**: All tokens processed together, attention mask controls visibility
- **Pros**: Simple, unified processing
- **Cons**: Longer sequence length, requires careful masking

## Key Implementation Details

### Why Concatenation Works (Single-Stream):

```python
# In OmniGen and FLUX single-stream:
sequence = [text_token_1, ..., text_token_N, time_token, img_token_1, ..., img_token_M]
# Position embeddings + RoPE distinguish modalities
# Attention mask controls which tokens can attend to which
# All processed through same self-attention mechanism
```

### Why Cross-Attention Works:

```python
# In PixArt:
# Q from image, K,V from text - explicit cross-modal attention
attn_output = softmax(Q_img @ K_text.T) @ V_text
```

### Why Double-Stream Works:

```python
# In SD3:
# Two separate Q,K,V projections but computed together
img_output = attention(Q_img, K_img, V_img)
txt_output = attention(Q_txt, K_txt, V_txt)
# But attention mechanism can see both streams internally
```

## Conclusion

Your figure **accurately represents** all architectures! The key insight is:

- **NO concatenation** in cross-attention models → separate modality processing
- **NO concatenation** in double-stream models → parallel stream processing
- **YES concatenation** in single-stream models → unified sequence processing

Each approach has different trade-offs for computational efficiency, parameter sharing, and modeling flexibility. Modern models like FLUX even combine both approaches to get the best of both worlds!
