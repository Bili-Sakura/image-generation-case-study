# Quick Reference: DiT Architecture Comparison

## 📊 At-A-Glance Comparison Table

| Model                  | Year    | Params | Architecture      | Text Handling         | Timestep            | Block Type             | ✅ Your Figure Shows |
| ---------------------- | ------- | ------ | ----------------- | --------------------- | ------------------- | ---------------------- | -------------------- |
| **PixArt-α/Σ**         | 2023-24 | 5.46B  | **Cross-Attn**    | Separate (cross-attn) | Time only           | Self-Attn + Cross-Attn | ✓ Correct            |
| **Lumina-T2I/Next**    | 2024    | 4.34B  | **Cross-Attn**    | Separate (cross-attn) | Time only           | Self-Attn + Cross-Attn | ✓ Correct            |
| **Stable Diffusion 3** | 2024    | 7.69B  | **Double-Stream** | Parallel streams      | **Time + Text**     | Joint Attn (⊙)         | ✓ Correct            |
| **FLUX.1-Dev**         | 2024    | 16.87B | **Hybrid**        | Double + Single       | **Time + Text**     | Mixed (⊙)              | ✓ Correct            |
| **CogView3-Plus**      | 2024    | 8.02B  | **Double-Stream** | Parallel streams      | Time + Size         | Joint Attn             | Similar to SD3       |
| **Hunyuan-DiT**        | 2024    | 3.61B  | **Cross-Attn**    | Separate (cross-attn) | Time + Text + Style | Self-Attn + Cross-Attn | Similar to PixArt    |
| **SANA**               | 2025    | 3.52B  | **Cross-Attn**    | Separate (cross-attn) | Time + Guidance     | Self-Attn + Cross-Attn | Similar to PixArt    |
| **CogView4-6B**        | 2025    | 6.00B  | **Double-Stream** | Concat → Split        | Time + Size         | Concat Joint Attn      | Similar to SD3       |
| **Qwen-Image**         | 2025    | 28.85B | **Double-Stream** | Parallel streams      | Time only           | Joint Attn             | Similar to SD3       |
| **OmniGen**            | N/A     | N/A    | **Single-Stream** | **Token Concat**      | **Time as Token**   | Unified Self-Attn (⊙)  | ✓ Correct            |

---

## 🎯 Three Core Architectures

### 1️⃣ Cross-Attention Architecture

**Used by**: PixArt, Lumina, Hunyuan-DiT, SANA

```
┌─────────────┐          ┌──────────────┐
│ Text Tokens │────┐     │ Noised Image │
└─────────────┘    │     └──────────────┘
                   │            │
                   │            ↓
                   │     ┌─────────────┐
                   │     │ Self-Attn   │
                   │     └─────────────┘
                   │            │
                   │            ↓
                   └─────→ Cross-Attn
                   (K,V)   (Q from img)
                               │
                               ↓
                          ┌─────────┐
                          │   FFN   │
                          └─────────┘
                               │
                               ↓
                           Output Image
```

**Key**: Text and image stay **SEPARATE**. Image queries attend to text.

---

### 2️⃣ Double-Stream Architecture

**Used by**: SD3, FLUX (early), CogView3/4, Qwen, HiDream

```
┌─────────────┐          ┌──────────────┐
│ Text Stream │          │ Image Stream │
└─────────────┘          └──────────────┘
       │                        │
       ↓                        ↓
┌─────────────┐          ┌──────────────┐
│ Norm + adaLN│          │ Norm + adaLN │
└─────────────┘          └──────────────┘
       │                        │
       └────────────┬───────────┘
                    ↓
            ┌──────────────┐
            │ Joint Attn ⊙ │  ← Both streams process together
            └──────────────┘
                    │
       ┌────────────┴───────────┐
       ↓                        ↓
┌─────────────┐          ┌──────────────┐
│ Text FFN    │          │ Image FFN    │
└─────────────┘          └──────────────┘
       │                        │
       ↓                        ↓
  Text Output              Image Output
```

**Key**: Two parallel streams with **joint attention** mechanism. Both get updated.

---

### 3️⃣ Single-Stream Architecture

**Used by**: OmniGen

```
┌─────┐  ┌──────┐  ┌───────┐
│Text │  │Time  │  │Image  │
│Token│  │Token │  │Tokens │
└─────┘  └──────┘  └───────┘
   │        │          │
   └────────┴──────────┘
            │
            ↓
    ┌───────────────┐
    │  CONCATENATE  │ ⊙
    └───────────────┘
            │
            ↓
    [T₁, T₂, ..., Tₙ, TIME, I₁, I₂, ..., Iₘ]
            │
            ↓
    ┌───────────────┐
    │  Self-Attn    │  ← Single attention over all tokens
    └───────────────┘
            │
            ↓
    ┌───────────────┐
    │     FFN       │
    └───────────────┘
            │
            ↓
      Extract Image Tokens → Output
```

**Key**: Everything is a token. **ONE sequence**. Processed together via self-attention.

---

## 🔑 Key Distinctions

| Feature                 | Cross-Attn           | Double-Stream   | Single-Stream        |
| ----------------------- | -------------------- | --------------- | -------------------- |
| **Modality Separation** | ✅ Complete          | ⚡ Parallel     | ❌ Unified           |
| **Text→Image Flow**     | One-way (cross-attn) | Bidirectional   | Implicit (attention) |
| **Concatenation**       | ❌ NO                | ❌ NO\*         | ✅ YES               |
| **Attention Ops/Block** | 2 (self + cross)     | 1 (joint)       | 1 (self over all)    |
| **Timestep**            | Modulation           | Time+Text fused | Token in sequence    |
| **Complexity**          | Medium               | High            | Low\*\*              |
| **Flexibility**         | Good                 | Better          | Best                 |

\* Except CogView4 variant which concat→attn→split  
\*\* Architecture is simple, but masking can be complex

---

## 💡 Understanding Timestep

### Time Only (PixArt, Lumina, Qwen)

```python
timestep → embedding → [shift, scale, gate] → modulate features
Text handled separately via attention
```

### Time + Text Fused (SD3, FLUX, Hunyuan)

```python
timestep_emb + pooled_text_emb → [shift, scale, gate]
Both global signals fused for modulation
Sequential text still in separate stream
```

### Time as Token (OmniGen)

```python
time_token = embed(timestep)
sequence = [text_tokens, time_token, image_tokens]
Process via attention (time is just another token)
```

---

## 🎓 For Your Audience

### **Your Figure is CORRECT!** ✅

1. **PixArt/Lumina** shows **separate** text input → Cross-attention ✓
2. **SD3** shows **two streams** with ⊙ symbol → Double-stream ✓
3. **FLUX** shows **mixed architecture** → Hybrid approach ✓
4. **OmniGen** shows **concatenation** ⊙ → Single-stream ✓
5. **Timestep labels** correctly distinguish:
   - "Time" alone = separate modulation
   - "Time + Text" = fused conditioning
   - Time with ⊙ = token concatenation

### **Key Takeaway**

> **There is NO conflict!** Different models use fundamentally different strategies:
>
> - **Cross-Attention**: Keep modalities SEPARATE
> - **Double-Stream**: Process in PARALLEL
> - **Single-Stream**: CONCATENATE everything
>
> Each approach has different trade-offs for efficiency, expressiveness, and simplicity.

---

## 📚 Reference

See `comprehensive_dit_architecture_comparison.md` for:

- Detailed code references with line numbers
- Complete implementation details
- Performance implications
- Architecture decision tree

**Code Base**: HuggingFace Diffusers (validated against actual implementations)
