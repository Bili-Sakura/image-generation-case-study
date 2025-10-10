# Closed-Source Image Generation API Guide

This guide explains how to use the closed-source image generation APIs integrated into this project.

## Supported Providers

### 1. OpenAI DALL-E

**Models:**
- `dall-e-2`: DALL-E 2 (faster, lower cost)
- `dall-e-3`: DALL-E 3 (higher quality, more control)

**Features:**
- Quality settings: `standard`, `hd`
- Style settings: `vivid` (hyper-real), `natural` (realistic)
- Max resolution: 1792x1792
- Support for specific sizes: 1024x1024, 1792x1024, 1024x1792

**API Documentation:** https://platform.openai.com/docs/guides/images

**Pricing:** Pay-per-image (varies by model and quality)

### 2. Google Imagen

**Models:**
- `imagegeneration@005`: Latest Imagen model via Vertex AI

**Features:**
- High-quality photorealistic generation
- Safety filtering options
- Max resolution: 1536x1536

**API Documentation:** https://cloud.google.com/vertex-ai/docs/generative-ai/image/overview

**Requirements:**
- Google Cloud Project ID
- Vertex AI API enabled

**Pricing:** Pay-per-image

### 3. Bytedance Cloud (Volcano Engine)

**Models:**
- `text2img-v1`: Bytedance text-to-image API

**Features:**
- Fast API response times
- Max resolution: 2048x2048
- Configurable sampling steps and guidance

**API Documentation:** https://www.volcengineapi.com/

**Pricing:** Pay-per-image

### 4. Kling AI

**Models:**
- `kling-v1`: Standard Kling model
- `kling-v1-pro`: Professional version (higher quality)

**Features:**
- High-quality generation
- Max resolution: 2048x2048

**API Documentation:** https://klingai.com/

**Pricing:** Pay-per-image

## Setup Instructions

### Step 1: Install Dependencies

```bash
# Install API dependencies
pip install -r requirements_api.txt
```

### Step 2: Configure API Keys

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:

```bash
# OpenAI DALL-E
OPENAI_API_KEY=sk-...

# Google Imagen
GOOGLE_API_KEY=...
GOOGLE_PROJECT_ID=your-project-id

# Bytedance Cloud
BYTEDANCE_API_KEY=...

# Kling AI
KLING_API_KEY=...
```

3. Load environment variables:

```bash
# Option 1: Export manually
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
# ... etc

# Option 2: Use a .env loader
# If using python-dotenv:
pip install python-dotenv
```

### Step 3: Verify Setup

Run the example script to test your API keys:

```bash
python example_api_generate.py
```

## Usage

### Web UI (Gradio)

1. Launch the application:

```bash
python run.py
```

2. Navigate to the **🔒 Closed-Source APIs** tab

3. Select a provider (OpenAI, Google, Bytedance, or Kling)

4. Enter your prompt and adjust settings

5. Click **🚀 Generate with API**

**Batch Comparison:**

Use the **🔄 API Comparison** tab to generate with multiple APIs simultaneously.

### Python API

#### Basic Usage

```python
from src.api_clients import get_api_client

# Get a client for any provider
client = get_api_client("openai")

# Generate an image
image, error = client.generate(
    prompt="A beautiful mountain landscape",
    width=1024,
    height=1024
)

if image:
    image.save("output.png")
    print("Success!")
else:
    print(f"Error: {error}")
```

#### Provider-Specific Examples

**OpenAI DALL-E with Advanced Settings:**

```python
from src.api_clients import OpenAIClient

client = OpenAIClient()
image, error = client.generate(
    prompt="A futuristic city at night",
    width=1024,
    height=1024,
    model="dall-e-3",
    quality="hd",      # "standard" or "hd"
    style="vivid"      # "vivid" or "natural"
)
```

**Google Imagen:**

```python
from src.api_clients import GoogleImagenClient

client = GoogleImagenClient()
image, error = client.generate(
    prompt="A photorealistic portrait",
    width=1024,
    height=1024,
    num_images=1
)
```

**Bytedance Cloud:**

```python
from src.api_clients import BytedanceClient

client = BytedanceClient()
image, error = client.generate(
    prompt="An anime character in a cyberpunk city",
    width=1024,
    height=1024
)
```

**Kling AI:**

```python
from src.api_clients import KlingClient

client = KlingClient()
image, error = client.generate(
    prompt="A magical fantasy scene",
    width=1024,
    height=1024,
    model="kling-v1-pro"  # or "kling-v1"
)
```

#### Batch Processing

```python
from src.api_clients import get_api_client

prompts = [
    "A serene lake at sunset",
    "A bustling marketplace",
    "A space station orbiting Earth"
]

providers = ["openai", "google", "kling"]

for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    for provider in providers:
        client = get_api_client(provider)
        image, error = client.generate(prompt=prompt)
        
        if image:
            filename = f"{provider}_{prompts.index(prompt)}.png"
            image.save(filename)
            print(f"  ✅ {provider}: {filename}")
        else:
            print(f"  ❌ {provider}: {error}")
```

### Using the Widget

The `closed_source_widget.py` provides a pre-built Gradio widget:

```python
from src.closed_source_widget import create_closed_source_widget
import gradio as gr

# Create the widget
widget = create_closed_source_widget()

# Use in your own Gradio app
with gr.Blocks() as app:
    gr.Markdown("# My Custom App")
    widget

app.launch()
```

## Error Handling

All API clients return a tuple of `(image, error)`:

- **Success:** `image` is a PIL Image, `error` is an empty string
- **Failure:** `image` is None, `error` contains the error message

```python
image, error = client.generate(prompt="...")

if image:
    # Success - process the image
    image.save("output.png")
else:
    # Error - handle the error
    print(f"Generation failed: {error}")
```

## Common Issues

### API Key Not Found

**Error:** "API key not found. Set XXXX_API_KEY environment variable."

**Solution:**
1. Make sure you've set the environment variable
2. Verify the variable name is correct
3. If using a .env file, ensure it's being loaded

### Google Project ID Missing

**Error:** "Google Project ID not found. Set GOOGLE_PROJECT_ID environment variable."

**Solution:**
1. Set `GOOGLE_PROJECT_ID` in addition to `GOOGLE_API_KEY`
2. Enable Vertex AI API in your Google Cloud Console

### Rate Limiting

**Error:** "Rate limit exceeded" or similar

**Solution:**
1. Implement retry logic with exponential backoff
2. Check your API tier/quota limits
3. Consider spreading requests across time

### Invalid Size

**Error:** Size not supported

**Solution:**
- OpenAI DALL-E 3: Use 1024x1024, 1792x1024, or 1024x1792
- Other APIs: Check provider-specific size limits

## Cost Considerations

Closed-source APIs charge per generation:

1. **OpenAI DALL-E:**
   - DALL-E 2: ~$0.02 per image
   - DALL-E 3 (standard): ~$0.04 per image
   - DALL-E 3 (HD): ~$0.08 per image

2. **Google Imagen:**
   - Varies by usage tier
   - Check current pricing at cloud.google.com

3. **Bytedance & Kling:**
   - Contact providers for pricing

**Tips to Reduce Costs:**
- Use lower quality settings when testing
- Start with DALL-E 2 instead of DALL-E 3
- Use open-source models for experimentation
- Monitor your usage via provider dashboards

## Best Practices

1. **API Key Security:**
   - Never commit API keys to version control
   - Use environment variables
   - Rotate keys periodically

2. **Error Handling:**
   - Always check for errors before using images
   - Implement retry logic for transient failures
   - Log errors for debugging

3. **Rate Limiting:**
   - Implement exponential backoff
   - Use async/concurrent requests carefully
   - Monitor your quota usage

4. **Cost Management:**
   - Set budget alerts in provider dashboards
   - Cache generated images
   - Use the most appropriate model for your needs

5. **Quality vs. Speed:**
   - Standard quality for testing
   - HD/Pro for final outputs
   - Use open-source models for rapid iteration

## Comparison: Open-Source vs. Closed-Source

| Aspect              | Open-Source Models       | Closed-Source APIs      |
| ------------------- | ------------------------ | ----------------------- |
| Cost                | GPU compute (one-time)   | Per-image pricing       |
| Setup               | Download models (~GB)    | API key only            |
| GPU Required        | Yes (8-16GB+ VRAM)       | No                      |
| Privacy             | Fully local              | Sent to provider        |
| Customization       | Full control             | Limited to API params   |
| Speed               | Depends on hardware      | Usually fast            |
| Quality             | Varies by model          | Generally high          |
| Offline Use         | Yes                      | No                      |

## Support

For issues specific to:
- **This integration:** Open a GitHub issue
- **API providers:** Contact their support channels
- **API keys/billing:** Use provider dashboards

## Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Google Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Gradio Documentation](https://gradio.app/docs)
- [PIL/Pillow Documentation](https://pillow.readthedocs.io/)
