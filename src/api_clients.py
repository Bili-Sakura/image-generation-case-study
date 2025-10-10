"""
API clients for closed-source image generation services.
Supports OpenAI DALL-E, Google Imagen, Bytedance, and Kling.
"""

import os
import requests
import base64
from io import BytesIO
from typing import Optional, Dict, Any, Tuple
from PIL import Image
import json
from abc import ABC, abstractmethod


class ImageGenerationClient(ABC):
    """Abstract base class for image generation API clients."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        **kwargs
    ) -> Tuple[Optional[Image.Image], str]:
        """Generate an image from a prompt.
        
        Args:
            prompt: Text prompt for image generation
            width: Image width
            height: Image height
            **kwargs: Additional parameters specific to the API
            
        Returns:
            Tuple of (PIL Image or None, error message or empty string)
        """
        pass


class OpenAIClient(ImageGenerationClient):
    """OpenAI DALL-E API client."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1/images/generations"
        
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        model: str = "dall-e-3",
        quality: str = "standard",
        style: str = "vivid",
        **kwargs
    ) -> Tuple[Optional[Image.Image], str]:
        """Generate image using DALL-E.
        
        Args:
            prompt: Text prompt
            width: Image width (1024, 1792 for DALL-E 3)
            height: Image height (1024, 1792 for DALL-E 3)
            model: Model name (dall-e-2, dall-e-3)
            quality: Image quality (standard, hd)
            style: Image style (vivid, natural)
            
        Returns:
            Tuple of (PIL Image or None, error message or empty string)
        """
        if not self.api_key:
            return None, "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
        
        try:
            # DALL-E 3 only supports specific sizes
            size = f"{width}x{height}"
            if model == "dall-e-3":
                if size not in ["1024x1024", "1792x1024", "1024x1792"]:
                    size = "1024x1024"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "prompt": prompt,
                "n": 1,
                "size": size,
                "quality": quality,
                "style": style,
                "response_format": "b64_json"  # Get base64 encoded image
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            b64_image = result["data"][0]["b64_json"]
            
            # Decode base64 to image
            image_data = base64.b64decode(b64_image)
            image = Image.open(BytesIO(image_data))
            
            return image, ""
            
        except requests.exceptions.RequestException as e:
            return None, f"OpenAI API error: {str(e)}"
        except Exception as e:
            return None, f"OpenAI error: {str(e)}"


class GoogleImagenClient(ImageGenerationClient):
    """Google Imagen API client."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Google Imagen client.
        
        Args:
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        # Using Vertex AI endpoint
        self.base_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/publishers/google/models/imagegeneration:predict"
        
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        **kwargs
    ) -> Tuple[Optional[Image.Image], str]:
        """Generate image using Google Imagen.
        
        Args:
            prompt: Text prompt
            width: Image width
            height: Image height
            num_images: Number of images to generate
            
        Returns:
            Tuple of (PIL Image or None, error message or empty string)
        """
        if not self.api_key:
            return None, "Google API key not found. Set GOOGLE_API_KEY environment variable."
        
        try:
            project_id = os.getenv("GOOGLE_PROJECT_ID")
            if not project_id:
                return None, "Google Project ID not found. Set GOOGLE_PROJECT_ID environment variable."
            
            url = self.base_url.format(project_id=project_id)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "instances": [{
                    "prompt": prompt
                }],
                "parameters": {
                    "sampleCount": num_images,
                    "aspectRatio": f"{width}:{height}",
                    "safetyFilterLevel": "block_some",
                    "personGeneration": "allow_adult"
                }
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            # Get first image from predictions
            if "predictions" in result and len(result["predictions"]) > 0:
                b64_image = result["predictions"][0]["bytesBase64Encoded"]
                image_data = base64.b64decode(b64_image)
                image = Image.open(BytesIO(image_data))
                return image, ""
            else:
                return None, "No image generated by Google Imagen"
            
        except requests.exceptions.RequestException as e:
            return None, f"Google Imagen API error: {str(e)}"
        except Exception as e:
            return None, f"Google Imagen error: {str(e)}"


class BytedanceClient(ImageGenerationClient):
    """Bytedance (Douyin/TikTok) image generation API client."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Bytedance client.
        
        Args:
            api_key: Bytedance API key. If None, reads from BYTEDANCE_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("BYTEDANCE_API_KEY")
        # Using Volcano Engine (Bytedance Cloud) endpoint
        self.base_url = "https://visual.volcengineapi.com"
        
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        **kwargs
    ) -> Tuple[Optional[Image.Image], str]:
        """Generate image using Bytedance API.
        
        Args:
            prompt: Text prompt
            width: Image width
            height: Image height
            
        Returns:
            Tuple of (PIL Image or None, error message or empty string)
        """
        if not self.api_key:
            return None, "Bytedance API key not found. Set BYTEDANCE_API_KEY environment variable."
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "req_key": "text2img",
                "prompt": prompt,
                "width": width,
                "height": height,
                "scale": 7.5,
                "seed": -1,
                "ddim_steps": 25
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/text2image",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") == 0 and "data" in result:
                # Get image URL or base64 data
                if "image" in result["data"]:
                    b64_image = result["data"]["image"]
                    image_data = base64.b64decode(b64_image)
                    image = Image.open(BytesIO(image_data))
                    return image, ""
                elif "image_url" in result["data"]:
                    img_response = requests.get(result["data"]["image_url"])
                    img_response.raise_for_status()
                    image = Image.open(BytesIO(img_response.content))
                    return image, ""
            
            return None, f"Bytedance API error: {result.get('message', 'Unknown error')}"
            
        except requests.exceptions.RequestException as e:
            return None, f"Bytedance API error: {str(e)}"
        except Exception as e:
            return None, f"Bytedance error: {str(e)}"


class KlingClient(ImageGenerationClient):
    """Kling AI image generation API client."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Kling client.
        
        Args:
            api_key: Kling API key. If None, reads from KLING_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("KLING_API_KEY")
        self.base_url = "https://api.klingai.com/v1/images/generations"
        
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        model: str = "kling-v1",
        **kwargs
    ) -> Tuple[Optional[Image.Image], str]:
        """Generate image using Kling AI.
        
        Args:
            prompt: Text prompt
            width: Image width
            height: Image height
            model: Model version
            
        Returns:
            Tuple of (PIL Image or None, error message or empty string)
        """
        if not self.api_key:
            return None, "Kling API key not found. Set KLING_API_KEY environment variable."
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": f"{width}:{height}",
                "n": 1,
                "response_format": "b64_json"
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if "data" in result and len(result["data"]) > 0:
                b64_image = result["data"][0]["b64_json"]
                image_data = base64.b64decode(b64_image)
                image = Image.open(BytesIO(image_data))
                return image, ""
            
            return None, "No image generated by Kling AI"
            
        except requests.exceptions.RequestException as e:
            return None, f"Kling API error: {str(e)}"
        except Exception as e:
            return None, f"Kling error: {str(e)}"


# Factory function to get the appropriate client
def get_api_client(provider: str) -> Optional[ImageGenerationClient]:
    """Get an API client for the specified provider.
    
    Args:
        provider: Provider name (openai, google, bytedance, kling)
        
    Returns:
        ImageGenerationClient instance or None
    """
    provider = provider.lower()
    
    clients = {
        "openai": OpenAIClient,
        "google": GoogleImagenClient,
        "bytedance": BytedanceClient,
        "kling": KlingClient,
    }
    
    client_class = clients.get(provider)
    if client_class:
        return client_class()
    
    return None
