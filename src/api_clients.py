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
from volcengine.visual.VisualService import VisualService


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


class SeedDreamClient(ImageGenerationClient):
    """SeedDream 3.0 图像生成客户端"""

    def __init__(self, access_key_id: Optional[str] = None, secret_access_key: Optional[str] = None):
        """
        初始化SeedDream客户端
        """
        self.service = VisualService()
        self.access_key_id = access_key_id or os.getenv("SEEDDREAM_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or os.getenv("SEEDDREAM_SECRET_ACCESS_KEY")
        
        if self.access_key_id and self.secret_access_key:
            self.service.set_ak(self.access_key_id)
            self.service.set_sk(self.secret_access_key)

    def generate(
        self,
        prompt: str,
        width: int = 1328,
        height: int = 1328,
        seed: int = -1,
        scale: float = 2.5,
        **kwargs
    ) -> Tuple[Optional[Image.Image], str]:
        
        if not self.access_key_id or not self.secret_access_key:
            return None, "SeedDream access key or secret not found."

        params = {
            "req_key": "high_aes_general_v30l_zt2i",  # SeedDream通用3.0模型
            "prompt": prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "scale": scale,
            "return_url": True,
        }

        try:
            response = self.service.high_aes_smart_drawing(params)
            result = json.loads(response)

            if result.get("code") == 10000:
                data = result.get("data", {})
                image_urls = data.get("image_urls", [])
                if image_urls:
                    image_url = image_urls[0]
                    res = requests.get(image_url)
                    res.raise_for_status()
                    image = Image.open(BytesIO(res.content))
                    return image, ""
                else:
                    return None, "No image URL found in SeedDream response."
            else:
                error_message = result.get("message", "Unknown error")
                return None, f"SeedDream API error: {error_message}"

        except Exception as e:
            return None, f"Error generating with SeedDream: {str(e)}"

# Factory function to get the appropriate client
def get_api_client(provider: str) -> Optional[ImageGenerationClient]:
    """Get an API client for the specified provider.
    
    Args:
        provider: Provider name (should be 'seeddream')
        
    Returns:
        ImageGenerationClient instance or None
    """
    if provider.lower() == "seeddream":
        return SeedDreamClient()
    
    return None
