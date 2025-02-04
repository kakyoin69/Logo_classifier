import tempfile
from urllib.parse import urlparse
from pathlib import Path
import numpy as np
import requests
from loguru import logger
from PIL import Image
from typing import List


class ImageUtils:
    @staticmethod
    def is_url(source: str) -> bool:
        """Check if the provided source is a URL."""
        try:
            result = urlparse(source)
            return result.scheme in ("http", "https")
        except Exception as e:
            logger.error(f"Error checking if source is a URL: {e}")
            return False

    def download_image(self, url: str) -> Path:
        """Download an image from a URL and save it to a temporary file."""
        try:
            logger.info(f"Downloading image from URL: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            suffix = Path(url).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_path = Path(temp_file.name)

            logger.info(f"Image downloaded and saved to: {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            raise

    def open_image(self, source: str) -> Image.Image:
        try:
            if self.is_url(source):
                image_path = self.download_image(source)
            else:
                image_path = Path(source)
                if not image_path.exists():
                    error_msg = f"Image file not found at: {source}"
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)

            image = Image.open(image_path).convert('RGB')  
            image_np = np.array(image)
            pil_img = Image.fromarray(image_np)
            return pil_img
        except Exception as e:
            logger.error(f"Failed to open image from {source}: {e}")
            raise

    def open_images(self, sources: List[str]) -> List[Image.Image]:
        """Open multiple images from a list of URLs or local file paths."""
        images = []
        for source in sources:
            try:
                image = self.open_image(source)
                images.append(image)
            except Exception as e:
                logger.error(f"Skipping source {source} due to error: {e}")
                continue  
        return images