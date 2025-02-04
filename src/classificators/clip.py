from loguru import logger
import torch
from PIL import Image
import clip

class Clip:
    """Classifier for detecting content"""
    def __init__(self, labels: list, device: str = None, clip_model_name: str = "ViT-B/32"):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model, self.preprocess = clip.load(clip_model_name, device=self.device)
        logger.info(f"Loaded CLIP model '{clip_model_name}' on device '{self.device}'")
        
        self.labels = labels 
        self.text_inputs = clip.tokenize(self.labels).to(self.device)

    def classify_image(self, image: Image.Image) -> dict:
        """Classify a single image using CLIP zero-shot classification."""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits_per_image, _ = self.model(image_input, self.text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        result = dict(zip(self.labels, probs))
        return result
