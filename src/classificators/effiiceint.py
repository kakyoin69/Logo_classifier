import torch
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

class Efficientnet:
    def __init__(self, model_path: str, classes_file: str, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = self._load_classes(classes_file)
        self.number_class = len(self.class_names)
        
        self.model = self._load_model(model_path)
        
 
    def _load_model(self, model_path: str):
        """Loading the EfficientNet-B0 model with pre-trained weights."""
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model._fc = torch.nn.Linear(model._fc.in_features, self.number_class)
        
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        model = model.to(self.device)
        model.eval()
        return model

    def _load_classes(self, classes_file: str) -> list:
        """Loading the list of classes from a text file."""
        try:
            with open(classes_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
            return class_names
        except FileNotFoundError:
            raise FileNotFoundError(f"Classes file not found: {classes_file}")
        except Exception as e:
            raise ValueError(f"Error reading the classes file: {e}")

    def predict_image_class(self, image: Image.Image) -> str:
        """Performing prediction for a single image."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])
        
        image_tensor = transform(image).unsqueeze(0)  
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():  # Отключаем вычисление градиентов
            outputs = self.model(image_tensor)
            _, predicted_idx = torch.max(outputs, 1)  # Получаем индекс максимального значения
        
        predicted_class = self.class_names[predicted_idx.item()]
        return predicted_class