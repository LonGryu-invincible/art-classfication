import torch
from torchvision import transforms
from PIL import Image
import io

class ImageProcessor:
    def __init__(self):
        # EfficientNet-B0 chuẩn yêu cầu 224x224
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def process_batch(self, list_bytes):
        tensors = []
        for img_bytes in list_bytes:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            tensors.append(self.transform(image))
        
        return torch.stack(tensors)