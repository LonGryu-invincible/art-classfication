from torchvision import transforms
from PIL import Image
import io
import torch

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((456, 456)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_batch(self, list_of_image_bytes):
        tensors = []
        for img_bytes in list_of_image_bytes:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            tensors.append(self.transform(img))
        
        # Chồng các ảnh lên nhau tạo thành Batch: [N, 3, 456, 456]
        return torch.stack(tensors)