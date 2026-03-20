import torch
import torch.nn as nn
import torchvision.models as models
import time

class ArtivaultModel:
    def __init__(self, model_path, class_names):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        
        # 1. Khởi tạo bộ khung EfficientNet-B0
        self.model = models.efficientnet_b0(weights=None)
        
        # 2. Sửa lại lớp Classifier để khớp với số lượng nhãn của bạn (19 nhãn)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
        
        # 3. Load trọng số từ file .pth
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, batch_tensor):
        start_time = time.time()
        batch_tensor = batch_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, indices = torch.max(probabilities, dim=1)
            
        total_time = (time.time() - start_time) * 1000
        
        results = []
        for i in range(len(indices)):
            results.append({
                "style": self.class_names[indices[i].item()],
                "confidence": f"{confidences[i].item() * 100:.2f}%"
            })
            
        return results, total_time