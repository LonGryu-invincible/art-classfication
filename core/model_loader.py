import torch
import torch.nn as nn
from torchvision import models
import time

class ArtivaultModel:
    def __init__(self, model_path, class_names):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        
        # 1. Khởi tạo đúng kiến trúc EfficientNet-B0
        self.model = models.efficientnet_b0(weights=None)
        
        # 2. Chỉnh sửa lớp cuối (Classifier) - B0 dùng 1280 features
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, len(class_names))
        )
        
        # 3. Nạp trọng số từ file .pth
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Xử lý trường hợp file .pth chứa cả thông tin train (epoch, optimizer...)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
                
            self.model.load_state_dict(state_dict)
            print(f"--- Đã nạp thành công model B0 từ {model_path} ---")
        except Exception as e:
            print(f"--- Lỗi nạp model: {e} ---")
            
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, batch_tensor):
        start_time = time.time()
        batch_tensor = batch_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Lấy top 1 kết quả
            conf, classes = torch.max(probabilities, dim=1)
            
        results = []
        for i in range(len(classes)):
            # CHỈNH SỬA TẠI ĐÂY: Trả về số để Frontend xử lý mượt hơn
            conf_percent = conf[i].item() * 100
            results.append({
                "style": self.class_names[classes[i].item()],
                "confidence": f"{conf_percent:.2f}" # Trả về "95.50" thay vì "95.50%"
            })
            
        latency = (time.time() - start_time) * 1000
        return results, latency