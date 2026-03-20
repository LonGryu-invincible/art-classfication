import torch
import timm
import torch.nn as nn

class ArtivaultModel:
    def __init__(self, model_path, class_names):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        
        # 1. Khởi tạo model bằng TIMM (phải khớp với lúc train)
        self.model = timm.create_model(
            'efficientnet_b0', 
            pretrained=False, 
            num_classes=len(class_names)
        )
        
        # 2. Load trọng số
        self.load_weights(model_path)
        self.model.to(self.device)
        self.model.eval()

    def load_weights(self, path):
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"--- SUCCESS: Weights loaded via TIMM from {path} ---")
        except Exception as e:
            print(f"--- ERROR: Cannot load weights: {e} ---")

    def predict_batch(self, batch_tensor):
        import time
        start_time = time.time()
        
        batch_tensor = batch_tensor.to(self.device)
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            # Chuyển đổi output thành xác suất (0 -> 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            results = []
            for probs in probabilities:
                conf, idx = torch.max(probs, dim=0)
                
                # --- SỬA TẠI ĐÂY: Nhân 100 để ra đơn vị % ---
                confidence_percent = float(conf.item()) * 100
                
                results.append({
                    "style": self.class_names[idx.item()],
                    "confidence": round(confidence_percent, 2) # Làm tròn 2 chữ số thập phân
                })
        
        total_time = (time.time() - start_time) * 1000
        return results, total_time