import torch
import torch.nn.functional as F
import time

class ArtivaultModel:
    def __init__(self, model_path, class_names):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = class_names
        self.model = self._load_model(model_path, len(class_names))

    def _load_model(self, path, num_classes):
        import timm
        model = timm.create_model('efficientnet_b5', pretrained=False, num_classes=num_classes)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict_batch(self, batch_tensor):
        batch_tensor = batch_tensor.to(self.device)
        
        # Đồng bộ GPU để đo thời gian chính xác
        if self.device == "cuda": torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            outputs = self.model(batch_tensor)
        
        if self.device == "cuda": torch.cuda.synchronize()
        total_latency = (time.time() - start) * 1000

        # Tính toán cho toàn bộ Batch
        probs = F.softmax(outputs, dim=1)
        confidences, indices = torch.max(probs, dim=1)

        results = []
        for i in range(len(indices)):
            results.append({
                "style": self.class_names[indices[i].item()],
                "confidence": float(confidences[i].item()), # Trả về float để FE dễ xử lý
                "latency_per_img_ms": float(total_latency / len(indices))
            })
        return results, total_latency