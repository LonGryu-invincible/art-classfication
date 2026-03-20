# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from typing import List
from core.model_loader import ArtivaultModel
from core.processor import ImageProcessor
import zipfile
import io
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import urllib.request

# Đường dẫn file trên server (giữ nguyên để code load model không bị lỗi)
MODEL_PATH = "models/best_model.pth"

# Link tải trực tiếp từ ID bạn vừa gửi
MODEL_URL = "https://drive.google.com/uc?export=download&id=1OS-FTl2VPkivyMowiT2AY1ORUI2xe98v"

def download_model():
    # Tạo thư mục models nếu chưa có
    if not os.path.exists("models"):
        os.makedirs("models")
        
    # Nếu chưa có file model thì mới tiến hành tải
    if not os.path.exists(MODEL_PATH):
        print("--- Đang tải model từ Google Drive (khoảng 109MB)... ---")
        try:
            # Tải file và lưu vào đúng đường dẫn MODEL_PATH
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("--- Tải model thành công! ---")
        except Exception as e:
            print(f"--- Lỗi khi tải model: {e} ---")

# Gọi hàm này trước khi thực hiện lệnh load model (ví dụ: torch.load)
download_model()
app = FastAPI()

# --- CẤU HÌNH THƯ MỤC STATIC & TEMPLATES ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- CẤU HÌNH GIỚI HẠN ---
MAX_FILES_ALLOWED = 20 # Đồng bộ với giới hạn ở JavaScript

# Danh sách nhãn chuẩn (Style)
ART_CLASSES = [
    "Abstract_Expressionism", "Art_Nouveau_Modern", "Baroque",
    "Color_Field_Painting", "Cubism", "Early_Renaissance",
    "Expressionism", "High_Renaissance", "Impressionism",
    "Mannerism_Late_Renaissance", "Minimalism", "Naive_Art_Primitivism",
    "Northern_Renaissance", "Pop_Art", "Realism", "Rococo",
    "Romanticism", "Symbolism", "Ukiyo_e"
]

# Khởi tạo Model và Processor
processor = ImageProcessor()
ai_vault = ArtivaultModel("models/best_model.pth", ART_CLASSES)

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    # 1. Bảo vệ Server: Chặn nếu vượt quá 20 ảnh
    if len(files) > MAX_FILES_ALLOWED:
        raise HTTPException(
            status_code=400, 
            detail=f"Quá giới hạn! Hệ thống chỉ xử lý tối đa {MAX_FILES_ALLOWED} ảnh mỗi lần."
        )

    # 2. Thu thập dữ liệu bytes
    list_bytes = []
    filenames = []
    for f in files:
        content = await f.read()
        list_bytes.append(content)
        filenames.append(f.filename)

    # 3. Xử lý AI
    try:
        # Tiền xử lý (Convert to Tensor)
        batch_tensor = processor.process_batch(list_bytes)
        
        # Dự đoán
        predictions, total_time = ai_vault.predict_batch(batch_tensor)

        # 4. Gộp kết quả với tên file
        final_output = []
        for i in range(len(filenames)):
            final_output.append({
                "filename": filenames[i],
                **predictions[i]
            })

        return {
            "total_images": len(files),
            "total_latency_ms": f"{total_time:.2f}",
            "results": final_output
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Lỗi trong quá trình xử lý AI.")

@app.post("/export-sorted-zip")
async def export_sorted_zip(files: List[UploadFile] = File(...)):
    # 1. Giới hạn tương tự để tránh lỗi Timeout hoặc tràn RAM
    if len(files) > 100: # Cho phép export nhiều hơn predict một chút nhưng vẫn cần giới hạn
         raise HTTPException(status_code=400, detail="Quá nhiều ảnh để đóng gói ZIP!")

    list_bytes = []
    for f in files:
        list_bytes.append(await f.read())

    # 2. Chạy lại dự đoán để lấy style phân loại vào folder
    batch_tensor = processor.process_batch(list_bytes)
    predictions, _ = ai_vault.predict_batch(batch_tensor)

    # 3. Tạo file ZIP trong bộ nhớ RAM (Memory)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for i, file in enumerate(files):
            style_name = predictions[i]['style']
            filename = file.filename
            
            # Cấu trúc: Tên_Trường_Phái / Tên_Ảnh.jpg
            zip_path = f"{style_name}/{filename}"
            zip_file.writestr(zip_path, list_bytes[i])

    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer, 
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": "attachment; filename=Artivault_Sorted_Archive.zip"}
    )