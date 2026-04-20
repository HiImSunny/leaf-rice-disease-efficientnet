# leaf-rice-disease-efficientnet

Giải pháp phân loại bệnh lúa (8 class) sử dụng EfficientNet-B3 + TTA — FPTU Cần Thơ Olympic AI 2026.

---

## Reproduce Instructions

### Yêu cầu môi trường

```bash
pip install timm torch torchvision tqdm scikit-learn pandas pillow numpy matplotlib
```

- Python ≥ 3.9, PyTorch ≥ 2.0
- Khuyến nghị có GPU (CUDA); chạy CPU sẽ rất chậm với `IMG_SIZE=336`
- Kaggle Notebook: bật **Internet on** ở Settings (để tải pretrained weights lần đầu)

---

### Cấu trúc thư mục dữ liệu

```
kaggle/input/competitions/fptu-can-tho-olympic-ai-2026/
├── label_mapping.json
├── train/
│   ├── <class_1>/
│   │   ├── img001.jpg
│   │   └── ...
│   └── <class_2>/
│       └── ...
└── test_speedup/
    ├── img_001.jpg
    └── ...
```

---

### Thứ tự chạy

| Bước | Cell | Mô tả | Output |
|------|------|--------|--------|
| 1 | **EDA** | Phân tích dataset, vẽ biểu đồ, hiển thị ảnh mẫu | `eda_class_distribution.png`, `eda_sample_images.png` |
| 2 | **Training** | Train EfficientNet-B3, early stopping, lưu best model | `kaggle/working/best_model.pth` |
| 3 | **TTA Inference** | Load best model, predict với 8 TTA transforms | `kaggle/working/submission.csv` |

> Cell **EDA** và **Training** phải chạy theo thứ tự.  
> Cell **TTA Inference** có thể chạy **độc lập** mà không cần chạy lại train, miễn là file `kaggle/working/best_model.pth` đã tồn tại.

---

### Các tham số cố định — không thay đổi để tái lập kết quả

```python
IMG_SIZE     = 336
BATCH_SIZE   = 28
EPOCHS       = 50
LR           = 2.8e-4
NUM_CLASSES  = 8
MODEL_NAME   = "efficientnet_b3"
test_size    = 0.18   # tỉ lệ val
random_state = 42     # seed toàn bộ
```

---

### Lưu ý quan trọng

- **Seed:** `seed_everything(42)` phải được gọi trước khi train. `seed_worker` và `torch.Generator().manual_seed(42)` trong DataLoader đảm bảo augmentation và shuffle nhất quán giữa các lần chạy.
- **Windows:** Đổi `num_workers=2` → `num_workers=0` để tránh lỗi multiprocessing.
- **Pretrained weights:** Lần đầu chạy `timm.create_model(..., pretrained=True)` sẽ tự tải từ [https://huggingface.co/timm/efficientnet_b3.ra2_in1k](https://huggingface.co/timm/efficientnet_b3.ra2_in1k) và cache lại — các lần sau không cần internet.
- **TTA Inference độc lập:** Cell TTA dùng `pretrained=False` và load thẳng từ `best_model.pth`, không tải lại weights từ internet.
