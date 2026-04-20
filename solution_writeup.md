# BÁO CÁO GIẢI PHÁP
**FPTU Can Tho Olympic AI 2026 – Rice Leaf Disease Classification**

---

## 1. Tổng quan giải pháp

Mục tiêu của bài toán là xây dựng mô hình phân loại 8 loại bệnh trên lá lúa, với yêu cầu đặc biệt về khả năng chịu đựng nhiễu cực mạnh trên tập test — bao gồm scanline, moiré pattern, chromatic aberration, color glitch, grid noise, blur và low-light.

Giải pháp được xây dựng xoay quanh bốn trụ cột chính:
- Backbone EfficientNet-B3 với kích thước đầu vào 336×336
- Strong Noise Augmentation kết hợp Noisy Validation
- Blend Score làm tiêu chí chọn checkpoint
- Test-Time Augmentation (TTA) có kiểm soát

---

## 2. Kiến trúc mạng

```python
model = timm.create_model("efficientnet_b3", 
                          pretrained=True, 
                          num_classes=8,
                          drop_rate=0.25, 
                          drop_path_rate=0.15)
```

EfficientNet-B3 được lựa chọn vì ba lý do chính. Thứ nhất, kiến trúc compound scaling cho phép mô hình nắm bắt đồng thời các chi tiết cục bộ (texture của vết bệnh) lẫn ngữ cảnh toàn bộ lá. Thứ hai, mô hình đạt cân bằng tốt giữa độ chính xác và tốc độ huấn luyện — điều quan trọng trong bối cảnh thi đấu có giới hạn thời gian. Thứ ba, drop rate và drop path rate cao giúp kiểm soát overfitting khi áp dụng augmentation nhiễu mạnh.

---

## 3. Chiến lược chống nhiễu (Noise Robustness)

Đây là thành phần cốt lõi và quan trọng nhất của giải pháp.

### Noise Augmentation trong Training

Hệ thống augmentation được thiết kế để mô phỏng gần như đầy đủ các dạng nhiễu xuất hiện trong tập test:

- **Scanline & grid/moiré:** `add_horizontal_stripe`, `add_grid_noise`, `add_moire_grid`
- **Lỗi quang học & lệch màu:** `add_chromatic_aberration`, `add_color_glitch`
- **Suy giảm hình ảnh:** `add_heavy_blur`, `add_extreme_low_light`
- **Hiệu ứng ánh sáng:** `add_vignette`, `add_darkening`

Mỗi loại nhiễu được áp dụng độc lập qua `RandomApply` với xác suất riêng (0.18–0.48), tránh xếp chồng nhiều transform cùng lúc để không gây over-augmentation.

### Noisy Validation & Blend Score

Điểm khác biệt nổi bật so với cách tiếp cận thông thường là việc xây dựng một tập validation nhiễu riêng biệt (`noisy_val_transform`) với mức nhiễu nhẹ hơn train. Checkpoint tốt nhất được chọn dựa trên:

$$\text{Blend Score} = 0.75 \times \text{Val F1} + 0.25 \times \text{Noisy Val F1}$$

Cơ chế này buộc mô hình phải duy trì hiệu suất tốt trên cả ảnh sạch lẫn ảnh nhiễu, thay vì chỉ tối ưu hóa trên validation set lý tưởng.

### Các kỹ thuật regularization bổ sung

- Label Smoothing = 0.09
- WeightedRandomSampler để xử lý mất cân bằng class
- OneCycleLR scheduler với learning rate cosine annealing
- Dropout và DropPath trong backbone

---

## 4. Test-Time Augmentation (TTA)

TTA được thực hiện với 8 biến thể, chia thành hai nhóm:
- **5 biến thể geometric:** Resize, Horizontal Flip, Vertical Flip, Multi-scale CenterCrop
- **3 biến thể nhiễu nhẹ:** Chromatic aberration, Horizontal stripe, Color glitch

Việc kết hợp hai nhóm này giúp tăng tính ổn định dự đoán trên ảnh nhiễu mà không gây ra phương sai quá cao trong xác suất đầu ra.

---

## 5. Pipeline huấn luyện

- **Input size:** 336×336
- **Optimizer:** AdamW, weight decay = 1e-4
- **Loss:** CrossEntropyLoss với label smoothing
- **Early stopping:** Dựa trên Blend Score, patience = 12
- **Checkpoint:** Lưu theo Blend Score cao nhất, thường rơi vào epoch 39–44

---

## 6. Kết quả

| Chỉ số | Giá trị |
|--------|---------|
| Best Blend Score (validation) | 0.9330 (epoch 45) |
| Best Public Score | 0.66376 |
| Submission nộp cuối cùng | 0.66376 |

---

## 7. Hạn chế và bài học rút ra

**Hạn chế chính:**
- Mô hình còn yếu ở một số class, đặc biệt là class 2 và class 5.
- Chưa mô phỏng được hiện tượng *inset* (picture-in-picture) — một dạng nhiễu phổ biến trong tập test nhưng khó tái tạo bằng augmentation thông thường.
- Chưa thử nghiệm curriculum learning (tăng cường độ nhiễu dần theo epoch) hay Mixup/CutMix.

**Bài học quan trọng:**
- Trong bài toán có nhiễu mạnh, **Noisy Validation + Blend Score** đóng vai trò quyết định hơn việc chỉ tối ưu Val F1 trên ảnh sạch.
- TTA đòi hỏi điều chỉnh tinh tế: quá nhẹ thì không đủ sức chịu nhiễu test, quá mạnh thì gây phương sai cao.
- Augmentation cần bao quát đúng các *artifact* đặc thù của bộ dữ liệu thay vì dùng noise tổng quát.

---

## 8. Định hướng cải tiến

- Thêm augmentation **Random Paste Small Crop** để mô phỏng hiện tượng inset.
- Áp dụng **Curriculum Noise** — tăng cường độ nhiễu tuyến tính theo epoch.
- **Ensemble** 2–3 checkpoint tốt nhất theo Blend Score.
- Nâng backbone lên **EfficientNet-B4/B5** hoặc **ConvNeXt** với input size 384.
- Thử nghiệm **Adaptive TTA** hoặc Test-Time Normalization để cải thiện độ ổn định.
