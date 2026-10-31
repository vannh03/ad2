# HƯỚNG DẪN 3 BƯỚC ĐO FPS & LATENCY TRÊN JETSON NANO
> **Dành riêng cho người mới bắt đầu:** Mọi cấu hình, chọn model, chọn GPU, tính toán độ trễ đã được tự động hóa 100%.

---

## 🎯 BƯỚC 1: Lấy Code về Jetson Nano
Trên màn hình Jetson Nano (kết nối qua RealVNC), mở **Terminal** và chạy:
```bash
# Cách 1: Sử dụng Git (Khuyên dùng)
cd /media/tx/disk    # Hoặc cd /home/jetson
git clone https://github.com/vannh03/ad2.git
cd ad2

# (Nếu đã clone rồi thì chỉ cần cập nhật code mới nhất):
# git pull origin main
```
*(Hoặc Cách 2: Nếu copy folder/zip từ máy tính sang qua RealVNC, mở Terminal và `cd` vào thư mục chứa code).*

---

## ⚙️ BƯỚC 2: Cài đặt Tự động (Chỉ chạy 1 lần đầu)
Trong thư mục code trên Jetson, chạy lệnh:
```bash
# Nếu máy báo đầy ổ đĩa (No space left), chạy script dọn dẹp trước:
bash clean_disk.sh

# Chạy cài đặt tự động môi trường:
bash setup_jetson.sh
```
*Script sẽ tự động: dọn dẹp bộ nhớ đệm, cấu hình ZRAM, cài đặt các thư viện hệ thống qua apt, kiểm tra mô hình và tối ưu hóa cho Jetson Nano.*

---

## 🚀 BƯỚC 3: Chạy và Lấy kết quả FPS & Latency
Khi cần đo hiệu năng, bạn chỉ cần gõ đúng 1 dòng:
```bash
python3 run.py
```
*(hoặc: `bash run_jetson.sh`)*

### 📊 Mẫu kết quả hiển thị trên màn hình:
```text
=================================================================
🚗 HỆ THỐNG CẢNH BÁO VA CHẠM THÔNG MINH (ADAS) - JETSON NANO
=================================================================
 • Thiết bị tính toán : NVIDIA Tegra X1 (device='0')
 • Mô hình YOLO       : models/detection/best.pt
 • Video kiểm thử     : wrong_way_NO20250603-083305-009849F_annotated.mp4
 • Kích thước ảnh     : 320x320 pixels
 • Số frame kiểm tra  : 100 frames
=================================================================
[CHUAN_XAC] Frame 100/100 | FPS:  24.8 | Latency:  40.3ms (YOLO: 37.1ms, Rule:  3.2ms)

=================================================================
📊 BẢNG TỔNG HỢP KẾT QUẢ ĐO ĐẠC TRÊN JETSON NANO
=================================================================
 • Tốc độ trung bình (Realtime FPS) : 24.85 FPS
 • Độ trễ xử lý trung bình (Latency): 40.24 ms
   ├─ Thời gian YOLO Detect & Track  : 37.41 ms
   └─ Thời gian Suy luận Luật + MLP : 2.83 ms
 • Độ trễ AI nhỏ nhất (Min Latency) : 36.10 ms
 • Độ trễ AI lớn nhất (Max Latency) : 51.20 ms
 • Độ trễ phân vị 95% (P95 Latency) : 44.30 ms
 • Độ trễ toàn trình (End-to-End)   : 43.15 ms
=================================================================
🎯 ĐÁNH GIÁ TIÊU CHUẨN THỜI GIAN THỰC:
  ✅ ĐẠT CHUẨN REALTIME TRÊN JETSON NANO (FPS >= 20, Latency < 50ms)
=================================================================
```
Bạn chỉ việc sao chép bảng số liệu này vào báo cáo đồ án hoặc slide thuyết trình!
