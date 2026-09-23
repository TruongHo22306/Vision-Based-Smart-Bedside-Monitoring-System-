# Patient Activity Monitoring Using Raspberry Pi 4 and a USB Webcam

Dự án mới hoàn toàn. Phạm vi dự kiến: đứng, ngồi, nằm, di chuyển và té ngã
theo thời gian; video có nhãn trước, webcam trên Pi sau. Chuyển động nghi
co giật là mở rộng tùy điều kiện. Hiện có M0 kiểm tra môi trường, M1.1 kiểm tra timeline và M1.2 tạo preview.

Từ PowerShell, tại workspace này:

```powershell
Set-Location D:\patient-monitoring-pi4
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -e .
.\.venv\Scripts\python.exe -m patient_monitoring --help
.\.venv\Scripts\python.exe -m patient_monitoring doctor
```

Yêu cầu Python >=3.10. Chỉ tạo `.venv` khi chưa có; máy hiện tại đã tạo bằng
Python 3.10.11. `doctor` dùng thư viện chuẩn; `inspect-video` dùng PyAV `av==16.1.0`.
Cài package cần setuptools; chỉ dùng wheel cho dependency/công cụ build.

- [Cài đặt Windows/VS Code và xử lý lỗi](docs/SETUP.md)
- [Trạng thái và bằng chứng thực tế](docs/STATUS.md)
- [Phạm vi và kế hoạch gốc](docs/PROJECT_PLAN.md)
- Mã CLI: `src/patient_monitoring/__main__.py`, hàm `main()` và `doctor()`.

`doctor` chỉ báo Python, đường dẫn interpreter, OS, virtualenv và gốc dự án.
Chương trình chưa nhận dạng hành động hay kiểm tra camera/GPU/Pi.

Kiểm tra video mới (chỉ chạy khi file đã có):

```powershell
.\.venv\Scripts\python.exe -m patient_monitoring inspect-video data/raw/pilot01.mp4 --output-dir outputs/pilot01
```

Xem [cách dùng và ý nghĩa metadata/timeline](docs/INSPECT_VIDEO.md).
M1.1 đã xác minh pilot 415 frame. M1.2 đã tạo preview; chờ sinh viên xem ảnh.
M1 chưa hoàn thành.

Tạo preview trong thư mục mới (thư mục dưới đã có output pilot):

```powershell
.\.venv\Scripts\python.exe -m patient_monitoring preview-video data/raw/pilot01.mp4 --output-dir outputs/pilot01_preview --samples 8
```

[Hướng dẫn preview và thống kê khoảng frame](docs/PREVIEW_VIDEO.md).
M1.2 thêm Pillow 12.1.1 từ wheel trong `.venv` để ghi ảnh.
