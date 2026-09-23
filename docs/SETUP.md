# Thiết lập M0 và M1.1 trên Windows / PowerShell

## 1. Kiểm tra công cụ

Mở PowerShell và chạy:

```powershell
Set-Location D:\patient-monitoring-pi4
python --version
python -c "import sys; print(sys.executable)"
git --version
code --version
code --list-extensions
```

Python phải >=3.10. Môi trường đã dùng ở M0: Python 3.10.11 (64 bit),
Git 2.47.1.windows.1, VS Code 1.137.0. Nếu thiếu Python hoặc Git, cài từ
python.org hoặc git-scm.com, mở lại PowerShell rồi chạy lại các lệnh trên.
Nếu thiếu VS Code, cài từ code.visualstudio.com. Không thay Python hệ thống
đang hoạt động để chạy dự án. Có thể dùng `py -0p` để liệt kê interpreter
và gọi trực tiếp một Python >=3.10 khi `python` không trỏ đúng phiên bản.

## 2. Tạo môi trường riêng và cài package

Chỉ chạy lệnh tạo môi trường nếu workspace chưa có `.venv`:

```powershell
python -m venv .venv
```

Cài package ở chế độ editable để chạy mã trong `src/`:

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -e .
```

Không cần activate hay đổi ExecutionPolicy; luôn gọi Python trong `.venv`.
M1.1 thêm `av==16.1.0` từ wheel; pip có thể cần mạng để tải PyAV, setuptools
và công cụ build. Nếu cài lỗi, lưu thông báo lỗi, khắc phục kết nối/quyền
hoặc thành phần bị thiếu rồi chạy lại đúng lệnh; chưa coi M0 là thành công.
Không biên dịch dependency từ source. Không cài ML hoặc tải model/dữ liệu trong M0/M1.1.

## 3. Chọn interpreter trong VS Code

Mở **File > Open Folder** và chọn `D:\patient-monitoring-pi4`.
Trong Extensions (`Ctrl+Shift+X`), kiểm tra/cài các extension:

- Python của Microsoft: `ms-python.python`.
- Pylance của Microsoft: `ms-python.vscode-pylance`.
- Codex của OpenAI: `openai.chatgpt`.

Nhấn `Ctrl+Shift+P`, chọn **Python: Select Interpreter** > **Enter interpreter
path**, chọn `D:\patient-monitoring-pi4\.venv\Scripts\python.exe`.
Mở terminal PowerShell tại thư mục dự án và chạy các lệnh kiểm tra bên dưới.
Việc extension xuất hiện trong danh sách chỉ xác nhận đã cài;
khả năng đăng nhập/hoạt động trong giao diện cần kiểm tra trong VS Code.

## 4. Kiểm tra khói

```powershell
.\.venv\Scripts\python.exe -m patient_monitoring --help
.\.venv\Scripts\python.exe -m patient_monitoring doctor
.\.venv\Scripts\python.exe -m pip check
```

`--help` phải hiển thị lệnh `doctor`. `doctor` phải báo Python của `.venv`,
`Virtualenv active: True` và `Project root: D:\patient-monitoring-pi4`.
OS và phiên bản Python là giá trị thực tế của máy đang chạy.
Không có thông tin camera/GPU/Pi vì M0 không kiểm tra những thiết bị đó.

## 5. Git và bước sau

M0 kiểm tra `git rev-parse --show-toplevel` trước khi khởi tạo; chỉ chạy
`git init` nếu không có repository tại đây hoặc thư mục cha.
Repository hiện tại đã được khởi tạo, chưa commit hoặc push.
Xem thay đổi bằng `git status --short`; `.venv`, video, kết quả sinh ra,
trọng số và secrets được loại khỏi Git qua `.gitignore`.

Khi M0 đã xác minh, cung cấp một video mới cho nhiệm vụ M1: đọc metadata
và xác minh timeline. M0 chưa thực hiện phần đó.

Nếu Git báo `detected dubious ownership` (repository được tạo bởi tài khoản
sandbox khác tài khoản Windows), dùng ngoại lệ cho đúng workspace trong
mỗi lệnh, không cần thay đổi cấu hình Git toàn cục:

```powershell
git -c safe.directory=D:/patient-monitoring-pi4 status --short
```

Khi kiểm tra M0, công cụ sandbox từng lỗi
`helper_unknown_error: setup refresh had errors`; các lệnh cài/kiểm tra sau đó
đã chạy thành công ngoài sandbox. VS Code CLI có cảnh báo quyền ghi log,
nhưng vẫn trả phiên bản và danh sách extension; giao diện chưa được kiểm tra.

Hướng dẫn M1.1: [inspect-video](INSPECT_VIDEO.md).

M1.2 thêm Pillow 12.1.1 từ wheel trong .venv; xem [preview-video](PREVIEW_VIDEO.md).
