# Chuẩn bị chuyển pilot sang Raspberry Pi — chưa benchmark

## Thông tin do người dùng xác nhận

Pi 4 RAM 8GB; Debian 12 Bookworm, aarch64, Python 3.11.2; SSH dự kiến
`truong@patient-pi4.local`. Webcam icSpring `32e6:9211`, `/dev/video0`, đã xem bằng ffplay.
MJPEG 1280×720 và 640×360 yêu cầu 30 FPS nhưng đo khoảng 15 FPS; hai lần capture
150 frame đều exit 0. Đây là thông tin người dùng cung cấp, không phải Codex tự kiểm chứng
và **chưa phải benchmark YOLO**. Task này chỉ chuẩn bị trên Windows, chưa SSH/cài/chạy trên Pi.

## Dependency: giữ nguyên Windows, ARM64 chưa xác minh

`pyproject.toml`: Python >=3.10; av==16.1.0, Pillow==12.1.1, ultralytics==8.4.152,
torch==2.9.1, torchvision==0.24.1; build backend setuptools>=64.
Python 3.11.2 đáp ứng ràng buộc khai báo, chưa chứng minh cả môi trường chạy được.

| Bằng chứng Windows | Việc cần kiểm tra trên Pi |
|---|---|
| Torch/Torchvision/PyAV/Pillow đang cài đều có wheel tag `cp310-cp310-win_amd64` | Không chuyển `.venv` hoặc các wheel này; cần Linux aarch64/CPython 3.11 tương ứng |
| Torch 2.9.1+cpu, Torchvision 0.24.1+cpu đang cài | Pyproject không ghim `+cpu`; không dùng freeze Windows làm lock ARM64; build/version hậu tố trên Pi có thể khác |
| `docs/POSE_PREVIEW.md` dùng `https://download.pytorch.org/whl/cpu` | Đây là CPU index, không phải bằng chứng mọi version/kiến trúc đều có wheel; không mặc định Pi có build Windows |
| Pyproject không có URL wheel x86 hoặc platform marker Windows | Pin vẫn có thể thiếu wheel ARM64 hoặc vướng dependency bắc cầu; phải resolver trên Pi |
| NumPy/OpenCV và dependency bắc cầu chưa khóa đầy đủ trong pyproject | Có thể giải ra phiên bản khác Windows; giữ report và phiên bản thực tế |
| Tài liệu cũ dùng `.venv\Scripts\python.exe`, đường dẫn `D:\...` | Pi dùng `.venv/bin/python`, đường dẫn `$HOME`; outputs Windows không nằm trong gói |

Không sửa pin Windows. Dry-run đầu dùng PyPI chính thức và pin hiện có; nếu thất bại,
dừng giữ log, không tự nới pin/đổi Python hệ thống/bỏ only-binary/build Torch từ source.
Nguồn: [pip install](https://pip.pypa.io/en/stable/cli/pip_install/) và
[PyTorch/Torchvision 2.9.1/0.24.1](https://pytorch.org/get-started/previous-versions/#v291).

## Gói và SHA-256

Thư mục ngoài source: `D:\patient-monitoring-transfer\pilot01-20260920\`.
Tên gói: `patient-monitoring-pi4-pilot01-20260920.tar.gz`.
Chỉ có src/tests `.py`, pyproject, AGENTS, README, .gitignore, tài liệu quy trình,
script đóng gói, `data/raw/pilot01.mp4` và `models/yolo11n-pose.pt`.
Không có .venv/.git/outputs cũ/cache/secrets/weights khác. Các outputs Windows
được STATUS nhắc tới chỉ là lịch sử, không có trên Pi sau giải nén.
Weights giữ đúng M2, SHA-256:
`869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`.

`*.files.sha256` liệt kê từng payload file, cũng được nhúng dưới tên `SHA256SUMS`.
Manifest không tự hash chính nó; `*.tar.gz.sha256` kiểm tra toàn archive gồm manifest.
`bundle_verification.json` ghi kiểm tra thực tế. Muốn tạo lại, chọn thư mục output mới:

```powershell
Set-Location D:\patient-monitoring-pi4
.\.venv\Scripts\python.exe tools\build_pi_bundle.py --output-dir D:\patient-monitoring-transfer\pilot01-20260920
```

## Windows PowerShell — người dùng chuyển gói bằng scp

```powershell
Set-Location D:\patient-monitoring-transfer\pilot01-20260920
$piBundle = "patient-monitoring-pi4-pilot01-20260920"
$expectedHash = ((Get-Content "$piBundle.tar.gz.sha256" -Raw) -split '\s+')[0]
if ((Get-FileHash "$piBundle.tar.gz" -Algorithm SHA256).Hash -ne $expectedHash) { throw "Archive hash mismatch" }
scp "$piBundle.tar.gz" "$piBundle.tar.gz.sha256" "$piBundle.files.sha256" "truong@patient-pi4.local:~/"
if ($LASTEXITCODE -ne 0) { throw "scp failed" }
```

Hostname/SSH chưa được kiểm chứng. Nếu mDNS không phân giải, dùng IP Pi do bạn xác nhận.
Nếu home Pi đã có gói cùng tên cần giữ, chọn tên/vị trí nhận mới trước khi scp.

## Terminal trên Pi — kiểm tra, giải nén, tạo .venv

Chạy trong terminal đã đăng nhập Pi. Khối sau từ chối mọi thư mục đích đã tồn tại
(kể cả symlink/thư mục rỗng), không ghi đè dữ liệu:

```bash
(
  set -eu
  cd "$HOME"
  bundle="patient-monitoring-pi4-pilot01-20260920"
  sha256sum -c "$bundle.tar.gz.sha256"
  target="$HOME/patient-monitoring-pi4"
  if [ -e "$target" ] || [ -L "$target" ]; then
    printf '%s\n' "Target exists; stop and choose a new destination."
    exit 1
  fi
  mkdir "$target"
  tar --extract --gzip --file="$HOME/$bundle.tar.gz" --directory="$target" --keep-old-files --no-same-owner
  cd "$target"
  cmp SHA256SUMS "$HOME/$bundle.files.sha256"
  sha256sum -c SHA256SUMS
  uname -m
  python3 --version
  python3 -m venv .venv
  .venv/bin/python -m pip --version
)
```

Nếu venv/ensurepip thiếu, dừng và lưu lỗi; không thay Python hệ thống.
Nếu giải nén/checksum/venv lỗi, giữ thư mục để kiểm tra, không chạy đè lên nó.

## Terminal trên Pi — dry-run trước khi cài dependency

Lệnh có thể tải metadata/wheel và chuẩn bị metadata cho project local nhưng không cài
bộ dependency runtime vào .venv. `--only-binary=:all:` buộc dependency từ index dùng wheel;
project Python local vẫn dùng setuptools để chuẩn bị metadata/editable package.

```bash
(
  set -eu
  cd "$HOME/patient-monitoring-pi4"
  mkdir -p reports
  .venv/bin/python -m pip --version
  .venv/bin/python -m pip --isolated --log reports/pi-dependency-dry-run.log install \
    --dry-run --ignore-installed --only-binary=:all: \
    --index-url https://pypi.org/simple \
    --report reports/pi-dependency-dry-run.json -e .
)
```

Resolver đạt khi exit 0 và report đủ pin, wheel Linux aarch64/CPython 3.11 phù hợp
hoặc Python dùng chung, không phải win_amd64/x86_64. Nếu lỗi hoặc pip chưa hỗ trợ tùy chọn,
giữ version/log để xử lý riêng; chưa cài thật, không tự đổi Python. Dry-run thành công
cũng chưa chứng minh import/runtime/hiệu năng. Hiện chưa khẳng định tương thích ARM64.

## Chỉ sau khi xem dry-run thành công — chưa thực hiện trong task này

Nếu quyết định cài thật, dùng cùng Python/index/pin và chỉ wheel dependency:

```bash
(
  set -eu
  cd "$HOME/patient-monitoring-pi4"
  .venv/bin/python -m pip --isolated install --only-binary=:all: \
    --index-url https://pypi.org/simple --report reports/pi-install.json -e .
  .venv/bin/python -m pip check
  .venv/bin/python -m patient_monitoring doctor
  .venv/bin/python -m pip freeze > reports/pi-installed-versions.txt
)
```

Dependency bắc cầu có thể thay đổi giữa dry-run và cài: so hai report, ghi phiên bản thực tế.
Không có lệnh inference/benchmark ở đây. M2 vẫn IN_PROGRESS.
