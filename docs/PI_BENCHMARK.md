# Benchmark pose ngắn trên Pi 4

## Bằng chứng người dùng cung cấp, chưa phải benchmark có kiểm soát

Pi 4 8GB, Bookworm aarch64, Python 3.11.2. Người dùng báo đã chạy YOLO11n-pose
CPU/float32, imgsz=640, batch=1, rect=False: pilot01 hoàn thành 415/415 frame,
421 detection, 6 frame nhiều detection, 0 zero detection; không thiếu/trùng/giảm timestamp.
Khoảng thời gian từ bắt đầu đến kết thúc là **677.096496 giây**, không phải inference-only
hoặc benchmark có kiểm soát. Nhiệt độ trước/sau 66.7/74°C; get_throttled đều 0xd0000.
Nguồn điện và tản nhiệt chưa xác nhận. Đây là thông tin trong trao đổi, không phải Codex tự đo Pi.

Theo [tài liệu Raspberry Pi](https://www.raspberrypi.com/documentation/computers/os.html#get_throttled),
bit 0/1/2/3 là undervoltage/frequency cap/throttled/soft temperature limit hiện tại;
16/17/18/19 là các cờ lịch sử tương ứng. 0xd0000 có bit lịch sử 16,18,19,
không có bit hiện tại 0–3 tại mẫu đó. Không xác định lúc nào xảy ra, nguyên nhân cụ thể,
hoặc suy ra không throttling giữa hai mẫu. Không reset các cờ trong benchmark.

## Hợp đồng đo

`benchmark-pose PATH --weights LOCAL.pt --imgsz 640 --warmup 3 --frames 30 --output-dir NEW_DIR`

- Nhận weights .pt local hoặc thư mục `*_ncnn_model` local đã export; không tải model/cài package tự động. Hash video và artifacts được ghi.
- CPU, float32, batch=1, rect=False; giữ M2.2 conf=0.25, IoU=0.7, max_det=300,
  augment=False, save=False. Không đổi thread counts, compile/export/fuse thủ công hoặc thêm tối ưu.
  Với NCNN, float32 là dtype đầu vào; precision nội bộ và thread NCNN được ghi riêng, không suy từ Torch. Xem [NCNN trial](PI_NCNN_TRIAL.md).
- imgsz phải dương và chia hết 32; frames>0, warmup>=0. Output phải chưa tồn tại.
- Decode tuần tự stream video đầu bằng PyAV, chuẩn bị đúng **N frame đầu** thành RGB trong RAM.
  Mặc định frame 0–29, giữ nguyên index, PTS, time_base và thời gian M1.1; không seek theo FPS.
  Thiếu N frame hoặc frame decode hỏng: incomplete, không báo đo đủ.
- Warmup chạy riêng, lần lượt dùng frame đã chuẩn bị (quay vòng nếu warmup>N).
  Frames đo luôn 0..N-1, không phụ thuộc imgsz hoặc số warmup. Đổi kích thước phải giữ cùng video/hash/N.
  Warmup=0 có thể tính cả khởi tạo predictor vào lần đo đầu; không nên dùng để so với cấu hình warmup=3.
- RAM chứa N ảnh RGB; mặc định 30×1280×720 xấp xỉ 83 MB dữ liệu pixel, chưa gồm model/overhead.
  Tập đầu video không đại diện toàn bộ tư thế/cảnh; benchmark ngắn không đo ổn định nhiệt dài hạn.

## Thời gian và file đầu ra

Tất cả thời gian wall-clock dùng `perf_counter`:

- `input_hashing_seconds`: hash đầu vào lần đầu.
- `data_preparation_seconds`: mở/decode N frame/chuyển RGB, trước khi nạp model.
- `model_loading_seconds`: import thư viện model, load weights, đưa CPU/float32;
  predictor lazy setup có thể nằm trong predict warmup đầu tiên.
  NCNN nạp Net trong giai đoạn model loading; không đưa chi phí nạp vào frame đo đầu tiên.
- `predict_seconds`: bao đúng mỗi lần `model.predict`, gồm preprocess/inference/postprocess
  và overhead của lời gọi; không gồm telemetry, ghi JSON, decode hoặc warmup khác.
- `total_benchmark_elapsed_seconds`: từ vào hàm benchmark sau import CLI/module đến sau cleanup/statistics;
  gồm hash, prep/load, warmup, telemetry, ghi giữa chừng; không gồm lần serialize summary cuối.

`frame_timings.jsonl` chứa warmup và measured có trường `phase`; mỗi dòng có index/PTS,
predict_seconds, số detection, `ultralytics_speed_ms` preprocess/inference/postprocess do thư viện báo.
Không đánh đồng các số thư viện báo với toàn bộ thời gian lời gọi Python.

`benchmark_summary.json` ghi status, cấu hình yêu cầu/thực tế, SHA-256, package versions,
device/dtype, Torch intraop/interop trước load, sau load/warmup/đo; selected frame metadata và thống kê.
Benchmark không gọi setters thread. Nếu thư viện tự thay đổi, các snapshot phản ánh điều đó.
`half=False` kế thừa M2.2; Ultralytics hiện tại cảnh báo deprecated và không trả lại key này,
nên kiểm tra dtype/backend thực tế. Không tự đổi sang tham số tối ưu khác.

Thống kê chỉ gồm các lần **measured thành công**: median và p95 nội suy tuyến tính
trên dãy tăng dần tại vị trí `(n-1)*q`, q=0.5/0.95, không làm tròn trước tổng hợp.
Throughput = số lần measured thành công / tổng predict_seconds của chính các lần đó.
Đây là **predict-call throughput**, không phải camera FPS hoặc full-pipeline FPS.
Run lỗi có `partial_run=true`, giữ dữ liệu đã ghi và trả exit 1; không so như run hoàn chỉnh.
Tiến độ in mỗi 5 frame đo. Không xuất ảnh/skeleton/khớp, không có action classification.

`telemetry.jsonl`: mẫu trước warmup, sau warmup và sau mỗi frame đo; hai lệnh vcgencmd
chạy tuần tự, timeout tối đa 1 giây/lệnh, ngoài predict timer. Ghi raw, nhiệt độ,
cờ hiện tại/lịch sử, thời điểm và lỗi. Thiếu công cụ (như Windows) ghi unavailable/null,
không thay bằng nhiệt độ 0 hoặc cờ false. Mẫu không quan sát được các sự kiện giữa lần lấy mẫu.

## Windows PowerShell — chuyển update (từng dòng)

Gói chỉ chứa 6 file mới/thay đổi và checksum; không video/weights/.venv/outputs/dependencies.
Đường dẫn: `D:\patient-monitoring-transfer\pi-benchmark-update-20260920\`.
Các lệnh sau do người dùng chạy; Codex chưa SSH hoặc áp dụng trên Pi.

```powershell
Set-Location D:\patient-monitoring-transfer\pi-benchmark-update-20260920
scp -4 -O pi-benchmark-update-20260920.tar.gz pi-benchmark-update-20260920.tar.gz.sha256 "truong@patient-pi4.local:~/"
```

Chỉ tiếp tục nếu lệnh trước exit 0. Nếu hostname không phân giải, dùng IP Pi do bạn xác nhận.
Nếu home Pi đã có archive cùng tên cần giữ, chọn vị trí nhận mới, không ghi đè archive cũ.

## Terminal trên Pi — xác minh, sao lưu rồi cập nhật (từng dòng)

Mỗi dòng chạy riêng, chỉ tiếp tục khi thành công. Thư mục staging phải mới:

```bash
cd "$HOME"
sha256sum -c pi-benchmark-update-20260920.tar.gz.sha256
mkdir "$HOME/pi-benchmark-update-20260920" && tar -xzf "$HOME/pi-benchmark-update-20260920.tar.gz" -C "$HOME/pi-benchmark-update-20260920" --keep-old-files --no-same-owner
cd "$HOME/pi-benchmark-update-20260920" && sha256sum -c UPDATE_SHA256SUMS
python3 "$HOME/pi-benchmark-update-20260920/tools/pi_benchmark_update.py" apply --project "$HOME/patient-monitoring-pi4"
```

Updater kiểm tra đúng allowlist/checksum/path rồi sao lưu **tất cả file đang tồn tại trước khi
thay bất kỳ file nào**, vào `~/patient-monitoring-pi4/backups/pi-benchmark-<UTC>-<id>/`.
`update_receipt.json` ghi hash trước/sau, file trước đó chưa tồn tại và trạng thái apply.
Nếu incomplete, dừng xem receipt; bản sao cũ giữ nguyên để phục hồi, không chạy benchmark tiếp.
Không thay pyproject, data, weights hoặc outputs cũ. Script chuẩn bị gói dùng `build --output-dir NEW_EXTERNAL_DIR`.

## Terminal trên Pi — benchmark 640 (từng dòng)

```bash
cd "$HOME/patient-monitoring-pi4"
.venv/bin/python -m patient_monitoring benchmark-pose --help
.venv/bin/python -m patient_monitoring benchmark-pose data/raw/pilot01.mp4 --weights models/yolo11n-pose.pt --imgsz 640 --warmup 3 --frames 30 --output-dir outputs/pi_benchmark_640_run01
```

Không cài lại dependency. Output phải mới; lần sau dùng run02. Khi so imgsz=320/640,
giữ video/frames/warmup, nguồn điện/tản nhiệt/tải nền; để máy về trạng thái nhiệt tương đương,
lưu nhiệt độ/cờ và thứ tự chạy. Không đổi thread counts giữa các lần một cách ngầm định.
Chưa xác nhận nguồn điện/tản nhiệt; không gọi kết quả là benchmark chuẩn trước khi ghi điều kiện này.

## Kiểm tra đã làm trên Windows

33 test pose liên quan PASS (9 test benchmark), gồm warmup exclusion, frame selection,
input sai, telemetry thiếu/timeout và phân biệt cờ lịch sử. Smoke test Windows 1 warmup + 5 measured
ở imgsz=640 hoàn thành; CPU/float32, 7 mẫu telemetry unavailable. Đây không phải benchmark Pi.
Kết quả: `outputs/benchmark_windows_smoke_20260920/`. M2 vẫn IN_PROGRESS.
