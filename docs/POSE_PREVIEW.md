# M2.1 — Pose smoke test trên preview PNG

Lệnh xử lý duy nhất các PNG được `samples.csv` tham chiếu, tối đa 8 dòng theo
quy ước preview. Không quét thêm ảnh. Frame index và actual timestamp lấy từ CSV;
actual timestamp là thời gian tương đối của video, không phải thời gian suy luận.
Đây là kiểm tra chạy mô hình, chưa chọn model cuối hoặc đánh giá accuracy.

## Cài đặt trong .venv và chạy

Giữ Python 3.10.11, PyAV 16.1.0 và Pillow 12.1.1. Dùng cặp CPU chính thức
[Torch 2.9.1 / Torchvision 0.24.1](https://pytorch.org/get-started/previous-versions/).
Ultralytics được ghim 8.4.152. Pip cũ 23.0.1 từng lỗi chuẩn hóa tên dependency
`typing-extensions`; nâng pip trong .venv lên 26.2.1 đã khắc phục lỗi này.

```powershell
cd D:\patient-monitoring-pi4
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: pip==26.2.1
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -e .
.\.venv\Scripts\python.exe -m patient_monitoring pose-preview outputs/pilot01_preview --output-dir outputs/pilot01_pose_m2_1
```

`outputs/pilot01_pose_m2_1` là output đã chạy thành công; khi chạy lại phải chọn
thư mục mới vì lệnh từ chối ghi đè báo cáo cũ. Mỗi PNG phân biệt
chỉ xử lý một lần. Nếu nhiều mốc target trùng PNG, `csv_row_indices` giữ tất cả
mapping. PNG nằm ngoài input directory hoặc frame/time xung đột bị từ chối.

Cấu hình: `yolo11n-pose.pt`, `device=cpu`, `imgsz=640`, `batch=1`, `rect=False`
(letterbox vuông), `half=False`, box confidence 0.25, NMS IoU 0.7, max_det 300,
không augment, không train. Ghi tất cả detection sau bộ lọc/NMS này; không chọn
riêng người đầu tiên. Tọa độ xuất về kích thước ảnh nguồn.

Ngưỡng vẽ mặc định 0.5, có thể đổi riêng cho hiển thị:

```powershell
.\.venv\Scripts\python.exe -m patient_monitoring pose-preview outputs/pilot01_preview --output-dir outputs/pilot01_pose_display07 --keypoint-confidence 0.7
```

Ngưỡng vẽ không thay box filter hoặc JSON. Chỉ vẽ điểm khi confidence >= ngưỡng;
chỉ vẽ cạnh khi cả hai đầu đạt. Không bù/ước lượng khớp thiếu. JSON giữ nguyên
17 x/y/confidence từ `result.keypoints.data`, không làm tròn hoặc xóa giá trị thấp.
Thứ tự khớp 0–16: nose, left/right eye, left/right ear, left/right shoulder,
left/right elbow, left/right wrist, left/right hip, left/right knee, left/right ankle.
Tham chiếu [Ultralytics Results](https://docs.ultralytics.com/reference/engine/results/)
và [YOLO11](https://docs.ultralytics.com/models/yolo11/).

## Weights và truy vết

Nguồn weights: [release chính thức Ultralytics v8.3.0](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt).
Lưu `models/yolo11n-pose.pt`, SHA-256:
`869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`.
File M2.1 hiện có đã khớp URL/tên/hash trên; CLI tải trực tiếp từ URL chính thức
khi chưa có file. Hash được kiểm tra trước khi load; mismatch không bị ghi đè. Nếu tải gián đoạn,
file `.part` có thể còn lại; xem lỗi, không coi là weights hoàn chỉnh.
Toàn bộ models/ được Git ignore; settings Ultralytics/Matplotlib được đặt tại đây,
không dùng cấu hình từ dự án cũ. Runtime không tự cài package còn thiếu.

## Output

- `pose_frame_*.png`: ảnh annotate giữ kích thước gốc, có nhãn frame/time/person count.
- `pose_contact_sheet.jpg`: hai hàng, giữ tỉ lệ ảnh, nhãn frame/time/person count.
- `poses.json`: `images` chứa mọi PNG đã xử lý, kể cả không phát hiện người.
  Mỗi image có đường dẫn, filename, SHA-256 nguồn, frame_index, actual_time_seconds,
  bản text timestamp nguyên từ CSV, csv_row_indices (0-based, không tính header),
  width/height, person_count, detection_status, annotated_image và people.
  Mỗi person có detection_index (chỉ trong ảnh, không phải tracking ID), class_id,
  box_xyxy, box_confidence và 17 keypoint với index/name/x/y/confidence.
  Zero detection: `person_count=0`, `detection_status=zero_detections`, `people=[]`;
  vẫn lưu ảnh/record. Không có keypoint cho box sẽ báo lỗi, không tự thêm khớp.
- `run_info.json`: status/error, UTC bắt đầu/kết thúc, model URL/path/SHA-256,
  Python và mọi package version, cấu hình yêu cầu/thực dùng, device/dtype thực,
  CSV/hash, số ảnh dự kiến/đã xử lý/zero detection, cảnh báo và human_pose_review.
  `runtime_unreported_config` liệt kê tham số được yêu cầu nhưng phiên bản
  Ultralytics không trả lại qua `predictor.args`; cấu hình yêu cầu vẫn được giữ
  trong `config` và dtype thực của model được ghi riêng.

Exit 0 chỉ khi mọi input đã xử lý và xuất xong. Exit 1 khi input/output/model/
import/inference lỗi; nếu đã tạo output, run_info và poses có thể lưu phần đã
xử lý, không phải kết quả thành công. Lỗi ghi ổ đĩa có thể ngăn lưu cả báo cáo lỗi.
Nguồn PNG/CSV được băm trước/sau. `outputs/m2_1_preservation.json` ghi hash các
artifact đã có trước lần chạy M2.1 để đối chiếu bảo toàn.

## Diễn giải và kiểm tra

Confidence là điểm mô hình, không phải accuracy. Chưa có nhãn keypoint thủ công
để đo độ chính xác. Người dùng đã hoàn tất xem preview M1.2 và xác nhận pilot
có frame chỉ thấy một phần cơ thể; HUMAN_POSE_REVIEW_PENDING sau smoke test.
Chưa kết luận pose phù hợp, không gán action, không xử lý thời gian, camera hoặc Pi.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m patient_monitoring doctor
.\.venv\Scripts\python.exe -m pip check
```

Test tổng hợp kiểm tra giữ nhiều người/17 khớp/confidence thấp, điều kiện cạnh,
CSV và zero-detection record; không phải dữ liệu nghiên cứu hoặc phép đo accuracy.
Kết quả thực tế của lần chạy xem `STATUS.md` và `run_info.json`.
