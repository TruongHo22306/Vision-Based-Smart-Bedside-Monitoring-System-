# M2.2 — Pose theo timestamp của toàn bộ video

Trong workspace `D:\patient-monitoring-pi4`, dùng đúng `.venv` đã xác minh ở M2.1:

```powershell
.\.venv\Scripts\python.exe -m patient_monitoring extract-poses data/raw/pilot01.mp4 --output-dir outputs/pilot01_poses_m2_2
```

Lệnh trên đã chạy; khi chạy lại hãy chọn tên output mới. Giữ model `models/yolo11n-pose.pt` và dependency M2.1;
không tải model hoặc tự cài thư viện. Kiểm tra SHA-256 weights trước khi load.
Cấu hình kế thừa M2.1: CPU, imgsz 640, batch 1, rect False, half False,
box confidence 0,25, NMS IoU 0,7, max_det 300. Đây chưa phải model cuối.

## Dữ liệu và thời gian

PyAV chọn video stream đầu tiên, decode tuần tự và suy luận từng frame.
`poses.jsonl` có một dòng cho mỗi frame đã decode trong lần chạy:

- `frame_index` bắt đầu từ 0, giữ thứ tự decode.
- `pts`, `time_base`, `presentation_time_seconds`, `relative_time_seconds`,
  `delta_seconds` dùng chính `TimelineAudit` của M1.1: thời điểm = PTS × time_base,
  tương đối với PTS hợp lệ đầu tiên; delta so với PTS hợp lệ trước đó.
- Timestamp thiếu được giữ `null`; timestamp trùng/giảm được giữ và thống kê.
  Không thay timestamp bằng FPS, tốc độ xử lý hoặc đồng hồ hệ thống; không lấy mẫu lại.
- Mỗi dòng lưu đường dẫn nguồn, stream index, kích thước ảnh, cờ frame hỏng,
  trạng thái inference, số người và toàn bộ `people` sau ngưỡng box/NMS.
- Mỗi người giữ box/confidence và đúng 17 bộ x/y/confidence trong tọa độ ảnh nguồn.
  `detection_index` chỉ là thứ tự detection trong frame, **không phải tracking ID**.
- Không thấy người: inference complete, `person_count=0`, `people=[]`,
  `detection_status=zero_detections`. Nhiều người được giữ nguyên tất cả.

## Cờ khớp độc lập — không sửa dự đoán

| Cờ | Định nghĩa |
|---|---|
| `finite_values` | Cả x, y và confidence đều hữu hạn |
| `confidence_pass` | Confidence hữu hạn và >= 0,5, độc lập với tọa độ |
| `at_or_outside_boundary` | Cả x/y hữu hạn và x<=0, x>=width, y<=0 hoặc y>=height |
| `passes_quality_gate` | Cả ba hữu hạn, confidence>=0,5 và 0<x<width, 0<y<height |

`finite_components` ghi riêng tính hữu hạn của x/y/confidence. Giá trị NaN/±Inf
được biểu diễn bằng JSON `null`; mọi giá trị hữu hạn giữ nguyên, kể cả âm,
nằm ngoài ảnh hoặc confidence thấp. Box/confidence không hữu hạn cũng có cờ riêng.
Không zero/clamp/xóa/nội suy điểm bị loại và không loại người vì khớp mặt không đạt.
Gate là bộ lọc bảo thủ tạm thời, **không phải accuracy hoặc bằng chứng khớp nhìn thấy**.

## Báo cáo và lỗi

`summary.json`: số decode/process/dòng ghi, EOF, frame hỏng, zero/multiple detection,
tổng detection, timestamp anomalies và lỗi theo giai đoạn. Mỗi khớp có số quan sát,
confidence thấp, tọa độ ở/ngoài biên, nonfinite và số lần đạt gate.
Các số là **count**, không phải tỷ lệ: mỗi người trong mỗi frame xử lý thành công
đóng góp một quan sát/khớp. Zero detection không thêm quan sát khớp. Low-confidence
chỉ đếm confidence hữu hạn <0,5; nonfinite đếm riêng. Các nhóm cờ có thể chồng lấp.

`run_info.json`: hash nguồn/weights thực tế, cấu hình yêu cầu/runtime, device/dtype,
Python và toàn bộ package versions. `half` vẫn là yêu cầu M2.1; phiên bản hiện tại
cảnh báo deprecated và không trả lại trường này trong runtime args.

Chỉ giữ frame/prediction hiện tại và bộ đếm cố định trong RAM; flush JSONL mỗi dòng,
checkpoint báo cáo mỗi 25 frame. Nếu ngắt cưỡng bức, báo cáo vẫn incomplete và bộ đếm
có thể chỉ phản ánh checkpoint gần nhất. Khi bắt được lỗi, ghi chính xác bộ đếm cuối.
Frame lỗi inference vẫn có dòng `inference_status=failed`, `person_count/people=null`;
không giả thành zero detection. Dừng tại lỗi, giữ dữ liệu đã ghi và trả exit 1.
Exit 0 chỉ khi EOF sạch, số process khớp decode, không có frame hỏng hoặc thiếu so
với số frame stream khai báo; không thể loại trừ truncation im lặng không phát hiện được.

Kiểm tra: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`
và `.\.venv\Scripts\python.exe -m pip check`.
Khi so timeline M1.1, đối chiếu cả sáu trường timestamp/index theo từng dòng;
pilot01 đã chạy đủ 415 dòng, khớp cả sáu trường với M1.1; zero detection 0 frame,
multiple detection 6 frame, tổng 421 detection. `verification.json` ghi kết quả đối chiếu
độc lập và kiểm tra bảo toàn artifact cũ. Human pose review M2.1 đã hoàn thành với
giới hạn partial-body; kết quả toàn video M2.2 còn cần xem thủ công.
