# M2.3 — Review dự đoán pose đã lưu

Chạy trong `D:\patient-monitoring-pi4` với `.venv` hiện có:

```powershell
.\.venv\Scripts\python.exe -m patient_monitoring review-poses data/raw/pilot01.mp4 --poses outputs/pilot01_poses_m2_2/poses.jsonl --summary outputs/pilot01_poses_m2_2/summary.json --output-dir outputs/pilot01_review_m2_3
```

Lệnh đã chạy. Khi chạy lại, chọn thư mục output mới; utility từ chối ghi đè.
Chỉ đọc dự đoán đã lưu; không import/chạy YOLO, tải model, đổi ngưỡng hay thêm dependency.

## Chọn và đối chiếu frame

- Lấy **mọi** dòng `person_count > 1`; kiểm tra số lượng bằng `summary.json`.
- Tìm các frame có ít nhất một gối/cổ chân `at_or_outside_boundary=true`.
  Chọn tối đa 6 frame riêng biệt gần nhất với các mốc PTS chia đều từ thời điểm
  ứng viên đầu đến cuối; mỗi lần chọn ứng viên chưa dùng, hòa thì lấy index nhỏ hơn.
- Decode video tuần tự bằng PyAV, so cả sáu trường index/PTS/time_base/thời gian
  với JSONL trên **mọi frame**, đồng thời kiểm tra kích thước, stream và hash nguồn.
  Không seek theo FPS, lấy mẫu lại hay chạy suy luận.
- Nếu sai lệch khi xuất, manifest ghi `export_status=incomplete` và lỗi; CLI trả exit 1.

## Đọc ảnh và manifest

- `frame_XXXXXX.png`: ảnh annotate ở độ phân giải nguồn, mỗi frame chỉ lưu một lần.
- `multi_detection_sheet.jpg`: toàn bộ frame nhiều detection, không bỏ box confidence thấp.
- `boundary_review_sheet.jpg`: tối đa 6 frame mẫu biên; đây không phải toàn bộ frame có lỗi.
- Mỗi detection có màu riêng, nhãn `D`, box confidence và số khớp đạt gate trên 17.
  `D` là index trong frame, **không phải tracking ID**; màu giống nhau không chứng minh cùng người.
- Chỉ nối cạnh khi cả hai khớp có cờ `passes_quality_gate=true` đã lưu.
  Dấu **X hồng** đánh dấu tọa độ hữu hạn của gối/cổ chân được gắn cờ biên,
  kèm tên khớp/confidence. Tọa độ ngoài ảnh chỉ được cắt vào biên để hiển thị;
  không thay đổi tọa độ, confidence hoặc cờ trong dự đoán/manifest.
- `review_manifest.json`: hash input, cách chọn, index/timestamp, lý do chọn,
  box/confidence và các giá trị khớp gốc của từng detection; kết luận người xem để `pending`.
  Giá trị nhãn ảnh làm tròn để đọc; giá trị JSON giữ nguyên.
- Gate là bộ lọc bảo thủ tạm thời; confidence không phải accuracy hoặc bằng chứng nhìn thấy khớp.
  Không kết luận false positive chỉ từ confidence. Partial-body vẫn là giới hạn cần xem xét.

## Lần chạy pilot01

415/415 frame khớp nguồn. Sáu frame nhiều detection: **2, 80, 92, 108, 109, 130**.
Trong 160 ứng viên biên, chọn **2, 69, 137, 191, 298, 339**; tổng cộng 11 PNG 1280×720.
Hai sheet lần lượt 1920×1196 và 1920×1052. Bằng chứng đối chiếu: `verification.json`
trong output M2.3. Visual review của người dùng vẫn pending; M2 vẫn IN_PROGRESS.

Kiểm tra liên quan: `python -m unittest discover -s tests -p 'test_pose*.py' -v`
và `python -m unittest discover -s tests -p test_video_inspection.py -v` trong `.venv`.
Bước tiếp theo: mở hai sheet, đối chiếu PNG nguồn độ phân giải đầy đủ và ghi nhận thủ công.
