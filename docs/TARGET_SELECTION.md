# M2.4 — Baseline thử nghiệm liên kết một người

Trong `D:\patient-monitoring-pi4`, dùng `.venv` hiện có:

```powershell
.\.venv\Scripts\python.exe -m patient_monitoring select-target outputs/pilot01_poses_m2_2/poses.jsonl --output-dir outputs/pilot01_target_m2_4 --seed-frame 0 --seed-detection 0 --min-iou 0.1 --ambiguity-margin 0.05 --max-gap 0.5 --video data/raw/pilot01.mp4 --review-frames 2 80 92 108 109 130
```

Lệnh đã chạy; chạy lại phải chọn output mới. Hai tham số seed bắt buộc.
`--video` và `--review-frames` chỉ dùng xuất ảnh **sau khi** chọn xong; có thể bỏ cả hai
để chỉ liên kết từ JSONL. Không import/chạy YOLO, tải model hoặc cài dependency.

## Quy tắc nhân quả và tham số

Chỉ dùng hình học box; không dùng detection index, box confidence hoặc chất lượng
khớp để xếp hạng. Detection index chỉ định detection của seed và tham chiếu nguồn từng frame,
không phải ID theo thời gian. Không hardcode D0 hoặc sáu frame review trong thuật toán.

1. Trước frame seed: null / `awaiting_seed`. Seed phải có timestamp và box hợp lệ.
2. Mỗi frame sau đó: tính IoU mọi box hợp lệ với **box được chấp nhận gần nhất**.
3. Chọn ứng viên IoU cao nhất nếu IoU >= `min_iou` và chênh lệch với ứng viên thứ hai
   >= `ambiguity_margin`. Hòa luôn bị từ chối, kể cả margin=0. Chỉ có một ứng viên
   hợp lệ thì margin không áp dụng; runner-up/separation ghi null.
4. Không có detection, box sai, IoU thấp hoặc mơ hồ: lựa chọn null với lý do;
   không cập nhật box/thời điểm tham chiếu và không nội suy khớp.
5. Thời gian PTS từ lần chấp nhận cuối **> max_gap**: `gap_timeout`, sau đó
   `association_ended` đến cuối file, kể cả khi xuất hiện box giống hệt. Đúng bằng max_gap
   vẫn được phép so khớp. Muốn tiếp tục phải chạy với seed mới và output mới.

Mặc định thử nghiệm: **IoU 0,1; margin 0,05; gap 0,5 giây**. Chưa được xác thực/tối ưu.
Box phải có bốn giá trị hữu hạn và `0<=x1<x2<=width`, `0<=y1<y2<=height`;
không clamp box sai. Timestamp phải khớp PTS×time_base và quy ước M1.1, tăng nghiêm ngặt.
Timestamp thiếu/sai/trùng/giảm tạo null, không làm mới tham chiếu; khi có timestamp hợp lệ
trở lại vẫn kiểm tra gap từ lần chấp nhận cuối. JSON phi chuẩn NaN/Infinity bị từ chối.
Frame index phải liên tục từ 0; nguồn/stream/kích thước không được đổi trong một chuỗi.
Seed không hợp lệ: không tự chọn người thay thế, báo run incomplete và giữ các dòng null.

## Đầu ra và giới hạn diễn giải

- `target_poses.jsonl`: mỗi frame một dòng, giữ sáu trường index/timestamp gốc,
  `selected_source_detection_index`, `selected_person`, status/reason,
  reference frame/time, elapsed gap và IoU của từng ứng viên/runner-up/separation.
  `selected_person` giữ nguyên box/confidence và toàn bộ 17 khớp/cờ; không loại partial-body.
- `selection_summary.json`: tham số, hash input, trạng thái thực thi, bộ đếm status,
  số selected/unresolved và các đoạn null liên tiếp. `gap_ranges` có thời điểm đầu/cuối,
  lần chấp nhận trước, thời điểm nối lại nếu có; không suy diễn thời gian khớp bị mất.
  Run complete chỉ nghĩa đã xử lý hết input, không chứng minh association đúng.
- `review/`: PNG độ phân giải nguồn, `reviewed_selection_sheet.jpg`,
  `unresolved_selection_sheet.jpg` và manifest. Box vàng là mục tiêu được chọn;
  null được ghi rõ. Chọn tối đa sáu frame unresolved theo thời gian; sheet trống ghi không có case.
- `manual_observations.json`: ghi riêng nhận xét do người dùng cung cấp:
  tại 2,80,92,108,109,130, D0 là người nhìn thấy; D1 có vẻ là false positive nền sát trái,
  với 0–1 khớp qua gate. Khớp ở biên có thể confidence cao. Đây là **quan sát thủ công**,
  không phải ground truth tự động, không đưa vào logic chọn mục tiêu.
- `verification.json`: đối chiếu độc lập timestamps, giá trị gốc, IoU, test và hash bảo toàn.

Pilot: 415 dòng, 1 seeded + 414 matched, không unresolved/gap. Sáu frame review đều chọn D0
bằng IoU; phù hợp sáu nhận xét thủ công, không phải accuracy trên toàn video.
88 quan sát khớp vừa ở biên vừa qua ngưỡng confidence vẫn được giữ nguyên trong mục tiêu.
Đây chỉ là baseline một người; box giao nhau/đổi kích thước/che khuất vẫn có thể gây đổi người.
Chưa phân loại hành động hoặc triển khai; M2 vẫn IN_PROGRESS.

Kiểm tra: `python -m unittest discover -s tests -p 'test_target*.py' -v` (13 test)
và `python -m unittest discover -s tests -p 'test_pose*.py' -v` (24 test), dùng `.venv`.
