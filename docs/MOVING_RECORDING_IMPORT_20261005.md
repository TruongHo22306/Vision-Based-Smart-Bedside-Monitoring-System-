# Nhập video mới — P001_SE04_V001

## Nguồn, ID và camera

- Nguồn: `C:/Users/asus/OneDrive/Pictures/Camera Roll/WIN_20261005_14_23_08_Pro.mp4`.
- Bản sao: `data/raw/P001/P001_SE04/P001_SE04_V001.mp4`, 15.423.119 byte.
- SHA-256 nguồn trước/sau và bản sao đều là
  `d379806e9897973bdf7a41ecc8529ff00ef025639b675697436d04ba4886dc92`.
- Đã đối chiếu hash các MP4 raw hiện có: chưa nhập. Copy độc quyền, không ghi đè,
  cắt/transcode hoặc đổi metadata video; giữ nguyên nguồn.
- User xác nhận **cùng người P001, buổi quay mới, camera đổi góc**; cấp session
  tiếp theo P001_SE04 và video đầu V001 theo protocol. Góc cụ thể, thiết bị và
  thông số camera khác chưa được cung cấp. Không suy giờ UTC từ filename.
- Không tạo quan hệ cùng buổi với các phiên cũ; SE02/SE03 vẫn giữ nhóm đã ghi.
  P001 vẫn là một người qua các phiên. Chưa gán tập hay nhãn thời gian cuối.

## Quan sát từ cuộc trao đổi, tách khỏi predictions

User cung cấp nhận xét sơ bộ: người đi về phía/rời xa camera và xoay người;
đầu tiến gần mép trên khi ở gần camera; các chi chồng lấp khi xoay.
Ghi riêng trong `outputs/fresh_20261005_142308_import/manual_observations.json`.
Đây không phải Codex tự review trong task này, ground truth hoặc annotation thời gian.
Không có ranh giới đoạn được cung cấp; `planned_activity` để trống vì chỉ có quan sát
sơ bộ, không xác nhận nội dung dự định quay. Manifest ghi rõ review/nhãn/split pending.

## Công cụ và kết quả thực tế

Đã đọc help ba CLI trước chạy. CLI cũ chỉ chấp nhận tối đa 8 mẫu; thay đổi tối thiểu
`__main__.py`, `video_preview.preview_video` và `pose_preview.read_samples` để nhận
tối đa 12, giữ mặc định 8 và thuật toán chọn theo PTS/two-row contact sheet hiện hữu.
21 test `test_*preview.py` PASS, gồm 12 mẫu và từ chối 13; không thêm dependency.

- Inspect complete: **439 frame**, H.264, **1280×720**, 0 frame hỏng.
- Timestamp thiếu/trùng/giảm đều **0**; time base 1/30000; PTS đầu 0, cuối 886103.
- Stream duration **29.570066667 s**, container **29.585541 s**, PTS span **29.536766667 s**.
  Reported rate ~14.84608 FPS không phải chứng minh FPS capture hoặc nhịp đều.
- Preview 12/12 time-distributed samples; index/time khớp timeline, chọn nearest PTS
  tại 12 thời điểm cách đều, gồm hai đầu. Không dùng frame_index/FPS để suy thời gian.
- PT640 complete 12/12 ảnh, **12 detection**, 0 mẫu zero/multiple detection.

| Frame | Thời gian thực (s) | Detection |
|---|---:|---:|
| 0 | 0.000000 | 1 |
| 40 | 2.688067 | 1 |
| 80 | 5.392133 | 1 |
| 120 | 8.080233 | 1 |
| 159 | 10.720300 | 1 |
| 199 | 13.408333 | 1 |
| 239 | 16.112433 | 1 |
| 279 | 18.816500 | 1 |
| 319 | 21.504600 | 1 |
| 358 | 24.128633 | 1 |
| 398 | 26.832667 | 1 |
| 438 | 29.536767 | 1 |

Existing local YOLO11n-pose, CPU float32 thực tế, imgsz640/batch1/rect=False,
conf=.25/IoU=.7/max_det=300, ngưỡng vẽ khớp .5. Hash weights M2 đã kiểm trước chạy;
không tải model. Package versions/config khớp preview trước. Giữ đủ 17 raw joints
mỗi detection; source PNG hashes khớp poses; tất cả ảnh đọc được đúng kích thước.
Cả ba CLI exit 0; cảnh báo `half` deprecated của Ultralytics không làm lỗi run.

## Đường dẫn và phần còn chờ

- `outputs/fresh_20261005_142308_import/`: manifest.csv, provenance.json,
  manual_observations.json, verification.json; annotations.csv/split_groups.csv chỉ header.
- `outputs/fresh_20261005_142308_inspect/`: metadata.json và timeline.csv.
- [Contact sheet gốc](../outputs/fresh_20261005_142308_preview/contact_sheet.jpg),
  samples.csv, 12 PNG và timing_summary.json trong cùng thư mục.
- [Contact sheet PT640](../outputs/fresh_20261005_142308_pose_pt640/pose_contact_sheet.jpg),
  poses.json/run_info.json và 12 PNG annotate trong cùng thư mục.

Chỉ inference trên 12 PNG; chưa inference toàn video hoặc huấn luyện. Visual pose review,
nhãn thời gian, split và hồ sơ consent riêng còn pending; việc xử lý được user yêu cầu.
Confidence/số detection không phải accuracy. Video dài hơn gợi ý 8–15 s được giữ nguyên.
PTS span không gồm thời gian hiển thị frame cuối, không chứng minh thời lượng capture ngoài đời.
M2/M3 IN_PROGRESS. Tiếp theo: review hai sheet, đặc biệt đoạn gần camera và khi xoay.
