# Nhập và kiểm tra video mới — P001_SE01_V001

## Nguồn, ID và bảo toàn

- Nguồn người dùng cung cấp: `C:/Users/asus/OneDrive/Pictures/Camera Roll/WIN_20260924_13_01_42_Pro.mp4`.
- Bản sao nguyên byte: `data/raw/P001/P001_SE01/P001_SE01_V001.mp4`, 10,129,801 byte.
- SHA-256 nguồn trước/sau và bản sao đều là `2ccad3c23be44d145c892d52eeccd0d3b4fea9d4b4dc98cf64299ec3cbd9a767`.
  Đích được tạo độc quyền, không ghi đè; file nguồn giữ nguyên, không đổi FPS/cắt/transcode.
- Người dùng xác nhận cùng người với pilot01, là chính người dùng. Cấp mã dự án `P001`, liên kết pilot01
  với cùng subject trong provenance mới; không sửa metadata/outputs pilot01 và không suy danh tính từ hình ảnh.
- `P001_SE01` là mã phiên quản lý được cấp cho video mới này. Phiên của pilot01 và quan hệ phiên quay
  chưa xác nhận; không gộp pilot01 vào SE01, không suy ngày quay UTC, camera/ánh sáng từ tên file.
- `planned_activity=standing` lấy từ yêu cầu người dùng; chưa gán nhãn standing cho toàn clip.
  Consent record riêng, annotation, split và review full-body/occlusion còn pending.
  Yêu cầu nhập/kiểm tra/pose của người dùng là căn cứ xử lý nhiệm vụ hiện tại.

## Kết quả đo độc lập bằng công cụ dự án

`inspect-video` decode hoàn tất **284 frame H.264, 1280×720**, không frame hỏng;
thiếu/trùng/giảm timestamp đều **0**. Time base `1/30000`, PTS đầu 0, cuối 572176.
Duration video stream **19.105866667 s**; container **19.178416 s**; span từ PTS đầu đến cuối
**19.072533333 s** (không gồm thời gian hiển thị frame cuối). Không gọi chúng là thời lượng capture thực đã xác minh.
Số đo phù hợp kiểm tra FFmpeg bên ngoài người dùng nêu, nhưng được tính lại tại máy bằng PyAV 16.1.0.
Clip dài hơn gợi ý 8–15 s của protocol; giữ nguyên và chờ review, không tự cắt để đạt mục tiêu.

Preview chọn 8 frame theo PTS: **0,40,81,121,162,202,242,283**. Đã đối chiếu index/thời gian/hash
PNG giữa timeline, samples.csv và poses.json. PT640 hoàn tất 8/8 ảnh; mỗi ảnh một detection/17 khớp,
không ảnh zero/multiple. Đây là số detection trên mẫu, không phải kết luận accuracy hoặc toàn bộ video.
Weights M2 được kiểm hash trước chạy; CPU float32, imgsz640, batch1, rect=False, conf=.25, IoU=.7,
ngưỡng hiển thị khớp .5 theo workflow pose-preview hiện hữu. Không đổi dependency hoặc tải weights.

## File kết quả và trạng thái review

- `outputs/fresh_20260924_130142_inspect/`: `metadata.json`, `timeline.csv`.
- [Contact sheet ảnh gốc](../outputs/fresh_20260924_130142_preview/contact_sheet.jpg), cùng 8 PNG, samples.csv và preview summary.
- [Contact sheet pose PT640](../outputs/fresh_20260924_130142_pose_pt640/pose_contact_sheet.jpg), cùng ảnh annotate, poses.json, run_info.json.
- `outputs/fresh_20260924_130142_import/`: `provenance.json`, `manifest.csv` (một video thực),
  `annotations.csv`/`split_groups.csv` chỉ header, và `verification.json`.

Các báo cáo inspect/preview giữ đường dẫn nguồn thực trên OneDrive vì được chạy khi chờ xác nhận ID.
Hash đồng nhất liên kết chúng với bản sao raw trong dự án. Không viết lại báo cáo hoặc chạy lại pose sau khi cấp ID.
Schema manifest lấy nguyên từ template; duration_sec là duration video stream; thời điểm quay UTC để trống.
`annotation_status=unannotated`, `dataset_role=unassigned`, `qc_status=needs_review`; chưa khóa split.
pilot01 vẫn development-only. Vì cùng subject P001, cần xét quan hệ này khi chia nhóm; video mới không
tự tạo người độc lập để đánh giá unseen-subject. Chưa quyết định clip mới thuộc tập nào.

**Tiếp theo:** review ảnh gốc và pose, xác nhận toàn thân/che khuất và đoạn hoạt động thực trước khi gán nhãn.
M2/M3 vẫn IN_PROGRESS; nhiệm vụ này chưa xác nhận mọi frame là standing hoặc pose đúng.
