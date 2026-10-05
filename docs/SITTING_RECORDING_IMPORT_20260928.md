# Nhập clip ngồi mới — P001_SE02_V001

## Nguồn và bảo toàn

- Nguồn chính xác: `C:/Users/asus/OneDrive/Pictures/Camera Roll/WIN_20260928_11_33_49_Pro.mp4`.
- Bản sao: `data/raw/P001/P001_SE02/P001_SE02_V001.mp4`, **7,362,247 byte**.
- SHA-256 nguồn trước/sau và bản sao cùng bằng `cfc0a516918758caf4d02c8d6851244ccc28f0afba83358817521370cacd8fcf`.
- Tạo file đích độc quyền, không ghi đè; giữ nguyên nguồn và byte video, không cắt/transcode/đổi FPS metadata.
- Người dùng xác nhận cùng người P001 (pilot01 và clip đứng trước), và xác nhận **buổi quay mới, dùng P001_SE02**.
  Video đầu phiên được cấp `P001_SE02_V001`; không suy ID, thời điểm quay UTC hoặc thông số camera từ filename/hình ảnh.
- `planned_activity=sitting` chỉ ghi nội dung người dùng mô tả; không phải nhãn cuối cho toàn clip.
  Visual pose review, nhãn thời gian, split và hồ sơ consent riêng còn pending; xử lý hiện tại theo yêu cầu người dùng.

## Kết quả CLI thực tế

Đã đọc `--help` của inspect-video, preview-video và pose-preview trước chạy. Không sửa code/dependency.

| Kiểm tra | Kết quả tại máy |
|---|---|
| Decode | Complete, **204 frame**, H.264, **1280×720**, 0 frame hỏng |
| PTS | Thiếu/trùng/giảm đều **0**; time base 1/30000, PTS đầu 0, cuối 410891 |
| Thời gian | Stream **13.729666667 s**; container **13.767146 s**; PTS span **13.696366667 s** |
| Preview | 8/8 mẫu: frame **0,29,58,87,116,145,174,203**; giữ thời gian thực trong samples.csv |
| PT640 | Complete 8/8 ảnh; **9 detection**, mỗi detection giữ đủ 17 khớp |
| Detection theo ảnh | **1,2,1,1,1,1,1,1**; frame 29 có 2 detection; 0 ảnh zero detection |

YOLO11n-pose weights M2 hiện hữu đã kiểm hash; CPU float32, imgsz640, batch1, rect=False,
conf=.25, IoU=.7, max_det=300; ngưỡng vẽ khớp .5 theo workflow pose-preview hiện tại.
Không tải model/thay dependency, không train hoặc benchmark Pi; không bỏ detection thứ hai.
Không coi số detection/confidence là accuracy; nhận xét visual pose vẫn pending.

Các lệnh đều exit 0. Không có lỗi decode/preview/inference. Ultralytics cảnh báo `half` deprecated;
giữ cấu hình hiện hữu và kiểm CPU/float32 thực tế. PTS span không gồm thời gian hiển thị frame cuối;
metadata/PTS không chứng minh thời lượng capture ngoài đời. Đây là giới hạn đo, không phải lỗi decode.

## Kết quả để review

- `outputs/fresh_20260928_113349_inspect/`: metadata.json và timeline.csv.
- [Contact sheet gốc](../outputs/fresh_20260928_113349_preview/contact_sheet.jpg): cùng samples.csv, 8 PNG và preview summary.
- [Contact sheet PT640](../outputs/fresh_20260928_113349_pose_pt640/pose_contact_sheet.jpg): cùng poses.json, run_info.json và ảnh annotate.
- `outputs/fresh_20260928_113349_import/`: provenance.json, manifest.csv, verification.json;
  annotations.csv và split_groups.csv chỉ có header, chưa có nhãn/nhóm được gán.

Inspect/preview chạy trên nguồn gốc trong lúc chờ xác nhận phiên; báo cáo giữ đường dẫn thực đó.
Hash đồng nhất liên kết kết quả với bản sao raw. Không chạy lại inference chỉ để đổi tên/path báo cáo.
Đã đối chiếu đủ 8 frame/index/timestamp/hash từ timeline qua PNG/poses, số detection và 17 khớp;
đã kiểm ảnh đầu ra đọc được đúng kích thước. duration_sec trong manifest dùng duration video stream.
`dataset_role=unassigned`, `annotation_status=unannotated`, `qc_status=needs_review`.
Người vẫn là P001 qua hai phiên; không dùng phiên mới để giả người mới khi chia tập. pilot01 development-only.

Tiếp theo: người dùng review ảnh gốc/pose, đặc biệt hai detection ở frame 29, rồi mới xác nhận nhãn đoạn thời gian.
M2/M3 vẫn IN_PROGRESS. Không ghi đè kết quả cũ hoặc thay PROJECT_PLAN.
