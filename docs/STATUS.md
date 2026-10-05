# Trạng thái dự án

- M0: COMPLETED; M1: IN_PROGRESS; M1.1 VERIFIED; M1.2 VISUAL_REVIEW_COMPLETED với partial-body.
- M2: IN_PROGRESS; M2.1 visual review hoàn thành; M2.2 EXTRACTION_VERIFIED.
- M2.3: nhận xét review thủ công do người dùng cung cấp; M2.4 COMPLETED_IN_PILOT_SCOPE.
- M3: IN_PROGRESS — SE01–SE04 đã có full pose/target; inventory cũ giữ như snapshot. SE04 user xác nhận 10 khoảng mô tả gần đúng tại outputs/P001_SE04_V001_temporal_notes_20261005/, chưa đổi sang frame/nhãn; xoay có thể chứa pause. Ranh giới chính xác/nhãn pending; giữ nhóm SE02/SE03, chưa split. M4–M8: NOT_STARTED.
- PROJECT_PLAN giữ nguyên; SHA-256 `A46D069392E3E86B6B22E4D62482199600DE130D76565950C9ACFDFFB92BC00C`.

## Windows và pilot đã kiểm tra
- Python 3.10.11 .venv; PyAV 16.1.0; Pillow 12.1.1; Ultralytics 8.4.152.
- Torch 2.9.1+cpu, Torchvision 0.24.1+cpu, NumPy 2.2.6, OpenCV 5.0.0.93; dependency giữ nguyên.
- Video data/raw/pilot01.mp4 H264, 415 frame 1280×720; timeline outputs/pilot01/.
- Không thiếu/trùng/giảm timestamp/frame hỏng; span 27.9207 s, không thay PTS bằng FPS.
- outputs/pilot01_preview/: 8 PNG/CSV/sheet; M1.2 review có toàn thân/partial-body.
- outputs/pilot01_pose_m2_1/: 8/8 PNG mỗi ảnh 1 detection/17 khớp; giữ outputs/pilot01_pose/ lỗi lịch sử.
- Weights M2 SHA-256 `869E83FCDFFDC7371FA4E34CD8E51C838CC729571D1635E5141E3075E9319DC0`.
- outputs/pilot01_poses_m2_2/: 415 dòng; 421 detection, 6 multiple, 0 zero; đủ 17 khớp mỗi detection.
- 7157 quan sát: 1226 confidence thấp, 812 ở/ngoài biên, 5843 đạt gate; nhóm có thể chồng lấp.
- outputs/pilot01_review_m2_3/: 11 PNG/hai sheet; đủ 6 frame multi; 6 mẫu biên từ 160 ứng viên.

## M2.4 hoàn thành trong pilot, chưa chứng minh accuracy toàn video
- Người dùng review ảnh trong trao đổi: frame 2,80,92,108,109,130 khung vàng chọn người áo 22; D1 nhầm nền sát trái.
- Codex chỉ ghi nhận, không tự kiểm chứng; nhận xét không phải ground truth toàn video.
- Seed 0/D0, min IoU 0.1, margin 0.05, gap 0.5 s; không hardcode D0/frame hoặc dùng quality chọn người.
- outputs/pilot01_target_m2_4/: 415 dòng = 1 seeded + 414 matched; 0 unresolved; giá trị gốc giữ nguyên.
- 88 khớp biên confidence cao giữ nguyên; chọn đúng người không chứng minh pose đúng.
- 13 test target + 24 test pose regression đã PASS; schema/hướng dẫn docs/TARGET_SELECTION.md.
- Ngưỡng chưa xác thực; IoU không đảm bảo danh tính qua che khuất/giao nhau/đổi box; partial-body còn tồn tại.
## Pi: thông tin do người dùng báo, Codex chưa tự kiểm chứng
- Pi 4 8GB, Bookworm aarch64, Python 3.11.2; SSH dự kiến truong@patient-pi4.local.
- Webcam icSpring 32e6:9211 /dev/video0; ffplay thấy hình; MJPEG 1280×720/640×360 yêu cầu 30, đo ~15 FPS.
- Hai capture 150 frame exit 0 chưa phải benchmark YOLO.
- Người dùng báo pilot PT640 CPU/float32/batch1/rectFalse: 415/415 frame, 421 detection, 6 multi, 0 zero, 0 lỗi timestamp; 677.096496 s đầu–cuối không phải inference-only/benchmark kiểm soát.
- Nhiệt độ trước/sau 66.7/74°C; get_throttled đều 0xd0000; nguồn điện/tản nhiệt chưa xác nhận.
- 0xd0000 có cờ lịch sử 16/18/19; không suy ra không throttling giữa các mẫu.
- Gói chuyển đầy đủ trước đó giữ nguyên ngoài source: D:/patient-monitoring-transfer/pilot01-20260920/.
## Benchmark / NCNN trial, M2 IN_PROGRESS
- User-reported Pi PT 640: throughput 0.96272884545/s, median 1.041947089 s; 320: 2.67687469585/s, median 0.3741457125 s.
- 3 warmup/30 measured, CPU float32/batch1/rectFalse; mỗi size một run, nhiệt độ đầu khác nhau; số liệu sơ bộ, nguồn/tản nhiệt chưa xác nhận.
- RGB/BGR audit Ultralytics 8.4.152: PyAV PIL RGB -> loader BGR -> predictor RGB đúng; không sửa màu/không hủy baseline cũ.
- Export Windows NCNN 320 thành công từ đúng weights M2: quantize=32, PNNX fp16=0; không tải model khác.
- .venv-ncnn là overlay riêng, đọc package nền; NCNN 1.0.20260526/PNNX 20260526; .venv/dependency hiện hữu giữ nguyên.
- models/ncnn_trial_320/export_info.json: source/artifact/exporter SHA-256, versions/settings; model yolo11n_pose_320_ncnn_model/.
- benchmark-pose nhận NCNN local, giữ frame/PTS/warmup/timing/telemetry; ghi thread/precision NCNN riêng, không giả định từ Torch.
- compare-pose-backends: cùng frame/pixel/config; đối chiếu box/17 khớp/confidence, không phải accuracy hoặc tracking.
- Windows: 41 test pose PASS; pip check cả hai môi trường PASS; smoke NCNN 1 warmup/2 measured hoàn thành.
- Windows outputs/ncnn_consistency_windows_20260923/: frame 0,1,2 có 1,1,2 detection/backend, 4 box match, max joint delta <0.000367 px; outputs/ncnn_smoke_windows_20260923/: CPU/Vulkan=False, telemetry unavailable, không phải benchmark Pi.
- docs/PI_NCNN_TRIAL.md: scp/checksum/backup/overlay; gói mã/artifacts tại D:/patient-monitoring-transfer/pi-ncnn-{update,artifacts}-20260923/; Codex chưa chạy Pi.
- Đã đọc file Pi pi-ncnn-results-20260923: comparison 3 frame/4 box match; NCNN benchmark 30/30 complete, median 0.482773 s, predict throughput 2.158753/s.
- Telemetry thực: current undervoltage/throttling 30/32 mẫu, gồm 29/30 mẫu sau frame đo; 0x50005. Không kết luận lợi thế tốc độ backend.
- outputs/pilot01_backend_visual_review_20260923/: 8 PNG ba cột PT640/PT320/NCNN320 + sheet; 8 dự đoán cũ xác minh, 16 predict mới Windows; raw/flags tách riêng.
- 44 test pose PASS, pip check hai môi trường PASS; 24 bản dự đoán/30 detection/510 khớp, đúng hash/index/time. Ghi chú docs/EXPERIMENT_NCNN_REVIEW_20260923.md.
- Review assistant định tính đã ghi docs/POSE_VISUAL_REVIEW_20260923.md, không ground truth; PT640 là tham chiếu offline tạm, chưa chốt triển khai; M2 IN_PROGRESS.
- 24/09: P001_SE01_V001 nhập nguyên byte theo protocol, hash nguồn/bản sao khớp; user xác nhận cùng người pilot01. 284 frame 1280×720; 0 lỗi timestamp; stream 19.105867 s/container 19.178416 s.
- SE01 PT640 complete 284/284 frame, 284 detection, 0 zero/multi; index/PTS khớp timeline, 0 lỗi timestamp; 1 seeded+283 matched, 0 unresolved/gap. Assistant review sheet 0,40,81,121,162,202,242,283: khung vàng chọn chủ thể 8/8, mỗi mẫu 1 detection; khớp chi hợp lý ở mức sheet, hai chân trong ảnh, tay/tư thế đổi nhẹ nhất là 121–162; không xác thực danh tính toàn video/accuracy/nhãn. Ghi riêng outputs/P001_SE01_V001_association_notes_20261005/manual_observations.json; docs/STANDING_TARGET_REVIEW_20261005.md. Nhãn/split pending; M2 IN_PROGRESS.
- 28/09: user xác nhận P001/buổi mới P001_SE02; nhập P001_SE02_V001, SHA-256 nguồn/bản sao khớp. 204 frame 1280×720, 0 lỗi timestamp/frame hỏng; stream 13.729667 s/container 13.767146 s.
- outputs/fresh_20260928_113349_{import,inspect,preview,pose_pt640}/ giữ nguyên; review định tính từ trao đổi: chủ thể ở 8/8 mẫu, frame 29 (1.952067 s) detection phụ nhầm đồ treo góc trái; ghi riêng manual_observations.json trong outputs/P001_SE02_V001_association_notes_20260928/, không ground truth/toàn-video.
- 28/09: P001_SE02_V001 PT640 204/204 frame, 214 detection, 0 zero/10 multi, 0 lỗi timestamp; target: 1 seeded + 203 matched, 0 unresolved. Review assistant trên sheet tải lên trong trao đổi (Codex chỉ ghi nhận): 12/12 khung vàng chọn người ngồi, không thấy đổi mục tiêu; D1 ở 2,5,28,29,51,52,57,61,67,121 nhầm đồ treo góc trái. Pose hợp lý ở mức sheet, chi chồng lấp hạn chế đánh giá khớp; gate không phải accuracy, không ground truth/toàn-video. docs/SITTING_TARGET_REVIEW_20260928.md; annotation/split pending; M2 IN_PROGRESS.
- 29/09: P001_SE03_V001 giữ cùng nhóm buổi SE02 (đổi góc); import/hash nguyên vẹn. PT640 588/588 frame, 588 detection, 0 zero/multi/lỗi PTS; seed 0/D0 đã xem, ngưỡng cũ: 1 seeded + 587 matched, 0 unresolved. outputs/P001_SE03_V001_{poses_pt640,target,association_notes}_20260929/; docs/LYING_TARGET_REVIEW_20260929.md. Assistant review sheet tải lên: 8/8 khung vàng chọn người nằm, không thấy đổi mục tiêu trong mẫu; khớp tay/thân chưa chắc. Codex chỉ ghi nhận; frame 419 toàn độ phân giải chưa được assistant review; không ground truth/xác thực danh tính toàn video/accuracy khớp. Nhãn/split pending; M2 IN_PROGRESS.
- 05/10: P001_SE04_V001 buổi mới/đổi góc; giữ nhóm SE02/SE03. PT640 complete 439/439 frame, 440 detection, 0 zero/1 multi (66); index/PTS toàn bộ khớp, 0 lỗi timestamp; 1 seeded+438 matched, 0 unresolved/gap. Assistant review sheet 13 mẫu và ảnh66 full-size: khung vàng chọn chủ thể 13/13; D1 ở66 (4.448100 s) nhầm nền tối mép phải, D0 là chủ thể; chồng chi/sát biên hạn chế review khớp, không ground truth toàn video. Ghi riêng outputs/P001_SE04_V001_association_notes_20261005/; docs/MOVING_TARGET_REVIEW_20261005.md. Mapping80=5.392133 s/81=5.456133 s đã xác minh; nguyên nhân báo cáo khác chưa xác nhận. Nhãn/split pending; M2 IN_PROGRESS.
