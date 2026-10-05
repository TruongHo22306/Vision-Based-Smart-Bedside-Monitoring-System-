# Nhập clip nằm — P001_SE03_V001

## Nguồn và ID

- Nguồn: `C:/Users/asus/OneDrive/Pictures/Camera Roll/WIN_20260928_12_15_19_Pro.mp4`.
- Bản sao nguyên byte: `data/raw/P001/P001_SE03/P001_SE03_V001.mp4`, 20.138.718 byte.
- SHA-256 nguồn trước/sau và bản sao: `b49700ec2be82a7ebba4124bb684016bc16373d7f1b678bc238a7203b907f406`.
- Chưa có bản nhập tương ứng; tạo file bằng chế độ độc quyền, không ghi đè hoặc transcode.
- Người dùng xác nhận cùng người P001, cùng buổi với clip ngồi nhưng khác góc camera.
  Theo protocol, đổi góc camera bắt đầu session dữ liệu mới: P001_SE03, video đầu V001.
  Đây không phải người mới hay bằng chứng về buổi quay độc lập. **Giữ P001_SE02 và P001_SE03
  cùng nhóm nếu sau này chia theo phiên**; nhóm subject P001 vẫn được ưu tiên. Chưa gán split.
- Không suy thời điểm UTC/thông số camera từ tên file. `planned_activity=lying` theo yêu cầu
  người dùng, không phải nhãn thời gian đã xác minh; hồ sơ consent riêng còn pending.

## Kiểm tra đã chạy

Đã đọc CLI `--help` trước chạy, dùng Python `.venv` và cấu hình hiện hữu.
Inspect/preview chạy trên nguồn trong lúc chờ xác nhận ID; hash khớp liên kết báo cáo với
bản sao raw. Giữ nguyên đường dẫn nguồn trong báo cáo, không chạy lại chỉ để đổi path.

| Kiểm tra | Kết quả thực tế |
|---|---|
| Inspect | Complete; H.264, 1280×720, **588 frame**, 0 frame hỏng |
| Timestamp | Thiếu/trùng/giảm đều **0**; time base 1/30000; PTS 0 đến 594735 |
| Thời gian | Stream 19.857833333333332 s; container 19.888896 s; PTS span 19.8245 s |
| Preview | 8 mẫu: **0,84,168,252,336,419,503,587** |
| PT640 | Complete 8/8; **8 detection**, mỗi mẫu 1; 0 mẫu zero/multiple detection |
| Đối chiếu | Index/thời gian khớp timeline/CSV/poses; hash PNG khớp; đủ 17 khớp mỗi detection |

YOLO11n-pose local weights M2 có SHA-256
`869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`.
CPU float32 thực tế, imgsz=640, batch=1, rect=False, conf=.25, IoU=.7,
max_det=300, ngưỡng vẽ khớp .5. Không tải weights hoặc thay dependency.
Cả ba CLI exit 0; không lỗi decode/preview/inference. Ultralytics cảnh báo `half`
deprecated; giữ cấu hình và xác minh dtype thực tế. Chỉ inference 8 PNG, không toàn video.

## File và review còn chờ

- `outputs/fresh_20260928_121519_import/`: manifest.csv, provenance.json, verification.json;
  annotations.csv và split_groups.csv chỉ có header.
- `outputs/fresh_20260928_121519_inspect/`: metadata.json, timeline.csv.
- [Contact sheet gốc](../outputs/fresh_20260928_121519_preview/contact_sheet.jpg),
  cùng samples.csv, 8 PNG và báo cáo preview.
- [Contact sheet PT640](../outputs/fresh_20260928_121519_pose_pt640/pose_contact_sheet.jpg),
  cùng poses.json, run_info.json và 8 PNG annotate.

Ảnh đầu ra đọc được; PNG 1280×720; sheet gốc 1536×552, sheet pose 1536×580.
Visual pose review, toàn thân/che khuất, nhãn thời gian cuối và split vẫn pending.
Detection/confidence không phải accuracy; PTS span không gồm thời gian hiển thị frame cuối,
metadata/PTS không chứng minh thời lượng capture ngoài đời. Clip dài hơn gợi ý 8–15 s được giữ nguyên.
Không inference toàn video, train hoặc benchmark Pi. M2 vẫn IN_PROGRESS.
Tiếp theo: xem hai sheet và ghi nhận vị trí khớp/toàn thân/che khuất trước khi gán nhãn.
