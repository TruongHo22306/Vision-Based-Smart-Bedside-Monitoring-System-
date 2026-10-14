# Xem baseline v002 và chuẩn bị buổi quay mới

## Bản review phát sinh

Chỉ dùng video nguồn, `target_poses.jsonl` và `predictions.jsonl` v002 đã lưu; không inference
hay tính lại nhãn. Mỗi video có skeleton theo **gate đã lưu** và khung vàng của target đã chọn.
Khớp không qua gate không được nối/vẽ thay thế. Đây không phải xác nhận joint accuracy.

Dải chữ được thêm **bên dưới** vùng ảnh, không cắt đầu/chân hoặc thu nhỏ ảnh gốc:

- `EXPERIMENTAL v002 PREDICTIONS - NOT GROUND TRUTH` luôn hiển thị.
- Video ID, frame index, source timestamp sáu chữ số thập phân và PTS/time_base gốc.
- Nhãn dự đoán, bằng chứng tư thế, lý do ngắn; giữ rõ cả nhãn `unknown`.
- Dòng riêng cho image motion và cảnh báo, kể cả khi nhãn vẫn là sitting/lying.
- Locomotion `candidate`/`unknown` không được diễn giải thành bước chân đã xác minh.

Bản xem video-only H.264,1280×900 (vùng nguồn1280×720 + footer180), CRF18/presetfast,
B-frames0, PyAV/libx264 hiện có. Không đổi dependency. Âm thanh không chép vào bản review;
nguồn vẫn giữ nguyên âm thanh và byte. Hình được encode lại nên không phải bản sao pixel lossless.

Giữ từng PTS,time_base,duration gốc; không đổi FPS, resample, bỏ/nhân frame hoặc rebase PTS.
Cách encode theo hồ sơ đã xác minh ở `SE04_focused_playback_20261006/review_manifest.json`
và `temporal_annotation_prep_20261005/P001_SE04_V001/review_copy_info.json`. Không tìm thấy
script encode cũ được lưu trong project; module mới tái hiện cách đó và tự giải mã lại để kiểm.

Mỗi thư mục sau có `review.mp4`, `review_manifest.json`, `frame_mapping.csv` và PNG kiểm bố cục:

| Video | Thư mục mới |
|---|---|
| SE01 | `outputs/P001_SE01_V001_action_review_v002_20261012/` |
| SE02 | `outputs/P001_SE02_V001_action_review_v002_20261012/` |
| SE03 | `outputs/P001_SE03_V001_action_review_v002_20261012/` |
| SE04 | `outputs/P001_SE04_V001_action_review_v002_20261012/` |

Kiểm tra từng dòng giữa decode nguồn, prediction và target gồm index, PTS/time_base, timestamp,
delta, đường dẫn nguồn, stream, kích thước, trạng thái và detection index đã chọn. Sau encode,
decode review để đối chiếu **tất cả PTS và duration, kể cả frame cuối**. Manifest chỉ ghi complete
khi kiểm qua; lỗi giữ incomplete và lý do. Chương trình từ chối ghi vào thư mục đã tồn tại.

Lệnh PowerShell tại project (ví dụ đã chạy; chọn thư mục mới nếu lặp lại):

```powershell
.\.venv\Scripts\python.exe -m patient_monitoring.action_review data/raw/P001/P001_SE01/P001_SE01_V001.mp4 --predictions outputs/P001_SE01_V001_action_baseline_v002_20261012/predictions.jsonl --targets outputs/P001_SE01_V001_target_20261005/target_poses.jsonl --output-dir outputs/P001_SE01_V001_action_review_v002_20261012
```

Xem ở1×; khi báo sai lệch ghi video ID, frame/PTS trên footer, nhãn và dòng cảnh báo.
Thanh playback có thể làm tròn giây. Việc decode/check PNG tự động không có nghĩa assistant
đã xem chuyển động liên tục hoặc xác nhận nhãn toàn video. Human playback review vẫn pending.

Kiểm tra thực tế ngày12/10: cả bốn export complete; giải mã lại **1.515/1.515frame**, mọi PTS,
time_base và duration khớp nguồn. Thời điểm kết thúc tính từ PTS+duration frame cuối:

| Video | Frame nguồn/prediction/target/output khớp | End theo frame (giây) |
|---|---:|---:|
| SE01 | 284 | 19.105866667 |
| SE02 | 204 | 13.729700000 |
| SE03 | 588 | 19.857833333 |
| SE04 | 439 | 29.570100000 |

End này không lấy từ duration container; metadata stream có thể lệch một tick so với frame cuối.
Đã xem ảnh bố cục giải mã từ cả bốn output, có SE03frame24 hiển thị lying cùng cảnh báo motion
riêng. Đây là kiểm bố cục bằng ảnh, không phải review toàn bộ chuyển động. 27test liên quan PASS
(4test exporter mới và23test baseline);546file được bảo vệ giữ nguyên SHA-256. Báo cáo kiểm tra
trong `outputs/baseline_v002_review_audit_20261012/verification.json`.

## Snapshot trước dữ liệu mới

Snapshot tại `outputs/baseline_v002_review_audit_20261012/v002_config_snapshot.json` ghi đầy đủ
config, mã nguồn, provenance của cả bốn run và timestamp lúc ghi. Hash cấu hình canonical JSON:

`ec8a41657c61207951957584fac50a9b6c68f29a7c25ae0716abaedc2068bbbc`

Mã v002: `beba80e800076df6cb8b27d568e81d981c016cc0f84d4c6e36703bd941f15c90`.
Snapshot được tạo **trước review dữ liệu quay mới**; chưa có video mới nào được nhập trong task.
Giữ cửa sổ1s/coverage80%, mọi threshold và decision policy; không sửa bộ phân loại.

**SE01–SE04 là development clips đã dùng để chẩn đoán**, không phải bộ test độc lập. pilot01
cũng là development-only. Buổi quay mới tạo thêm bằng chứng nhưng không tự trở thành test set.
Nếu dùng clip mới để chỉnh ngưỡng/luật, chúng cũng trở thành development data; cần dữ liệu khác
để đánh giá độc lập sau đó. Không random-split frame; giữ cùng nhóm buổi/nguồn và subject theo
protocol. SE02/SE03 vẫn cùng recording occasion. Task này không gán hoặc khóa split.

## Checklist một recording occasion MỚI — chưa có bản ghi hoàn thành

Trước quay:

- [ ] Xác nhận người tham gia và đồng ý. Cùng người giữ subject_id cũ (nếu là bạn đã xác nhận,
  dùng P001); người khác cần mã riêng, không suy từ tên file.
- [ ] Ghi buổi quay mới thật: ngày/giờ và múi giờ, subject_id, session_id theo `<subject>_SEnn`,
  recording-occasion grouping. Chỉ cấp số còn trống khi có thông tin thật; chưa cấp sẵn ID ở đây.
- [ ] Camera cố định trong cả buổi; thấy trọn đầu, tay, gối và bàn chân, có khoảng trống ở mép;
  kiểm cả đường đi, ghế và giường. Nếu đổi góc/bố trí đáng kể, ghi session mới nhưng giữ liên hệ
  cùng buổi quay. Tránh tiến quá gần làm mất đầu/chân.
- [ ] Ghi điều kiện **thực tế**: camera/thiết bị, vị trí/góc/độ cao/khoảng cách nếu biết, ánh sáng,
  nền, quần áo, độ phân giải và FPS yêu cầu. FPS/thời lượng/frame count đo sau khi có file;
  chưa đo thì để trống, không điền giá trị dự kiến.

Quay **bảy clip riêng, mỗi clip khoảng10–20giây**, giữ camera/bố trí; đây là nội dung dự định,
không tự xác nhận nhãn temporal:

| Clip dự định | Việc thực hiện |
|---|---|
| Đứng yên | Đứng tự nhiên, không đi, giữ toàn thân trong ảnh. |
| Đứng cử động tay/đổi dáng | Vẫy/nâng tay nhẹ hoặc đổi trọng tâm; không bước đi. Ghi lại nếu vô tình bước. |
| Đi gần/xa camera | Đi về phía camera rồi đi xa, trong vùng toàn thân nhìn thấy; giữ cả đoạn quay người. |
| Đi ngang | Đi từ trái sang phải hoặc ngược lại, không di chuyển camera. |
| Đi–dừng–đi tiếp | Đi vài bước, dừng đứng rõ vài giây rồi khởi động đi lại; mốc thật xác định khi review. |
| Ngồi cử động tay | Ngồi ổn định trên ghế/giường, cử động tay tự nhiên; giữ hông, gối, bàn chân nhìn thấy. |
| Nằm điều chỉnh nhẹ | Nằm trên giường, tự điều chỉnh tay/chân hoặc dáng nằm nhẹ; không thu ngã/co giật. |

Sau quay:

- [ ] Giữ video gốc, không cắt/transcode/đổi FPS; ghi tên/đường dẫn gốc và điều kiện từng clip.
- [ ] Chỉ sau khi file thật tồn tại mới nhập không ghi đè, cấp video_id `<session>_Vnnn`,
  kiểm hash và thêm manifest. Không tạo hàng giả từ checklist này.
- [ ] `planned_activity` chỉ là bối cảnh. Ghi full-body/occlusion thực tế, cử động ngoài dự định,
  thay đổi camera và uncertain intervals; để nhãn cuối và split pending.
- [ ] Gán nhãn độc lập từ **video gốc không có nhãn dự đoán**, giữ timestamp thật và ghi nguồn
  review, rồi mới so với v002 đã snapshot để giảm thiên lệch. Không lấy confidence làm ground truth.

Theo [protocol hiện hữu](DATASET_RECORDING_PROTOCOL.md); không sửa template/annotation hiện tại.
Cùng một người ở buổi mới không chứng minh khả năng tổng quát sang người chưa gặp.
