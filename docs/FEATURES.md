# Per-frame features v001 — saved target poses only

Phạm vi: hình học2D và sai phân lùi từ `target_poses.jsonl`, không inference, nhãn hoạt động,
cửa sổ, smoothing, interpolation hoặc training. Đây là chuẩn bị feature, không hoàn tất M2/M4.
PROJECT_PLAN yêu cầu giữ dịch chuyển trong ảnh và tính tốc độ theo timestamp thực; chưa có
đặc tả feature riêng trước nhiệm vụ này. Công thức và quy ước bên dưới là `per_frame_v001`.

## CLI và đầu ra

Trong `D:\patient-monitoring-pi4`, dùng `.venv` hiện có:

```powershell
.\.venv\Scripts\python.exe -m patient_monitoring extract-features outputs/P001_SE01_V001_target_20261005/target_poses.jsonl --video-id P001_SE01_V001 --output-dir outputs/P001_SE01_V001_features_v001_20261007 --max-time-gap 0.5
```

Lệnh trên đã chạy; khi chạy lại phải chọn **thư mục mới**. `--video-id` bắt buộc vì schema target
hiện chưa lưu ID này trong từng dòng. Caller phải dùng ID đã xác minh; không đoán ID từ tên file.
Nếu input có `video_id` thì phải trùng tham số. Không đổi hoặc nối video/nguồn vào cùng chuỗi.

- `features.jsonl`: một dòng cho mỗi frame đầu vào, kể cả target null/unresolved.
  Giữ nguyên frame_index, pts, time_base, presentation_time_seconds, relative_time_seconds,
  delta_seconds, source_path, stream_index, width/height; thêm video_id và feature_version.
  Giữ target status/reason và detection index nguồn (chỉ cục bộ frame, không phải tracking ID).
- Mỗi feature có `{value, valid, reason}`: lỗi/thiếu thì value=null, valid=false và lý do.
  Giá trị0 hợp lệ vẫn là0, ví dụ góc thân thẳng; không thay dữ liệu thiếu bằng0.
- `joints`:17 slot theo COCO index/name, `pixel_xy`, `body_xy`, `source_quality` sao chép các cờ
  finite_components, finite_values, confidence_pass, at_or_outside_boundary, passes_quality_gate.
  Không sửa raw predictions; thiếu joint/mask thì invalid, không tạo tọa độ thay thế.
- `summary.json`: số dòng, số feature hợp lệ, lý do invalid, số chuẩn hóa hợp lệ từng joint,
  số frame đủ17joint và số frame có toàn bộ9feature hình học. Mẫu số là **mọi dòng đã ghi**,
  kể cả missing/unresolved; mỗi joint có1slot/frame. Đây không phải accuracy.
- `provenance.json`: phiên bản feature, ID/nguồn/hash input/output/mã/helper, cấu hình,
  Python/project/package versions, trạng thái complete/incomplete; không đọc model hay nhãn.
  Lỗi parse/schema/disk hoặc input thay đổi giữ run incomplete; lỗi JSON không phải frame0detection.
  Dòng có timestamp/joint/target thiếu vẫn được xuất, không bị lọc bỏ.

## Quality và công thức hình học

Chỉ dùng joint có **`passes_quality_gate is True` đã lưu** và x/y/confidence hữu hạn.
Không tính lại hoặc đổi threshold0.5/boundary gate của extraction. Kiểm tra hữu hạn bảo vệ
schema mâu thuẫn, không phải confidence threshold mới. Facial joint thiếu không loại toàn người.
Trục ảnh: x sang phải, y xuống dưới, tọa độ pixel gốc; left/right theo tên khớp mô hình.

Gọi LHip/RHip là hai hông, LShoulder/RShoulder là hai vai:

- `hip_midpoint_px` H=(LHip+RHip)/2, cần cả hai hông hợp lệ.
- `shoulder_midpoint_px` S=(LShoulder+RShoulder)/2, cần cả hai vai hợp lệ.
- `torso_scale_px` s=||S-H||₂, đơn vị pixel. Thiếu midpoint hoặc s≤1e-9px thì null cùng lý do.
  1e-9px chỉ là bảo vệ số học cho đoạn suy biến, không phải ngưỡng chất lượng đã hiệu chuẩn.
- Joint chuẩn hóa `body_xy`=(joint-H)/s, không có đơn vị; **không xoay** hệ tọa độ.
  Một joint phải qua mask và scale phải hợp lệ. Scale lỗi làm mọi body_xy null, nhưng các
  feature khác như vị trí hông trong ảnh vẫn có thể hợp lệ độc lập.
- Góc gối trái/phải: u=hip-knee, v=ankle-knee;
  theta=acos(clamp((u/||u||)·(v/||v||),-1,1))·180/pi.
  Cần3joint tương ứng hợp lệ, mỗi đoạn dài>1e-9px. **Đơn vị độ, miền0–180; chân thẳng=180°**.
  Clamp chỉ áp dụng cosine do sai số số học, không clamp tọa độ gốc.
- `torso_inclination_deg`: d=S-H; atan2(|dx|,-dy)·180/pi.
  Độ nghiêng **không dấu so với hướng lên của ảnh**:0° thẳng hướng lên,90° ngang,
  180° đảo hướng xuống. Không phải góc so với trọng lực đã hiệu chuẩn.
- `hip_image_xy`=(Hx/W,Hy/H); giữ riêng với pose chuẩn hóa thân.
- `box_size_px`=(x2-x1,y2-y1); `box_size_image`=((x2-x1)/W,(y2-y1)/H).
  Dùng box target đã chọn, kiểm tra hữu hạn/thứ tự/bounds bằng validator hiện có;
  không áp ngưỡng confidence box mới. Thiếu hông không làm box-size tự mất.

## Sai phân lùi và điểm ngắt

Hai feature `hip_image_velocity_per_second`, `box_size_image_velocity_per_second` dùng:

`(q_current - q_previous) / dt`, với `dt=PTS_current*time_base_current - PTS_previous*time_base_previous`.

Mỗi vector có2thành phần. Đơn vị: phần chiều rộng/chiều cao ảnh mỗi giây; dấu giữ nguyên.
Đây là tốc độ thay đổi tọa độ/kích thước ảnh, **không phải mét/giây**, không phải nhãn moving.
Tính dt bằng Fraction chính xác rồi đổi sang float cho phép chia; không dùng FPS, thời gian inference
hay trường delta_seconds làm mẫu số. Trường delta_seconds gốc vẫn được bảo toàn để truy nguồn.

Chỉ lấy **dòng trước liền kề**, không tìm ngược tới dòng hợp lệ gần nhất. Sai phân null nếu:

- Frame index không liên tiếp hoặc video nguồn/stream/kích thước ảnh thay đổi.
- Một trong hai target không phải seeded/matched có selected_person, hoặc association_ended.
- Frame hiện tại là seed mới, có track_break rõ ràng, track_id/association_segment_id được cung cấp
  thay đổi, hoặc reference_frame_index không trỏ tới frame trước. Không dùng detection_index làm identity.
- PTS/time_base thiếu, timestamp_valid nguồn false, presentation_time không khớp PTS (sai số số học1e-9s),
  timestamp trùng/giảm so high-water của nguồn, hoặc timestamp của frame trước không hợp lệ.
- dt>**max_time_gap**, mặc định thử nghiệm0.5s; dt=0.5s được phép. Tham số CLI cấu hình được;
  độc lập với max-gap association, không sửa dự đoán/ngưỡng liên kết đã lưu.
- Feature q tương ứng của một trong hai frame không hợp lệ. Thiếu hip chỉ chặn sai phân hip;
  box vẫn có thể tính nếu đủ điều kiện. Không nối qua một frame có joint cần thiết bị mất.

Frame đầu không có sai phân. Sau một điểm ngắt, frame hợp lệ hiện tại có thể làm đầu chuỗi mới
cho frame kế tiếp; tuyệt đối không tạo sai phân bắc qua điểm ngắt. Bộ nhớ chỉ giữ dòng trước.
`temporal_pair` ghi tính hợp lệ liên kết, lý do, index trước và dt khi tính được; feature còn có gate riêng.
Feature tĩnh hợp lệ không đồng nghĩa timestamp/đạo hàm hợp lệ. Không thể phát hiện đổi người **không
được đánh dấu** chỉ từ dữ liệu này; IoU association vẫn là baseline thử nghiệm.

## Các lần chạy thực tế và kiểm tra

Mỗi hàng có `features.jsonl`, `summary.json`, `provenance.json` trong thư mục output ghi dưới:

| Video | Input target directory | Output directory |
|---|---|---|
| P001_SE01_V001 | outputs/P001_SE01_V001_target_20261005 | outputs/P001_SE01_V001_features_v001_20261007 |
| P001_SE02_V001 | outputs/P001_SE02_V001_target_20260928 | outputs/P001_SE02_V001_features_v001_20261007 |
| P001_SE03_V001 | outputs/P001_SE03_V001_target_20260929 | outputs/P001_SE03_V001_features_v001_20261007 |
| P001_SE04_V001 | outputs/P001_SE04_V001_target_20261005 | outputs/P001_SE04_V001_features_v001_20261007 |

| Video | Dòng | Hông/vai/scale/gối trái/gối phải/nghiêng thân/box: mỗi feature | Sai phân hông/box: mỗi feature | Đủ17joint chuẩn hóa |
|---|---:|---:|---:|---:|
| SE01 | 284 | 284 | 283 | 284 |
| SE02 | 204 | 204 | 203 | 0 |
| SE03 | 588 | 588 | 587 | 51 |
| SE04 | 439 | 439 | 438 | 235 |

Tất cả4run complete. Các khớp bị gate loại vẫn null trong feature; counts cao của feature thân
không chứng minh khớp đúng hoặc người được chọn đúng toàn video. Có thể tính feature thân dù
không đủ17joint, như SE02. Sai phân frame đầu mỗi video bị loại đúng quy tắc.

Kiểm tra thực:21test mới và toàn bộ106test unittest PASS; `pip check` PASS; doctor Python3.10.11,
.venv đúng dự án. Đối chiếu toàn bộ1515dòng với target/timeline, tất cả source quality flags,
chuẩn hóa và sai phân theo dt thật; hash453file được bảo vệ (outputs cũ/data/models/PLAN/pyproject) không đổi.
Báo cáo: `outputs/feature_extraction_v001_20261007/verification.json`.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_pose_features.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

## Giới hạn

Phối cảnh, góc camera/roll, che khuất và khoảng cách làm thay đổi góc2D, scale và box dù hoạt động
không đổi. Scale thân ngắn vẫn có thể hợp lệ số học nhưng khuếch đại nhiễu. Pose jitter bị khuếch đại
bởi phép chia dt; không smoothing hoặc hiệu chuẩn3D. Chất lượng mask/confidence không phải accuracy.
Chưa có nhãn được duyệt cho việc học. Không đọc/sửa annotation v001/v002, không tạo cửa sổ/split;
các loại trừ temporal annotation vẫn phải được tôn trọng khi có nhiệm vụ huấn luyện trong tương lai.
