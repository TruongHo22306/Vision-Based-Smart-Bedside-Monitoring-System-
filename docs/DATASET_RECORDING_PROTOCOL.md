# M3 — Protocol quay và gán nhãn dữ liệu mới, bản 0.1

## Phạm vi và trạng thái

Theo PROJECT_PLAN: M1 kiểm tra video mới; M3 quản lý người/phiên/video, nhãn theo thời gian và
khóa train/validation/test theo nhóm trước huấn luyện. **Protocol và mẫu trống đã chuẩn bị;
chưa có video mới, nhãn đã duyệt hoặc bộ chia tập được khóa trong nhiệm vụ này.** M2 vẫn IN_PROGRESS.
PT640 chỉ là tham chiếu offline tạm thời; xem [review định tính](POSE_VISUAL_REVIEW_20260923.md).

Thu bốn hoạt động `standing`, `sitting`, `lying`, `moving`; `moving` ban đầu nghĩa là đi lại
trong vùng quan sát. Quay các chuyển tiếp sinh hoạt thường ngày thành video riêng.
Ngã là **sự kiện theo thời gian**, tách khỏi nhãn hoạt động. Chưa hướng dẫn/thu ngã ở đây;
quy ước sự kiện và protocol thu thuộc M5. Chuyển động nghi co giật thuộc M8, chưa triển khai.

## Bố trí và đợt quay đầu

1. Chỉ ghi người đồng ý tham gia; dùng mã thay tên thật trong dữ liệu làm việc.
2. Cố định webcam, không đổi vị trí/zoom giữa đứng, ngồi, nằm và đi trong cùng phiên.
   Khung hình cần bao được người đứng, ghế, giường và đường đi; giữ đầu, bàn tay, đầu gối và
   hai bàn chân trong ảnh, có khoảng trống ở mép. Khi nằm, phải thấy trọn chân; tránh giường/ghế che khớp.
3. Kiểm tra ánh sáng và hình gốc bằng mắt trước quay. Ghi điều kiện camera, nền, góc nhìn,
   khoảng cách và độ phân giải thật; FPS yêu cầu không thay thế FPS/timeline đo được.
4. Bắt đầu bằng **một clip đứng 8–15 giây** của phiên mới để kiểm tra toàn thân/timeline.
   Giữ vài giây tư thế ổn định; không cố tạo chuyển động ngã. Giữ nguyên phần đầu/cuối video gốc.
5. Sau khi clip đầu đạt kiểm tra bố cục và thời gian, quay lần lượt một clip ngồi, nằm và đi lại,
   mỗi clip khoảng 8–15 giây, cùng camera. Tư thế ngồi có thể hơi nghiêng để nhìn rõ gối/cổ chân.
   Đây là bốn clip kiểm tra quy trình dự kiến, chưa phải bộ dữ liệu đủ để đánh giá mô hình.
6. Sau bốn clip đầu, có thể quay riêng chuyển tiếp đứng→ngồi và ngồi→đứng tự nhiên;
   giữ đoạn ổn định trước/sau. Đánh dấu `recording_kind=transition`, không gán một nhãn ổn định cho toàn video.

Theo kế hoạch, sau kiểm tra có thể mở rộng nhóm thăm dò khoảng 20 clip (5 mỗi hoạt động).
M3 hướng tới 5–8 người và ít nhất hai phiên khác nhau nếu tổ chức được; đây là mục tiêu lập lịch,
không phải dữ liệu hiện có hoặc điều kiện đã đạt. Điều chỉnh quy mô theo lỗi/độ đa dạng, không chỉ số frame.

## Định danh, bản gốc và manifest

- `subject_id`: mã không chứa danh tính, dạng `Pnnn`; cùng một người luôn giữ cùng mã qua mọi phiên.
- `session_id`: duy nhất toàn dự án, dạng `<subject_id>_SEnn`; một buổi/cấu hình camera ổn định.
  Bắt đầu phiên mới khi đổi buổi, góc camera hoặc bố trí đáng kể; không đổi mã người để tạo người “mới”.
- `video_id`: `<session_id>_Vnnn`, duy nhất và không đổi. Số thứ tự chỉ cấp khi thật sự ghi nhận video.
  Ví dụ cú pháp `P001_SE01_V001` **chỉ minh họa, chưa cấp ID hoặc xác nhận có bản ghi**.
- Lưu video nguyên bản tại `data/raw/<subject_id>/<session_id>/<video_id>.<đuôi gốc>`.
  Nếu đổi tên bản sao nhập dự án, giữ `original_filename`, đường dẫn nguồn và SHA-256; byte phải như bản gốc.
  Không cắt/nối/transcode, thay FPS metadata hoặc ghi đè. Bản phát sinh sau này phải trỏ về `recording_group_id` gốc.
- `recording_group_id`: ID video quay liên tục ban đầu; các đoạn cắt/bản sao/cửa sổ cùng nguồn giữ ID này.
  Mỗi video mới độc lập dùng chính `video_id`; không ghép các bản cắt thành các “recording” độc lập để chia tập.
- Một hàng trong [dataset_manifest_template.csv](dataset_manifest_template.csv) cho mỗi video thật.
  `planned_activity` là nội dung dự định quay, **không phải nhãn đã xác minh**; transition có thể để trống trường này.
- `recorded_at_utc`: ISO 8601 UTC nếu biết thời điểm quay thật; nếu chưa biết để trống, ghi vào `notes`.
  Không lấy giờ sửa file làm giờ quay. Đường dẫn dùng tương đối trong dự án; lưu `source_description`, `sha256`
  để truy nguồn. Trường chưa đo để trống, không điền 0 hoặc số dự kiến thay số đo.
- `duration_sec`, `width`, `height`, `decoded_frame_count`, `first_pts`, `time_base` điền sau kiểm tra video thực.
  Ghi `timestamp_status=unchecked|valid|needs_review`; timestamp thiếu/trùng/giảm phải review, không suy từ FPS.
- `consent_status=pending|confirmed|not_permitted`; chỉ quay/đưa vào sử dụng khi được phép.
  `recording_kind=activity|transition`; `dataset_role=unassigned|development_only|evaluation_candidate`.
  `annotation_status=unannotated|draft|needs_review|reviewed`; `qc_status=unchecked|pass|needs_review`.
- `camera_notes`, `scene_notes`, `full_body_notes`, `occlusion_notes` ghi điều kiện thật, vùng bị cắt/che và đoạn
  thời gian nếu thấy; không suy từ confidence. Clip có lỗi được giữ, đánh dấu lý do, không âm thầm loại bỏ.

## Gán nhãn theo đoạn thời gian

Xem **video gốc**, không dùng pose, confidence hoặc nội dung dự định quay làm nhãn mặc định.
Dùng [segment_annotations_template.csv](segment_annotations_template.csv), nhiều hàng có thể trỏ cùng video.
`segment_id=<video_id>_A<nnn>` duy nhất; `annotator_id`/`reviewer_id` là mã người gán/kiểm tra nhãn thật.
Để trống reviewer/date khi chưa duyệt; không tự điền đã reviewed.

- `start_sec`/`end_sec`: giây tương đối từ **PTS hợp lệ đầu tiên** của video, khoảng nửa mở `[start,end)`;
  `start_frame_index`/`end_frame_index_exclusive` là index decode 0-based, end là frame đầu không thuộc đoạn.
  Mốc đầu và ranh giới nội bộ lấy từ timeline PTS thực; không lấy `frame_index/30` hay thời gian chạy inference.
  Ở cuối video, end index bằng decoded count; end time dùng PTS frame cuối cộng duration frame nếu có/đã xác minh.
  Nếu chưa xác định được end time/PTS, để trống và `needs_review`; không đoán thời lượng frame cuối.
- Các đoạn hoạt động đã duyệt không chồng nhau trên cùng video một người. Bao phủ thời gian có thể gán nhãn,
  kể cả `unknown`/`no_person`; đoạn chưa làm không được xem là nhãn mặc định hoặc dữ liệu âm.

| `activity_label` | Quy ước bắt đầu/kết thúc theo quan sát |
|---|---|
| `standing` | Đứng trên chân, không đi dịch chuyển; bắt đầu khi đã đứng ổn định, kết thúc khi bắt đầu đi hoặc đổi tư thế. Cử động tay không tự thành moving. |
| `sitting` | Cơ thể ở tư thế ngồi, vùng chậu được ghế/giường đỡ; bắt đầu khi ngồi ổn định, kết thúc khi bắt đầu rời tư thế. |
| `lying` | Cơ thể nằm được bề mặt đỡ; bắt đầu khi đã nằm ổn định, kết thúc khi bắt đầu ngồi/dậy. Nằm không tự tạo nhãn ngã. |
| `moving` | Đi lại có bước chân và dịch chuyển trong vùng camera; từ bước đi đầu đến khi ngừng đi. Đoạn dừng đứng được tách; không đồng nhất di chuyển ảnh do camera với đi. |
| `unknown` | Người hiện diện nhưng hoạt động hoặc ranh giới không thể xác định chắc; lưu `uncertainty_reason`, không ép vào một trong bốn lớp. |
| `no_person` | Không thấy người trong ảnh theo review nguồn; không suy từ việc model trả zero detection. |

- `segment_kind=activity|transition|uncertain|no_person`. Transition là từ frame đầu rời tư thế ổn định
  đến frame đầu đạt tư thế mới; đặt `activity_label=unknown`, ghi `transition_from`/`transition_to` khi xác định được.
  Tách các đoạn ổn định trước/sau thành hàng riêng. Transition không thành lớp thứ năm của bộ phân loại hiện tại.
- Nếu ranh giới mơ hồ, đặt một đoạn `uncertain` bao vùng đó thay vì giả độ chính xác từng frame;
  `label_certainty=clear|uncertain`, `candidate_labels` ngăn bởi `|` nếu có, và lý do ngắn.
- `full_body=yes|no|uncertain`; `occlusion=none|partial|severe|uncertain`; `affected_regions` liệt kê vùng/khớp,
  ghi chi tiết trong `notes`. Partial-body không tự làm mất nhãn hoạt động nếu nguồn vẫn đủ rõ;
  trường hợp không đủ bằng chứng phải unknown. Tách đoạn khi mức che khuất thay đổi đáng kể.
- Hoạt động và sự kiện là hai kênh riêng. Đợt này dùng `event_annotation_status=deferred`, `event_id` để trống.
  Không suy “không ngã” từ event_id trống. Schema/mốc sự kiện và hướng dẫn quay ngã để M5 xác định.
- Chỉ đổi `annotation_status` thành `reviewed` sau xem lại video, sửa ranh giới và ghi reviewer/date thực.
  Nếu tự gán rồi tự kiểm, ghi cùng mã và nêu rõ self-review; không gọi là đồng thuận hai người.

## Chia tập và chống rò rỉ

1. **pilot01 chỉ dùng phát triển**: đã dùng để kiểm tra/tinh chỉnh pipeline, xem pose và thử backend.
   Video, frame, bản cắt, pose, cửa sổ và biến thể của nó không được vào validation/test khóa để báo cáo đánh giá.
   Không tạo ID mới cho pilot01 để đổi vai trò; truy nguồn bằng hash/recording_group_id.
2. Không random-split frame hoặc cửa sổ. Toàn bộ một recording và các bản phát sinh thuộc cùng tập;
   mọi video của một session cùng nhóm. Chia nhóm **trước** khi cắt đoạn/cửa sổ hoặc tăng cường dữ liệu.
3. Khi đủ người, nhóm ưu tiên là `subject_id`: mọi phiên của cùng người chỉ vào một tập train/validation/test.
   Khi chỉ có một người/thiếu nhóm, chưa thể đánh giá người mới. Nếu sau này dùng session-disjoint để thăm dò,
   phải ghi rõ đánh giá người đã biết; cùng session không xuất hiện ở nhiều tập. Một phiên duy nhất chưa đủ chia ba tập.
4. Chuẩn bị [split_groups_template.csv](split_groups_template.csv) nhưng **chưa gán tập/khóa split**.
   `group_type=subject|session|recording`, `group_id` là ID thật; một chính sách chia thống nhất trong một dataset version.
   `split=unassigned|development_only|train|validation|test`, `lock_status=draft|locked`; ngày/reviewer trống đến khi khóa.
   `group_type=recording` chỉ dùng ghi ngoại lệ development-only như pilot, không dùng để lách nhóm subject/session.
5. Trước huấn luyện M4: kiểm tra ID/hash nguồn không giao tập, độ phủ lớp/người/phiên, nhãn đã review;
   lưu danh sách nhóm, dataset version, quy tắc/seed nếu dùng ngẫu nhiên ở **cấp nhóm**, và thời điểm khóa.
   Chưa có tỷ lệ, nhóm, seed hay mục tiêu định lượng nào được chốt ở task này.
6. Chỉ dùng train/validation để chọn ngưỡng/tham số; fit chuẩn hóa trên train, không dùng test cho lựa chọn.
   Test chỉ mở khi cấu hình đã khóa. Các trường hợp unknown/che khuất/không dự đoán được phải được thống kê riêng.

## Checklist đầu tiên — chưa đánh dấu hoàn thành

- [ ] Người tham gia đồng ý; cấp subject/session/video ID mới, không liên hệ dữ liệu dự án khác.
- [ ] Camera cố định nhìn trọn vùng đứng, ghế, giường; thử đầu/chân không chạm mép, ghi bố cục/ánh sáng.
- [ ] Quay **một video đứng 8–15 giây**, thấy toàn thân; giữ file gốc chưa chỉnh sửa.
- [ ] Nhập bản gốc vào đường dẫn theo ID, tính SHA-256; thêm một hàng manifest sau khi có file thật.
- [ ] Kiểm tra phát video, PTS/thời lượng, full-body/occlusion; chưa đạt thì ghi lý do và giữ bản gốc.
- [ ] Chỉ sau kiểm tra đầu tiên mới quay ba clip ngồi/nằm/đi; gán nhãn nguồn và chuyển tiếp riêng sau review.

Ba template hiện chỉ có header, **không có dòng dữ liệu/nhãn/split giả**. Khi bắt đầu nhập liệu,
sao chép template thành các file quản lý có dataset version; không ghi dữ liệu vào chính template.
