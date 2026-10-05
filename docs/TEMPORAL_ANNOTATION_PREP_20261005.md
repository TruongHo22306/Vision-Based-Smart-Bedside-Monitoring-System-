# Chuẩn bị annotation thời gian — chưa xác nhận nhãn

Đã đọc protocol, template và bốn manifest nhập thật; chỉ đọc đường dẫn nguồn ngoài
project từ provenance đã lưu, không truy cập dự án cũ hay video nguồn ngoài project.
Chi tiết nguồn, hash, timeline, pose/target, grouping và bước thiếu:
[INVENTORY.md](../outputs/temporal_annotation_prep_20261005/INVENTORY.md) và inventory.json.

| Video ID | Bối cảnh, chưa là nhãn cuối | Frame đã đối chiếu timeline | Pose preview | Full pose / target |
|---|---|---:|---:|---|
| P001_SE01_V001 | standing dự định | 284 | 8 | Chưa có cả hai |
| P001_SE02_V001 | sitting dự định | 204 | 8 | Complete / 204 selected |
| P001_SE03_V001 | lying dự định | 588 | 8 | Complete / 588 selected |
| P001_SE04_V001 | đi gần/xa và xoay, quan sát cuộc trao đổi | 439 | 12 | Complete / 439 selected |

Mọi imported source nằm tại `data/raw/P001/<session_id>/<video_id>.mp4`.
Hash cả bốn video khớp manifest; giải mã tuần tự lại cả bốn xác minh mọi index/PTS
với timeline, không timestamp thiếu/trùng/giảm. Các full pose/target hiện hữu được
đối chiếu số dòng/index/PTS. Không giả clip đứng đã có full-video extraction.
Tất cả còn thiếu review temporal liên tục, nhãn thời gian cuối và split đã khóa.
Review mẫu pose trước đó không thay thế các bước này.

## Nhóm buổi quay và chống rò rỉ

- SE01 là session đã cấp cho clip đứng; liên hệ buổi quay với pilot chưa biết.
- **SE02 và SE03 cùng buổi**, đổi góc nên tách session dữ liệu; phải giữ cùng nhóm.
- SE04 là buổi mới theo user; không suy nhóm cùng buổi với các phiên cũ.
- Cả bốn là P001; nhóm subject vẫn được ưu tiên nếu sau này chia tập.
  pilot01 development-only, không đặt lại ID/vai trò để đưa vào evaluation.
- Không tạo split; `split_groups_unassigned.csv` chỉ có header, không khóa nhóm.

## Review P001_SE04_V001

Không có bề mặt công cụ cho Codex quan sát phát video liên tục trong phiên này.
Giải mã tuần tự là kiểm tra kỹ thuật, không phải bằng chứng xem chuyển động.
Đã thực hiện fallback được yêu cầu: [video review có timestamp](../outputs/temporal_annotation_prep_20261005/P001_SE04_V001/P001_SE04_V001_timestamped_review.mp4).

- Giữ toàn bộ 439 frame theo đúng thứ tự; decode lại output xác nhận **mọi PTS và
  duration từng frame giống nguồn**, không resample theo FPS cố định.
- Vùng ảnh 1280×720 giữ nguyên bố cục; thêm dải chữ bên dưới thành 1280×816.
  Encode H.264 bằng PyAV/libx264 có sẵn, video-only; bản gốc/âm thanh không sửa.
- Dải chữ có video ID, index, thời gian tương đối và PTS/time base. Ảnh kiểm bố cục
  không được dùng để xác định ranh giới. Hash/settings/kiểm timing nằm trong
  `P001_SE04_V001/review_copy_info.json` của bộ review.
- Phạm vi kỹ thuật `[0,29.570100)` s, frame `[0,439)`, gồm duration frame cuối
  được decode xác minh; không phải một đoạn hoạt động đề xuất. Duration stream
  metadata=29.570066667 s lệch một tick 1/30000 s, được ghi rõ trong hồ sơ.

## Annotation nháp và bằng chứng còn thiếu

`outputs/temporal_annotation_prep_20261005/<video_id>/annotations_proposed.csv`
cho mỗi video giữ nguyên schema template, **header-only: 0 proposed/0 confirmed**.
`annotation_provenance.json` ghi trạng thái `draft_pending_human_review`, nguồn/hash,
phạm vi cần xem, ranh giới ứng viên rỗng và reviewer chưa có. Toàn bộ phạm vi chưa
review được ghi riêng `unreviewed_unannotated`; không biến thành unknown/moving.
Chưa có căn cứ để đặt walking/stationary/transition intervals hoặc localized uncertain
intervals. Không suy từ confidence, gate, sheet thưa hoặc tên video.

[REVIEW_GUIDE.md](../outputs/temporal_annotation_prep_20261005/REVIEW_GUIDE.md) hướng dẫn
đọc frame/PTS, moving/standing/transition/uncertain theo protocol. Xoay không phải
lớp riêng; ranh giới mơ hồ phải ghi rõ khi review. Không dùng nhãn sự kiện trong bước này.

Không inference, đổi ngưỡng, dependency, weights, PROJECT_PLAN hoặc dữ liệu/kết quả cũ.
M2 vẫn IN_PROGRESS. Bước tiếp theo: user xem liên tục bản review SE04, đánh dấu frame
bắt đầu/dừng đi cùng những vùng chưa chắc; sau đó mới có thể ghi các đoạn draft.
