# M1.2 — Preview video và thống kê khoảng thời gian frame

Dùng Python 3.10.11 trong `.venv`; giữ PyAV 16.1.0. Chỉ thêm Pillow 12.1.1
để ghi PNG/JPEG và nhãn; wheel `cp310-win_amd64`, Requires-Python `>=3.10`
đã được kiểm tra tại máy. Không cần NumPy/ML.
Nguồn tương thích: [Pillow 12.1.1 trên PyPI](https://pypi.org/project/pillow/12.1.1/).

```powershell
cd D:\patient-monitoring-pi4
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: Pillow==12.1.1
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -e .
.\.venv\Scripts\python.exe -m patient_monitoring preview-video data/raw/pilot01.mp4 --output-dir outputs/pilot01_preview --samples 8
```

`--samples` mặc định 8, nhận 2–8 để có cả hai đầu timeline. Output phải là thư mục
chưa tồn tại. Pilot preview trên đã tạo; khi chạy lại chọn tên output mới.
Không ghi đè nguồn, báo cáo M1.1 hoặc output preview cũ.

## Cách chọn và giới hạn bộ nhớ

Lượt decode đầu gọi lại `inspect_video()` vào thư mục tạm, không sửa báo cáo M1.1.
Tính `min(samples, số frame có timestamp hợp lệ)` mốc cách đều theo thời gian
PTS đầu đến cuối, không cách đều theo index/FPS. Quét CSV tạm để chọn frame gần
nhất mỗi mốc bằng phân số `pts * time_base`; không nội suy frame.
Khi khoảng cách bằng nhau, chọn index nhỏ hơn; mốc đầu/cuối ghim đúng frame
hợp lệ đầu/cuối, kể cả khi timestamp trùng nhau.

Nhiều mốc có thể cùng chọn một frame: chỉ ghi một PNG cho index đó, giữ mapping
cho từng mốc trong CSV và cảnh báo số frame phân biệt thấp hơn số yêu cầu.
Một frame hợp lệ vẫn xuất được một ảnh (cả hai đầu trùng frame); không có frame
hợp lệ thì báo lỗi. Timestamp giảm vẫn được thống kê nhưng chặn preview vì
không có timeline tăng để phân bố mốc. Timestamp thiếu không được điền bằng FPS.

Lượt decode thứ hai xuất tuần tự các frame được chọn. PNG giữ kích thước frame;
contact sheet thu nhỏ vừa ô, giữ tỉ lệ rộng/cao, không crop. Không tự xoay theo
metadata hay sửa hình học nguồn. Contact sheet luôn hai hàng, tối đa bốn cột;
stream ngắn có thể có ô trống. Nhãn ghi index thực và thời gian tương đối 6 số lẻ.

RAM chỉ giữ tối đa 8 lựa chọn, một ảnh đang xuất và contact sheet. Timeline và
interval được lưu tạm trên đĩa; SQLite dùng cache 2 MiB và sort trên đĩa để lấy
median. Dung lượng tạm tăng theo số frame, được dọn khi xong. Hai lượt decode
đổi thêm thời gian đọc để không giữ mọi ảnh trong RAM.

## samples.csv

| Trường | Ý nghĩa |
|---|---|
| `target_time_seconds` | Mốc lấy mẫu tương đối so với timestamp hợp lệ đầu tiên, đơn vị giây. |
| `actual_time_seconds` | Timestamp của frame được chọn trừ timestamp hợp lệ đầu tiên, giây; không phải thời gian xử lý. |
| `frame_index` | Index decode bắt đầu từ 0, cùng quy ước M1.1. |
| `image_filename` | Tên PNG trong thư mục output; có thể lặp khi nhiều mốc chọn cùng frame. |

## timing_summary.json

| Trường | Ý nghĩa |
|---|---|
| `interval_count` | Số khoảng giữa hai timestamp hợp lệ liên tiếp; bằng max(số timestamp hợp lệ − 1, 0). |
| `min_interval_seconds`, `median_interval_seconds`, `max_interval_seconds` | Min/median/max của delta, đơn vị giây. Median lấy trung bình hai giá trị giữa khi số khoảng chẵn; null khi không có khoảng. |
| `missing_timestamp_count` | Frame thiếu PTS hoặc time base hợp lệ, cùng định nghĩa M1.1. |
| `duplicate_timestamp_count`, `decreasing_timestamp_count` | Số delta bằng 0/âm so với timestamp hợp lệ liền trước; vẫn tính các delta này vào thống kê. |
| `actual_decoded_count`, `decode_completed`, `decode_status` | Kết quả lượt kiểm tra đầu, không thay thế `preview_status`. |
| `first_valid_timestamp`, `last_valid_timestamp`, `timestamp_span_seconds` | Mốc đầu/cuối và last-minus-first; không phải full duration video. |
| `requested_samples`, `target_count`, `distinct_sample_count` | Số yêu cầu, số mốc thực tế và số index phân biệt được chọn. |
| `exported_image_count` | Số PNG xác nhận sau khi lượt xuất kết thúc; nếu xuất lỗi, xem thông báo và coi ảnh còn lại là output một phần. |
| `preview_status`, `error`, `warnings` | `complete` hoặc `failed`, lỗi/cảnh báo. Complete chỉ xác nhận tạo output, không xác nhận phù hợp thị giác. |
| `source_path`, `pyav_version`, `pillow_version` | Nguồn và phiên bản dùng để tạo preview. |

Delta qua frame thiếu vẫn nối hai timestamp hợp lệ liền trước/sau. Average FPS
trong metadata không chứng minh khoảng cách frame cố định. Timestamp trong file
không tự xác nhận thời lượng quay ngoài đời.

Exit 0 khi tạo preview xong (có thể ít ảnh hơn yêu cầu); exit 1 khi thiếu input,
output đã có, decode lỗi, không có timestamp hợp lệ hoặc timeline giảm; exit 2
khi sai cú pháp. Nếu xử lý lỗi sau khi tạo output, xem `timing_summary.json` và
coi mọi ảnh còn lại là một phần. Lỗi quyền/ổ đĩa có thể ngăn cả việc ghi summary.

## Kiểm tra đã chạy trên pilot

- 25 tests đạt: kiểm tra M1.1 và chọn nearest với timestamp không đều, stream
  ngắn, một frame, tie, timestamp thiếu/giảm, median chẵn/lẻ và bảo vệ output.
- Preview: index 0, 59, 118, 177, 237, 296, 355, 414; đủ 8 PNG 1280×720.
- Contact sheet 1536×552, 2 hàng × 4 cột; đã kiểm tra bố cục/nhãn.
- Đối chiếu cả 8 index/time và nearest target bằng PTS/time_base từ timeline M1.1.
- 414 khoảng: min 0.0634 s, median 0.0640 s, max 0.0806 s; thiếu/trùng/giảm = 0.
- SHA-256 nguồn, metadata/timeline M1.1 và PROJECT_PLAN không đổi.
- Đánh giá bằng mắt của sinh viên vẫn PENDING; chưa kết luận video phù hợp.
