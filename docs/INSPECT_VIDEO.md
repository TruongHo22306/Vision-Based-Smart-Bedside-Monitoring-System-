# M1.1 — Kiểm tra video và timeline

Dùng video mới do sinh viên cung cấp. M1.1 đã xác minh pilot01: 415 frame, timestamp không thiếu/trùng/giảm.
Không tạo dataset thay thế, trích pose hoặc nhận dạng hành động.

## Cài và chạy (PowerShell tại gốc workspace)

Giữ Python 3.10.11 và `.venv` hiện có. Chỉ cài dependency/công cụ build từ wheel:

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: av==16.1.0
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -e .
.\.venv\Scripts\python.exe -m patient_monitoring inspect-video data/raw/pilot01.mp4 --output-dir outputs/pilot01
```

Nếu không có wheel phù hợp, dừng cài và ghi lỗi; không bỏ `--only-binary`.
Hai lệnh cài đã chạy với `--disable-pip-version-check --retries 0 --timeout 20`.
Nếu đường dẫn có khoảng trắng, đặt trong dấu ngoặc kép.
Chọn thư mục output chưa chứa `metadata.json` hoặc `timeline.csv`; lệnh từ chối
mọi báo cáo đã có và không ghi đè video đầu vào. Dùng thư mục mới khi chạy lại.

Exit 0: decoder đọc hết, không phát hiện lỗi/khung hỏng/thiếu frame so với count
đã báo. Đây không phải xác nhận timeline đúng hoặc bản ghi gốc nguyên vẹn.
Exit 1: input/output không hợp lệ, thiếu video stream hoặc decode chưa hoàn tất.
Exit 2: sai cú pháp dòng lệnh. Khi thiếu file, không tạo báo cáo.
Khi mở/decode lỗi nhưng output ghi được, giữ CSV một phần và metadata mô tả lỗi.
Nếu ổ đĩa/quyền ghi lỗi, báo cáo cũng có thể chưa ghi đủ; đọc lỗi terminal.

## metadata.json

Giá trị không biết là JSON `null`; không suy ra từ FPS hoặc tốc độ xử lý.

| Trường | Ý nghĩa |
|---|---|
| `source_path`, `pyav_version` | Đường dẫn nguồn tuyệt đối và phiên bản PyAV thực tế. |
| `selected_stream_index` | Index trong container của video stream đầu tiên; có thể khác 0 nếu audio đứng trước. |
| `codec`, `width`, `height` | Codec và kích thước do codec context báo, không đo lại từng frame. |
| `reported_frame_rate`, `reported_frame_rate_fraction` | `stream.average_rate`, dạng số và phân số; chỉ metadata, không dùng tạo timeline. |
| `reported_frame_count` | `stream.frames`; giá trị 0 nghĩa là chưa biết và được lưu null. |
| `actual_decoded_count` | Số frame thực sự decoder đã trả về, gồm cả frame thiếu timestamp/hỏng. |
| `container_duration_seconds` | Duration container / `av.time_base`; có thể bao gồm stream khác. |
| `stream_duration_seconds`, `stream_time_base` | Duration video stream nhân time base và phân số time base của stream. |
| `first_valid_timestamp`, `last_valid_timestamp` | Timestamp hợp lệ đầu/cuối theo thứ tự decode: index, PTS, time base, giây; không phải min/max. |
| `timestamp_span_seconds` | Thời gian hợp lệ cuối trừ đầu, không cộng thời gian hiển thị frame cuối; có thể âm khi timeline giảm. Không phải full duration. |
| `missing_timestamp_count` | Số frame thiếu PTS hoặc thiếu/time base không dương; giữ nguyên PTS/time base quan sát được. |
| `duplicate_timestamp_count`, `decreasing_timestamp_count` | Số lần thời gian bằng/nhỏ hơn timestamp hợp lệ liền trước, bỏ qua frame thiếu khi so sánh. Không đếm trùng với toàn bộ lịch sử. |
| `corrupt_frame_count` | Số frame có cờ `is_corrupt` từ decoder. |
| `decode_completed`, `decode_status` | Boolean và `complete`, `incomplete`, `failed` (lỗi trước frame đầu), `no_video_stream`. |
| `error`, `warnings` | Lỗi cụ thể nếu có; cảnh báo count/timestamp và giới hạn diễn giải. |

Count báo lớn hơn count decode được coi là chưa hoàn tất để tránh kết luận
thành công với kết quả một phần; chính metadata count cũng có thể sai.
EOF không có lỗi vẫn không chứng minh file không bị cắt nếu thiếu metadata đối chiếu.
Timestamp lỗi không tự biến thành lỗi decode: cần đọc cả các bộ đếm và warnings.

## timeline.csv

Ghi tuần tự từng frame, không giữ toàn bộ frame hay timeline trong RAM.
Ô trống là chưa biết; index bắt đầu từ 0 theo thứ tự decoder trả về.

| Cột | Ý nghĩa |
|---|---|
| `frame_index` | Thứ tự frame được decode. |
| `pts`, `time_base` | Giá trị gốc trên frame PyAV; time base dạng phân số. |
| `presentation_time_seconds` | Chính xác về phép tính: `frame.pts * frame.time_base`, xuất giây dạng số thập phân. |
| `relative_time_seconds` | Thời gian hiện tại trừ timestamp hợp lệ đầu tiên, kể cả khi mốc đầu khác 0. |
| `delta_seconds` | Thời gian hiện tại trừ timestamp hợp lệ liền trước; trống ở frame hợp lệ đầu hoặc frame thiếu timestamp. |

Phép trừ/so sánh thực hiện bằng phân số trước khi xuất số thập phân.
Khi thiếu timestamp, không dùng `frame_index/FPS`, DTS hoặc đồng hồ xử lý để bù.
Dữ liệu là timestamp trên frame do PyAV/FFmpeg trả về, không phải kiểm chứng
độc lập timestamp camera gốc. Thời gian trong file không đủ chứng minh thời lượng
thực ngoài đời; cần đối chiếu quy trình quay/đồng hồ độc lập khi kiểm tra video thật.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m patient_monitoring --help
.\.venv\Scripts\python.exe -m patient_monitoring doctor
.\.venv\Scripts\python.exe -m patient_monitoring inspect-video --help
.\.venv\Scripts\python.exe -m pip check
```

Fixture timestamp/frame được tổng hợp trong test. WAV cực ngắn và file lỗi chỉ
được tạo trong thư mục tạm để kiểm tra PyAV thật rồi dọn; không phải dữ liệu
nghiên cứu và không thay thế việc kiểm tra video mới.

Tham chiếu API: [PyAV Frame](https://pyav.org/docs/stable/api/frame.html),
[Stream](https://pyav.org/docs/stable/api/stream.html),
[Container](https://pyav.org/docs/stable/api/container.html).
Kết quả chạy tại workspace dùng phiên bản ghim 16.1.0; xem `STATUS.md`.
