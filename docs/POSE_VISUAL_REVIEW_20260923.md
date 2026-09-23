# Review pose định tính — 23/09/2026

## Nguồn và phạm vi

Ghi nhận **review bằng mắt của assistant trong cuộc trao đổi**, với các nhận xét được người dùng
cung cấp lại trong yêu cầu này. Không phải ground truth, gán nhãn khớp thủ công, đo accuracy,
hay một lần kiểm chứng độc lập mới. Chỉ áp dụng tám frame 0,59,118,177,237,296,355,414.
Ảnh nguồn: [backend_review_sheet.jpg](../outputs/pilot01_backend_visual_review_20260923/backend_review_sheet.jpg)
và các `comparison_frame_*.png` trong cùng thư mục.

| Frame/phạm vi | Quan sát định tính đã ghi nhận |
|---|---|
| Cả tám frame | PT320 và NCNN320 trông rất giống nhau. Sự đồng thuận không chứng minh accuracy. |
| 59,118,237 | Cả hai bản 320 có detection ở mép trái được đánh giá bằng mắt là phát hiện nhầm nền. Đây là nhận xét review, không phải nhãn tự động/ground truth. |
| 59,118,177 | Có giới hạn partial-body; không xem khớp thiếu/ngoài ảnh là vị trí cơ thể đã được xác minh. |
| 237 | Có khác biệt thấy được ở chi dưới giữa PT640 và PT320; chưa chứng minh bản nào chính xác hơn. |
| 296,355,414 | Bộ khung tư thế ngồi có thể nhận ra trên ảnh tổng quan; chưa xác nhận từng khớp hoặc độ ổn định theo thời gian. |

Không suy rộng các nhận xét này thành độ chính xác toàn video, các đối tượng khác hoặc bệnh nhân mới.
Không đưa nhận xét false positive vào thuật toán, sửa dự đoán, xóa detection hay tạo nhãn huấn luyện.
Các file output cũ, kể cả trạng thái `human_review: pending` tại thời điểm tạo, được giữ nguyên.
Tài liệu này là bản ghi review bổ sung; không biến các JSON cũ thành nhãn chuẩn.

## Quyết định tạm thời và điều kiện còn mở

- Giữ **YOLO11n-pose PT640 làm tham chiếu offline tạm thời** để phát triển/đối chiếu; không gọi là ground truth.
- Chưa chọn model/backend/kích thước ảnh cuối cùng cho triển khai Pi.
- **M2 vẫn IN_PROGRESS**. Theo bảng mốc M2 trong PROJECT_PLAN, còn phải tổng hợp bằng chứng chất lượng/thời gian
  và quyết định cấu hình tạm thời đủ căn cứ; review tám ảnh không tự hoàn thành mọi điều kiện.
  So sánh kích thước/backend của cùng YOLO11n chưa thay thế yêu cầu so sánh hai ứng viên pose trong kế hoạch.
  Đo Pi đã có current undervoltage/throttling; không kết luận lợi thế tốc độ backend từ các run đó.
- Chuẩn bị protocol M3 có thể thực hiện ngay; việc này không miễn các điều kiện M2 và không xác nhận M3 hoàn tất.

Xem [ghi chú thí nghiệm gốc](EXPERIMENT_NCNN_REVIEW_20260923.md) và [protocol dữ liệu mới](DATASET_RECORDING_PROTOCOL.md).
