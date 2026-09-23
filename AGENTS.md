# Hướng dẫn làm việc

- Dự án mới tại workspace hiện tại; giữ nguyên `docs/PROJECT_PLAN.md`.
- Không tìm, đọc, sao chép, import hoặc tái sử dụng
  `patient-action-monitoring-starter`, môi trường, dữ liệu hoặc kết quả cũ.
- Đích: Raspberry Pi 4 + webcam USB; bắt đầu bằng video có nhãn, CPU.
- Cốt lõi: đứng, ngồi, nằm, di chuyển và phát hiện té ngã theo thời gian.
- Hoạt động hiện tại và sự kiện ngã là hai trường riêng;
  `no_person` và `unknown` là trạng thái vận hành.
- Chuyển động nghi co giật là mở rộng tùy điều kiện, không phải chẩn đoán.
- Mỗi nhiệm vụ đọc `docs/STATUS.md` trước, rồi chỉ phần kế hoạch và mã liên quan.
- Chỉ sửa phạm vi được giao; không tự triển khai mốc tiếp theo.
- Giữ bộ khung nhỏ; chỉ thêm dependency khi nhiệm vụ cần.
- M0 không cài ML, tải model/dữ liệu, huấn luyện, mở camera hay tạo UI/backend.
- Dùng Python của `.venv`; không thay đổi Python hệ thống.
- Kiểm tra bằng lệnh thực tế; không tạo bộ test chỉ để kiểm tra bộ khung.
- Không bịa dữ liệu, phần cứng, số liệu, benchmark hoặc kết quả kiểm tra.
- Phân biệt đã chạy thành công, đã chạy lỗi và chưa kiểm tra.
- Không commit hoặc push nếu chưa được yêu cầu.
- Cập nhật STATUS theo bằng chứng; AGENTS và STATUS tối đa 60 dòng/file.
- Trả kết quả ngắn bằng tiếng Việt: việc đã làm, `path:line` hoặc tên hàm/mục,
  lệnh và kết quả thật, phần còn thiếu, một hành động tiếp theo cho sinh viên.
