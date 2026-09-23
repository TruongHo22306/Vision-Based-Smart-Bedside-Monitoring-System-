**Kế hoạch dự án mới: Giám sát hành động bệnh nhân bằng Raspberry Pi 4 và webcam USB**

Phiên bản 1.0 — 14/09/2026. Đây là kế hoạch thực hiện, chưa phải báo cáo kết quả nghiên cứu. Thời lượng bên dưới là ước lượng để sắp xếp công việc; ngày nộp và số giờ thực tế mỗi tuần chưa được xác nhận.

**Dự án bắt đầu từ số 0 theo yêu cầu của bạn.**

Tạo một thư mục và lịch sử Git mới; tên đề xuất là `patient-monitoring-pi4`. Trên Windows có thể đặt tại `D:\patient-monitoring-pi4` nếu đường dẫn này chưa có dự án khác. Không dùng `patient-action-monitoring-starter` làm nền, không đọc hoặc sao chép mã, cấu hình, môi trường, dữ liệu S01, đặc trưng đã trích xuất hay kết quả cũ vào dự án mới. Toàn bộ trạng thái thực hiện của dự án mới bắt đầu là `NOT_STARTED`.

Các bài viết và repository bạn cung cấp vẫn là tài liệu tham khảo. Việc khởi tạo mới không yêu cầu tự huấn luyện mô hình pose từ đầu: ta có thể sử dụng thư viện và trọng số công khai, ghi rõ nguồn, rồi tự xây dựng chương trình xử lý và đánh giá của mình.

Tên đề tài tiếng Anh tạm dùng: **Patient Activity Monitoring Using Raspberry Pi 4 and a USB Webcam**. Tên chính thức, mã sinh viên, người hướng dẫn và thông tin trang ký sẽ điền theo thông tin bạn xác nhận khi bắt đầu report.

**Phạm vi cần hoàn thành gồm năm hành động và hai giai đoạn vận hành.**

| Nội dung | Phạm vi phiên bản đầu |
|---|---|
| Phần cứng đích | Raspberry Pi 4, webcam USB; suy luận cuối cùng chạy trên Pi |
| Môi trường phát triển | Laptop và VS Code; bắt đầu bằng CPU, kiểm tra phần cứng thực tế trước khi chọn tăng tốc |
| Hoạt động thường ngày | Đứng, ngồi, nằm, di chuyển; ban đầu định nghĩa di chuyển là đi lại trong vùng camera quan sát |
| Sự kiện theo thời gian | Té ngã/nghi té ngã, có thời điểm phát hiện và nhật ký sự kiện |
| Mở rộng | Chuyển động nghi co giật, chỉ thực hiện sau khi phần cốt lõi ổn định và có dữ liệu phù hợp |
| Giai đoạn A | Ảnh để kiểm tra pose; video có nhãn để xây dựng và đánh giá nhận dạng hành động |
| Giai đoạn B | Webcam trực tiếp trên Pi; giữ cùng quy trình xử lý và cách định nghĩa nhãn |
| Bối cảnh ban đầu | Một người trong vùng quan sát, camera cố định, góc nhìn và khoảng cách được mô tả |
| Bối cảnh mở rộng | Nhiều người, che khuất, camera bệnh viện hoặc luồng mạng được kiểm tra ở thí nghiệm riêng |
| Đầu ra | Nhãn hoạt động, trạng thái không đủ dữ liệu, cảnh báo sự kiện, thời gian, log và hình/video minh họa |

`no_person` và `unknown` là trạng thái vận hành bổ sung. Người đang nằm có thể đồng thời đang trong giai đoạn sau một sự kiện ngã, nên lưu hoạt động hiện tại và sự kiện thành hai trường riêng. Không gắn nhãn ngã chỉ vì một ảnh cho thấy người nằm ngang. Mở rộng co giật được mô tả là nhận biết chuyển động nghi ngờ từ video, không diễn giải thành chẩn đoán.

Camera bệnh viện không phải điều kiện để hoàn thành bản đầu. Thử nghiệm với webcam ở bối cảnh kiểm soát là mốc chính; dữ liệu bệnh viện, nếu có quyền sử dụng, là bước đánh giá khả năng áp dụng sang bối cảnh mới.

**Nền kỹ thuật đề xuất là mã ứng dụng viết mới, sử dụng Ultralytics YOLO Pose làm thư viện.**

Đây là lựa chọn thiết kế để bắt đầu công việc, chưa phải kết luận mô hình tối ưu. Tài liệu chính thức xác nhận YOLO Pose trả về tọa độ khớp và thông tin confidence, hỗ trợ ảnh, video và luồng trực tiếp [1]. Hai ứng viên nhỏ để so sánh ban đầu là `yolo11n-pose.pt` và `yolo26n-pose.pt` [1, 2]. Chưa mặc định chọn một phiên bản hoặc tần suất lấy mẫu từ dự án cũ.

| Nguồn đã trao đổi | Vai trò trong dự án mới |
|---|---|
| TensorFlow image classification cho Pi [6] | Tham khảo tổ chức camera và suy luận; không dùng mô hình phân loại ảnh mặc định để gán nhãn hành động |
| `rpi_pose_estimation` [7] | Tham khảo lấy khớp và tích hợp phần cứng; không sao chép môi trường cài đặt cũ |
| `rpi_road_object_detection` [8] | Tham khảo đọc camera, vẽ kết quả, lưu đầu ra |
| Hackster/Sony fall detection [9, 10] | Tham khảo đặc trưng hình học; phần triển khai AI Camera cần phần cứng khác webcam USB |
| Ultralytics [1–3] | Thư viện suy luận và hướng xuất mô hình để thử trên Pi |

Nhận xét về các repo mẫu dựa trên phần đọc nguồn đã có trong cuộc trao đổi. Kế hoạch này không ghi nhận bất kỳ repo nào đã được chạy thành công trên thiết bị của bạn.

Chương trình được xây dựng theo các phần nhỏ, thêm khi đến đúng mốc:

1. Đọc video hoặc webcam và giữ timestamp của khung hình.
2. Lấy pose, khung bao và confidence; chọn đúng người trong vùng quan sát.
3. Lưu chuỗi khớp, đánh dấu dữ liệu thiếu và tính đặc trưng hình học/chuyển động.
4. Nhận dạng đứng, ngồi, nằm, di chuyển; bổ sung logic theo thời gian cho té ngã.
5. Hiển thị kết quả, ghi sự kiện và đo hiệu năng toàn bộ quy trình.

Phiên bản đầu dùng quy tắc hình học dễ giải thích làm baseline. Sau khi có tập dữ liệu phù hợp, so sánh với một bộ phân loại nhẹ, chẳng hạn Random Forest trên đặc trưng cửa sổ thời gian. Đây là ứng viên thí nghiệm, không mặc định thêm LSTM/TCN trước khi biết baseline sai ở đâu.

Khi chuẩn hóa pose theo cơ thể, vẫn giữ đặc trưng dịch chuyển trong ảnh và thay đổi kích thước khung bao để phân tích di chuyển. Tốc độ phải tính theo chênh lệch timestamp thực tế. Không nối chuỗi qua hai video, hai người hoặc khoảng mất dấu dài. Chỉ dùng khung hình hiện tại và quá khứ cho phiên bản sẽ chạy trực tiếp; không dùng thông tin tương lai để tạo kết quả có vẻ nhanh hơn thực tế.

NCNN là định dạng chạy mô hình cần thử trên Pi [3]. Trước khi dùng, so sánh đầu ra pose sau xuất với bản gốc trên cùng ảnh/video và ghi lại phiên bản thư viện. Các bảng benchmark trong hướng dẫn Pi hiện tại chủ yếu dùng Pi 5 và mô hình detection; không chuyển những số liệu đó thành FPS của pose trên Pi 4.

**Lộ trình đề xuất kéo dài khoảng 12 tuần, cộng 1–2 tuần dự phòng.**

Ước lượng này giả định khoảng 12–15 giờ mỗi tuần và có thể sắp xếp dùng Pi sớm. Ta điều chỉnh lịch theo deadline thực tế, nhưng giữ các điều kiện kiểm tra bên dưới. Mỗi mốc lớn được chia tiếp thành những nhiệm vụ Codex nhỏ.

| Mốc | Thời gian dự kiến | Công việc và thứ bạn cần hiểu | Bằng chứng hoàn thành |
|---|---|---|---|
| M0 — Khởi tạo | Tuần 1 | Tạo workspace mới; cài/kiểm tra VS Code, Python, Git, Codex; tạo `.venv`; chạy một chương trình nhỏ | Lệnh kiểm tra môi trường chạy được bằng Python của dự án; có hướng dẫn cài đặt tái lập và trạng thái công việc |
| M1 — Video và dữ liệu thăm dò | Tuần 2 | Hiểu frame, FPS và timestamp; đọc video mới; ghi metadata; kiểm tra clip trước khi thu nhiều | Một nhóm video mới đọc được; thời gian không bị suy ra từ tốc độ xử lý; có danh sách lỗi clip |
| M2 — Pose và thử Pi sớm | Tuần 3 | Trích khớp; hiển thị bộ khung; so sánh hai ứng viên pose trên cùng dữ liệu; thử xuất/chạy một cấu hình trên Pi | Video minh họa, dữ liệu khớp, bảng đo chất lượng và thời gian; quyết định tạm thời về model/backend/kích thước ảnh |
| M3 — Dataset và cách đánh giá | Tuần 4–5 | Viết quy ước nhãn, thu dữ liệu mới, gán nhãn đoạn thời gian, quản lý người/phiên/video và chia tập | Manifest có nguồn và thời gian; nhãn được kiểm tra; tập train/validation/test được khóa theo nhóm trước huấn luyện |
| M4 — Bốn hoạt động thường ngày | Tuần 6–7 | Làm baseline đứng/ngồi/nằm/di chuyển; đo lỗi; thử bộ phân loại nhẹ khi đủ dữ liệu | Confusion matrix, F1 từng lớp, macro-F1 và video lỗi; cấu hình được lựa chọn bằng validation |
| M5 — Té ngã theo chuỗi | Tuần 8–9 | Phân tích chuyển động xuống thấp và diễn biến trước/sau; phân biệt nằm chủ động, ngồi nhanh, cúi người; chống báo lặp | Danh sách TP/FP/FN theo sự kiện, độ trễ phát hiện, số cảnh báo nhầm trên mỗi giờ video và các tình huống bỏ sót |
| M6 — Webcam trên Pi | Tuần 10–11 | Chạy toàn bộ quy trình trên Pi; xử lý khung hình mới, giới hạn hàng đợi, hiển thị/log; kiểm tra mất người và ngắt camera | Demo trực tiếp, log hiệu năng, kiểm tra chạy liên tục tối thiểu 30 phút; mô tả lỗi còn tồn tại |
| M7 — Đánh giá và bảo vệ | Tuần 12 | Chạy đánh giá cuối với cấu hình đã khóa; hoàn thiện report, hình, bảng, demo và câu hỏi bảo vệ | Kết quả tái lập gắn với phiên bản mã; report hoàn chỉnh; demo trực tiếp và video dự phòng |
| M8 — Mở rộng tùy điều kiện | Sau phần cốt lõi | Chuyển động nghi co giật hoặc thử bối cảnh camera khác | Thí nghiệm riêng, nguồn dữ liệu rõ ràng, giới hạn và chỉ số riêng |

Nếu chưa có Pi ở M2, vẫn tiếp tục được phần video và dữ liệu, nhưng ghi rõ phần hiệu năng thiết bị còn chưa được xác minh. Nếu pose không phù hợp về tốc độ hoặc chất lượng, quay lại M2 để sửa cấu hình hoặc kiểm tra một phương án nhẹ khác trước khi đầu tư thêm vào mô hình hành động.

**Dữ liệu sẽ được thu mới và phải phục vụ cả phát triển lẫn kiểm tra sai.**

Ở M1, có thể bắt đầu với khoảng 20 clip ngắn: 5 clip cho mỗi nhóm đứng, ngồi, nằm, di chuyển, mỗi clip khoảng 8–15 giây. Đây là gợi ý để kiểm tra đường đi dữ liệu, không phải kích thước dataset đủ cho kết luận nghiên cứu. Mỗi video được lưu nguyên bản; không sửa FPS metadata để làm thời lượng có vẻ đúng.

Ở M3, mở rộng số người, phiên thu, góc nhìn, ánh sáng và tình huống dựa trên lỗi quan sát được. Mục tiêu ban đầu để lập lịch là 5–8 người, ít nhất hai phiên khác nhau nếu có thể tổ chức; số clip cần thiết được điều chỉnh theo độ đa dạng và các lớp còn thiếu. Khi chỉ có một người, báo cáo kết quả trong phạm vi người/phiên đã đo và không tuyên bố tổng quát sang bệnh nhân mới.

Với dữ liệu ngã, chỉ sử dụng mô phỏng có kiểm soát hoặc dữ liệu được phép sử dụng phù hợp; không tự thực hiện cú ngã nguy hiểm để thu video. Luôn có ví dụ không ngã nhưng dễ gây nhầm như chủ động nằm xuống, cúi nhặt đồ và ngồi nhanh. Nên có video sinh hoạt liên tục để đo cảnh báo nhầm, không chỉ những clip cắt sẵn quanh sự kiện.

| Thành phần dữ liệu | Nội dung cần ghi |
|---|---|
| Manifest video | `video_id`, đường dẫn, `subject_id`, `session_id`, thời lượng, độ phân giải, nguồn, điều kiện quay |
| Nhãn thời gian | `video_id`, `start_sec`, `end_sec`, nhãn hoạt động; sự kiện có `event_id`, mốc bắt đầu/kết thúc theo quy ước |
| Quy ước nhãn | Khi nào bắt đầu/kết thúc hành động; xử lý chuyển tiếp, ra khỏi ảnh, che khuất và trường hợp khó xác định |
| Dữ liệu pose | Video/frame/timestamp, người được theo dõi, khung bao, tọa độ và confidence, mặt nạ dữ liệu thiếu |
| Phân chia tập | Danh sách nhóm train/validation/test được lưu thành file; chia trước khi cắt cửa sổ thời gian |
| Dấu vết thí nghiệm | Phiên bản dataset, cấu hình, seed, mã phiên bản, model và phiên bản thư viện |

Không đưa các frame hoặc cửa sổ chồng lấn của cùng video vào nhiều tập. Khi đủ người, tách theo người để đánh giá người chưa xuất hiện trong huấn luyện. Chọn ngưỡng và tham số bằng train/validation; test cuối chỉ được dùng sau khi khóa lựa chọn. Ghi riêng những trường hợp không dự đoán được, không âm thầm bỏ chúng khỏi báo cáo.

**Việc đánh giá phải đo được cả nhận dạng, sự kiện và tốc độ thực tế.**

| Câu hỏi | Cách đo |
|---|---|
| Pose có sử dụng được không? | Tỷ lệ mất người/khớp, kiểm tra ảnh chồng bộ khung; một tập nhỏ được kiểm tra vị trí khớp bằng tay nếu cần so sánh chất lượng |
| Hoạt động nào đang bị nhầm? | Precision/recall/F1 từng lớp, macro-F1, confusion matrix; báo cáo số mẫu và đơn vị là frame/cửa sổ/đoạn |
| Có phát hiện đúng sự kiện ngã không? | Precision và recall theo sự kiện, TP/FP/FN và quy tắc ghép dự đoán với sự kiện thật |
| Cảnh báo có gây phiền không? | Số cảnh báo nhầm chia cho tổng giờ video theo dõi, có video sinh hoạt thường ngày và cơ chế gộp cảnh báo lặp |
| Cảnh báo có đủ nhanh không? | Độ trễ từ mốc sự kiện được định nghĩa tới thời điểm phát cảnh báo; báo cáo trung vị và p95 khi đủ mẫu |
| Pi có xử lý theo kịp không? | FPS xử lý toàn bộ quy trình, thời gian từng bước, độ trễ quan sát được, số frame bị bỏ, RAM và nhiệt độ |
| Kết quả có lặp lại được không? | Chạy lại cùng dataset/cấu hình/phiên bản và lưu file kết quả |

Confidence của pose không phải độ chính xác nhận dạng hành động. Khi không có nhãn khớp bằng tay, không gọi tỷ lệ confidence cao là pose accuracy. FPS capture, FPS suy luận và FPS xử lý toàn bộ quy trình được đặt tên riêng. Nếu không đo được thời điểm phơi sáng của webcam, ghi rõ phép đo độ trễ bắt đầu từ lúc chương trình nhận frame.

Tại M2–M3, chốt mục tiêu định lượng về F1, recall ngã, cảnh báo nhầm và tốc độ theo kết quả thăm dò, điều kiện thiết bị và yêu cầu của người hướng dẫn. Ghi chúng trước đánh giá test cuối. Sau đó báo cáo cả phần đạt và chưa đạt; không thay đổi tiêu chí để làm đẹp kết quả test.

Đối chứng tối thiểu dự kiến: cùng dataset và cách chia tập, so sánh quy tắc trên từng khung hình với cách dùng thông tin theo thời gian. So sánh thêm bộ phân loại nhẹ nếu đủ dữ liệu. Với Pi, kiểm tra ảnh đầu vào/backend khác nhau bằng cùng protocol; khi đã giải quyết được rủi ro cụ thể thì chuyển sang mốc tiếp theo.

**Report được viết khi có đủ bằng chứng cho từng phần.**

Đã xem mục lục ở trang PDF 4–5 và các phần phương pháp, triển khai, đánh giá, kết luận của `ITCSIU22282_NguyenVuPhuongNghi.docx.pdf`, một luận văn 71 trang về hệ thống hỗ trợ chọn đề tài. Ta giữ cấu trúc sáu chương làm mẫu trình bày và điều chỉnh nội dung cho thị giác máy tính. Đây là mẫu bạn cung cấp, chưa phải xác nhận về mọi quy định nộp luận văn hiện hành của khoa.

| Chương theo mẫu | Nội dung dành cho đề tài mới | Khi bắt đầu viết thành bản nháp |
|---|---|---|
| 1. Introduction | Bối cảnh, vấn đề, mục tiêu, phạm vi, câu hỏi nghiên cứu, đóng góp dự kiến | Sau khi phạm vi M0–M1 được xác định |
| 2. Literature Review | Nghiên cứu nhận dạng hoạt động/té ngã từ video và pose; triển khai trên thiết bị biên; so sánh dữ liệu, phương pháp và giới hạn | Khi có đủ bài nghiên cứu gốc đã đọc và bảng so sánh; tutorial/repo dùng làm nguồn kỹ thuật bổ trợ |
| 3. Methodology | Yêu cầu, pipeline, thu/gán nhãn/chia dữ liệu, pose, đặc trưng, mô hình, sự kiện, protocol đánh giá | Sau M2–M3 khi thiết kế và protocol ổn định |
| 4. Implementation and Results | Cấu trúc chương trình, môi trường, triển khai Pi, kiểm thử chức năng, kết quả đo và minh họa | Theo các đợt thí nghiệm đủ dữ liệu ở M4–M6; cập nhật kết quả cuối tại M7 |
| 5. Discussion, Comparison and Evaluation | So sánh baseline, phân tích lỗi, đánh đổi tốc độ/chất lượng, giới hạn dữ liệu và bối cảnh áp dụng | Sau khi có các thí nghiệm so sánh hoàn chỉnh |
| 6. Conclusion and Future Work | Kết luận dựa trên kết quả, phần đã/chưa đạt, hướng mở rộng | Sau đánh giá cuối |

Phần đầu và cuối giữ các mục cần thiết như trang bìa, trang xác nhận, acknowledgments, abstract, mục lục, danh sách hình/bảng, references và appendix. Abstract hoàn thiện sau khi có kết quả. Không sao chép tên tác giả, người hướng dẫn, nội dung hoặc tài liệu tham khảo của đề tài mẫu sang đề tài mới.

Chương 3 ưu tiên sơ đồ xử lý dữ liệu, sơ đồ trạng thái phát hiện sự kiện và protocol đánh giá. Chương 4 mô tả video/pose/inference/Pi; không ép thành frontend/backend chỉ để giống luận văn mẫu. Chỉ thêm thiết kế cơ sở dữ liệu hoặc sơ đồ lớp khi hệ thống thực sự cần và có triển khai tương ứng.

Sau mỗi nhiệm vụ chỉ cập nhật ngắn trạng thái và bằng chứng thực nghiệm. Sau một cụm thí nghiệm đủ kết luận mới viết các đoạn report liên quan. Mỗi số liệu trong report cần tìm lại được file kết quả, cấu hình và phiên bản mã. Nội dung chưa đo được ghi là dự kiến hoặc chưa đánh giá.

**Cách phối hợp: mình hướng dẫn, Codex thực hiện nhiệm vụ được giới hạn, bạn kiểm tra và nắm được cách giải thích.**

Codex trong IDE có thể làm việc với project đang mở, giải thích và chỉnh sửa mã trong ngữ cảnh editor [4]. `AGENTS.md` là nơi đặt hướng dẫn chung để Codex đọc khi bắt đầu công việc [5]. Chạy Codex trong thư mục dự án mới trên máy bạn giúp giảm việc sao chép mã vào chat. Cuộc trò chuyện này chỉ có thể kiểm tra những file hoặc kết quả được bạn chia sẻ/kết nối; không mặc định nhìn thấy ổ đĩa trên máy bạn.

Mỗi nhiệm vụ theo một vòng lặp:

1. Mình giải thích mục tiêu, kiến thức cần hiểu và đầu ra mong đợi.
2. Mình đưa prompt Codex với mã nhiệm vụ, phạm vi, tiêu chí xong và lệnh kiểm tra.
3. Codex đọc những file cần thiết, sửa phạm vi đó và chạy kiểm tra phù hợp.
4. Bạn chạy phần cần thiết trên thiết bị thật, gửi bản tóm tắt kết quả hoặc lỗi đầu tiên.
5. Mình xem bằng chứng, giải thích thay đổi và đưa nhiệm vụ kế tiếp hoặc sửa đúng vấn đề còn lại.

Giao việc ngắn dựa trên các file trong repo, không lặp toàn bộ lịch sử chat trong mỗi prompt. Codex đọc `AGENTS.md` và `docs/STATUS.md` trước; chỉ đọc phần liên quan trong `docs/PROJECT_PLAN.md` và mã nguồn khi nhiệm vụ cần. Giữ hướng dẫn và trạng thái ngắn, khoảng 40–60 dòng mỗi file. Không yêu cầu nó đọc lại mọi PDF hoặc quét toàn bộ project cho một thay đổi nhỏ. Không tự mở nhiều agent cho công việc có thể xử lý gọn trong một nhiệm vụ.

| File dự kiến trong repo | Mục đích |
|---|---|
| `AGENTS.md` | Quy tắc ổn định: dự án mới, phạm vi thay đổi, cách xác minh, không bịa số liệu, mẫu trả kết quả |
| `docs/PROJECT_PLAN.md` | Bản kế hoạch này, được sao chép vào repo mới |
| `docs/STATUS.md` | Mốc hiện tại, đã xác minh, còn vướng và nhiệm vụ tiếp theo; giữ ngắn |
| `docs/DECISIONS.md` | Các quyết định thật sự ảnh hưởng mô hình, dataset, cách đánh giá hoặc cấu hình |
| `docs/experiments/` | Kết quả thí nghiệm và liên kết log/bảng/hình, tạo khi có thí nghiệm |
| `docs/report/` | Các bản nháp chương, tạo khi đủ thông tin để viết |

Bản trả kết quả của Codex nên gồm: việc đã làm; file và dòng hoặc tên hàm đã thay đổi; lệnh kiểm tra và kết quả thật; phần chưa kiểm tra; bước cần bạn chạy nếu liên quan webcam/Pi. Chỉ cần gửi lại phần này cùng file/hình liên quan, không cần dán toàn bộ code.

Cam kết về cách làm việc: giữ các quyết định đã xác nhận trong tài liệu dự án; giải thích trước khi giao nhiệm vụ; dùng bằng chứng để chọn bước tiếp theo; luôn ghi đúng đường dẫn và vị trí sửa; phân biệt đề xuất với kết quả đã chạy; viết report theo từng cụm thông tin đủ chín. Mục tiêu là một đồ án bạn tự giải thích và tái lập được. Điểm số vẫn phụ thuộc tiêu chí và đánh giá của hội đồng.

**Bước bắt đầu hiện tại là M0: tạo môi trường mới và chạy kiểm tra môi trường.**

Trên Windows, tạo thư mục mới bằng File Explorer, ví dụ `D:\patient-monitoring-pi4`, rồi mở đúng thư mục đó bằng **VS Code → File → Open Folder** [11]. Dùng extension **Python** và **Pylance** của Microsoft, cùng extension **Codex** chính thức của OpenAI. Nếu chưa có Python hoặc Git, nhiệm vụ M0 sẽ ghi chính xác thành phần còn thiếu và hướng dẫn cài đặt trước khi chạy tiếp; không coi setup là xong khi chưa kiểm tra được.

Lưu bản kế hoạch này vào `docs/PROJECT_PLAN.md` trong thư mục mới trước khi giao prompt. Tên thư mục và đường dẫn Windows ở đây là đề xuất, chưa được tạo trên máy của bạn trong cuộc trò chuyện này.

Prompt đầu tiên để dán vào Codex:

```text
Task M0: bootstrap a fresh Python thesis project in the CURRENT workspace.

Read only the scope and M0 sections of docs/PROJECT_PLAN.md if present.
Goal: patient activity monitoring on
Raspberry Pi 4 + USB webcam, with labeled videos first. Core: sitting, lying,
standing, moving, and temporal fall detection; suspected convulsive motion
is optional. Execute only M0.

This is a NEW project. Do not search, read, copy, import, or reuse anything
from patient-action-monitoring-starter or any previous dataset/results.
Inspect only the current root and applicable instructions. Preserve the
supplied plan. If an unrelated codebase exists here, report the conflict
without overwriting it.

Create the smallest useful scaffold:
- AGENTS.md: concise project scope, fresh-start rule, focused changes,
  honest verification, no fabricated data/results, and path:line/anchor
  change reporting. Future tasks read STATUS first, then only the relevant
  plan section and source files. Keep AGENTS and STATUS <=60 lines each.
- README.md and docs/SETUP.md: beginner Windows/PowerShell setup, VS Code
  interpreter selection, Python/Pylance/Codex extensions, and exact commands.
- docs/STATUS.md: current facts, verification, blockers, next task (M1).
- Minimal pyproject.toml; src/patient_monitoring/__init__.py and __main__.py.
- A standard-library CLI supporting --help and doctor. doctor reports only
  observed Python version/executable, OS, virtualenv status and project root;
  it must not invent camera, GPU or Raspberry Pi availability.
- .gitignore excluding virtualenvs, caches, raw videos, generated outputs,
  model weights and secrets. Keep code, config and documentation trackable.

Inspect installed Python/Git. Use an available Python >=3.10 and create a
local .venv if possible; document the version. Do not change system Python.
Install this minimal package into .venv and run its --help and doctor there.
If prerequisites or package installation are blocked, still create the
scaffold and report the exact blocker; do not claim verification passed.

Do not install ML packages, download models/data, train, access a camera,
build a UI/backend, or implement M1+. Initialize local Git only if no Git
repository or enclosing repository exists; do not commit or push.

Verify the documented smoke commands; no scaffolding-only test suite.
Update STATUS with what actually ran. Return <=250 words in Vietnamese:
summary, changed paths with lines/anchors, verification, blockers, and
the single next action the student should take. Do not print entire files.
```

Sau khi M0 thành công, nhiệm vụ M1 sẽ bắt đầu bằng đọc một video mới do bạn cung cấp, hiển thị metadata và xác minh timeline. Việc tải model pose và phân loại hành động thuộc các mốc sau.

**Nguồn để tra cứu và đưa vào tài liệu dự án khi sử dụng.**

[1] Ultralytics, [Pose Estimation](https://docs.ultralytics.com/tasks/pose/). Đã đối chiếu ngày 14/09/2026: đầu ra khớp, model pose, nguồn đầu vào và khả năng export.

[2] Ultralytics, [YOLO11](https://docs.ultralytics.com/models/yolo11/). Đã đối chiếu ngày 14/09/2026: có nhóm YOLO11-pose gồm bản nano.

[3] Ultralytics, [Raspberry Pi Guide](https://docs.ultralytics.com/guides/raspberry-pi/). Đã đối chiếu ngày 14/09/2026: hướng triển khai và NCNN; cần đọc đúng phần cứng/task của benchmark.

[4] OpenAI, [Codex IDE extension](https://learn.chatgpt.com/docs/codex/ide). Đã đối chiếu ngày 14/09/2026.

[5] OpenAI, [Custom instructions with AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md). Đã đối chiếu ngày 14/09/2026.

[6] TensorFlow, [Raspberry Pi image classification example](https://github.com/tensorflow/examples/tree/a9265a997d0410c431a292e3553646b3b655ea1f/lite/examples/image_classification/raspberry_pi). Nguồn và phần đánh giá đã được trao đổi trước kế hoạch này.

[7] ecd1012, [rpi_pose_estimation](https://github.com/ecd1012/rpi_pose_estimation/tree/45d0c2bcd7d9464615f6222ff217889c154937b4). Nguồn và phần đánh giá đã được trao đổi trước kế hoạch này.

[8] ecd1012, [rpi_road_object_detection](https://github.com/ecd1012/rpi_road_object_detection/tree/3341a1ad013731cc4dd52b4234dfa1992e59fe14). Nguồn và phần đánh giá đã được trao đổi trước kế hoạch này.

[9] Hackster, [Using Raspberry Pi AI Camera for fall detection](https://www.hackster.io/540813/using-raspberry-pi-ai-camera-for-fall-detection-a772cb). Nguồn và phần đánh giá đã được trao đổi trước kế hoạch này.

[10] Sony Semiconductor Solutions, [Fall detection source](https://github.com/SonySemiconductorSolutions/aitrios-rpi-sandbox/blob/main/examples/falldown-detection/src/main.py). Nguồn đã trao đổi; cần ghim commit cụ thể nếu trích mã trong report.

[11] Microsoft, [Getting Started with Python in VS Code](https://code.visualstudio.com/docs/python/python-tutorial). Đã đối chiếu ngày 14/09/2026: mở workspace, môi trường ảo và chọn Python interpreter.

Mẫu report: `ITCSIU22282_NguyenVuPhuongNghi.docx.pdf`, bản PDF 71 trang bạn cung cấp; mục lục trang 4–5. Các nguồn kỹ thuật trên không thay thế việc đọc và kiểm chứng bài nghiên cứu gốc cho Literature Review.
