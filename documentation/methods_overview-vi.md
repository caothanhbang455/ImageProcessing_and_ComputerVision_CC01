# Tìm hiểu các phương pháp Document Unwarping

Một image chụp trang giấy từ góc nghiêng thường khiến trang trông giống hình thang. Nếu giấy vẫn phẳng, homography có thể sửa phần lớn perspective distortion. Nhưng với một trang sách bị cong ở gáy, sửa bốn góc chưa đủ: những dòng chữ ở giữa trang vẫn có thể bị uốn cong.

Document Unwarping vì thế không có một cách xử lý phù hợp cho mọi trường hợp. Classical computer vision dựa vào những đặc điểm có thể tìm được trên image, như góc trang, boundary hoặc text lines. Deep learning học cách dự đoán geometry từ dữ liệu. Một số phương pháp kết hợp cả hai.

Tài liệu này giải thích những hướng tiếp cận đó và cách cân nhắc khi chọn method. Phần viết và chạy code nằm trong [Hướng dẫn sử dụng repository](implementation_guide-vi.md). Cách đánh giá method được trình bày trong [Thực nghiệm và đánh giá](experiments_and_evaluation-vi.md).

## 1. Cần sửa distortion nào?

Điều đầu tiên nên xem là image bị biến dạng theo cách nào. Một image có thể gặp nhiều vấn đề cùng lúc, nhưng chúng không có chung một cách giải quyết.

**Perspective distortion.** Khi chụp lệch góc, cạnh gần camera thường trông dài hơn cạnh xa. Một tờ giấy hình chữ nhật có thể thành hình thang. Nếu mặt giấy phẳng và xác định được bốn góc, một phép homography thường là điểm bắt đầu hợp lý.

**Page curvature.** Một trang sách cong gần gáy có thể làm text lines bị uốn, dù đường viền bên ngoài đã được chỉnh. Đây là nonrigid distortion: mỗi vùng của trang cần mức chỉnh khác nhau. Một homography duy nhất không xử lý được toàn bộ vấn đề này.

**Folds và creases.** Nếp gấp tạo ra thay đổi đột ngột trên bề mặt giấy. Nếu cố làm mượt geometry quá mạnh, vùng gần nếp gấp có thể bị kéo giãn hoặc biến dạng. Nội dung bị che khuất cũng không thể được khôi phục chỉ bằng cách di chuyển pixel.

**Missing boundaries và background phức tạp.** Có khi giấy bị cắt mất một phần khỏi khung hình, hoặc màu giấy quá giống background. Bản thân điều này chưa chắc là distortion của trang, nhưng khiến việc xác định geometry khó hơn. Shadows và ánh sáng không đều cũng có thể làm page detector nhận sai boundary.

Cần phân biệt **sửa geometry** với **cải thiện appearance**. Tăng contrast, làm rõ chữ hoặc xóa shadow có thể khiến image dễ đọc hơn, nhưng không nhất thiết làm text lines thẳng hơn. OCR cũng là một mục tiêu khác: OCR tốt hơn không tự động chứng minh geometry chính xác hơn.

## 2. Classical methods: dùng geometry có thể quan sát được

Classical methods thường sử dụng những quy tắc hình học và đặc điểm trực tiếp trên image, không cần train một neural model. Chúng dễ kiểm tra từng bước, nhưng phụ thuộc vào độ tin cậy của các đặc điểm đã tìm được.

### Page detection và homography

Cách quen thuộc nhất là tìm vùng document, xác định bốn góc rồi dùng homography để đưa vùng đó về hình chữ nhật. Với trang gần như phẳng, đây là một baseline đơn giản và có thể chạy nhanh.

Tuy nhiên, cần tách hai vấn đề: **phát hiện đúng trang giấy** và **biến đổi đúng geometry**. Nếu detector chọn nhầm một vùng background làm góc trang, homography có được tính chính xác cũng sẽ tạo output sai. Vì vậy, khi một kết quả không tốt, nên xem cả bốn góc mà detector đã chọn trước khi sửa thuật toán biến đổi.

Homography cũng có giới hạn rõ ràng. Nó thay đổi toàn bộ image theo một phép biến đổi phối cảnh. Nó không thể kéo thẳng một dòng chữ cong ở giữa trang theo cách độc lập với những vùng khác.

### Text lines và các đặc điểm của layout

Nhiều document có text lines, cột và đường viền tương đối đều. Nếu các đường này bị cong, chúng có thể cung cấp thông tin về distortion bên trong trang, kể cả khi boundary không rõ.

Một method có thể phát hiện text lines, ước lượng độ cong rồi điều chỉnh geometry để các đường trở nên thẳng hơn. Cách này hữu ích khi cần xử lý curvature, nhưng không phải document nào cũng có đủ text. Image nhiều biểu đồ, bảng, hình minh họa hoặc chữ quá mờ có thể khiến việc phát hiện đường không đáng tin cậy.

Ngoài ra, làm mọi đường trông thật thẳng chưa chắc là cách chỉnh đúng. Một đường cong trong sơ đồ vốn đã cong từ đầu thì không nên bị ép thành đường thẳng.

### Nonrigid correction

Khi distortion thay đổi theo từng vùng, method cần một phép biến đổi linh hoạt hơn homography. Có thể dùng control points, grid, mesh hoặc mô hình bề mặt giấy để xác định vị trí mới cho các phần khác nhau của image.

Thông thường, output được tạo bằng một **backward map**: mỗi pixel trên output cho biết cần lấy pixel ở đâu trên input. Map có thể được xây dựng bằng geometry thủ công hoặc do model dự đoán. Khi sampling, interpolation quyết định cách tính giá trị pixel ở những tọa độ không nguyên.

Một map linh hoạt hơn cũng đi kèm rủi ro: local stretching, vùng không hợp lệ, nếp gấp của map hoặc mất nội dung ngoài trang. Chỉnh cho grid mượt hơn không phải lúc nào cũng đúng, vì đôi khi những thay đổi mạnh phản ánh curvature thật.

## 3. Deep learning: học cách dự đoán geometry

Deep-learning methods sử dụng model đã train để suy ra cách chỉnh image. Thay vì phải tìm chính xác mọi góc hoặc text line bằng rule cố định, model có thể học từ appearance, layout và các distortion trong training data.

Điều model dự đoán không nhất thiết là output image. Nhiều phương pháp dự đoán **dense backward map** hoặc **coarse grid**, rồi dùng sampling để tạo image cuối cùng. Một số model còn dự đoán 3D shape hoặc các bước geometry trung gian.

Bốn hướng nghiên cứu có thể giúp hình dung sự khác biệt:

| Method | Ý tưởng chính |
|---|---|
| [DewarpNet](https://openaccess.thecvf.com/content_ICCV_2019/html/Das_DewarpNet_Single-Image_Document_Unwarping_With_Stacked_3D_and_2D_Regression_ICCV_2019_paper.html) | Ước lượng hình dạng 3D của trang trước khi tính mapping để làm phẳng |
| [UVDoc](https://github.com/tanguymagne/UVDoc) | Dự đoán 2D và 3D grids, dùng geometry để tạo output |
| [DocScanner](https://github.com/fh2019ustc/DocScanner) | Kết hợp tìm vùng document với việc chỉnh geometry qua nhiều bước |
| [RDGR](https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_Revisiting_Document_Image_Dewarping_by_Grid_Regularization_CVPR_2022_paper.html) | Kết hợp dự đoán đã học với những ràng buộc về text, boundary và grid |

Đây là các ví dụ về cách thiết kế method, không phải bảng xếp hạng chất lượng. Các model có thể dùng training data, checkpoint và evaluation protocol khác nhau, nên không thể chỉ nhìn tên model để biết method nào tốt hơn.

Training data ảnh hưởng trực tiếp đến kết quả. Model quen với những trang giấy cong mượt trên dữ liệu synthetic có thể gặp khó khi image thật có shadow nặng, nếp gấp bất thường hoặc missing boundaries. Điều đó không có nghĩa deep learning luôn kém trên image thật; chỉ có nghĩa khả năng generalize phải được kiểm tra, không nên mặc định.

Với UVDoc, cần phân biệt chất lượng **geometry prediction** và bước **sampling**. Nếu output bị méo, vấn đề có thể đến từ grid được dự đoán, cách chuyển đổi tọa độ hoặc interpolation. Không phải lỗi nào cũng nằm ở neural network.

## 4. Hybrid methods: kết hợp nhưng không tự động tốt hơn

Classical và deep learning có thể bổ sung cho nhau. Chẳng hạn, homography có thể xử lý global perspective trước, sau đó neural model xử lý curvature còn lại. Hoặc một predicted grid được điều chỉnh bằng boundary hay text-line constraints.

Cách kết hợp này nghe hợp lý, nhưng kết quả không được đảm bảo. Nếu sửa perspective trước UVDoc, input của model đã khác với image mà model thường nhận. Thêm một bước resampling cũng có thể làm chữ mờ đi. Với một số sample, bước mới giúp được; với sample khác, nó có thể làm kết quả tệ hơn.

Khi thử hybrid, phải xác định rõ các phép biến đổi được áp dụng theo thứ tự nào, output cuối cùng tương ứng với input ban đầu ra sao, và mỗi bước có thay đổi geometry hay chỉ thay đổi appearance. Nếu các map cần ghép với nhau, phải kiểm tra chiều mapping và hệ tọa độ trước khi sampling.

Do đó, hybrid là một **hướng cần thực nghiệm**, không phải một cải tiến đã được chứng minh. Baseline gốc cần được giữ nguyên để biết phần mới thực sự tạo ra khác biệt gì.

## 5. Chọn method theo tình huống

| Tình huống | Hướng nên thử trước | Cần kiểm tra thêm |
|---|---|---|
| Trang phẳng, thấy rõ bốn góc | Page detection và homography | Detector có chọn đúng góc không? |
| Trang sách cong, text lines bị uốn | Nonrigid method hoặc learned geometry | Có còn curvature và local stretching không? |
| Boundary bị thiếu hoặc lẫn background | Model tận dụng nhiều image cues, hoặc detector phù hợp | Có bị crop mất nội dung không? |
| Trang có nhiều text lines rõ | Text-line constraints | Các đường được phát hiện có thật sự là text không? |
| Có predicted grid nhưng output bị méo cục bộ | Kiểm tra grid, map convention và sampling | Lỗi prediction hay lỗi chuyển đổi tọa độ? |

Bảng này chỉ giúp chọn điểm bắt đầu. Không có method nào chắc chắn thắng trên mọi dataset. Một method có thể sửa perspective tốt nhưng không giữ được toàn bộ trang; một method khác có thể làm text thẳng nhưng khiến vài ký tự bị kéo giãn.

Khi đánh giá output, đừng chỉ nhìn một ảnh đẹp nhất. Nên xem thêm những sample thất bại và những chỗ method xử lý chưa tốt: sai boundary, còn curvature, local stretching, clipping, blur hoặc background artifacts. Từ đó mới xác định phần nào của method cần cải thiện.

Các file và function cần dùng để implement method được giới thiệu trong [Hướng dẫn sử dụng repository](implementation_guide-vi.md). Để biết một thay đổi có thực sự cải thiện kết quả hay không, xem [Thực nghiệm và đánh giá](experiments_and_evaluation-vi.md).
