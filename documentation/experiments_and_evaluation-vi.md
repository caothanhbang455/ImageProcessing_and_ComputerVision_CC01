# Thực nghiệm và đánh giá kết quả

Hai method có thể tạo ra output nhìn tương tự nhau nhưng đạt evaluation scores khác nhau. Ngược lại, một image trông rõ chữ hơn chưa chắc có geometry chính xác hơn. Contrast, cropping và alignment đều có thể ảnh hưởng đến một số metrics, dù các dòng chữ vẫn cong như cũ.

Vì vậy, trước khi so sánh hai method, cần xác định experiment muốn kiểm tra điều gì, dataset nào phù hợp, metric nào trả lời đúng câu hỏi đó và những trường hợp failed sẽ được ghi nhận ra sao. Tài liệu này tập trung vào cách thiết kế và đọc kết quả experiment, không hướng dẫn lại cách chạy Python. Phần chạy code nằm trong [Hướng dẫn sử dụng repository](implementation_guide-vi.md); quy định chính xác của từng benchmark nằm trong [Evaluation protocols](reference/evaluation_protocols.md).

## 1. Experiment đang trả lời câu hỏi gì?

Giả sử có hai output từ cùng một image: một output dùng homography, output còn lại dùng UVDoc. Nếu chỉ hỏi “cái nào tốt hơn?”, rất khó chọn cách đánh giá.

Câu hỏi cụ thể hơn sẽ hữu ích: method nào làm text lines thẳng hơn, method nào giữ được nhiều nội dung hơn, method nào xử lý được nhiều sample hơn, hoặc method nào chạy nhanh hơn? Những câu hỏi đó không nhất thiết có cùng câu trả lời.

Tương tự, nếu thêm CLAHE trước UVDoc, output có thể trông rõ hơn vì contrast thay đổi. Muốn kết luận geometry tốt hơn, phải đo geometry, không chỉ dựa vào việc image nhìn đẹp hơn hoặc similarity score tăng.

Một experiment hợp lý cần nêu rõ phần thay đổi và điều muốn kiểm tra. Khi có thể, chỉ thay **một yếu tố mỗi lần**. Nếu vừa đổi checkpoint, vừa resize input, vừa thay interpolation, sẽ khó biết yếu tố nào khiến score thay đổi.

## 2. Chọn dataset phù hợp

Các dataset hiện có không phục vụ cùng một mục đích:

| Dataset | Dùng để làm gì | Cần lưu ý |
|---|---|---|
| UVDoc Final | Phát triển hoặc train method cần annotation hình học | Split hiện tại tách base geometry, nhưng có texture/background trùng nhau |
| UVDoc Benchmark | Đo độ thẳng của các line và chuẩn bị texture output | Có 50 sample; line metric yêu cầu coarse grid phù hợp |
| DocUNet | Đánh giá trên image chụp thật và reference scan | 130 image từ 65 document; document 64 có vấn đề orientation |
| DIR300 | Kiểm tra trên một tập image thật khác | 300 cặp input/reference, protocol riêng |

Cần phân biệt training set, validation set và benchmark. Training data dùng để train model; validation hỗ trợ chọn config hoặc checkpoint; benchmark dùng để đánh giá phương án đã chọn.

Những sample đã được xem nhiều lần trong lúc phát triển method không còn là các ví dụ hoàn toàn chưa từng tiếp xúc. Vẫn có thể đưa chúng vào full benchmark chính thức, nhưng khi nói về khả năng generalize thì phải giải thích rõ chúng đã được sử dụng như thế nào.

Với UVDoc Final, split đã lưu gồm 17.992 training image và 2.008 validation image, tách theo base geometry. Điều đó không đảm bảo texture và background hoàn toàn khác nhau. Hơn nữa, split này không chứng minh official pretrained UVDoc checkpoint chưa từng được train trên những image tương ứng.

Với DocUNet, hai góc chụp của cùng một document không phải hai tài liệu độc lập. Khi phân tích kết quả theo document, cần giữ thông tin nhóm 65 document gốc. Xem [Dataset protocols](reference/dataset_protocols.md) nếu cần kiểm tra cách ghép sample và reference.

## 3. So sánh công bằng là như thế nào?

Điểm bắt đầu là **cùng danh sách sample** và **cùng evaluation protocol**. Không nên so sánh một method chạy trên toàn bộ dataset với một method chỉ chạy trên vài image đã chọn mà không ghi rõ sự khác biệt.

Tuy nhiên, công bằng không có nghĩa hai method bắt buộc dùng cùng architecture hoặc trả output cùng kích thước. Homography có thể tạo image đã crop sát trang; neural model có thể giữ kích thước gần input. Việc crop, resize hay align để tính metric phải tuân theo protocol đã chọn, không điều chỉnh tùy hứng để output của một method trông tốt hơn.

Reference image chỉ được sử dụng trong evaluation theo đúng quy định. Nếu lấy reference scan để xác định góc hoặc chọn crop tốt nhất ngay trong inference, đó đã là một experiment khác, cần nói rõ method có sử dụng reference.

Khi lưu kết quả, nên ghi những thông tin có thể ảnh hưởng đến phép so sánh: sample IDs, input variant, model/checkpoint, config, preprocessing, interpolation, output size và protocol. Nếu so runtime, cần ghi cả hardware và cách bắt đầu/kết thúc phép đo.

Một ví dụ dễ kiểm soát là thay interpolation: giữ nguyên backward map, chỉ đổi bước sampling. Khi đó, khác biệt ở output có thể liên hệ trực tiếp với interpolation. Nếu grid cũng thay đổi, không thể tách riêng hai tác động.

## 4. Hiểu các metrics chính

Không có một metric duy nhất nói lên toàn bộ chất lượng document unwarping.

| Metric | Đo điều gì? | Hướng tốt hơn |
|---|---|---|
| MS-SSIM | Mức độ giống nhau về cấu trúc giữa output và reference theo protocol | Cao hơn |
| LD | Local distortion theo cách tính của reference benchmark | Thấp hơn |
| AD | Aligned distortion theo reference benchmark | Thấp hơn |
| UVDoc line deviation | Những line vốn thẳng còn lệch bao nhiêu sau khi chỉnh | Thấp hơn |
| OCR CER | Tỷ lệ ký tự nhận diện sai so với text reference phù hợp | Thấp hơn |
| Runtime | Thời gian xử lý trong phạm vi đã xác định | Thấp hơn |

**MS-SSIM không chỉ đo geometry.** Image được tăng contrast, crop khác hoặc align khác vẫn có thể thay đổi MS-SSIM. Vì vậy, MS-SSIM cao hơn chưa đủ để khẳng định phương pháp sửa curvature tốt hơn.

**LD và AD không phải phép so pixel đơn giản.** Benchmark dùng những bước alignment và thuật toán tham chiếu cụ thể. LD theo một số benchmark lịch sử phụ thuộc vào SIFT flow. Không thể thay bằng optical flow khác rồi giữ nguyên tên metric và so trực tiếp với paper.

**Line deviation không nói hết chất lượng image.** Text lines có thể thẳng hơn nhưng một phần trang bị crop mất, hoặc ký tự bị kéo giãn ở những vùng khác. Cần xem cả output, không chỉ một giá trị trung bình.

OCR CER cũng cần thận trọng: phải có text reference đủ tin cậy, cùng OCR engine và quy định normalize text. OCR chạy trên reference scan không tự động tạo ra ground truth. Với repository hiện tại, OCR nên được xem là hướng đánh giá bổ sung, không phải benchmark đã được xác minh.

## 5. Khi nào tính được score?

Mỗi dataset có evaluation protocol riêng. Trước khi chạy, cần kiểm tra loại output mà metric yêu cầu.

| Dataset | Protocol | Điều kiện quan trọng |
|---|---|---|
| UVDoc Benchmark | `uvdoc_lines_official_n25_v1` | Cần predicted coarse grid theo official UVDoc convention |
| UVDoc Benchmark | `uvdoc_texture_prepare_v1` | Cần grid để tạo texture output |
| UVDoc Benchmark | `uvdoc_visual_matlab_v1` | Visual reference metrics cần MATLAB và các dependencies tương ứng |
| DocUNet | `docunet_matlab_legacy_v1` | Chuẩn bị output/reference theo protocol cũ |
| DocUNet | `docunet_matlab_corrected64_v1` | Hai sample của document 64 cần xử lý orientation riêng |
| DIR300 | `dir300_matlab_v1` | Chuẩn bị dữ liệu cho MATLAB MS-SSIM và LD; không có AD trong wrapper đã chọn |

Ví dụ, một classical method chỉ trả output image vẫn có thể đi qua bước reference preparation của DocUNet hoặc DIR300. Nhưng image đó không đủ để chạy UVDoc official line metric, vì metric cần **coarse grid**, không chỉ image đã chỉnh hoặc một dense map bất kỳ.

Trong DocUNet, `64_1` và `64_2` có vấn đề orientation so với reference. Corrected protocol yêu cầu inference từ input đã xử lý đúng orientation. Không nên mặc định xoay output sau inference là tương đương.

Điểm dễ nhầm nhất là **preparation thành công không có nghĩa score đã được tính**. Colab đã hoàn thành UVDoc line evaluation trên 50 sample và đối chiếu chính xác với official implementation trên hai sample được chọn. Nhưng các phép đánh giá dùng MATLAB chưa được tính trong môi trường đó. Trạng thái `UNAVAILABLE_DEPENDENCY` không phải là điểm 0, cũng không phải lỗi hình học của method.

## 6. Đọc bảng kết quả cho đúng

Đừng chỉ nhìn average. Trước khi so sánh score, cần biết mỗi method đã thử bao nhiêu sample, chạy thành công bao nhiêu sample, và có bao nhiêu sample thực sự được tính metric.

Giả sử method A tạo được output cho 100/100 image, còn method B chỉ thành công 70/100. Nếu chỉ tính average trên 70 image thành công, B có thể được lợi vì đã bỏ qua những case khó. Phải báo cả **success rate** lẫn **metric coverage**.

Khi so sánh trực tiếp hai method, nên tính chênh lệch trên **cùng những sample có score hợp lệ ở cả hai bên**. Sau đó trình bày số sample không tính được metric của từng method để người đọc không nhầm paired result với toàn bộ benchmark.

Bảng báo cáo có thể như sau. Các dấu `...` chỉ là chỗ điền kết quả, không phải số liệu đã đo:

| Method | Attempted | Success | Valid metric | Mean | Median |
|---|---:|---:|---:|---:|---:|
| A | ... | ... | ... | ... | ... |
| B | ... | ... | ... | ... | ... |

Mean cho biết xu hướng chung, nhưng có thể che khuất những sample rất kém. Median, phân bố score, số sample được cải thiện và số sample tệ hơn sẽ giúp hiểu rõ hơn. Với DocUNet, khi muốn rút ra kết luận ở mức document, nên xét cả nhóm 65 document chứ không coi 130 views là độc lập hoàn toàn.

## 7. Xem output để hiểu method lỗi ở đâu

Một bảng metric không nói được toàn bộ nguyên nhân. Khi có score bất thường, nên xem input, output, reference và cách alignment đã được dùng.

Các vấn đề đáng kiểm tra gồm:

| Hiện tượng | Có thể liên quan đến |
|---|---|
| Boundary bị chọn sai | Page detection |
| Trang vẫn nghiêng hoặc thành hình thang | Residual perspective |
| Text lines vẫn cong | Residual curvature |
| Chữ bị kéo giãn | Local stretching hoặc map không hợp lệ |
| Mất một phần trang | Cropping, mapping hoặc page localization |
| Chữ bị mờ, viền có artifacts | Sampling, interpolation hoặc appearance processing |
| Score thấp dù output trông hợp lý | Reference alignment, orientation hoặc khác biệt appearance |

Đây là gợi ý khi review bằng mắt, không phải nhãn lỗi tự động đã có trong repository. Một sample có thể gặp nhiều vấn đề cùng lúc. Đặc biệt, không nên chỉ dựa vào một score thấp để kết luận rằng method bị sai geometry; reference hoặc alignment cũng có thể là nguyên nhân.

Khi chọn image để trình bày, nên có cả trường hợp thành công, thất bại và những case mà hai method cho kết quả khác nhau. Chỉ đưa vài output đẹp nhất sẽ không thể hiện được độ tin cậy của method.

## 8. Viết kết quả experiment

Một phần báo cáo tốt không cần dài, nhưng phải trả lời được các câu hỏi: đã so sánh method nào, trên dataset và sample nào, dùng checkpoint/config gì, tính metric theo protocol nào, có bao nhiêu output hợp lệ và giới hạn của kết quả là gì.

Có ba mức kết luận cần phân biệt:

- **Implementation đã được kiểm tra:** Một function hoặc metric khớp reference trên các test đã chọn. Đây là bằng chứng code thực hiện đúng trường hợp được kiểm tra.
- **Experiment cho thấy một khác biệt:** Method A đạt score khác method B trên một tập sample và protocol cụ thể. Đây là kết quả đo được trong điều kiện đã nêu.
- **Kết luận rộng hơn:** Method tốt hơn trong một nhóm điều kiện hoặc có khả năng generalize. Cần bằng chứng đủ rộng, không thể suy ra từ vài sample được chọn.

Chẳng hạn, nếu một thay đổi giúp UVDoc line deviation thấp hơn trên một nhóm image nhưng làm tăng clipping ở vài case khác, báo cáo nên nêu cả hai. Kết luận phù hợp là method cải thiện metric đó trong điều kiện đã thử, kèm những hạn chế quan sát được, chứ không phải “method mới tốt hơn UVDoc”.

Cuối cùng, mỗi bảng score nên liên kết được với run tương ứng trong `runs/`, nơi có config, per-sample results và status. Muốn biết cách tạo các output này, xem [Hướng dẫn sử dụng repository](implementation_guide-vi.md). Muốn kiểm tra cách tính metric chính xác, xem [Evaluation protocols](reference/evaluation_protocols.md).
