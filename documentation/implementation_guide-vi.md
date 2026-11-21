# Hướng dẫn sử dụng và phát triển Document Unwarping

Repository đã có sẵn các phần đọc dataset, xử lý geometry, chạy UVDoc và evaluation. Khi thử một method mới, không cần viết lại những phần này. Chỉ cần một Python function nhận image và trả về `MethodResult`, sau đó dùng `run_method()` để chạy trên các sample đã chọn.

Tài liệu này tập trung vào cách sử dụng repository: tìm đúng file, đọc image, viết method, chạy thử và xem output. Nếu muốn hiểu các hướng xử lý khác nhau hoặc cách so sánh kết quả, xem thêm [Tổng quan các phương pháp](methods_overview-vi.md) và [Thực nghiệm và đánh giá](experiments_and_evaluation-vi.md).

## 1. Repository có những phần nào?

Các thư mục chính được chia theo công việc:

```text
Document Unwarping/
├── configs/          # Dataset, model và evaluation settings
├── data/             # Dataset readers, sample records, image loading
├── geometry/         # Coordinates, grids, homography, sampling
├── models/           # MethodResult và model implementations
├── evaluation/       # Metrics, evaluation protocols và runner
├── scripts/          # Chạy method, UVDoc và evaluation
├── tests/            # Automated tests
├── notebooks/        # Colab experiments
├── documentation/    # Guides và tài liệu tham khảo
├── dataset/          # Dữ liệu tải về, không đưa lên Git
├── checkpoints/      # Model checkpoints, không đưa lên Git
└── runs/             # Output của các lần chạy, không đưa lên Git
```

`data/` và `dataset/` dễ bị nhầm. `data/` chứa Python code để đọc và quản lý dataset. `dataset/` là nơi lưu image, annotation và các file đã tải về. Tương tự, `models/` chứa implementation của model, còn `checkpoints/` lưu weights.

Nếu viết một classical method, có thể tạo `models/classical/` để đặt implementation. Neural model mới thường nằm trong `models/deep/`. Các function xử lý geometry có thể dùng lại ở nhiều method thì nên đặt trong `geometry/`, thay vì copy sang từng model.

`notebooks/` phù hợp để thử nghiệm và xem output. Phần xử lý cần chạy nhiều lần nên được viết trong file `.py`; notebook chỉ import và gọi function. Các file hướng dẫn hiện tại nằm ở `documentation/`, protocol chính xác nằm trong [`documentation/reference/`](reference/), còn báo cáo cũ được giữ ở [`documentation/archive/`](archive/).

## 2. Đọc image từ dataset

Dự án sử dụng manifest để xác định từng sample, thay vì quét folder rồi ghép image theo thứ tự file. Manifest ghi ID, đường dẫn input, reference nếu có và annotation liên quan.

Ví dụ chạy trên Google Colab. Trước tiên mount Google Drive và chuyển vào thư mục dự án:

```python
from google.colab import drive

drive.mount("/content/drive")
%cd "/content/drive/MyDrive/Document Unwarping"
```

Sau đó đọc một image từ DIR300:

```python
from pathlib import Path
from IPython.display import display
from PIL import Image

from data.manifests import load_records
from data.paths import PathResolver
from data.images import read_image

ROOT = Path("/content/drive/MyDrive/Document Unwarping")
records = load_records("dir300", ROOT, variant="dist")
sample = records[0]

image_path = PathResolver(ROOT).resolve(sample.input_path)
image = read_image(image_path, mode="RGB")

print("Sample:", sample.sample_key)
print("Shape:", image.shape)
display(Image.fromarray(image))
```

`load_records()` trả về các `SampleRecord`. `sample_key` là ID duy nhất mà repository dùng để nối input với output và metric của cùng một sample. `sample_id` là ID theo dataset gốc; `variant` cho biết đang chọn loại input nào. `input_path` và `reference_path` là đường dẫn tương đối; `PathResolver` tìm file thực tế trên Drive hoặc Colab.

Các dataset đang được hỗ trợ:

| Dataset | Tên truyền vào `load_records()` | `variant` |
|---|---|---|
| UVDoc Final | `uvdoc_final` | `rendered` |
| UVDoc Benchmark | `uvdoc_benchmark` | `rendered` |
| DocUNet | `docunet` | `crop` hoặc `original` |
| DIR300 | `dir300` | `dist` |

Không nên tự ghép input với reference bằng thứ tự file. Chẳng hạn, DocUNet có nhiều ảnh chụp của cùng một trang, còn DIR300 ghép cặp theo ID. Quy định của từng dataset nằm trong [Dataset protocols](reference/dataset_protocols.md).

## 3. Thêm một method mới

Method đơn giản nhất là một function nhận image RGB dạng NumPy array (`uint8`) và trả về `MethodResult`:

```python
from models.interfaces import MethodResult

def identity_method(image) :
    return MethodResult(status="OK", image=image.copy())
```

Function này không chỉnh image. Nó chỉ giúp kiểm tra toàn bộ quy trình, từ đọc input đến lưu output. Khi có thuật toán thật, thay `image.copy()` bằng kết quả xử lý.

`MethodResult` có một số field quan trọng:

| Field | Ý nghĩa |
|---|---|
| `status` | `OK`, `FAILED`, `UNAVAILABLE` hoặc `REVIEW_REQUIRED` |
| `image` | Output image khi xử lý thành công |
| `failure_reason` | Lý do khi `status` không phải `OK` |
| `backward_map` | Dense map, nếu method trả thêm geometry |
| `map_space` | Hệ tọa độ của `backward_map` |
| `valid_mask` | Mask cho biết pixel nào của output hợp lệ |
| `metadata` | Thông tin bổ sung nếu cần |

Method không nhất thiết phải trả về map hoặc mask. Nếu chỉ tạo ra image đã chỉnh, `MethodResult(status="OK", image=output)` là đủ, miễn là output hợp lệ và có dtype `uint8`.

Khi method không tìm được document boundary, nên báo lỗi rõ ràng:

```python
return MethodResult(
    status="FAILED",
    image=None,
    failure_reason="No valid page boundary found",
)
```

Không nên trả lại input image với status `OK` chỉ để che một lần xử lý thất bại. Nếu thật sự muốn dùng identity fallback, hãy coi đó là một lựa chọn riêng và ghi rõ trong config.

Với neural model, nên load model và checkpoint **một lần** trước khi chạy các sample. Function đưa vào `run_method()` chỉ làm nhiệm vụ xử lý một image. Nếu sử dụng pretrained model, ghi tên model, checkpoint và những settings quan trọng vào `config` để sau này biết run đó dùng đúng phiên bản nào.

## 4. Geometry: vài quy ước không nên nhầm

NumPy image thường có shape `(height, width, channels)`, nhưng tọa độ một pixel lại viết theo thứ tự `(x, y)`. `x` đi từ trái sang phải, `y` đi từ trên xuống dưới.

`backward_map` có shape `(H, W, 2)`. Ở mỗi vị trí trên output, map cho biết tọa độ tương ứng cần lấy trên input image:

```text
backward_map[y_out, x_out] = (x_input, y_input)
```

Đó là lý do gọi là *backward*: map đi từ output về input, không phải từ input sang output.

Ví dụ identity map:

```python
import numpy as np
from models.interfaces import MethodResult

def identity_with_map(image) :
    h, w = image.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)
    backward_map = np.stack((x, y), axis=-1)

    return MethodResult(
        status="OK",
        image=image.copy(),
        backward_map=backward_map,
        map_space="source_pixels",
    )
```

Với `source_pixels`, tọa độ trong map tính theo pixel của input. Với `normalized_minus1_plus1`, tọa độ nằm trong khoảng từ `-1` đến `1`. Hai cách biểu diễn không thể thay thế nhau chỉ bằng cách đổi tên `map_space`.

Dense map trong ví dụ cũng **khác** coarse grid của UVDoc. Vì thế một method trả `backward_map` chưa chắc dùng được UVDoc line metric; metric đó yêu cầu grid đúng theo official protocol.

Ngoài ra, interpolation, padding và `align_corners` có thể khiến output khác nhau dù dùng cùng geometry. Nếu một experiment muốn so sánh geometric correction, cần tránh thay đổi các settings này ngoài chủ đích. Chi tiết nằm trong [Geometry conventions](reference/geometry_conventions.md).

## 5. Chạy method và xem output

`scripts/run_method.py` cung cấp function `run_method()`. Ví dụ dưới đây tự tạo một run mới trên hai sample DIR300:

```python
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

from data.manifests import load_records
from models.interfaces import MethodResult
from scripts.run_method import run_method

ROOT = Path("/content/drive/MyDrive/Document Unwarping")

def identity_method(image) :
    return MethodResult(status="OK", image=image.copy())

records = load_records("dir300", ROOT, variant="dist")[:2]
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
output_dir = ROOT / "runs/inference" / ("identity_" + run_id)

result = run_method(
    records=records,
    method=identity_method,
    root=ROOT,
    output_dir=output_dir,
    method_id="identity_example",
    config={"purpose": "interface check"},
)
print(result)
```

Mỗi run phải có output directory mới; `run_method()` không ghi đè kết quả cũ. Nó xử lý từng sample, lưu image khi thành công và ghi lại lý do nếu thất bại.

```text
runs/inference/<run_id>/
├── config.json
├── provenance.json
├── environment.json
├── summary.json
├── samples.csv
├── artifacts.csv
└── predictions/<method_id>/
    ├── images/
    ├── dense_maps/     # Nếu có backward_map hoặc valid_mask
    └── metadata/       # Nếu có metadata
```

Sau khi chạy xong, mở `samples.csv` để xem status của từng sample. Output image nằm trong `predictions/<method_id>/images/`. `artifacts.csv` ghi đường dẫn, dung lượng và SHA-256 hash của các file đã lưu. `config.json` và `provenance.json` giúp xác định run đã dùng method và dataset nào.

Nếu muốn chạy official UVDoc, dùng workflow riêng trong `scripts/run_inference.py`, vì model này còn lưu coarse grid để phục vụ UVDoc Benchmark. Không cần sửa official UVDoc implementation để thêm một method độc lập.

## 6. Đánh giá kết quả

Inference và evaluation là hai việc khác nhau. Sau khi có output, `evaluate_inference_run()` đọc các file đã lưu rồi xử lý chúng theo evaluation protocol được chọn.

| Dataset | Protocol | Cần lưu ý |
|---|---|---|
| UVDoc Benchmark | `uvdoc_lines_official_n25_v1` | Cần coarse grid tương thích với UVDoc |
| UVDoc Benchmark | `uvdoc_texture_prepare_v1` | Chuẩn bị output texture từ grid của UVDoc |
| UVDoc Benchmark | `uvdoc_visual_matlab_v1` | Reference visual metrics cần MATLAB |
| DocUNet | `docunet_matlab_legacy_v1` | Reference preparation theo quy định cũ |
| DocUNet | `docunet_matlab_corrected64_v1` | Document 64 có vấn đề orientation cần xử lý riêng |
| DIR300 | `dir300_matlab_v1` | Reference preparation cho MS-SSIM và LD |

Ví dụ dùng DIR300 run vừa tạo để chuẩn bị evaluation:

```python
from evaluation.runner import evaluate_inference_run

result = evaluate_inference_run(
    root=ROOT,
    inference_run=output_dir,
    protocol_id="dir300_matlab_v1",
    output_dir=ROOT / "runs/evaluation" / ("identity_" + run_id),
)
print(result["status_counts"])
```

Nếu chưa có MATLAB và các dependencies cần thiết, bước preparation vẫn có thể thành công nhưng **chưa có MS-SSIM hoặc LD score**. Đừng xem trạng thái `OK` của preparation là bằng chứng method đạt kết quả tốt.

Trong evaluation directory, `samples.csv` cho biết từng sample được xử lý ra sao; `metrics/per_sample.csv` ghi metric và lý do thiếu score; `metrics/summary.csv` chứa phần tổng hợp. `report.md` giúp xem nhanh run đó đã làm gì.

Colab đã chạy đủ 50 sample của UVDoc line metric và đối chiếu thành công với official implementation trên hai sample được chọn. Reference metrics cần MATLAB chưa được tính ở môi trường đó. Xem [Evaluation protocols](reference/evaluation_protocols.md) để biết rõ input và giới hạn của từng protocol, hoặc [Thực nghiệm và đánh giá](experiments_and_evaluation-vi.md) nếu cần so sánh các method.

## 7. Test và xử lý lỗi

Sau khi sửa method, chạy các test có sẵn trong Colab:

```python
%cd "/content/drive/MyDrive/Document Unwarping"
!python -m pytest -q tests/
```

Lần kiểm tra đầy đủ gần nhất ghi nhận **95 tests passed**. Con số này chỉ phản ánh phiên bản đã kiểm tra, không đảm bảo những thay đổi sau đó luôn hoạt động.

Khi gặp lỗi, bắt đầu từ nơi gần vấn đề nhất:

| Vấn đề | Kiểm tra |
|---|---|
| Không tìm thấy input image | `sample.input_path`, `data/paths.py`, folder dataset trên Drive |
| Input và reference không khớp | `sample_key`, `reference_path`, [Dataset protocols](reference/dataset_protocols.md) |
| Method trả `FAILED` | `failure_reason` trong `samples.csv` và function của method |
| Sai shape hoặc tọa độ map | `models/interfaces.py`, [Geometry conventions](reference/geometry_conventions.md) |
| UVDoc line metric không chạy | Kiểm tra coarse grid, không chỉ output image |
| Chưa có MS-SSIM hoặc LD | Kiểm tra MATLAB, SIFT flow và status trong evaluation output |
| `%cd` hoặc `!python` báo lỗi trong PowerShell | Đây là lệnh dành cho Colab notebook, không phải PowerShell |

`dataset/`, `runs/`, `checkpoints/` và những file local không nên được commit lên Git. Khi cập nhật documentation hoặc code, chỉ stage đúng file đã thay đổi. Các notebook và báo cáo cũ được lưu để tra cứu, nhưng không cần đọc hết chúng trước khi thử một method mới.
