# EDA — Document Unwarping

Hai notebook độc lập, chạy CPU, không sửa ảnh nguồn:

- `01_docunet_eda.ipynb`: `data/DocUNet/crop/` và `data/DocUNet/scan/`.
- `02_dir300_eda.ipynb`: `data/DIR300/dist/` và `data/DIR300/gt/`.

## Cài đặt

Chép `.gitignore` và `requirements-eda.txt` vào thư mục gốc repo; chép cả `notebooks/` vào repo.
Giữ README gốc hiện tại của nhóm. Nếu đã có `.gitignore`, ghép các rule vào file hiện có.

```text
ImageProcessing_and_ComputerVision_CC01/
  .gitignore
  README.md                         # README hiện tại của nhóm
  requirements-eda.txt
  notebooks/
    README.md                       # hướng dẫn này
    01_docunet_eda.ipynb
    02_dir300_eda.ipynb
  data/
    DocUNet/
      crop/crop/...                  # cũng hỗ trợ crop/... không lồng
      scan/scan/...                  # cũng hỗ trợ scan/... không lồng
    DIR300/
      dist/...
      gt/...
  raw/                              # ZIP tải về
  reports/eda/                      # sinh ra khi chạy notebook
```

Từ terminal tại thư mục gốc repo:

```bash
python -m pip install -r requirements-eda.txt
python -m jupyter lab
```

Hoặc mở notebook bằng VS Code, chọn Python kernel đã cài dependency và Run All.
Notebook tự tìm repo từ thư mục đang chạy. Nếu không tìm được, sửa `PROJECT_ROOT` ở cell cấu hình
bằng `Path(r'D:\HCMUT\HK261\CV\ImageProcessing_and_ComputerVision_CC01')`.
Không đổi tên các ảnh có hậu tố `copy`: notebook đã hỗ trợ tên như `10_1 copy.png`.

## Các phần có sẵn

1. Kiểm kê số lượng, cấu trúc thư mục, ID và view.
2. Kiểm tra cặp input–reference, tên lạ, ID trùng, ID/cặp thiếu và lỗi đọc ảnh.
3. Thống kê W/H, aspect ratio, diện tích pixel, megapixels, dung lượng, mode và số kênh.
4. Mẫu ngẫu nhiên và cặp ảnh–reference cạnh nhau, giữ đúng tỷ lệ ảnh.
5. Ảnh min/max W, H, W×H và dung lượng; có bảng toàn bộ trường hợp đồng hạng.
6. Thống kê RGB, saturation, độ sáng grayscale, tương phản std và P95−P05, tỷ lệ vùng tối/sáng.
7. Histogram, scatter W/H, ảnh cực trị về ánh sáng/tương phản/màu; so sánh kích thước từng cặp.
8. Xuất CSV, PNG và metadata tái lập vào `reports/eda/<dataset>/`.

W/H, diện tích và dung lượng đo từ file gốc; W/H đã áp dụng EXIF orientation khi có.
Phép đo màu/ánh sáng là ước lượng trên thumbnail cạnh dài tối đa 512 px, alpha đặt trên nền trắng.
Mỗi reference chỉ xuất hiện một lần trong thống kê riêng, kể cả DocUNet dùng một reference cho hai input.
Chỉ số sáng/tương phản mô tả dữ liệu, không đo chất lượng unwarping; không tự tính SSIM/LD giữa input thô và GT.

Nếu dữ liệu đang tải, notebook vẫn phân tích các file hiện có. Tải xong cần Run All lại để cập nhật audit.
Không dùng số ảnh có trong cây thư mục cũ làm kết luận về dữ liệu cuối cùng.

## Kiểm tra trước khi push

```bash
git status --short
git check-ignore -v data/DocUNet/crop/crop/10_1\ copy.png
git check-ignore -v data/DIR300/dist/1.png
git check-ignore -v raw/crop.zip
```

Trên Windows CMD/PowerShell, lệnh kiểm tra đường dẫn có dấu cách nên viết:

```powershell
git check-ignore -v "data/DocUNet/crop/crop/10_1 copy.png"
```

Nếu dữ liệu đã được `git add`/commit trước khi tạo `.gitignore`, rule mới không tự bỏ theo dõi.
Kiểm tra rồi bỏ các thư mục dữ liệu khỏi Git index (vẫn giữ nguyên file trên máy):

```bash
git ls-files -- data raw datasets
git rm -r --cached --ignore-unmatch -- data raw datasets
```

Lệnh trên thay đổi Git index, không xóa ảnh khỏi ổ đĩa. Nó cũng không xóa dữ liệu trong lịch sử commit cũ.
Chỉ thực hiện nếu các thư mục đó đang được Git theo dõi.

Trước khi commit notebook, Clear All Outputs và Save để tránh đưa ảnh nhúng dung lượng lớn vào Git.
Để commit các file mới này, dùng đường dẫn cụ thể:

```bash
git add .gitignore requirements-eda.txt notebooks/
git status --short
git commit -m "chore: ignore datasets and add DocUNet DIR300 EDA notebooks"
```

Gói này chỉ chứa notebook, cấu hình và hướng dẫn; không chứa dataset hoặc kết quả EDA thật.
