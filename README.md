# 🛒 Analyze & Forecast Sale System (Hệ thống Phân tích & Dự đoán Doanh số Bán hàng)

> Đồ án tốt nghiệp / Xây dựng hệ thống phân tích dữ liệu thương mại điện tử và dự đoán doanh số sản phẩm sử dụng Machine Learning và Web Dashboard tích hợp.

---

## 📌 Giới thiệu Đề tài
Hệ thống được xây dựng nhằm giải quyết bài toán khai thác dữ liệu từ các nền tảng thương mại điện tử (Dữ liệu Amazon Electronics). Đồ án cung cấp một giải pháp toàn diện từ khâu thu thập, làm sạch dữ liệu, phân tích khám phá (EDA), huấn luyện các mô hình học máy (Machine Learning) cho đến việc triển khai giao diện trực quan hóa dữ liệu và dự đoán thời gian thực.

---

## 🛠️ Các Tính năng Chính của Hệ thống
1. **Dashboard Phân tích Trực quan (Storytelling EDA):**
   * Tổng quan các chỉ số KPI doanh số, sản lượng, rating.
   * Biểu đồ phân tích danh mục sản phẩm bán chạy nhất.
   * Khảo sát các yếu tố ảnh hưởng đến sản lượng bán ra.
   * Phân tích cơ cấu khoảng giá sản phẩm và scatter plot tương quan giữa Rating & Doanh số.
2. **Giải thích Mô hình Học máy (Model Interpretability):**
   * Tích hợp biểu đồ **Feature Importance** và phân tích chiều sâu bằng **SHAP (SHapley Additive exPlanations)** giúp minh bạch hóa cách mô hình đưa ra dự đoán.
3. **Dự đoán Doanh số Thông minh:**
   * Cho phép nhập liệu hoặc **tải lên file CSV** để dự đoán hàng loạt sản lượng bán ra của sản phẩm dựa trên mô hình **Random Forest Regressor** đã được tối ưu.
   * Cho phép xuất kết quả dự đoán ra file CSV trực tiếp từ giao diện web.

---

## 🗂️ Cấu trúc Thư mục Dự án
```text
Analyze&ForecastSaleSystem/
│
├── data/                      # Chứa các tập dữ liệu thô, dữ liệu sạch và biểu đồ SHAP/Feature Importance
├── notebooks/                 # Các file Jupyter Notebook (EDA, làm sạch dữ liệu, huấn luyện mô hình)
├── saved_artifacts/           # Chứa file mô hình đã huấn luyện (.joblib)
├── src/                       # Mã nguồn các module chức năng (Dashboard, Data Cleaner, Feature Prep)
├── app.py                     # File chính chạy ứng dụng Streamlit Dashboard
├── requirements.txt           # Danh sách các thư viện Python phụ thuộc
└── README.md                  # Tài liệu giới thiệu dự án
