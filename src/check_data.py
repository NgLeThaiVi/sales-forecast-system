from pathlib import Path
import pandas as pd
from pandas.testing import assert_frame_equal
from data_cleaner import AmazonDataCleaner

# 1. Định vị đường dẫn
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "amazon_products_sales_data_uncleaned.csv"
CLEANED_FILE = PROJECT_ROOT / "data" / "amazon_electronics_data_cleaned.csv"

try:
    # 2. Đọc dữ liệu
    df_raw = pd.read_csv(RAW_FILE)
    df_notebook = pd.read_csv(CLEANED_FILE)

    # Bỏ cột index rác (Unnamed: 0) nếu có
    if "Unnamed: 0" in df_notebook.columns:
        df_notebook = df_notebook.drop(columns=["Unnamed: 0"])

    # 🔧 ĐỒNG BỘ KIỂU DỮ LIỆU NGÀY THÁNG CỦA DF_NOTEBOOK
    # Vì đọc từ CSV nên ngày tháng bị tụt về dạng string, cần đưa về datetime chuẩn
    if "delivery_date" in df_notebook.columns:
        df_notebook["delivery_date"] = pd.to_datetime(df_notebook["delivery_date"], errors="coerce")

    if "data_collected_at" in df_notebook.columns:
        df_notebook["data_collected_at"] = pd.to_datetime(df_notebook["data_collected_at"], errors="coerce")

    # 3. Chạy qua Class AmazonDataCleaner
    cleaner = AmazonDataCleaner()
    df_class = cleaner.transform(df_raw)

    # 4. So sánh đối chiếu
    assert_frame_equal(df_notebook, df_class, check_dtype=False)
    print("\n✅ HOÀN HẢO! Code trong file data_cleaner.py khớp 100% dữ liệu với Notebook!")

except AssertionError as e:
    print("\n❌ VẪN CÒN SỰ KHÁC BIỆT:")
    print(e)