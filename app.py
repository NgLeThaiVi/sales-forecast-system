from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- IMPORT CÁC MODULE XỬ LÝ & ĐỒ HỌA TỪ SRC ---
try:
  from src.data_cleaner import AmazonDataCleaner
except ImportError:
  AmazonDataCleaner = None

try:
  from src.feature_prep import MODEL_FEATURES, FeaturePreparer
except ImportError:
  FeaturePreparer = None
  MODEL_FEATURES = [
      'product_rating',
      'total_reviews',
      'discounted_price',
      'original_price',
      'discount_percentage',
      'is_best_seller',
      'is_sponsored',
      'has_coupon',
      'buy_box_availability',
      'product_category',
  ]

# Import module vẽ biểu đồ dashboard
try:
  from src.dashboard_plots import render_dashboard
except ImportError:
  render_dashboard = None

# --- CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(
    page_title='Amazon Electronics Sales Dashboard & Prediction',
    page_icon='🛒',
    layout='wide',
)

# --- XÁC ĐỊNH ĐƯỜNG DẪN DỰ ÁN ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / 'data' / 'amazon_electronics_data_cleaned.csv'
MODEL_PATH = (
    PROJECT_ROOT
    / 'saved_artifacts'
    / 'random_forest_regressor_best_sales_model.joblib'
)


# --- NẠP DỮ LIỆU VÀ MODEL (VỚI CACHE) ---
@st.cache_data
def load_data():
  if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    if 'original_price' in df.columns and 'discounted_price' in df.columns:
      df['discount_percent'] = (
          (df['original_price'] - df['discounted_price'])
          / df['original_price']
          * 100
      ).clip(lower=0)
    return df
  return None


@st.cache_resource
def load_model():
  if MODEL_PATH.exists():
    try:
      return joblib.load(MODEL_PATH), None
    except Exception as e:
      return None, str(e)
  return None, 'File mô hình không tồn tại trong thư mục saved_artifacts/.'


df_clean = load_data()
model, model_error = load_model()

# ==============================================================================
# SIDEBAR NAVIGATION & FILTERS
# ==============================================================================
st.sidebar.title('📌 Điều Hướng System')
page = st.sidebar.radio(
    'Chọn chức năng:',
    ['📊 Dashboard Tổng quan', '🔮 Dự đoán Doanh số (Import CSV)'],
)

st.sidebar.markdown('---')

# Bộ lọc tương tác (chỉ áp dụng cho trang Dashboard)
filtered_df = df_clean.copy() if df_clean is not None else None

if page == '📊 Dashboard Tổng quan' and df_clean is not None:
  st.sidebar.subheader('🎯 Bộ Lọc Dữ Liệu Dashboard')

  if 'product_category' in df_clean.columns:
    categories = ['Tất cả'] + sorted(
        df_clean['product_category'].dropna().unique().tolist()
    )
    selected_cat = st.sidebar.selectbox('Danh mục sản phẩm:', categories)
    if selected_cat != 'Tất cả':
      filtered_df = filtered_df[filtered_df['product_category'] == selected_cat]

# ==============================================================================
# TRANG 1: DASHBOARD TỔNG QUAN
# ==============================================================================
if page == '📊 Dashboard Tổng quan':
  st.title('📊 Dashboard Phân Tích Dữ Liệu Amazon Electronics')
  st.write(
      'Phân tích toàn diện quy mô, đặc trưng thương mại điện tử và các yếu tố'
      ' ảnh hưởng đến doanh số.'
  )

  if df_clean is None:
    st.error(
        f'❌ Không tìm thấy file dữ liệu đã làm sạch tại: `{DATA_PATH}`. Vui'
        ' lòng kiểm tra lại thư mục `data/`.'
    )
  elif len(filtered_df) == 0:
    st.warning('⚠️ Không có sản phẩm nào thỏa mãn bộ lọc hiện tại!')
  elif render_dashboard is None:
    st.error(
        '❌ Không thể tải module `dashboard_plots.py`! Kiểm tra file trong'
        ' thư mục `src/`.'
    )
  else:
    # Truyền df_clean (gốc) và filtered_df (đã lọc) vào hàm render_dashboard
    render_dashboard(df_clean, filtered_df, PROJECT_ROOT)

# ==============================================================================
# TRANG 2: DỰ ĐOÁN DOANH SỐ (IMPORT FILE CSV)
# ==============================================================================
elif page == '🔮 Dự đoán Doanh số (Import CSV)':
  st.title('🔮 Dự Đoán Doanh Số Bằng Tải File CSV')
  st.write(
      'Hệ thống tự động kết hợp **`data_cleaner.py`** và **`feature_prep.py`** '
      'để biến đổi bất kỳ file CSV nào thành 10 đặc trưng chuẩn cho mô hình.'
  )

  # --- KHU VỰC TẢI DOWN SAMPLE TEMPLATE ---
  with st.expander(
      '📄 Chưa có file chạy thử? Tải file CSV mẫu (Sample Template) tại đây'
  ):
    st.write('File mẫu đã chứa đầy đủ 10 cột chuẩn đầu vào cho mô hình ML:')
    sample_data = pd.DataFrame({
        'product_title': [
            'Laptop Dell XPS 13',
            'iPhone 15 Pro',
            'Sony WH-1000XM5',
            'Canon EOS R6',
            'Echo Dot 5th Gen',
        ],
        'product_category': [
            'Laptops',
            'Phones',
            'Headphones',
            'Cameras',
            'Smart Home',
        ],
        'original_price': [1200.0, 800.0, 150.0, 600.0, 50.0],
        'discounted_price': [999.0, 699.0, 99.0, 499.0, 35.0],
        'discount_percentage': [16.75, 12.63, 34.00, 16.83, 30.00],
        'product_rating': [4.5, 4.2, 4.7, 4.1, 4.0],
        'total_reviews': [1500, 850, 3200, 410, 620],
        'is_best_seller': [1, 0, 1, 0, 0],
        'is_sponsored': [0, 1, 0, 1, 0],
        'has_coupon': [1, 0, 1, 0, 1],
        'buy_box_availability': [1, 1, 1, 1, 1],
    })
    st.dataframe(sample_data, use_container_width=True)

    csv_sample = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='📥 Tải xuống File CSV Mẫu (sample_amazon_data.csv)',
        data=csv_sample,
        file_name='sample_amazon_data.csv',
        mime='text/csv',
    )

  st.markdown('---')

  if model is None:
    st.error(f'❌ Không thể tải mô hình ML! Chi tiết lỗi: {model_error}')
  elif FeaturePreparer is None:
    st.error(
        '❌ Thư viện `feature_prep.py` bị thiếu! Vui lòng kiểm tra lại import'
        ' từ thư mục `src/`.'
    )
  else:
    uploaded_file = st.file_uploader(
        'Upload file CSV cần dự đoán (Hỗ trợ file thô hoặc file đã làm sạch):',
        type=['csv'],
    )

    if uploaded_file is not None:
      try:
        input_raw_df = pd.read_csv(uploaded_file)
        st.success(
            f'✅ Đã tải file thành công! Số lượng dòng thô: {len(input_raw_df):,}'
        )

        # --- PIPELINE 2 BƯỚC ĐỒNG BỘ ---
        with st.spinner('🔄 Đang xử lý làm sạch & biến đổi đặc trưng...'):
          # BƯỚC 1: Xử lý dữ liệu thô bằng AmazonDataCleaner (nếu có)
          if AmazonDataCleaner is not None and (
              'title' in input_raw_df.columns
              or 'product_title' in input_raw_df.columns
              or 'listed_price' in input_raw_df.columns
          ):
            cleaner = AmazonDataCleaner()
            processed_df = cleaner.transform(input_raw_df)
          else:
            processed_df = input_raw_df.copy()

          # BƯỚC 2: Tái tạo đặc trưng và trích xuất đúng 10 cột X bằng FeaturePreparer
          preparer = FeaturePreparer()
          X_input = preparer.transform(processed_df)

        st.subheader('📋 10 Đặc trưng $X$ đã chuẩn hóa sẵn sàng cho Model:')
        st.dataframe(X_input.head(10), use_container_width=True)

        if st.button('🚀 Tiến hành Dự đoán Hàng loạt', type='primary'):
          with st.spinner(
              'Đang tính toán dự đoán qua Random Forest Model...'
          ):
            # Dự đoán
            predictions = model.predict(X_input)
            predictions = np.maximum(0, predictions).round().astype(int)

            result_df = processed_df.copy()

            # 1. Thêm cột dự đoán 'predict_purchase'
            result_df['predict_purchase'] = predictions

            # 2. Xóa các cột target gốc / cột trung gian thừa
            cols_to_drop = [
                'purchased_last_month',
                'predicted_purchased_last_month',
                'discount_percent',
            ]
            for col in cols_to_drop:
              if col in result_df.columns:
                result_df.drop(columns=[col], inplace=True)

            # 3. Sắp xếp giảm dần theo cột 'predict_purchase'
            result_df = result_df.sort_values(
                by='predict_purchase', ascending=False
            ).reset_index(drop=True)

            # 4. Xác định tên cột sản phẩm trong dữ liệu
            cols = result_df.columns.tolist()
            display_cols = []

            if 'product_title' in cols:
              display_cols.append('product_title')
            elif 'title' in cols:
              display_cols.append('title')

            display_cols.append('predict_purchase')

            # Tạo bản chuẩn bị sẵn toàn bộ cột cho file tải xuống
            remaining_cols = [c for c in cols if c not in display_cols]
            download_df = result_df[display_cols + remaining_cols]

            # --- HIỂN THỊ KẾT QUẢ RÚT GỌN CHỈ 2 CỘT ---
            st.balloons()
            st.success('🎉 Dự đoán hoàn tất thành công!')

            st.subheader(
                '📊 Kết Quả Dự Đoán Doanh Số (Sắp xếp giảm dần theo'
                ' predict_purchase)'
            )

            # Chỉ hiển thị duy nhất 2 cột trên Web UI
            st.dataframe(result_df[display_cols], use_container_width=True)

            # Nút tải file CSV chứa đầy đủ toàn bộ thuộc tính
            csv_output = download_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='📥 Tải xuống File Kết quả Đầy đủ Thuộc tính (CSV)',
                data=csv_output,
                file_name='amazon_sales_predictions_full.csv',
                mime='text/csv',
            )

      except Exception as ex:
        st.error(
            f'⚠️ Có lỗi xảy ra trong quá trình xử lý hoặc dự đoán: {str(ex)}'
        )