from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(
    page_title='Amazon Electronics Sales Dashboard & Prediction',
    page_icon='🛒',
    layout='wide',
    initial_sidebar_state='expanded',
)

# --- THIẾT KẾ GIAO DIỆN SÁNG CAO CẤP THEO CHUẨN 60-30-10 ---
st.markdown(
    """
    <style>
    /* 60% Nền chung ứng dụng: Tăng độ xám sâu hơn chút (#f1f5f9) để tách biệt hoàn toàn với khối trắng của ảnh/biểu đồ */
    .stApp {
        background-color: #f1f5f9;
        color: #1e293b;
    }

    /* SIDEBAR TÔNG MÀU SÁNG: Xanh pastel nhạt tinh tế */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e2e8f0 0%, #cbd5e1 100%) !important;
        border-right: 1px solid #94a3b8;
        padding-top: 1rem;
    }

    /* Tiêu đề trong Sidebar sắc nét */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #0f172a !important;
        font-size: 1.15rem !important;
        border-bottom: 2px solid #94a3b8;
        padding-bottom: 0.5rem;
    }

    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #1e293b !important;
    }

    /* Tùy chỉnh lựa chọn Radio trong Sidebar */
    [data-testid="stSidebar"] .stRadio > div {
        background-color: #ffffff !important;
        padding: 12px !important;
        border-radius: 10px !important;
        border: 1px solid #64748b !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    [data-testid="stSidebar"] .stRadio label {
        font-weight: 600 !important;
        color: #0f172a !important;
        cursor: pointer;
    }

    /* Tiêu đề chính trên Main Area */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }

    /* PHÂN MÀU 10% ACCENT CHO CÁC Ô METRIC CARD */
    [data-testid="stHorizontalBlock"] > div:nth-child(1) [data-testid="stMetric"] {
        border-left: 4px solid #2563eb !important;
        background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%) !important;
    }
    [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stMetric"] {
        border-left: 4px solid #16a34a !important;
        background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%) !important;
    }
    [data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetric"] {
        border-left: 4px solid #ea580c !important;
        background: linear-gradient(135deg, #ffffff 0%, #fff7ed 100%) !important;
    }
    [data-testid="stHorizontalBlock"] > div:nth-child(4) [data-testid="stMetric"] {
        border-left: 4px solid #ca8a04 !important;
        background: linear-gradient(135deg, #ffffff 0%, #fefce8 100%) !important;
    }
    [data-testid="stHorizontalBlock"] > div:nth-child(5) [data-testid="stMetric"] {
        border-left: 4px solid #9333ea !important;
        background: linear-gradient(135deg, #ffffff 0%, #faf5ff 100%) !important;
    }

    /* Thiết kế chung cho các ô Metric có đổ bóng nổi bật khỏi nền web */
    [data-testid="stMetric"] {
        border: 1px solid #cbd5e1 !important;
        padding: 16px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 800 !important;
    }

    /* Khối Card trắng cho Expander & File Uploader */
    [data-testid="stExpander"], [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        color: #1e293b !important;
    }

    [data-testid="stExpander"] details, [data-testid="stExpander"] summary {
        background-color: #ffffff !important;
    }

    [data-testid="stExpander"] summary p, 
    [data-testid="stExpander"] summary span, 
    [data-testid="stExpander"] summary svg {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #f8fafc !important;
        border: 2px dashed #94a3b8 !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploader"] section * {
        color: #334155 !important;
    }

    /* Tabs màu sắc */
    button[data-baseweb="tab"] p, button[data-baseweb="tab"] span {
        color: #475569 !important;
        font-weight: 500 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] p, button[data-baseweb="tab"][aria-selected="true"] span {
        color: #2563eb !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom-color: #2563eb !important;
    }

    /* Nút bấm chính */
    .stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.25);
        transition: all 0.2s ease-in-out;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.35);
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

try:
    from src.dashboard_plots import render_dashboard
except ImportError:
    render_dashboard = None

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

                with st.spinner('🔄 Đang xử lý làm sạch & biến đổi đặc trưng...'):
                    if AmazonDataCleaner is not None and (
                            'title' in input_raw_df.columns
                            or 'product_title' in input_raw_df.columns
                            or 'listed_price' in input_raw_df.columns
                    ):
                        cleaner = AmazonDataCleaner()
                        processed_df = cleaner.transform(input_raw_df)
                    else:
                        processed_df = input_raw_df.copy()

                    preparer = FeaturePreparer()
                    X_input = preparer.transform(processed_df)

                st.subheader('📋 10 Đặc trưng $X$ đã chuẩn hóa sẵn sàng cho Model:')
                st.dataframe(X_input.head(10), use_container_width=True)

                if st.button('🚀 Tiến hành Dự đoán Hàng loạt', type='primary'):
                    with st.spinner(
                            'Đang tính toán dự đoán qua Random Forest Model...'
                    ):
                        predictions = model.predict(X_input)
                        predictions = np.maximum(0, predictions).round().astype(int)

                        result_df = processed_df.copy()
                        result_df['predict_purchase'] = predictions

                        cols_to_drop = [
                            'purchased_last_month',
                            'predicted_purchased_last_month',
                            'discount_percent',
                        ]
                        for col in cols_to_drop:
                            if col in result_df.columns:
                                result_df.drop(columns=[col], inplace=True)

                        result_df = result_df.sort_values(
                            by='predict_purchase', ascending=False
                        ).reset_index(drop=True)

                        cols = result_df.columns.tolist()
                        display_cols = []

                        if 'product_title' in cols:
                            display_cols.append('product_title')
                        elif 'title' in cols:
                            display_cols.append('title')

                        display_cols.append('predict_purchase')

                        remaining_cols = [c for c in cols if c not in display_cols]
                        download_df = result_df[display_cols + remaining_cols]

                        st.balloons()
                        st.success('🎉 Dự đoán hoàn tất thành công!')

                        st.subheader(
                            '📊 Kết Quả Dự Đoán Doanh Số (Sắp xếp giảm dần theo'
                            ' predict_purchase)'
                        )
                        st.dataframe(result_df[display_cols], use_container_width=True)

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