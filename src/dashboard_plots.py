import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# --- CẤU HÌNH PHONG CÁCH TỐI GIẢN (STORYTELLING WITH DATA) ---
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

FIG_SIZE = (7, 4.2)


def setup_storytelling_ax(ax):
  """Hàm phụ trợ: Xóa viền thừa và đường lưới nhiễu."""
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['bottom'].set_color('#888888')
  ax.grid(False)
  ax.tick_params(axis='both', which='both', length=0)


def format_large_number(num):
  """Định dạng số lớn gọn gàng."""
  if num >= 1_000_000:
    return f'{num / 1_000_000:.2f}M'
  elif num >= 1_000:
    return f'{num / 1_000:.1f}K'
  return f'{int(num)}'


def render_kpis(filtered_df: pd.DataFrame):
  """Hiển thị các chỉ số KPI chính."""
  total_products = len(filtered_df)
  total_sales = (
      filtered_df['purchased_last_month'].sum()
      if 'purchased_last_month' in filtered_df.columns
      else 0
  )
  avg_price = (
      filtered_df['discounted_price'].mean()
      if 'discounted_price' in filtered_df.columns
      else 0
  )
  avg_rating = (
      filtered_df['product_rating'].mean()
      if 'product_rating' in filtered_df.columns
      else 0
  )
  avg_discount = (
      filtered_df['discount_percent'].mean()
      if 'discount_percent' in filtered_df.columns
      else 0
  )

  kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
  kpi1.metric('📦 Tổng sản phẩm', f'{total_products:,}')
  kpi2.metric('📈 Tổng doanh số', f'{format_large_number(total_sales)} SP')
  kpi3.metric('💵 Giá TB', f'${avg_price:.2f}')
  kpi4.metric('⭐ Rating TB', f'{avg_rating:.2f} / 5.0')
  kpi5.metric('🏷️ Chiết khấu TB', f'{avg_discount:.1f}%')


# ---KHỐI 1: TẤT CẢ DANH MỤC BÁN CHẠY (CỐ ĐỊNH FULL DATA) ---
def render_category_sales(df_full: pd.DataFrame):
  """STORYTELLING: Hiển thị TẤT CẢ danh mục toàn hệ thống cố định."""
  st.subheader('🔥 Top Danh Mục Bán Chạy')
  if (
      'product_category' in df_full.columns
      and 'purchased_last_month' in df_full.columns
  ):
    cat_sales = (
        df_full.groupby('product_category')['purchased_last_month']
        .sum()
        .reset_index()
        .sort_values(by='purchased_last_month', ascending=False)
    )

    if len(cat_sales) == 0:
      return

    fig, ax = plt.subplots(figsize=(10, 4))

    categories = [
        cat[:12] + '...' if len(cat) > 12 else cat
        for cat in cat_sales['product_category']
    ]
    values = cat_sales['purchased_last_month'].values
    colors = ['#2CA02C'] + ['#D3D3D3'] * (len(cat_sales) - 1)

    bars = ax.bar(categories, values, color=colors, width=0.55)

    max_val = values[0]
    for bar in bars:
      height = bar.get_height()
      ax.text(
          bar.get_x() + bar.get_width() / 2,
          height + (max_val * 0.02),
          f'{format_large_number(height)}',
          ha='center',
          va='bottom',
          fontsize=8.5,
          color='#333333',
          fontweight='bold' if height == max_val else 'normal',
      )

    setup_storytelling_ax(ax)
    ax.set_yticks([])
    ax.tick_params(axis='x', rotation=25, labelsize=8.5)

    top_cat = cat_sales.iloc[0]['product_category']
    ax.set_title(
        f'Danh mục "{top_cat}" dẫn đầu hoàn toàn về sản lượng trên tổng số'
        f' {len(cat_sales)} danh mục',
        loc='left',
        fontsize=11,
        fontweight='bold',
        pad=15,
        color='#111111',
    )
    plt.tight_layout()
    st.pyplot(fig)


# --- KHỐI 2: YẾU TỐ ẢNH HƯỞNG ĐẾN LƯỢNG BÁN ---
def render_feature_impact(filtered_df: pd.DataFrame):
  """STORYTELLING: Biểu đồ phân cực trực quan hóa yếu tố làm TĂNG / GIẢM doanh số."""
  st.subheader('🎯 Các Yếu Tố Ảnh Hưởng Đến Lượng Bán')

  numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()

  if 'purchased_last_month' in numeric_cols and len(numeric_cols) > 1:
    corr = (
        filtered_df[numeric_cols]
        .corr()[['purchased_last_month']]
        .drop(index='purchased_last_month')
    )

    name_mapping = {
        'product_rating': 'Điểm Đánh Giá (Rating)',
        'total_reviews': 'Số Lượng Đánh Giá (Reviews)',
        'discounted_price': 'Giá Sau Giảm',
        'original_price': 'Giá Gốc',
        'product_price': 'Giá Gốc',
        'discount_percent': 'Mức Chiết Khấu (%)',
        'discount_percentage': 'Mức Chiết Khấu (%)',
        'is_best_seller': 'Nhãn Best Seller',
        'is_sponsored': 'Sản Phẩm Tài Trợ (Sponsored)',
        'has_coupon': 'Có Mã Giảm Giá (Coupon)',
        'buy_box_availability': 'Sẵn Hàng (Buy Box)',
    }

    corr.index = [name_mapping.get(col, col) for col in corr.index]
    corr = corr.groupby(corr.index).mean()
    corr = corr.sort_values(by='purchased_last_month', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 4.2))

    colors = [
        '#2CA02C' if val > 0 else '#E15759'
        for val in corr['purchased_last_month']
    ]

    bars = ax.barh(
        corr.index, corr['purchased_last_month'], color=colors, height=0.5
    )

    ax.axvline(0, color='#888888', linewidth=1, linestyle='--')

    min_val = corr['purchased_last_month'].min()
    max_val = corr['purchased_last_month'].max()
    ax.set_xlim(
        min_val * 1.55 if min_val < 0 else -0.1,
        max_val * 1.35 if max_val > 0 else 0.1,
    )

    for bar in bars:
      width = bar.get_width()
      if width >= 0:
        offset = 0.015
        ha = 'left'
        label_text = f'+{width:.2f}'
      else:
        offset = -0.015
        ha = 'right'
        label_text = f'{width:.2f}'

      ax.text(
          width + offset,
          bar.get_y() + bar.get_height() / 2,
          label_text,
          ha=ha,
          va='center',
          fontsize=9,
          fontweight='bold',
          color='#333333',
      )

    setup_storytelling_ax(ax)
    ax.set_xticks([])
    ax.set_xlabel('')

    ax.text(
        0.0,
        -0.12,
        '◄ Tác động Tiêu cực (Làm giảm lượng bán)',
        transform=ax.transAxes,
        color='#E15759',
        fontweight='bold',
        fontsize=8.5,
    )
    ax.text(
        0.65,
        -0.12,
        'Tác động Tích cực (Thúc đẩy lượng bán) ►',
        transform=ax.transAxes,
        color='#2CA02C',
        fontweight='bold',
        fontsize=8.5,
    )

    ax.set_title(
        'Tương quan giữa các thuộc tính sản phẩm và sản lượng bán ra',
        loc='left',
        fontsize=11,
        fontweight='bold',
        pad=15,
        color='#111111',
    )
    plt.tight_layout()
    st.pyplot(fig)


# --- KHỐI 3A: CƠ CẤU THEO KHOẢNG GIÁ ---
def render_price_distribution(filtered_df: pd.DataFrame):
  """STORYTELLING: Biểu đồ Vành Khuyên thể hiện thị phần các khoảng giá."""
  st.subheader('🏷️ Cơ Cấu Theo Khoảng Giá')

  price_col = (
      'discounted_price'
      if 'discounted_price' in filtered_df.columns
      else ('product_price' if 'product_price' in filtered_df.columns else None)
  )

  if price_col:
    prices = filtered_df[price_col].dropna()
    if len(prices) == 0:
      return

    bins = [0, 25, 50, 100, 200, 500, np.inf]
    labels = [
        '< $25',
        '$25 - $50',
        '$50 - $100',
        '$100 - $200',
        '$200 - $500',
        '> $500',
    ]

    price_range = (
        pd.cut(prices, bins=bins, labels=labels)
        .value_counts()
        .reindex(labels)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    max_label = price_range.idxmax()
    max_val = price_range.max()
    total_val = price_range.sum()
    max_pct = (max_val / total_val) * 100 if total_val > 0 else 0

    colors = [
        '#2CA02C' if label == max_label else c
        for label, c in zip(
            labels,
            ['#2CA02C', '#A0C4DF', '#B0C4DE', '#C0D4EE', '#D0E4FE', '#E0E0E0'],
        )
    ]

    wedges, texts, autotexts = ax.pie(
        price_range.values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor='w', linewidth=2),
    )

    for autotext, label in zip(autotexts, labels):
      if label == max_label:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9.5)
      else:
        autotext.set_color('#333333')
        autotext.set_fontsize(8)

    ax.text(
        0,
        0.05,
        f'{max_pct:.1f}%',
        ha='center',
        va='center',
        fontsize=16,
        fontweight='bold',
        color='#2CA02C',
    )
    ax.text(
        0,
        -0.18,
        f'Thuộc phân khúc\n{max_label}',
        ha='center',
        va='center',
        fontsize=8.5,
        color='#555555',
    )

    ax.set_title(
        f'Phân khúc {max_label} chiếm tỷ trọng lớn nhất ({format_large_number(max_val)} SP)',
        loc='left',
        fontsize=11,
        fontweight='bold',
        pad=15,
        color='#111111',
    )

    plt.tight_layout()
    st.pyplot(fig)


# --- KHỐI 3B: TOP SẢN PHẨM DOANH THU ---
def render_top_best_sellers(filtered_df: pd.DataFrame):
  """STORYTELLING: Lollipop Chart thể hiện doanh thu Top 5 sản phẩm."""
  st.subheader('🏆 Top Sản Phẩm Doanh Thu')

  title_col = (
      'product_title'
      if 'product_title' in filtered_df.columns
      else ('title' if 'title' in filtered_df.columns else None)
  )

  if title_col and 'purchased_last_month' in filtered_df.columns:
    df_temp = filtered_df.copy()
    price_col = (
        'discounted_price'
        if 'discounted_price' in df_temp.columns
        else ('product_price' if 'product_price' in df_temp.columns else None)
    )

    if price_col:
      df_temp['est_revenue'] = (
          df_temp['purchased_last_month'] * df_temp[price_col]
      )
      sort_col = 'est_revenue'
      val_prefix = '$'
    else:
      sort_col = 'purchased_last_month'
      val_prefix = ''

    top_products = (
        df_temp.sort_values(by=sort_col, ascending=False).head(5).copy()
    )

    if len(top_products) == 0:
      return

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    top_products = top_products.iloc[::-1]

    titles = [
        row[title_col][:22] + '...' if len(row[title_col]) > 22 else row[title_col]
        for _, row in top_products.iterrows()
    ]
    values = top_products[sort_col].values
    y_pos = np.arange(len(titles))

    max_val = values[-1]

    for y, val in zip(y_pos, values):
      is_top1 = val == max_val
      line_color = '#2CA02C' if is_top1 else '#B0C4DE'
      dot_color = '#2CA02C' if is_top1 else '#708090'

      ax.hlines(
          y=y,
          xmin=0,
          xmax=val,
          color=line_color,
          alpha=0.8,
          linewidth=2 if is_top1 else 1.2,
      )
      ax.plot(val, y, 'o', markersize=9 if is_top1 else 7, color=dot_color)

      ax.text(
          val + (max_val * 0.03),
          y,
          f'{val_prefix}{format_large_number(val)}',
          ha='left',
          va='center',
          fontsize=8.5,
          fontweight='bold' if is_top1 else 'normal',
          color='#333333',
      )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(titles, fontsize=8.5)
    setup_storytelling_ax(ax)
    ax.set_xticks([])
    ax.set_xlim(0, max_val * 1.28)

    ax.set_title(
        'Top 1 tạo cách biệt doanh thu lớn',
        loc='left',
        fontsize=11,
        fontweight='bold',
        pad=15,
        color='#111111',
    )
    plt.tight_layout()
    st.pyplot(fig)


# --- KHỐI 4: RATING & DOANH SỐ (FULL DATASET & DYNAMIC SCALE BƯỚC NHẢY 0.2) ---
def render_rating_reviews_scatter(filtered_df: pd.DataFrame):
  """STORYTELLING: Scatter plot trực quan hóa toàn bộ sản phẩm."""
  st.subheader('⭐ Rating & Doanh Số')
  if (
      'product_rating' in filtered_df.columns
      and 'purchased_last_month' in filtered_df.columns
  ):
    # Lấy TOÀN BỘ các dòng dữ liệu hợp lệ (loại bỏ dòng bị thiếu rating)
    valid_df = filtered_df.dropna(subset=['product_rating']).copy()

    if len(valid_df) == 0:
      st.warning('Không có dữ liệu Rating hợp lệ!')
      return

    fig, ax = plt.subplots(figsize=(10, 4))

    # Vẽ toàn bộ dữ liệu lên biểu đồ scatter
    sns.scatterplot(
        data=valid_df,
        x='product_rating',
        y='purchased_last_month',
        color='#A9A9A9',
        alpha=0.4,
        s=30,  # Giảm nhẹ kích thước hạt để tránh đè lấp khi dữ liệu đông
        ax=ax,
    )

    # Xác định các sản phẩm đột phá trên TOÀN BỘ tập dữ liệu
    high_sales_thresh = valid_df['purchased_last_month'].quantile(0.80)
    highlights = valid_df[
        (valid_df['product_rating'] >= 4.2)
        & (valid_df['purchased_last_month'] >= high_sales_thresh)
    ]

    ax.scatter(
        highlights['product_rating'],
        highlights['purchased_last_month'],
        color='#2CA02C',
        s=45,
        alpha=0.9,
        label='Sản phẩm đột phá',
    )

    setup_storytelling_ax(ax)

    # === TỰ ĐỘNG SCALE THEO MIN-MAX CỦA TOÀN BỘ DỮ LIỆU VỚI BƯỚC NHẢY 0.2 ===
    min_rating = valid_df['product_rating'].min()
    max_rating = valid_df['product_rating'].max()

    # Làm tròn min xuống bội số của 0.2 và max lên bội số của 0.2
    start_tick = np.floor(min_rating / 0.2) * 0.2
    end_tick = np.ceil(max_rating / 0.2) * 0.2

    # Trường hợp tất cả điểm bằng nhau (min == max)
    if start_tick == end_tick:
      start_tick = max(0.0, start_tick - 0.2)
      end_tick = min(5.0, end_tick + 0.2)

    # Tạo dải mốc vạch chạy từ start_tick đến end_tick với bước nhảy 0.2
    custom_ticks = np.arange(start_tick, end_tick + 0.001, 0.2)

    # Đặt giới hạn trục X mở rộng nhẹ 0.05 ở 2 đầu để không bị che điểm sát lề
    ax.set_xlim(start_tick - 0.05, end_tick + 0.05)
    ax.set_xticks(custom_ticks)

    # Hiển thị nhãn vạch chuẩn 1 chữ số thập phân (Ví dụ: 3.2, 3.4, 3.6, 3.8, 4.0, ...)
    ax.set_xticklabels([f'{x:.1f}' for x in custom_ticks], fontsize=8)

    ax.set_xlabel('Điểm Đánh Giá (Rating)', color='#555555', fontsize=9)
    ax.set_ylabel('Sản lượng bán ra', color='#555555', fontsize=9)

    ax.set_title(
        f'Rating & Doanh số trên tổng số {len(valid_df):,} sản phẩm',
        loc='left',
        fontsize=11,
        fontweight='bold',
        pad=15,
        color='#111111',
    )
    ax.legend(frameon=False, loc='upper left', fontsize=8.5)
    plt.tight_layout()
    st.pyplot(fig)



# --- KHỐI GIẢI THÍCH MÔ HÌNH BẰNG SHAP & FEATURE IMPORTANCE ---
def render_model_explainability(project_root):
  """Hiển thị các biểu đồ giải thích mô hình ML (SHAP & Feature Importance) từ file ảnh có sẵn."""
  st.subheader('🧠 Giải Thích Mô Hình Học Máy (Model Interpretability)')
  st.write(
      'Sử dụng các kỹ thuật **Feature Importance** và **SHAP (SHapley Additive'
      ' exPlanations)** để minh bạch hóa cách mô hình Random Forest đưa ra dự'
      ' đoán.'
  )

  data_dir = project_root / 'data'
  img_fi = data_dir / 'feature_importance.png'
  img_shap_bar = data_dir / 'shap_bar_plot.png'
  img_shap_summary = data_dir / 'shap_summary_plot.png'

  with st.expander(
      '🔍 Bấm vào đây để xem chi tiết phân tích SHAP & Feature Importance',
      expanded=False,
  ):
    tab1, tab2, tab3 = st.tabs(
        ['📊 Feature Importance', '📈 SHAP Summary', '📊 SHAP Bar Plot']
    )

    with tab1:
      st.markdown(
          '**Mức độ đóng góp của từng thuộc tính vào mô hình Random Forest:**'
      )
      if img_fi.exists():
        st.image(
            str(img_fi),
            caption='Random Forest Feature Importance',
            use_container_width=True,
        )
      else:
        st.info('Không tìm thấy file feature_importance.png trong thư mục data/')

    with tab2:
      st.markdown(
          '**Tác động chi tiết của từng giá trị đặc trưng (Cao/Thấp) tới kết'
          ' quả dự đoán:**'
      )
      if img_shap_summary.exists():
        st.image(
            str(img_shap_summary),
            caption='SHAP Summary Plot',
            use_container_width=True,
        )
      else:
        st.info('Không tìm thấy file shap_summary_plot.png trong thư mục data/')

    with tab3:
      st.markdown(
          '**Tầm quan trọng trung bình của các đặc trưng theo giá trị SHAP:**'
      )
      if img_shap_bar.exists():
        st.image(
            str(img_shap_bar),
            caption='SHAP Bar Plot',
            use_container_width=True,
        )
      else:
        st.info('Không tìm thấy file shap_bar_plot.png trong thư mục data/')



# ==============================================================================
# ĐIỀU HÀNH VẼ ĐÚNG CÁC KHỐI THEO THỨ TỰ YÊU CẦU
# ==============================================================================
def render_dashboard(
    df_full: pd.DataFrame, filtered_df: pd.DataFrame, project_root=None
):
  """Hàm tổng hợp hiển thị toàn bộ trang Dashboard theo thứ tự Khối 1 -> 4."""
  # KPI Overview Header
  render_kpis(filtered_df)
  st.markdown('---')

  # KHỐI 1: Danh mục bán chạy các sản phẩm
  render_category_sales(df_full)
  st.markdown('---')

  # KHỐI 2: Yếu tố ảnh hưởng
  render_feature_impact(filtered_df)
  st.markdown('---')

  # KHỐI 3: Cơ cấu theo khoảng giá & Top sản phẩm
  col_k3_1, col_k3_2 = st.columns(2)
  with col_k3_1:
    render_price_distribution(filtered_df)
  with col_k3_2:
    render_top_best_sellers(filtered_df)
  st.markdown('---')

  # KHỐI 4: Rating & Doanh số
  render_rating_reviews_scatter(filtered_df)
  st.markdown('---')

  # KHỐI NÂNG CẤP: Giải thích mô hình ML (SHAP & Feature Importance)
  if project_root is not None:
    render_model_explainability(project_root)
    st.markdown('---')

  # XEM TRƯỚC DỮ LIỆU
  st.subheader(
      f'🔍 Xem trước Dữ liệu Đã Lọc ({len(filtered_df):,} sản phẩm - Top 100'
      ' dòng)'
  )
  st.dataframe(filtered_df.head(100), use_container_width=True)