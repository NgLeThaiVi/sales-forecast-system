"""
feature_prep.py
---------------
Module chuẩn bị đặc trưng (Feature Preparation) cho Mô hình ML.
Nhận đầu vào là DataFrame đã làm sạch từ `data_cleaner.py`
và trả về đúng 10 cột đặc trưng X hoàn chỉnh cho Pipeline.
"""

from typing import List
import numpy as np
import pandas as pd

# Danh sách chuẩn 10 cột đặc trưng X mà Cell 3 đã dùng để Train Model
MODEL_FEATURES: List[str] = [
    # Numerical (5 cột)
    'product_rating',
    'total_reviews',
    'discounted_price',
    'original_price',
    'discount_percentage',
    # Categorical (5 cột)
    'is_best_seller',
    'is_sponsored',
    'has_coupon',
    'buy_box_availability',
    'product_category',
]


class FeaturePreparer:

  def __init__(self, required_features: List[str] = None):
    self.features = required_features or MODEL_FEATURES

  def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    """Xử lý Feature Engineering và đảm bảo đủ 10 cột X cho Model."""
    df_out = df.copy()

    # 1. Feature Engineering: Tính discount_percentage nếu chưa có
    if 'discount_percentage' not in df_out.columns:
      if (
          'original_price' in df_out.columns
          and 'discounted_price' in df_out.columns
      ):
        df_out['discount_percentage'] = np.where(
            df_out['original_price'] > 0,
            (
                (df_out['original_price'] - df_out['discounted_price'])
                / df_out['original_price']
            )
            * 100,
            0.0,
        )
      else:
        df_out['discount_percentage'] = 0.0

    # 2. Đảm bảo các cột cờ (Boolean / Binary Flag) không bị missing
    flag_cols = [
        'is_best_seller',
        'is_sponsored',
        'has_coupon',
        'buy_box_availability',
    ]
    for col in flag_cols:
      if col not in df_out.columns:
        df_out[col] = 0
      else:
        df_out[col] = df_out[col].fillna(0)

    # 3. Đảm bảo cột product_category không bị missing
    if 'product_category' not in df_out.columns:
      df_out['product_category'] = 'Unknown'
    else:
      df_out['product_category'] = df_out['product_category'].fillna('Unknown')

    # 4. Đảm bảo các cột giá/rating số cơ bản không bị thiếu cột
    num_cols = [
        'product_rating',
        'total_reviews',
        'discounted_price',
        'original_price',
    ]
    for col in num_cols:
      if col not in df_out.columns:
        df_out[col] = 0.0

    # Trả về DataFrame chứa đúng 10 cột X theo đúng thứ tự
    return df_out[self.features]