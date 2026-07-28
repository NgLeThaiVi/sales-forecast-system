import pandas as pd
import numpy as np
import re
from typing import Dict, List


class AmazonDataCleaner:
    """
    Một class để làm sạch và tiền xử lý dữ liệu sản phẩm Amazon.
    Class này đóng gói tất cả các bước làm sạch và kỹ thuật đặc trưng
    được thực hiện trong notebook, đảm bảo khả năng tái sử dụng và độ bền.
    """

    def __init__(self):
        """
        Khởi tạo Class AmazonDataCleaner.
        Định nghĩa các từ khóa danh mục và URL cơ sở của Amazon.
        """
        self.category_keywords: Dict[str, List[str]] = {
            'Laptops': [
                'laptop', 'notebook', 'macbook', 'chromebook', 'ultrabook', 'acer', 'asus', 'dell', 'lenovo', 'hp',
                'core',
                'intel', 'ryzen', 'surface', 'thinkpad', 'ideapad'
            ],
            'Phones': [
                'phone', 'iphone', 'smartphone', 'samsung', 'android', 'galaxy', 'pixel', 'oneplus', 'xiaomi', 'oppo',
                'realme', 'huawei', 'vivo', 'nokia', 'motorola'
            ],
            'Headphones': [
                'headphone', 'headset', 'earphone', 'earbuds', 'airpods', 'beats', 'sony wh', 'wireless buds',
                'neckband'
            ],
            'Chargers & Cables': [
                'charger', 'charging', 'cable', 'adapter', 'dock', 'usb c', 'type c', 'lightning', 'power adapter',
                'usb cable'
            ],
            'Cameras': [
                'camera', 'dslr', 'mirrorless', 'canon', 'nikon', 'gopro', 'instax', 'webcam', 'camcorder',
                'security camera'
            ],
            'Storage': [
                'ssd', 'hard drive', 'memory card', 'flash drive', 'pendrive', 'hdd', 'storage', 'micro sd', 'sd card'
            ],
            'Smart Home': [
                'alexa', 'echo', 'smart plug', 'smart bulb', 'smart home', 'nest', 'homekit', 'smart switch'
            ],
            'TV & Display': [
                'monitor', 'display', 'tv', 'screen', 'projector', 'oled', 'led', 'curved monitor', 'uhd', '4k'
            ],
            'Power & Batteries': [
                'battery', 'power bank', 'rechargeable', 'aa', 'aaa', 'portable power', 'cell'
            ],
            'Networking': [
                'wifi', 'router', 'modem', 'ethernet', 'access point', 'mesh', 'network switch'
            ],
            'Wearables': [
                'smartwatch', 'fitness band', 'fitbit', 'watch', 'garmin', 'amazfit'
            ],
            'Speakers': [
                'speaker', 'soundbar', 'subwoofer', 'bluetooth speaker', 'party speaker', 'home theater'
            ],
            'Printers & Scanners': [
                'printer', 'scanner', 'inkjet', 'laserjet', 'photocopier', 'all in one printer'
            ],
            'Gaming': [
                'gaming console', 'playstation', 'ps5', 'ps4', 'xbox', 'nintendo', 'joystick', 'controller',
                'gaming mouse',
                'gaming keyboard', 'gaming chair'
            ],
            'Other Electronics': []
        }
        self.AMAZON_BASE_URL: str = "https://www.amazon.com"

        # Định nghĩa thứ tự các cột đầu ra mong muốn để đảm bảo tính nhất quán
        self.target_columns: List[str] = [
            'product_title', 'product_rating', 'total_reviews',
            'purchased_last_month', 'discounted_price', 'original_price',
            'is_best_seller', 'is_sponsored', 'has_coupon', 'buy_box_availability',
            'delivery_date', 'sustainability_tags', 'product_image_url',
            'product_page_url', 'data_collected_at', 'product_category',
            'discount_percentage'
        ]

    def _clean_text(self, text: str) -> str:
        """
        Hàm trợ giúp để làm sạch văn bản bằng cách chuyển sang chữ thường
        và loại bỏ các ký tự không phải chữ cái/số.
        """
        if pd.isna(text):  # Xử lý NaN
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    def _assign_category_simple(self, title: str) -> str:
        """
        Gán một danh mục cho sản phẩm dựa trên tiêu đề của nó
        và các từ khóa được xác định trước.
        """
        title_clean = self._clean_text(title)
        for category, keywords in self.category_keywords.items():
            for kw in keywords:
                if kw in title_clean:
                    return category
        return 'Other Electronics'

    def _clean_price_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch các cột liên quan đến giá: 'current/discounted_price' và 'price_on_variant'.
        """
        if 'price_on_variant' in df.columns:
            # Trích xuất giá thực tế từ 'price_on_variant'
            df['price_on_variant'] = df['price_on_variant'].astype(str).str.split(":").str.get(1)
            # Đặt NaN cho các hàng không chứa ký hiệu '$'
            df.loc[~df['price_on_variant'].astype(str).str.contains(r'\$', na=False), 'price_on_variant'] = np.nan
            # Loại bỏ khoảng trắng và chỉ lấy phần đầu tiên nếu có nhiều giá trị
            df['price_on_variant'] = df['price_on_variant'].astype(str).str.strip().str.split(" ").str.get(0)

        if 'current/discounted_price' in df.columns and 'price_on_variant' in df.columns:
            # Điền các giá trị thiếu trong 'current/discounted_price' bằng 'price_on_variant'
            df['current/discounted_price'] = df['current/discounted_price'].fillna(df['price_on_variant'])

        if 'current/discounted_price' in df.columns:
            # Làm sạch 'current/discounted_price' và chuyển đổi sang float
            df['current/discounted_price'] = df['current/discounted_price'].astype(str).str.replace(r"\$", "",
                                                                                                    regex=True).str.replace(
                r",", "").astype(float)

        return df

    def _clean_rating(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch cột 'rating'.
        """
        if 'rating' in df.columns:
            df['rating'] = df['rating'].astype(str).str.replace(r"out of 5 stars", "").str.strip().astype(float)
        return df

    def _clean_number_of_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch cột 'number_of_reviews'.
        """
        if 'number_of_reviews' in df.columns:
            df['number_of_reviews'] = df['number_of_reviews'].astype(str).str.replace(",", "").str.strip().astype(float)
        return df

    def _clean_bought_in_last_month(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'bought_in_last_month' not in df.columns:
            return df

        # Bước 1: Ép về string, xóa cụm "+ bought in past month", strip khoảng trắng và đổi 'K' thành '000'
        s = (
            df['bought_in_last_month']
            .astype(str)
            .str.replace('+ bought in past month', '', regex=False)
            .str.strip()
            .str.replace('K', '000', regex=False)
        )

        # Bước 2: Chỉ giữ lại chuỗi chữ số thuần (isdigit), còn lại gán np.nan và ép về kiểu 'Int64'
        is_digit_mask = s.str.isdigit().fillna(False)
        df['bought_in_last_month'] = (
            s.where(is_digit_mask, np.nan)
            .astype('Int64')
        )

        return df

    def _clean_listed_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """Làm sạch cột 'listed_price' và xử lý an toàn các chuỗi không phải số như 'No Discount'."""
        if 'listed_price' in df.columns:
            # 1. Chuyển sang string và làm sạch ký tự $, phẩy, khoảng trắng
            listed = (
                df['listed_price']
                .astype(str)
                .str.replace('$', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.strip()
            )

            # 2. Ép kiểu số an toàn với pd.to_numeric:
            # Tham số errors='coerce' sẽ TỰ ĐỘNG biến các chuỗi chữ như 'No Discount', 'nan', 'None' thành NaN mà KHÔNG BỊ BÁO LỖI
            df['listed_price'] = pd.to_numeric(listed, errors='coerce')

            # 3. Nếu 'listed_price' bị NaN (do là No Discount), điền bằng giá 'current/discounted_price'
            if 'current/discounted_price' in df.columns:
                df['listed_price'] = df['listed_price'].fillna(
                    df['current/discounted_price']
                )

        return df

    def _clean_delivery_details(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch cột 'delivery_details'.
        """
        if 'delivery_details' in df.columns:
            # Trích xuất phần ngày từ 'delivery_details' và bỏ qua tên ngày
            df['delivery_details'] = df['delivery_details'].astype(str).str.extract(
                r'(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)?,?\s*(\w+\s+\d{1,2})')
            # Chuyển đổi 'delivery_details' sang datetime, thêm năm mặc định (2025)
            df['delivery_details'] = pd.to_datetime(df['delivery_details'] + ' 2025', errors='coerce')
        return df

    def _complete_product_url(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hoàn thiện URL sản phẩm bằng cách thêm URL cơ sở của Amazon nếu cần.
        """
        if 'product_url' in df.columns:
            df['product_url'] = df['product_url'].apply(
                lambda x: self.AMAZON_BASE_URL + x
                if pd.notna(x) and not str(x).startswith(("http://", "https://"))
                else x
            )
        return df

    def _clean_collected_at(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuyển đổi cột 'collected_at' sang định dạng datetime.
        """
        if 'collected_at' in df.columns:
            df['collected_at'] = pd.to_datetime(df['collected_at'], errors='coerce')
        return df

    def _create_category_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo cột 'category' mới dựa trên cột 'title'.
        """
        if 'title' in df.columns:
            df['category'] = df['title'].apply(self._assign_category_simple)
        else:
            # Nếu cột 'title' không tồn tại, tạo cột 'category' với giá trị mặc định
            df['category'] = 'Unknown'
        return df

    def _create_discount_percentage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo cột 'discount_percentage' dựa trên 'listed_price' và 'current/discounted_price'.
        Khớp chính xác 100% logic tính toán và giữ nguyên NaN của Notebook.
        """
        if 'listed_price' in df.columns and 'current/discounted_price' in df.columns:
            # Ép kiểu số an toàn trước khi tính toán
            listed = pd.to_numeric(df['listed_price'], errors='coerce')
            current = pd.to_numeric(df['current/discounted_price'], errors='coerce')

            # Tính discount_percentage và làm tròn 2 chữ số thập phân
            discount_calc = ((listed - current) / listed) * 100
            df['discount_percentage'] = discount_calc.round(2)
        else:
            df['discount_percentage'] = np.nan

        return df

    def _drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xóa các cột không cần thiết sau khi tiền xử lý.
        """
        columns_to_drop = ['price_on_variant']
        for col in columns_to_drop:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        return df

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Đổi tên các cột để rõ ràng và nhất quán.
        """
        rename_map = {
            'title': 'product_title',
            'rating': 'product_rating',
            'number_of_reviews': 'total_reviews',
            'bought_in_last_month': 'purchased_last_month',
            'current/discounted_price': 'discounted_price',
            'listed_price': 'original_price',
            'is_couponed': 'has_coupon',
            'delivery_details': 'delivery_date',
            'sustainability_badges': 'sustainability_tags',
            'image_url': 'product_image_url',
            'product_url': 'product_page_url',
            'collected_at': 'data_collected_at',
            'category': 'product_category'
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        return df

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Thực hiện toàn bộ quy trình làm sạch và tiền xử lý dữ liệu thô.

        Args:
            df_raw (pd.DataFrame): DataFrame thô chưa được làm sạch.

        Returns:
            pd.DataFrame: DataFrame đã được làm sạch và tiền xử lý.
        """
        df = df_raw.copy()  # Làm việc trên một bản sao của DataFrame gốc

        # Bước 1: Làm sạch các cột liên quan đến giá
        df = self._clean_price_columns(df)

        # Bước 2: Làm sạch cột 'rating'
        df = self._clean_rating(df)

        # Bước 3: Làm sạch cột 'number_of_reviews'
        df = self._clean_number_of_reviews(df)

        # Bước 4: Làm sạch cột 'bought_in_last_month'
        df = self._clean_bought_in_last_month(df)

        # Bước 5: Làm sạch cột 'listed_price' (phụ thuộc vào discounted_price)
        df = self._clean_listed_price(df)

        # Bước 6: Làm sạch cột 'delivery_details'
        df = self._clean_delivery_details(df)

        # Bước 7: Hoàn thiện URL sản phẩm
        df = self._complete_product_url(df)

        # Bước 8: Làm sạch cột 'collected_at'
        df = self._clean_collected_at(df)

        # Bước 9: Tạo cột 'category' (kỹ thuật đặc trưng)
        df = self._create_category_feature(df)

        # Bước 10: Tạo cột 'discount_percentage' (kỹ thuật đặc trưng)
        df = self._create_discount_percentage(df)

        # Bước 11: Xóa các cột không cần thiết
        df = self._drop_unnecessary_columns(df)

        # Bước 12: Đổi tên các cột
        df = self._rename_columns(df)

        # Bước cuối: Đảm bảo các cột đầu ra và thứ tự giống như dữ liệu đã huấn luyện
        # Thêm các cột bị thiếu với NaN và sắp xếp lại
        for col in self.target_columns:
            if col not in df.columns:
                df[col] = np.nan  # Hoặc giá trị mặc định phù hợp khác

        # Sắp xếp lại DataFrame theo thứ tự cột mong muốn
        df = df[self.target_columns]

        return df