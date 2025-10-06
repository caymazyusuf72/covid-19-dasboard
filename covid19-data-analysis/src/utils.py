"""
COVID-19 Data Analysis Project - Utility Functions
=================================================

Bu modül proje genelinde kullanılan yardımcı fonksiyonları içerir.
"""

import os
import yaml
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import requests


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    YAML yapılandırma dosyasını yükler.
    
    Args:
        config_path (str): Yapılandırma dosyasının yolu
        
    Returns:
        Dict: Yapılandırma verilerini içeren sözlük
        
    Raises:
        FileNotFoundError: Yapılandırma dosyası bulunamazsa
        yaml.YAMLError: YAML parsing hatası
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Yapılandırma dosyası bulunamadı: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML dosyası okunamadı: {e}")


def setup_logging(config: Optional[Dict] = None) -> logging.Logger:
    """
    Logging sistemini yapılandırır.
    
    Args:
        config (Dict, optional): Logging yapılandırması
        
    Returns:
        logging.Logger: Yapılandırılmış logger
    """
    if config is None:
        config = load_config()
    
    log_config = config.get('logging', {})
    
    # Log dizini oluştur
    log_file = log_config.get('file', 'logs/covid_analysis.log')
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Logger yapılandırması
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging sistemi başlatıldı")
    
    return logger


def create_directories(paths: List[str]) -> None:
    """
    Belirtilen dizinleri oluşturur (yoksa).
    
    Args:
        paths (List[str]): Oluşturulacak dizin listesi
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Dizin oluşturuldu: {path}")


def safe_request(url: str, timeout: int = 30, retries: int = 3) -> Optional[requests.Response]:
    """
    Güvenli HTTP isteği yapar.
    
    Args:
        url (str): İstek yapılacak URL
        timeout (int): Timeout süresi (saniye)
        retries (int): Tekrar deneme sayısı
        
    Returns:
        requests.Response or None: HTTP yanıtı veya None (hata durumunda)
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            print(f"İstek hatası (Deneme {attempt + 1}/{retries}): {e}")
            if attempt == retries - 1:
                return None
    return None


def validate_date_format(date_string: str, format_str: str = "%Y-%m-%d") -> bool:
    """
    Tarih formatını doğrular.
    
    Args:
        date_string (str): Doğrulanacak tarih string'i
        format_str (str): Beklenen format
        
    Returns:
        bool: Tarih formatı geçerli mi?
    """
    try:
        datetime.strptime(date_string, format_str)
        return True
    except ValueError:
        return False


def format_large_number(number: Union[int, float], precision: int = 1) -> str:
    """
    Büyük sayıları okunabilir formatta gösterir.
    
    Args:
        number (Union[int, float]): Formatlanacak sayı
        precision (int): Ondalık basamak sayısı
        
    Returns:
        str: Formatlanmış sayı string'i
        
    Examples:
        >>> format_large_number(1000000)
        '1.0M'
        >>> format_large_number(1500)
        '1.5K'
    """
    if pd.isna(number):
        return "N/A"
    
    number = float(number)
    
    if abs(number) >= 1e9:
        return f"{number / 1e9:.{precision}f}B"
    elif abs(number) >= 1e6:
        return f"{number / 1e6:.{precision}f}M"
    elif abs(number) >= 1e3:
        return f"{number / 1e3:.{precision}f}K"
    else:
        return f"{number:.{precision}f}"


def calculate_percentage_change(current: float, previous: float) -> float:
    """
    Yüzde değişim hesaplar.
    
    Args:
        current (float): Mevcut değer
        previous (float): Önceki değer
        
    Returns:
        float: Yüzde değişim
    """
    if previous == 0:
        return 0.0 if current == 0 else float('inf')
    
    return ((current - previous) / previous) * 100


def get_date_range(start_date: str, end_date: Optional[str] = None) -> List[str]:
    """
    Belirtilen tarih aralığındaki tarihleri döndürür.
    
    Args:
        start_date (str): Başlangıç tarihi (YYYY-MM-DD)
        end_date (str, optional): Bitiş tarihi. None ise bugün
        
    Returns:
        List[str]: Tarih listesi
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
    
    date_list = []
    current_date = start
    
    while current_date <= end:
        date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    return date_list


def clean_country_name(country_name: str) -> str:
    """
    Ülke adını temizler ve standartlaştırır.
    
    Args:
        country_name (str): Ham ülke adı
        
    Returns:
        str: Temizlenmiş ülke adı
    """
    # Yaygın ülke adı eşleştirmeleri
    country_mapping = {
        'US': 'United States',
        'USA': 'United States',
        'UK': 'United Kingdom',
        'South Korea': 'Korea, South',
        'Korea, South': 'South Korea',
        'Iran (Islamic Republic of)': 'Iran',
        'Russian Federation': 'Russia',
        'Viet Nam': 'Vietnam',
        'Syrian Arab Republic': 'Syria',
        'Venezuela (Bolivarian Republic of)': 'Venezuela',
        'Bolivia (Plurinational State of)': 'Bolivia',
        'Tanzania, United Republic of': 'Tanzania',
        'North Macedonia': 'Macedonia',
        'Czechia': 'Czech Republic'
    }
    
    # Temizleme
    cleaned = country_name.strip()
    
    # Eşleştirme varsa değiştir
    return country_mapping.get(cleaned, cleaned)


def save_to_cache(data: pd.DataFrame, filename: str, cache_dir: str = ".cache") -> None:
    """
    Veriyi cache'e kaydeder.
    
    Args:
        data (pd.DataFrame): Kaydedilecek veri
        filename (str): Dosya adı
        cache_dir (str): Cache dizini
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    filepath = os.path.join(cache_dir, filename)
    data.to_pickle(filepath)
    print(f"Veri cache'e kaydedildi: {filepath}")


def load_from_cache(filename: str, cache_dir: str = ".cache", max_age_hours: int = 1) -> Optional[pd.DataFrame]:
    """
    Cache'den veri yükler.
    
    Args:
        filename (str): Dosya adı
        cache_dir (str): Cache dizini
        max_age_hours (int): Maksimum yaş (saat)
        
    Returns:
        pd.DataFrame or None: Cache'deki veri veya None
    """
    filepath = os.path.join(cache_dir, filename)
    
    if not os.path.exists(filepath):
        return None
    
    # Dosya yaşını kontrol et
    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(filepath))
    if file_age.total_seconds() > max_age_hours * 3600:
        print(f"Cache dosyası çok eski: {filepath}")
        return None
    
    try:
        data = pd.read_pickle(filepath)
        print(f"Cache'den veri yüklendi: {filepath}")
        return data
    except Exception as e:
        print(f"Cache dosyası okunamadı: {e}")
        return None


if __name__ == "__main__":
    # Test fonksiyonları
    print("COVID-19 Veri Analizi - Utility Functions Test")
    
    # Yapılandırma yükle
    try:
        config = load_config()
        print("✓ Yapılandırma dosyası başarıyla yüklendi")
    except Exception as e:
        print(f"✗ Yapılandırma yüklenemedi: {e}")
    
    # Büyük sayı formatlama testi
    test_numbers = [1000, 15000, 1000000, 2500000, 1000000000]
    for num in test_numbers:
        formatted = format_large_number(num)
        print(f"{num:,} -> {formatted}")