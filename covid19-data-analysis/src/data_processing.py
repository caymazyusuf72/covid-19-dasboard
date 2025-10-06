"""
COVID-19 Data Processing Module
==============================

Bu modül COVID-19 verilerinin yüklenmesi, temizlenmesi ve ön işlenmesi için fonksiyonlar içerir.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Union
import os
import sys

# Kendi modülümüzü import edelim
sys.path.append(os.path.dirname(__file__))
from utils import (
    load_config, safe_request, clean_country_name, 
    save_to_cache, load_from_cache, setup_logging
)


# Logger kurulumu
logger = logging.getLogger(__name__)


class CovidDataLoader:
    """COVID-19 verilerini yüklemek için ana sınıf."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        CovidDataLoader'ı başlatır.
        
        Args:
            config_path (str): Yapılandırma dosyasının yolu
        """
        self.config = load_config(config_path)
        self.data_sources = self.config.get('data_sources', {})
        self.cache_settings = self.config.get('cache', {})
        
        logger.info("CovidDataLoader başlatıldı")
    
    
    def load_disease_sh_data(self, cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Disease.sh API'sinden COVID-19 verilerini yükler.
        
        Args:
            cache (bool): Veriyi cache'e kaydet/yükle
            
        Returns:
            pd.DataFrame or None: COVID-19 verileri veya None (hata durumunda)
        """
        cache_filename = "disease_sh_data.pkl"
        
        # Önce cache'den dene
        if cache:
            cached_data = load_from_cache(
                cache_filename,
                cache_dir=self.cache_settings.get('directory', '.cache'),
                max_age_hours=self.cache_settings.get('ttl', 3600) // 3600
            )
            if cached_data is not None:
                return cached_data
        
        try:
            base_url = self.data_sources.get('disease_sh_api', 'https://disease.sh/v3/covid-19')
            
            # Ülkeler verisi
            countries_url = f"{base_url}/countries"
            response = safe_request(countries_url)
            
            if not response or response.status_code != 200:
                logger.error("Disease.sh API'sinden veri alınamadı")
                return None
            
            countries_data = response.json()
            
            # DataFrame'e dönüştür
            df = pd.DataFrame(countries_data)
            
            # Sütun adlarını standartlaştır
            df = self._standardize_disease_sh_columns(df)
            
            # Veri temizleme
            df = self._clean_disease_sh_data(df)
            
            logger.info(f"Disease.sh'dan {len(df)} ülke verisi yüklendi")
            
            # Cache'e kaydet
            if cache:
                save_to_cache(df, cache_filename, self.cache_settings.get('directory', '.cache'))
            
            return df
            
        except Exception as e:
            logger.error(f"Disease.sh veri yükleme hatası: {e}")
            return None
    
    
    def load_johns_hopkins_data(self, data_type: str = 'confirmed', cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Johns Hopkins GitHub verilerini yükler.
        
        Args:
            data_type (str): 'confirmed', 'deaths', veya 'recovered'
            cache (bool): Veriyi cache'e kaydet/yükle
            
        Returns:
            pd.DataFrame or None: Zaman serisi verileri veya None
        """
        valid_types = ['confirmed', 'deaths', 'recovered']
        if data_type not in valid_types:
            logger.error(f"Geçersiz data_type: {data_type}. Geçerli tipler: {valid_types}")
            return None
        
        cache_filename = f"jhu_{data_type}_data.pkl"
        
        # Önce cache'den dene
        if cache:
            cached_data = load_from_cache(
                cache_filename,
                cache_dir=self.cache_settings.get('directory', '.cache'),
                max_age_hours=self.cache_settings.get('ttl', 3600) // 3600
            )
            if cached_data is not None:
                return cached_data
        
        try:
            base_url = self.data_sources.get('jhu_base_url', '')
            file_name = self.data_sources.get(f'jhu_{data_type}', '')
            
            if not base_url or not file_name:
                logger.error(f"Johns Hopkins {data_type} URL'i yapılandırmada bulunamadı")
                return None
            
            url = f"{base_url}/{file_name}"
            response = safe_request(url, timeout=30)
            
            if not response or response.status_code != 200:
                logger.error(f"Johns Hopkins {data_type} verisi alınamadı")
                return None
            
            # CSV'yi oku
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            # Veri işleme
            df = self._process_johns_hopkins_data(df, data_type)
            
            logger.info(f"Johns Hopkins {data_type} verisi yüklendi: {df.shape}")
            
            # Cache'e kaydet
            if cache:
                save_to_cache(df, cache_filename, self.cache_settings.get('directory', '.cache'))
            
            return df
            
        except Exception as e:
            logger.error(f"Johns Hopkins {data_type} veri yükleme hatası: {e}")
            return None
    
    
    def _standardize_disease_sh_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Disease.sh verilerinin sütun adlarını standartlaştırır."""
        column_mapping = {
            'country': 'country',
            'cases': 'total_cases',
            'deaths': 'total_deaths',
            'recovered': 'total_recovered',
            'active': 'active_cases',
            'critical': 'critical_cases',
            'todayCases': 'new_cases',
            'todayDeaths': 'new_deaths',
            'todayRecovered': 'new_recovered',
            'casesPerOneMillion': 'cases_per_million',
            'deathsPerOneMillion': 'deaths_per_million',
            'population': 'population',
            'continent': 'continent',
            'countryInfo': 'country_info'
        }
        
        # Mevcut sütunları eşleştir
        available_columns = {old: new for old, new in column_mapping.items() if old in df.columns}
        df = df.rename(columns=available_columns)
        
        return df
    
    
    def _clean_disease_sh_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Disease.sh verilerini temizler."""
        # Ülke adlarını temizle
        if 'country' in df.columns:
            df['country'] = df['country'].apply(clean_country_name)
        
        # Sayısal sütunlarda NaN'ları 0 ile değiştir
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Negatif değerleri 0 yap (veri hatası olabilir)
        for col in numeric_columns:
            df.loc[df[col] < 0, col] = 0
        
        # Güncellenme tarihi ekle
        df['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        
        return df
    
    
    def _process_johns_hopkins_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Johns Hopkins verilerini işler ve long format'a dönüştürür."""
        # Gereksiz sütunları kaldır
        if 'Lat' in df.columns:
            df = df.drop(['Lat', 'Long'], axis=1)
        
        # Sütun adlarını standartlaştır
        df = df.rename(columns={
            'Province/State': 'province_state',
            'Country/Region': 'country'
        })
        
        # Ülke adlarını temizle
        df['country'] = df['country'].apply(clean_country_name)
        
        # Tarih sütunlarını bul (4. sütundan itibaren tarih sütunları)
        date_columns = df.columns[2:]  # Province/State ve Country/Region'dan sonraki sütunlar
        
        # Long format'a dönüştür
        df_long = df.melt(
            id_vars=['country', 'province_state'],
            value_vars=date_columns,
            var_name='date',
            value_name=data_type
        )
        
        # Tarihleri datetime'a dönüştür
        df_long['date'] = pd.to_datetime(df_long['date'], format='%m/%d/%y')
        
        # Province/State boş olanları kaldır veya ülke bazında grupla
        df_country = df_long.groupby(['country', 'date'])[data_type].sum().reset_index()
        
        # Sayısal değerleri temizle
        df_country[data_type] = pd.to_numeric(df_country[data_type], errors='coerce').fillna(0)
        df_country.loc[df_country[data_type] < 0, data_type] = 0
        
        # Tarihe göre sırala
        df_country = df_country.sort_values(['country', 'date']).reset_index(drop=True)
        
        return df_country
    
    
    def combine_data_sources(self, include_time_series: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Farklı veri kaynaklarını birleştirir.
        
        Args:
            include_time_series (bool): Zaman serisi verilerini dahil et
            
        Returns:
            Dict[str, pd.DataFrame]: Birleştirilmiş veriler
        """
        combined_data = {}
        
        # Disease.sh snapshot verisi
        disease_data = self.load_disease_sh_data()
        if disease_data is not None:
            combined_data['current_snapshot'] = disease_data
            logger.info("Disease.sh snapshot verisi eklendi")
        
        # Johns Hopkins zaman serileri
        if include_time_series:
            for data_type in ['confirmed', 'deaths', 'recovered']:
                jhu_data = self.load_johns_hopkins_data(data_type)
                if jhu_data is not None:
                    combined_data[f'time_series_{data_type}'] = jhu_data
                    logger.info(f"Johns Hopkins {data_type} zaman serisi eklendi")
        
        return combined_data
    
    
    def get_country_data(self, country: str, data_sources: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Belirli bir ülkenin verilerini getirir.
        
        Args:
            country (str): Ülke adı
            data_sources (List[str], optional): Kullanılacak veri kaynakları
            
        Returns:
            Dict[str, pd.DataFrame]: Ülke verileri
        """
        country_data = {}
        clean_country = clean_country_name(country)
        
        if not data_sources:
            data_sources = ['current_snapshot', 'time_series_confirmed', 'time_series_deaths', 'time_series_recovered']
        
        combined_data = self.combine_data_sources(include_time_series=True)
        
        for source_name in data_sources:
            if source_name in combined_data:
                df = combined_data[source_name]
                
                if 'country' in df.columns:
                    country_df = df[df['country'] == clean_country].copy()
                    if not country_df.empty:
                        country_data[source_name] = country_df
                        logger.info(f"{clean_country} için {source_name} verisi bulundu ({len(country_df)} kayıt)")
                    else:
                        logger.warning(f"{clean_country} için {source_name} verisi bulunamadı")
        
        return country_data


def calculate_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mevcut verilerden türetilmiş metrikleri hesaplar.
    
    Args:
        df (pd.DataFrame): COVID-19 verileri
        
    Returns:
        pd.DataFrame: Türetilmiş metriklerle birlikte veriler
    """
    df = df.copy()
    
    # Case Fatality Rate (CFR)
    if 'total_cases' in df.columns and 'total_deaths' in df.columns:
        df['case_fatality_rate'] = np.where(
            df['total_cases'] > 0,
            (df['total_deaths'] / df['total_cases']) * 100,
            0
        )
    
    # Recovery Rate
    if 'total_cases' in df.columns and 'total_recovered' in df.columns:
        df['recovery_rate'] = np.where(
            df['total_cases'] > 0,
            (df['total_recovered'] / df['total_cases']) * 100,
            0
        )
    
    # Active cases (eğer mevcut değilse hesapla)
    if 'active_cases' not in df.columns and all(col in df.columns for col in ['total_cases', 'total_deaths', 'total_recovered']):
        df['active_cases'] = df['total_cases'] - df['total_deaths'] - df['total_recovered']
        df['active_cases'] = df['active_cases'].clip(lower=0)  # Negatif değerleri 0 yap
    
    return df


def calculate_time_series_metrics(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """
    Zaman serisi verilerinden türetilmiş metrikleri hesaplar.
    
    Args:
        df (pd.DataFrame): Zaman serisi verileri (country, date, value_column sütunları olmalı)
        value_column (str): Değer sütununun adı
        
    Returns:
        pd.DataFrame: Türetilmiş metriklerle birlikte veriler
    """
    df = df.copy().sort_values(['country', 'date'])
    
    # Günlük yeni vakalar
    df[f'new_{value_column}'] = df.groupby('country')[value_column].diff().fillna(0)
    df[f'new_{value_column}'] = df[f'new_{value_column}'].clip(lower=0)  # Negatif değerleri 0 yap
    
    # 7 günlük hareketli ortalama
    df[f'{value_column}_7day_avg'] = df.groupby('country')[f'new_{value_column}'].rolling(
        window=7, min_periods=1
    ).mean().reset_index(drop=True)
    
    # 14 günlük hareketli ortalama
    df[f'{value_column}_14day_avg'] = df.groupby('country')[f'new_{value_column}'].rolling(
        window=14, min_periods=1
    ).mean().reset_index(drop=True)
    
    # Büyüme oranı (yüzde değişim)
    df[f'{value_column}_growth_rate'] = df.groupby('country')[value_column].pct_change(periods=1).fillna(0) * 100
    
    return df


if __name__ == "__main__":
    # Test kodu
    print("🦠 COVID-19 Data Processing Test")
    print("=" * 40)
    
    try:
        # Data loader'ı başlat
        loader = CovidDataLoader()
        
        # Disease.sh verisi test et
        print("\n🔗 Disease.sh verisi yükleniyor...")
        disease_data = loader.load_disease_sh_data()
        if disease_data is not None:
            print(f"✅ {len(disease_data)} ülke verisi yüklendi")
            print(f"Sütunlar: {', '.join(disease_data.columns[:5])}...")
            
            # Türetilmiş metrikleri hesapla
            disease_data_with_metrics = calculate_derived_metrics(disease_data)
            print(f"✅ Türetilmiş metrikler eklendi")
        
        # Johns Hopkins verisi test et
        print("\n🔗 Johns Hopkins confirmed verisi yükleniyor...")
        jhu_confirmed = loader.load_johns_hopkins_data('confirmed')
        if jhu_confirmed is not None:
            print(f"✅ {len(jhu_confirmed)} kayıt yüklendi")
            print(f"Tarih aralığı: {jhu_confirmed['date'].min()} - {jhu_confirmed['date'].max()}")
            
            # Zaman serisi metrikleri hesapla
            jhu_with_metrics = calculate_time_series_metrics(jhu_confirmed, 'confirmed')
            print(f"✅ Zaman serisi metrikleri eklendi")
        
        # Türkiye verisi test et
        print("\n🇹🇷 Türkiye verisi test ediliyor...")
        turkey_data = loader.get_country_data('Turkey')
        for source, data in turkey_data.items():
            print(f"   {source}: {len(data)} kayıt")
        
        print("\n🎉 Tüm testler başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"\n💥 Test hatası: {e}")