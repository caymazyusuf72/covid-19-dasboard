"""
COVID-19 Data Analysis Module
=============================

Bu modül COVID-19 verilerinin istatistiksel analizini yapmak için fonksiyonlar içerir.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
import warnings
from datetime import datetime, timedelta
import logging
import os
import sys

# Kendi modülümüzü import edelim
sys.path.append(os.path.dirname(__file__))
from utils import load_config, format_large_number, calculate_percentage_change

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class CovidAnalyzer:
    """COVID-19 verilerini analiz etmek için ana sınıf."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        CovidAnalyzer'ı başlatır.
        
        Args:
            config_path (str): Yapılandırma dosyasının yolu
        """
        try:
            self.config = load_config(config_path)
            self.analysis_config = self.config.get('analysis', {})
        except:
            self.analysis_config = {
                'default_countries': ['Turkey', 'United States', 'Germany', 'Italy', 'Spain'],
                'moving_average_windows': [7, 14, 30]
            }
        
        logger.info("CovidAnalyzer başlatıldı")
    
    
    def calculate_global_stats(self, df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """
        Global COVID-19 istatistiklerini hesaplar.
        
        Args:
            df (pd.DataFrame): COVID-19 snapshot verisi
            
        Returns:
            Dict: Global istatistikler
        """
        stats = {
            'total_countries': len(df),
            'total_cases': df.get('total_cases', pd.Series()).sum(),
            'total_deaths': df.get('total_deaths', pd.Series()).sum(),
            'total_recovered': df.get('total_recovered', pd.Series()).sum(),
            'total_active': df.get('active_cases', pd.Series()).sum(),
            'global_cfr': 0.0,  # Case Fatality Rate
            'global_recovery_rate': 0.0,
            'countries_with_new_cases': 0,
            'countries_with_new_deaths': 0
        }
        
        # CFR hesapla
        if stats['total_cases'] > 0:
            stats['global_cfr'] = (stats['total_deaths'] / stats['total_cases']) * 100
        
        # Recovery rate hesapla
        if stats['total_cases'] > 0:
            stats['global_recovery_rate'] = (stats['total_recovered'] / stats['total_cases']) * 100
        
        # Yeni vakası olan ülkeler
        if 'new_cases' in df.columns:
            stats['countries_with_new_cases'] = len(df[df['new_cases'] > 0])
        
        # Yeni ölümü olan ülkeler
        if 'new_deaths' in df.columns:
            stats['countries_with_new_deaths'] = len(df[df['new_deaths'] > 0])
        
        # Aktif vaka kontrolü
        if stats['total_active'] <= 0 and all(col in df.columns for col in ['total_cases', 'total_deaths', 'total_recovered']):
            stats['total_active'] = max(0, stats['total_cases'] - stats['total_deaths'] - stats['total_recovered'])
        
        return stats
    
    
    def analyze_country_trends(self, time_series_df: pd.DataFrame, country: str, 
                              days_back: int = 30) -> Dict[str, Union[float, int, str]]:
        """
        Belirli bir ülkenin trendlerini analiz eder.
        
        Args:
            time_series_df (pd.DataFrame): Zaman serisi verisi
            country (str): Analiz edilecek ülke
            days_back (int): Kaç gün geriye bakılacağı
            
        Returns:
            Dict: Trend analizi sonuçları
        """
        country_data = time_series_df[time_series_df['country'] == country].copy()
        
        if country_data.empty:
            return {'error': f'{country} için veri bulunamadı'}
        
        country_data = country_data.sort_values('date').tail(days_back)
        
        analysis = {
            'country': country,
            'analysis_period': f'{days_back} gün',
            'data_points': len(country_data),
            'date_range': f"{country_data['date'].min().strftime('%Y-%m-%d')} - {country_data['date'].max().strftime('%Y-%m-%d')}"
        }
        
        # Trend analizi için sütunları kontrol et
        for metric in ['confirmed', 'deaths', 'recovered']:
            if metric in country_data.columns:
                analysis.update(self._analyze_metric_trend(country_data, metric, days_back))
        
        return analysis
    
    
    def _analyze_metric_trend(self, data: pd.DataFrame, metric: str, days_back: int) -> Dict[str, Union[float, str]]:
        """Belirli bir metrik için trend analizi yapar."""
        results = {}
        
        if len(data) < 2:
            return {f'{metric}_trend': 'Insufficient data'}
        
        # Günlük yeni değerler
        new_metric = f'new_{metric}'
        if new_metric in data.columns:
            recent_avg = data[new_metric].tail(7).mean()
            previous_avg = data[new_metric].head(7).mean() if len(data) >= 14 else recent_avg
            
            results[f'{metric}_daily_avg_recent'] = round(recent_avg, 2)
            results[f'{metric}_daily_avg_previous'] = round(previous_avg, 2)
            results[f'{metric}_daily_change_pct'] = round(calculate_percentage_change(recent_avg, previous_avg), 2)
        
        # Toplam değer trendi
        if len(data) >= 7:
            # Son 7 gün ile önceki 7 gün karşılaştırması
            recent_total = data[metric].iloc[-1]
            week_ago_total = data[metric].iloc[-8] if len(data) >= 8 else data[metric].iloc[0]
            
            results[f'{metric}_total_current'] = int(recent_total)
            results[f'{metric}_total_week_ago'] = int(week_ago_total)
            results[f'{metric}_total_change'] = int(recent_total - week_ago_total)
            results[f'{metric}_total_change_pct'] = round(calculate_percentage_change(recent_total, week_ago_total), 2)
        
        # Trend yönü
        if len(data) >= 3:
            recent_values = data[metric].tail(3).values
            if len(recent_values) == 3:
                if recent_values[2] > recent_values[1] > recent_values[0]:
                    trend = "Artış"
                elif recent_values[2] < recent_values[1] < recent_values[0]:
                    trend = "Azalış"
                else:
                    trend = "Karışık"
                
                results[f'{metric}_trend_direction'] = trend
        
        return results
    
    
    def compare_countries(self, df: pd.DataFrame, countries: List[str], 
                         metrics: List[str] = ['total_cases', 'total_deaths', 'cases_per_million']) -> pd.DataFrame:
        """
        Ülkeleri belirtilen metriklerde karşılaştırır.
        
        Args:
            df (pd.DataFrame): COVID-19 verisi
            countries (List[str]): Karşılaştırılacak ülkeler
            metrics (List[str]): Karşılaştırma metrikleri
            
        Returns:
            pd.DataFrame: Karşılaştırma tablosu
        """
        comparison_data = []
        
        for country in countries:
            country_data = df[df['country'] == country]
            
            if country_data.empty:
                logger.warning(f"{country} için veri bulunamadı")
                continue
            
            row_data = {'country': country}
            
            for metric in metrics:
                if metric in country_data.columns:
                    value = country_data[metric].iloc[0]
                    row_data[metric] = value
                else:
                    row_data[metric] = np.nan
            
            comparison_data.append(row_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            # Her metrik için sıralama ekle
            for metric in metrics:
                if metric in comparison_df.columns:
                    comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=False, method='dense')
        
        return comparison_df
    
    
    def calculate_severity_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ülkeler için COVID-19 şiddet indeksi hesaplar.
        
        Args:
            df (pd.DataFrame): COVID-19 verisi
            
        Returns:
            pd.DataFrame: Şiddet indeksi eklenmiş veri
        """
        df = df.copy()
        
        # Gerekli sütunları kontrol et
        required_cols = ['cases_per_million', 'deaths_per_million']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < 2:
            logger.warning("Şiddet indeksi için gerekli sütunlar bulunamadı")
            return df
        
        # Normalize edilmiş değerler (0-1 arası)
        for col in available_cols:
            col_max = df[col].max()
            if col_max > 0:
                df[f'{col}_normalized'] = df[col] / col_max
            else:
                df[f'{col}_normalized'] = 0
        
        # Basit şiddet indeksi (ağırlıklı ortalama)
        df['severity_index'] = (
            df.get('cases_per_million_normalized', 0) * 0.6 +  # Vaka oranı ağırlığı
            df.get('deaths_per_million_normalized', 0) * 0.4   # Ölüm oranı ağırlığı
        ) * 100  # 0-100 arası ölçek
        
        # Şiddet kategorisi
        df['severity_category'] = pd.cut(
            df['severity_index'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['Çok Düşük', 'Düşük', 'Orta', 'Yüksek', 'Çok Yüksek']
        )
        
        return df
    
    
    def detect_outbreak_patterns(self, time_series_df: pd.DataFrame, 
                                country: str, threshold_multiplier: float = 2.0) -> Dict[str, List]:
        """
        Salgın patlaması desenlerini tespit eder.
        
        Args:
            time_series_df (pd.DataFrame): Zaman serisi verisi
            country (str): Analiz edilecek ülke
            threshold_multiplier (float): Patlama tespit eşiği
            
        Returns:
            Dict: Tespit edilen patlamalar
        """
        country_data = time_series_df[time_series_df['country'] == country].copy()
        
        if country_data.empty or 'new_confirmed' not in country_data.columns:
            return {'error': f'{country} için uygun veri bulunamadı'}
        
        country_data = country_data.sort_values('date')
        
        # 7 günlük hareketli ortalama
        country_data['ma_7'] = country_data['new_confirmed'].rolling(window=7, min_periods=1).mean()
        
        # Baseline (genel ortalama)
        baseline = country_data['ma_7'].mean()
        threshold = baseline * threshold_multiplier
        
        # Patlama noktalarını tespit et
        outbreak_periods = []
        in_outbreak = False
        outbreak_start = None
        
        for idx, row in country_data.iterrows():
            if row['ma_7'] > threshold and not in_outbreak:
                # Patlama başlangıcı
                outbreak_start = row['date']
                in_outbreak = True
                
            elif row['ma_7'] <= threshold and in_outbreak:
                # Patlama bitişi
                outbreak_periods.append({
                    'start_date': outbreak_start,
                    'end_date': row['date'],
                    'duration_days': (row['date'] - outbreak_start).days,
                    'peak_cases': country_data[
                        (country_data['date'] >= outbreak_start) & 
                        (country_data['date'] <= row['date'])
                    ]['new_confirmed'].max()
                })
                in_outbreak = False
        
        # Eğer hala patlama devam ediyorsa
        if in_outbreak:
            outbreak_periods.append({
                'start_date': outbreak_start,
                'end_date': country_data['date'].iloc[-1],
                'duration_days': (country_data['date'].iloc[-1] - outbreak_start).days,
                'peak_cases': country_data[country_data['date'] >= outbreak_start]['new_confirmed'].max(),
                'status': 'ongoing'
            })
        
        return {
            'country': country,
            'baseline_daily_cases': round(baseline, 2),
            'outbreak_threshold': round(threshold, 2),
            'total_outbreaks': len(outbreak_periods),
            'outbreak_periods': outbreak_periods
        }
    
    
    def calculate_reproduction_number(self, time_series_df: pd.DataFrame, 
                                    country: str, serial_interval: int = 7) -> Dict[str, float]:
        """
        Basit üreme sayısı (R) tahmini hesaplar.
        
        Args:
            time_series_df (pd.DataFrame): Zaman serisi verisi
            country (str): Analiz edilecek ülke
            serial_interval (int): Serial interval (gün)
            
        Returns:
            Dict: R sayısı tahminleri
        """
        country_data = time_series_df[time_series_df['country'] == country].copy()
        
        if country_data.empty or 'new_confirmed' not in country_data.columns:
            return {'error': f'{country} için uygun veri bulunamadı'}
        
        country_data = country_data.sort_values('date')
        
        if len(country_data) < serial_interval * 2:
            return {'error': 'R hesabı için yeterli veri yok'}
        
        # Basit R hesaplama: (Son N gün ortalama) / (Önceki N gün ortalama)
        recent_avg = country_data['new_confirmed'].tail(serial_interval).mean()
        previous_avg = country_data['new_confirmed'].iloc[-(serial_interval*2):-serial_interval].mean()
        
        r_estimate = recent_avg / previous_avg if previous_avg > 0 else 0
        
        # R yorumu
        if r_estimate > 1.2:
            interpretation = "Hızla yayılıyor"
        elif r_estimate > 1.0:
            interpretation = "Yavaşça artıyor"
        elif r_estimate > 0.8:
            interpretation = "Kontrol altında"
        else:
            interpretation = "Azalıyor"
        
        return {
            'country': country,
            'r_estimate': round(r_estimate, 2),
            'recent_avg_cases': round(recent_avg, 2),
            'previous_avg_cases': round(previous_avg, 2),
            'interpretation': interpretation,
            'calculation_period': f'{serial_interval} gün'
        }
    
    
    def generate_country_report(self, snapshot_df: pd.DataFrame, 
                               time_series_df: Optional[pd.DataFrame], 
                               country: str) -> Dict[str, Union[str, float, Dict]]:
        """
        Kapsamlı ülke raporu oluşturur.
        
        Args:
            snapshot_df (pd.DataFrame): Mevcut durum verisi
            time_series_df (pd.DataFrame, optional): Zaman serisi verisi
            country (str): Rapor edilecek ülke
            
        Returns:
            Dict: Detaylı ülke raporu
        """
        report = {
            'country': country,
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'data_sources': []
        }
        
        # Snapshot analizi
        country_snapshot = snapshot_df[snapshot_df['country'] == country]
        if not country_snapshot.empty:
            report['data_sources'].append('snapshot')
            data = country_snapshot.iloc[0]
            
            report['current_status'] = {
                'total_cases': int(data.get('total_cases', 0)),
                'total_deaths': int(data.get('total_deaths', 0)),
                'total_recovered': int(data.get('total_recovered', 0)),
                'active_cases': int(data.get('active_cases', 0)),
                'new_cases': int(data.get('new_cases', 0)),
                'new_deaths': int(data.get('new_deaths', 0)),
                'cfr': round((data.get('total_deaths', 0) / max(data.get('total_cases', 1), 1)) * 100, 2),
                'cases_per_million': round(data.get('cases_per_million', 0), 1),
                'deaths_per_million': round(data.get('deaths_per_million', 0), 1)
            }
        
        # Zaman serisi analizi
        if time_series_df is not None:
            report['data_sources'].append('time_series')
            
            # Trend analizi
            trend_analysis = self.analyze_country_trends(time_series_df, country, days_back=30)
            if 'error' not in trend_analysis:
                report['trend_analysis'] = trend_analysis
            
            # Patlama analizi
            outbreak_analysis = self.detect_outbreak_patterns(time_series_df, country)
            if 'error' not in outbreak_analysis:
                report['outbreak_analysis'] = outbreak_analysis
            
            # R sayısı
            r_analysis = self.calculate_reproduction_number(time_series_df, country)
            if 'error' not in r_analysis:
                report['reproduction_analysis'] = r_analysis
        
        return report


def quick_analysis(snapshot_df: pd.DataFrame, country: str = 'Turkey') -> None:
    """
    Hızlı analiz için yardımcı fonksiyon.
    
    Args:
        snapshot_df (pd.DataFrame): Mevcut durum verisi
        country (str): Analiz edilecek ülke
    """
    analyzer = CovidAnalyzer()
    
    print(f"🔍 {country} için hızlı analiz yapılıyor...")
    
    # Global istatistikler
    global_stats = analyzer.calculate_global_stats(snapshot_df)
    print(f"\n🌍 Global İstatistikler:")
    print(f"   Toplam Ülke: {global_stats['total_countries']:,}")
    print(f"   Toplam Vaka: {format_large_number(global_stats['total_cases'])}")
    print(f"   Toplam Ölüm: {format_large_number(global_stats['total_deaths'])}")
    print(f"   CFR: {global_stats['global_cfr']:.2f}%")
    
    # Ülke karşılaştırması
    countries = ['Turkey', 'Germany', 'United States', 'Italy']
    comparison = analyzer.compare_countries(snapshot_df, countries)
    if not comparison.empty:
        print(f"\n🏆 Ülke Karşılaştırması:")
        for _, row in comparison.iterrows():
            print(f"   {row['country']}: {format_large_number(row.get('total_cases', 0))} vaka")
    
    print("\n✅ Hızlı analiz tamamlandı!")


if __name__ == "__main__":
    # Test kodu
    print("🔍 COVID-19 Analysis Test")
    print("=" * 40)
    
    # Test verisi oluştur
    test_data = pd.DataFrame({
        'country': ['Turkey', 'Germany', 'USA', 'Italy', 'Spain'],
        'total_cases': [17232066, 38437756, 103436829, 25603510, 13980340],
        'total_deaths': [102174, 174979, 1127152, 192474, 121760],
        'total_recovered': [16000000, 37000000, 100000000, 24000000, 13000000],
        'active_cases': [1129892, 1262777, 2309677, 1411036, 858580],
        'new_cases': [0, 0, 0, 0, 0],
        'new_deaths': [0, 0, 0, 0, 0],
        'cases_per_million': [204129, 459763, 312691, 435069, 296304],
        'deaths_per_million': [1210, 2094, 3407, 3272, 2583],
        'population': [84340000, 83600000, 331000000, 59110000, 47200000]
    })
    
    try:
        analyzer = CovidAnalyzer()
        
        # Global istatistikler testi
        print("\n🌍 Global istatistikler hesaplanıyor...")
        global_stats = analyzer.calculate_global_stats(test_data)
        print(f"✅ Toplam ülke: {global_stats['total_countries']}")
        print(f"✅ Global CFR: {global_stats['global_cfr']:.2f}%")
        
        # Ülke karşılaştırması testi
        print("\n🏆 Ülke karşılaştırması yapılıyor...")
        comparison = analyzer.compare_countries(
            test_data, 
            ['Turkey', 'Germany', 'USA'], 
            ['total_cases', 'total_deaths', 'cases_per_million']
        )
        print(f"✅ {len(comparison)} ülke karşılaştırıldı")
        
        # Şiddet indeksi testi
        print("\n⚠️ Şiddet indeksi hesaplanıyor...")
        severity_data = analyzer.calculate_severity_index(test_data)
        print(f"✅ Şiddet indeksi eklendi")
        print(f"   Türkiye şiddet indeksi: {severity_data[severity_data['country'] == 'Turkey']['severity_index'].iloc[0]:.1f}")
        
        print("\n🎉 Tüm analiz testleri başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"\n💥 Analiz test hatası: {e}")