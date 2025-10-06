"""
COVID-19 Data Visualization Module
==================================

Bu modül COVID-19 verilerini görselleştirmek için fonksiyonlar içerir.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
import warnings
from datetime import datetime, timedelta
import os
import sys

# Kendi modülümüzü import edelim
sys.path.append(os.path.dirname(__file__))
from utils import load_config, format_large_number

warnings.filterwarnings('ignore')

# Matplotlib Türkçe karakter desteği
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
sns.set_style("whitegrid")

class CovidVisualizer:
    """COVID-19 verilerini görselleştirmek için ana sınıf."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        CovidVisualizer'ı başlatır.
        
        Args:
            config_path (str): Yapılandırma dosyasının yolu
        """
        try:
            self.config = load_config(config_path)
            self.colors = self.config.get('visualization', {}).get('colors', {})
            self.figure_size = self.config.get('visualization', {}).get('figure_size', [12, 8])
        except:
            # Eğer config yüklenemezse varsayılan değerler
            self.colors = {
                'confirmed': '#1f77b4',
                'deaths': '#d62728', 
                'recovered': '#2ca02c',
                'active': '#ff7f0e'
            }
            self.figure_size = [12, 8]
    
    
    def plot_country_summary(self, df: pd.DataFrame, country: str, save_path: Optional[str] = None) -> plt.Figure:
        """
        Belirli bir ülkenin özet istatistiklerini görselleştirir.
        
        Args:
            df (pd.DataFrame): COVID-19 snapshot verisi
            country (str): Ülke adı
            save_path (str, optional): Grafiği kaydetmek için dosya yolu
            
        Returns:
            plt.Figure: Matplotlib figure objesi
        """
        # Ülke verisini filtrele
        country_data = df[df['country'] == country]
        
        if country_data.empty:
            print(f"⚠️ {country} için veri bulunamadı")
            return None
        
        data = country_data.iloc[0]
        
        # Figure oluştur
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'🦠 COVID-19 Özet İstatistikleri - {country}', fontsize=16, fontweight='bold')
        
        # 1. Toplam Rakamlar (Bar Chart)
        categories = ['Toplam\nVaka', 'Aktif\nVaka', 'İyileşen', 'Vefat']
        values = [
            data.get('total_cases', 0),
            data.get('active_cases', 0),
            data.get('total_recovered', 0),
            data.get('total_deaths', 0)
        ]
        colors = [self.colors.get('confirmed', '#1f77b4'),
                 self.colors.get('active', '#ff7f0e'),
                 self.colors.get('recovered', '#2ca02c'),
                 self.colors.get('deaths', '#d62728')]
        
        bars = axes[0, 0].bar(categories, values, color=colors, alpha=0.8)
        axes[0, 0].set_title('Toplam Rakamlar', fontweight='bold')
        axes[0, 0].set_ylabel('Kişi Sayısı')
        
        # Bar'ların üzerine değerleri yaz
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           format_large_number(value),
                           ha='center', va='bottom', fontweight='bold')
        
        # Y eksenini formatla
        axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_large_number(x)))
        
        
        # 2. Günlük Yeni Vakalar
        daily_categories = ['Yeni\nVaka', 'Yeni\nİyileşen', 'Yeni\nVefat']
        daily_values = [
            data.get('new_cases', 0),
            data.get('new_recovered', 0),
            data.get('new_deaths', 0)
        ]
        daily_colors = [colors[0], colors[2], colors[3]]
        
        bars2 = axes[0, 1].bar(daily_categories, daily_values, color=daily_colors, alpha=0.8)
        axes[0, 1].set_title('Günlük Yeni Rakamlar', fontweight='bold')
        axes[0, 1].set_ylabel('Kişi Sayısı')
        
        for bar, value in zip(bars2, daily_values):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           format_large_number(value),
                           ha='center', va='bottom', fontweight='bold')
        
        axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_large_number(x)))
        
        
        # 3. Oranlar (Pie Chart)
        if data.get('total_cases', 0) > 0:
            pie_labels = ['İyileşen', 'Aktif', 'Vefat']
            pie_values = [
                data.get('total_recovered', 0),
                data.get('active_cases', 0),
                data.get('total_deaths', 0)
            ]
            pie_colors = [colors[2], colors[1], colors[3]]
            
            # Sıfır olmayan değerleri filtrele
            pie_data = [(label, value, color) for label, value, color in zip(pie_labels, pie_values, pie_colors) if value > 0]
            if pie_data:
                labels, values, colors_filtered = zip(*pie_data)
                
                wedges, texts, autotexts = axes[1, 0].pie(values, labels=labels, colors=colors_filtered, 
                                                         autopct='%1.1f%%', startangle=90)
                axes[1, 0].set_title('Vaka Dağılımı', fontweight='bold')
            else:
                axes[1, 0].text(0.5, 0.5, 'Veri Yok', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Vaka Dağılımı', fontweight='bold')
        
        
        # 4. Milyon Kişi Başına Oranlar
        per_million_categories = ['Vaka/1M', 'Vefat/1M']
        per_million_values = [
            data.get('cases_per_million', 0),
            data.get('deaths_per_million', 0)
        ]
        per_million_colors = [colors[0], colors[3]]
        
        bars3 = axes[1, 1].bar(per_million_categories, per_million_values, color=per_million_colors, alpha=0.8)
        axes[1, 1].set_title('Milyon Kişi Başına', fontweight='bold')
        axes[1, 1].set_ylabel('Kişi/1M')
        
        for bar, value in zip(bars3, per_million_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:,.0f}',
                           ha='center', va='bottom', fontweight='bold')
        
        
        # Genel düzenlemeler
        plt.tight_layout()
        
        # Kaydet
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Grafik kaydedildi: {save_path}")
        
        return fig
    
    
    def plot_time_series(self, df: pd.DataFrame, countries: List[str], 
                        value_column: str = 'confirmed', 
                        show_moving_average: bool = True,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Ülkeler için zaman serisi grafiği çizer.
        
        Args:
            df (pd.DataFrame): Zaman serisi verisi
            countries (List[str]): Görselleştirilecek ülkeler
            value_column (str): Görselleştirilecek sütun
            show_moving_average (bool): Hareketli ortalama göster
            save_path (str, optional): Grafiği kaydetmek için dosya yolu
            
        Returns:
            plt.Figure: Matplotlib figure objesi
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        colors_cycle = plt.cm.Set1(np.linspace(0, 1, len(countries)))
        
        for i, country in enumerate(countries):
            country_data = df[df['country'] == country].sort_values('date')
            
            if country_data.empty:
                print(f"⚠️ {country} için {value_column} verisi bulunamadı")
                continue
            
            color = colors_cycle[i]
            
            # 1. Kümülatif veriler
            ax1.plot(country_data['date'], country_data[value_column], 
                    label=country, color=color, linewidth=2, alpha=0.8)
            
            # 2. Günlük yeni veriler
            new_col = f'new_{value_column}'
            if new_col in country_data.columns:
                ax2.plot(country_data['date'], country_data[new_col],
                        label=f'{country} (Günlük)', color=color, alpha=0.6)
                
                # Hareketli ortalama
                if show_moving_average:
                    avg_col = f'{value_column}_7day_avg'
                    if avg_col in country_data.columns:
                        ax2.plot(country_data['date'], country_data[avg_col],
                               label=f'{country} (7 gün ort.)', color=color, linewidth=2)
        
        # Grafik düzenlemeleri
        ax1.set_title(f'Kümülatif {value_column.title()} Vakaları', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Toplam Vaka Sayısı')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_large_number(x)))
        
        ax2.set_title(f'Günlük Yeni {value_column.title()} Vakaları', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Tarih')
        ax2.set_ylabel('Günlük Yeni Vaka')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_large_number(x)))
        
        # X ekseni tarih formatı
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Zaman serisi grafiği kaydedildi: {save_path}")
        
        return fig
    
    
    def plot_top_countries(self, df: pd.DataFrame, metric: str = 'total_cases', 
                          top_n: int = 10, save_path: Optional[str] = None) -> plt.Figure:
        """
        Belirtilen metrikte en yüksek değerlere sahip ülkeleri gösterir.
        
        Args:
            df (pd.DataFrame): COVID-19 snapshot verisi
            metric (str): Sıralama metriği
            top_n (int): Gösterilecek ülke sayısı
            save_path (str, optional): Grafiği kaydetmek için dosya yolu
            
        Returns:
            plt.Figure: Matplotlib figure objesi
        """
        if metric not in df.columns:
            print(f"⚠️ {metric} sütunu verisetinde bulunamadı")
            return None
        
        # En yüksek değerlere sahip ülkeleri al
        top_countries = df.nlargest(top_n, metric)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Horizontal bar chart
        bars = ax.barh(range(len(top_countries)), top_countries[metric], 
                      color=self.colors.get(metric.split('_')[-1], '#1f77b4'), alpha=0.8)
        
        # Ülke isimlerini y eksenine ekle
        ax.set_yticks(range(len(top_countries)))
        ax.set_yticklabels(top_countries['country'])
        
        # Değerleri bar'ların üzerine yaz
        for i, (bar, value) in enumerate(zip(bars, top_countries[metric])):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   format_large_number(value),
                   ha='left', va='center', fontweight='bold', fontsize=10)
        
        ax.set_xlabel(f'{metric.replace("_", " ").title()}')
        ax.set_title(f'🌍 En Yüksek {metric.replace("_", " ").title()} - İlk {top_n} Ülke', 
                    fontsize=14, fontweight='bold')
        
        # X ekseni formatı
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_large_number(x)))
        
        # Y eksenini ters çevir (en yüksek değer üstte)
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Top ülkeler grafiği kaydedildi: {save_path}")
        
        return fig
    
    
    def plot_correlation_matrix(self, df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
        """
        COVID-19 metriklerinin korelasyon matrisini görselleştirir.
        
        Args:
            df (pd.DataFrame): COVID-19 verisi
            save_path (str, optional): Grafiği kaydetmek için dosya yolu
            
        Returns:
            plt.Figure: Matplotlib figure objesi
        """
        # Sayısal sütunları seç
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # İlginç metrikleri seç
        interesting_metrics = [col for col in numeric_columns if any(keyword in col.lower() 
                             for keyword in ['cases', 'deaths', 'recovered', 'active', 'rate', 'million'])]
        
        if len(interesting_metrics) < 2:
            print("⚠️ Korelasyon analizi için yeterli sayısal sütun yok")
            return None
        
        correlation_data = df[interesting_metrics].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Heatmap
        sns.heatmap(correlation_data, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('🔍 COVID-19 Metrikleri Korelasyon Matrisi', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Korelasyon matrisi kaydedildi: {save_path}")
        
        return fig
    
    
    def create_dashboard_summary(self, snapshot_df: pd.DataFrame, time_series_df: Optional[pd.DataFrame] = None,
                                countries: List[str] = ['Turkey', 'United States', 'Germany'],
                                save_dir: str = "reports") -> Dict[str, plt.Figure]:
        """
        Kapsamlı dashboard özeti oluşturur.
        
        Args:
            snapshot_df (pd.DataFrame): Mevcut durum verisi
            time_series_df (pd.DataFrame, optional): Zaman serisi verisi
            countries (List[str]): Analiz edilecek ülkeler
            save_dir (str): Grafiklerin kaydedileceği dizin
            
        Returns:
            Dict[str, plt.Figure]: Oluşturulan figürlerin sözlüğü
        """
        figures = {}
        
        # Dizini oluştur
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 1. Global top ülkeler
        figures['top_cases'] = self.plot_top_countries(
            snapshot_df, 'total_cases', top_n=15,
            save_path=os.path.join(save_dir, 'top_countries_cases.png')
        )
        
        figures['top_deaths'] = self.plot_top_countries(
            snapshot_df, 'total_deaths', top_n=15,
            save_path=os.path.join(save_dir, 'top_countries_deaths.png')
        )
        
        # 2. Ülke özetleri
        for country in countries:
            if not snapshot_df[snapshot_df['country'] == country].empty:
                figures[f'{country.lower()}_summary'] = self.plot_country_summary(
                    snapshot_df, country,
                    save_path=os.path.join(save_dir, f'{country.lower()}_summary.png')
                )
        
        # 3. Zaman serisi (eğer veri varsa)
        if time_series_df is not None:
            available_countries = [c for c in countries if not time_series_df[time_series_df['country'] == c].empty]
            if available_countries:
                figures['time_series_confirmed'] = self.plot_time_series(
                    time_series_df, available_countries, 'confirmed',
                    save_path=os.path.join(save_dir, 'time_series_confirmed.png')
                )
        
        # 4. Korelasyon analizi
        figures['correlation'] = self.plot_correlation_matrix(
            snapshot_df,
            save_path=os.path.join(save_dir, 'correlation_matrix.png')
        )
        
        print(f"🎉 Dashboard özeti oluşturuldu! {len(figures)} grafik {save_dir} dizininde")
        
        return figures


def create_quick_visualization(snapshot_data: pd.DataFrame, country: str = 'Turkey') -> None:
    """
    Hızlı görselleştirme için yardımcı fonksiyon.
    
    Args:
        snapshot_data (pd.DataFrame): Mevcut durum verisi
        country (str): Görselleştirilecek ülke
    """
    visualizer = CovidVisualizer()
    
    print(f"📊 {country} için hızlı görselleştirme oluşturuluyor...")
    
    # Ülke özeti
    fig = visualizer.plot_country_summary(snapshot_data, country)
    if fig:
        plt.show()
    
    print("✅ Görselleştirme tamamlandı!")


if __name__ == "__main__":
    # Test kodu
    print("📊 COVID-19 Visualization Test")
    print("=" * 40)
    
    # Basit test verisi oluştur
    test_data = pd.DataFrame({
        'country': ['Turkey', 'Germany', 'USA', 'Italy', 'Spain'],
        'total_cases': [17232066, 38437756, 103436829, 25603510, 13980340],
        'total_deaths': [102174, 174979, 1127152, 192474, 121760],
        'total_recovered': [16000000, 37000000, 100000000, 24000000, 13000000],
        'active_cases': [1129892, 1262777, 2309677, 1411036, 858580],
        'new_cases': [0, 0, 0, 0, 0],
        'new_deaths': [0, 0, 0, 0, 0],
        'new_recovered': [0, 0, 0, 0, 0],
        'cases_per_million': [204129, 459763, 312691, 435069, 296304],
        'deaths_per_million': [1210, 2094, 3407, 3272, 2583]
    })
    
    try:
        visualizer = CovidVisualizer()
        
        # Ülke özeti testi
        print("\n🇹🇷 Türkiye özet grafiği oluşturuluyor...")
        fig1 = visualizer.plot_country_summary(test_data, 'Turkey')
        if fig1:
            print("✅ Türkiye özeti başarıyla oluşturuldu")
            plt.close(fig1)  # Belleği temizle
        
        # Top ülkeler testi
        print("\n🌍 Top ülkeler grafiği oluşturuluyor...")
        fig2 = visualizer.plot_top_countries(test_data, 'total_cases', top_n=5)
        if fig2:
            print("✅ Top ülkeler grafiği başarıyla oluşturuldu")
            plt.close(fig2)
        
        # Korelasyon testi
        print("\n🔍 Korelasyon matrisi oluşturuluyor...")
        fig3 = visualizer.plot_correlation_matrix(test_data)
        if fig3:
            print("✅ Korelasyon matrisi başarıyla oluşturuldu")
            plt.close(fig3)
        
        print("\n🎉 Tüm görselleştirme testleri başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"\n💥 Görselleştirme test hatası: {e}")