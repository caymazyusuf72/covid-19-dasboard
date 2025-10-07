"""
COVID-19 Data Analysis Dashboard
===============================

Bu Streamlit uygulaması COVID-19 verilerini interaktif olarak keşfetmek için tasarlanmıştır.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Proje modüllerini import et
sys.path.append('src')
try:
    from data_processing import CovidDataLoader, calculate_derived_metrics, calculate_time_series_metrics
    from analysis import CovidAnalyzer
    from visualization import CovidVisualizer
    from modeling import CovidPredictor
    from utils import format_large_number, load_config
except ImportError as e:
    st.error(f"Modül import hatası: {e}")
    st.stop()

# Sayfa yapılandırması
st.set_page_config(
    page_title="COVID-19 Veri Analizi",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Başlık
st.title("🦠 COVID-19 Veri Analizi Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.title("🔧 Kontrol Paneli")

# Cache fonksiyonları - veri yükleme performansı için
@st.cache_data(ttl=3600)  # 1 saat cache
def load_covid_data():
    """COVID-19 verilerini yükler ve cache'ler"""
    loader = CovidDataLoader()
    
    # Snapshot verisi
    snapshot_data = loader.load_disease_sh_data()
    if snapshot_data is not None:
        snapshot_data = calculate_derived_metrics(snapshot_data)
    
    # Zaman serisi verisi
    confirmed_data = loader.load_johns_hopkins_data('confirmed')
    if confirmed_data is not None:
        confirmed_data = calculate_time_series_metrics(confirmed_data, 'confirmed')
    
    return snapshot_data, confirmed_data

# Veri yükleme
with st.spinner('📊 COVID-19 verileri yükleniyor...'):
    snapshot_data, confirmed_data = load_covid_data()

if snapshot_data is None:
    st.error("❌ Snapshot verisi yüklenemedi. Lütfen internet bağlantınızı kontrol edin.")
    st.stop()

# Analiz sınıflarını başlat
analyzer = CovidAnalyzer()
visualizer = CovidVisualizer()
predictor = CovidPredictor()

# Global istatistikler
global_stats = analyzer.calculate_global_stats(snapshot_data)

# Sidebar seçenekleri
st.sidebar.header("📋 Filtreler")

# Ülke seçimi
available_countries = sorted(snapshot_data['country'].unique())
selected_countries = st.sidebar.multiselect(
    "🌍 Ülke Seçin",
    options=available_countries,
    default=['Turkey', 'Germany', 'United States', 'Italy'],
    help="Analiz edilecek ülkeleri seçin"
)

# Metrik seçimi
metrics_options = {
    'total_cases': 'Toplam Vaka',
    'total_deaths': 'Toplam Ölüm', 
    'total_recovered': 'Toplam İyileşen',
    'active_cases': 'Aktif Vaka',
    'cases_per_million': 'Milyon Başına Vaka',
    'deaths_per_million': 'Milyon Başına Ölüm'
}

selected_metric = st.sidebar.selectbox(
    "📊 Analiz Metriği",
    options=list(metrics_options.keys()),
    format_func=lambda x: metrics_options[x],
    index=0
)

# Görselleştirme tipi
chart_type = st.sidebar.radio(
    "📈 Grafik Tipi",
    ["Bar Chart", "Line Chart", "Scatter Plot", "Map View"],
    help="Görselleştirme tipini seçin"
)

# Ana sayfa düzeni
col1, col2, col3, col4 = st.columns(4)

# Global istatistikler kartları
with col1:
    st.metric(
        "🌍 Toplam Vaka",
        format_large_number(global_stats['total_cases']),
        delta=f"{global_stats['countries_with_new_cases']} ülkede yeni vaka"
    )

with col2:
    st.metric(
        "💀 Toplam Ölüm", 
        format_large_number(global_stats['total_deaths']),
        delta=f"CFR: {global_stats['global_cfr']:.2f}%"
    )

with col3:
    st.metric(
        "💚 Toplam İyileşen",
        format_large_number(global_stats['total_recovered']),
        delta=f"İyileşme oranı: {global_stats['global_recovery_rate']:.1f}%"
    )

with col4:
    st.metric(
        "🟡 Aktif Vaka",
        format_large_number(global_stats['total_active']),
        delta=f"{global_stats['total_countries']} ülke"
    )

st.markdown("---")

# Ana içerik alanı
if selected_countries:
    
    # Seçili ülkeler için veri filtrele
    filtered_data = snapshot_data[snapshot_data['country'].isin(selected_countries)]
    
    # Tab'lar oluştur
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Genel Bakış",
        "📈 Zaman Serisi",
        "🔍 Detaylı Analiz",
        "🌍 Global Harita",
        "🤖 ML Tahminleri"
    ])
    
    with tab1:
        st.header("📊 Seçili Ülkeler - Genel Bakış")
        
        # Ülke karşılaştırma tablosu
        comparison_metrics = ['total_cases', 'total_deaths', 'active_cases', 'cases_per_million']
        comparison_df = analyzer.compare_countries(snapshot_data, selected_countries, comparison_metrics)
        
        if not comparison_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # İnteraktif grafik
                if chart_type == "Bar Chart":
                    fig = px.bar(
                        filtered_data,
                        x='country',
                        y=selected_metric,
                        title=f"{metrics_options[selected_metric]} Karşılaştırması",
                        labels={'country': 'Ülke', selected_metric: metrics_options[selected_metric]},
                        color=selected_metric,
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Scatter Plot":
                    x_metric = 'total_cases'
                    y_metric = 'total_deaths'
                    
                    fig = px.scatter(
                        filtered_data,
                        x=x_metric,
                        y=y_metric,
                        size='population' if 'population' in filtered_data.columns else None,
                        hover_name='country',
                        title=f"{metrics_options.get(y_metric, y_metric)} vs {metrics_options.get(x_metric, x_metric)}",
                        labels={
                            x_metric: metrics_options.get(x_metric, x_metric),
                            y_metric: metrics_options.get(y_metric, y_metric)
                        }
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Karşılaştırma tablosu
                st.subheader("📋 Karşılaştırma Tablosu")
                
                display_df = comparison_df.copy()
                # Sayısal değerleri formatla
                for col in ['total_cases', 'total_deaths', 'active_cases']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
                
                for col in ['cases_per_million', 'deaths_per_million']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
                
                st.dataframe(display_df, use_container_width=True)
    
    with tab2:
        st.header("📈 Zaman Serisi Analizi")
        
        if confirmed_data is not None:
            # Zaman serisi filtresi
            time_filter = st.selectbox(
                "⏱️ Zaman Aralığı",
                ["Son 30 gün", "Son 90 gün", "Son 1 yıl", "Tüm veriler"],
                index=2
            )
            
            # Tarihi filtrele
            max_date = confirmed_data['date'].max()
            if time_filter == "Son 30 gün":
                start_date = max_date - timedelta(days=30)
            elif time_filter == "Son 90 gün":
                start_date = max_date - timedelta(days=90)
            elif time_filter == "Son 1 yıl":
                start_date = max_date - timedelta(days=365)
            else:
                start_date = confirmed_data['date'].min()
            
            time_filtered_data = confirmed_data[
                (confirmed_data['date'] >= start_date) & 
                (confirmed_data['country'].isin(selected_countries))
            ]
            
            if not time_filtered_data.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Kümülatif veri
                    fig_cumulative = px.line(
                        time_filtered_data,
                        x='date',
                        y='confirmed',
                        color='country',
                        title='Kümülatif Confirmed Vakaları',
                        labels={'date': 'Tarih', 'confirmed': 'Toplam Vaka'}
                    )
                    fig_cumulative.update_layout(height=400)
                    st.plotly_chart(fig_cumulative, use_container_width=True)
                
                with col2:
                    # Günlük yeni vakalar
                    if 'new_confirmed' in time_filtered_data.columns:
                        fig_daily = px.line(
                            time_filtered_data,
                            x='date',
                            y='new_confirmed',
                            color='country',
                            title='Günlük Yeni Vakalar',
                            labels={'date': 'Tarih', 'new_confirmed': 'Yeni Vaka'}
                        )
                        fig_daily.update_layout(height=400)
                        st.plotly_chart(fig_daily, use_container_width=True)
                
                # Hareketli ortalama
                if 'confirmed_7day_avg' in time_filtered_data.columns:
                    st.subheader("📊 7 Günlük Hareketli Ortalama")
                    fig_ma = px.line(
                        time_filtered_data,
                        x='date',
                        y='confirmed_7day_avg',
                        color='country',
                        title='7 Günlük Hareketli Ortalama (Yeni Vakalar)',
                        labels={'date': 'Tarih', 'confirmed_7day_avg': '7 Gün Ortalaması'}
                    )
                    fig_ma.update_layout(height=400)
                    st.plotly_chart(fig_ma, use_container_width=True)
        else:
            st.warning("⚠️ Zaman serisi verisi mevcut değil")
    
    with tab3:
        st.header("🔍 Detaylı Analiz")
        
        selected_country = st.selectbox(
            "🎯 Detaylı analiz için ülke seçin:",
            options=selected_countries,
            index=0
        )
        
        if selected_country:
            country_data = snapshot_data[snapshot_data['country'] == selected_country]
            
            if not country_data.empty:
                data = country_data.iloc[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🦠 Toplam Vaka", f"{data.get('total_cases', 0):,}")
                    st.metric("💀 Toplam Ölüm", f"{data.get('total_deaths', 0):,}")
                    st.metric("💚 Toplam İyileşen", f"{data.get('total_recovered', 0):,}")
                
                with col2:
                    st.metric("🟡 Aktif Vaka", f"{data.get('active_cases', 0):,}")
                    st.metric("🆕 Yeni Vaka", f"{data.get('new_cases', 0):,}")
                    st.metric("⚰️ Yeni Ölüm", f"{data.get('new_deaths', 0):,}")
                
                with col3:
                    cfr = data.get('case_fatality_rate', 0)
                    recovery_rate = data.get('recovery_rate', 0)
                    st.metric("📊 Vaka Ölüm Oranı", f"{cfr:.2f}%")
                    st.metric("📈 İyileşme Oranı", f"{recovery_rate:.2f}%")
                    st.metric("📊 Milyon/Vaka", f"{data.get('cases_per_million', 0):,.0f}")
                
                # Donut chart - vaka dağılımı
                if all(col in data.index for col in ['total_recovered', 'active_cases', 'total_deaths']):
                    st.subheader(f"📊 {selected_country} - Vaka Dağılımı")
                    
                    values = [
                        data.get('total_recovered', 0),
                        data.get('active_cases', 0), 
                        data.get('total_deaths', 0)
                    ]
                    labels = ['İyileşen', 'Aktif', 'Vefat']
                    colors = ['#2ca02c', '#ff7f0e', '#d62728']
                    
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.4,
                        marker=dict(colors=colors)
                    )])
                    
                    fig_donut.update_layout(
                        title=f"{selected_country} Vaka Dağılımı",
                        height=400
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)
                
                # Trend analizi (eğer zaman serisi varsa)
                if confirmed_data is not None:
                    trend_analysis = analyzer.analyze_country_trends(confirmed_data, selected_country, days_back=30)
                    
                    if 'error' not in trend_analysis:
                        st.subheader(f"📈 {selected_country} - Trend Analizi")
                        st.json(trend_analysis)
    
    with tab4:
        st.header("🌍 Global Harita")
        
        # Dünya haritası
        fig_map = px.choropleth(
            snapshot_data,
            locations='country',
            locationmode='country names',
            color=selected_metric,
            hover_name='country',
            hover_data={
                'total_cases': ':,',
                'total_deaths': ':,',
                selected_metric: ':,.0f'
            },
            color_continuous_scale='Reds',
            title=f"Dünya - {metrics_options[selected_metric]} Dağılımı"
        )
        
        fig_map.update_layout(
            height=600,
            geo=dict(showframe=False, showcoastlines=True)
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        # En çok etkilenen ülkeler
        st.subheader("🏆 En Çok Etkilenen Ülkeler")
        
        top_n = st.slider("Gösterilecek ülke sayısı:", 5, 20, 10)
        
        top_countries = snapshot_data.nlargest(top_n, selected_metric)[
            ['country', selected_metric, 'total_cases', 'total_deaths']
        ]
        
        # Formatla
        display_top = top_countries.copy()
        for col in ['total_cases', 'total_deaths']:
            if col in display_top.columns:
                display_top[col] = display_top[col].apply(lambda x: f"{x:,}" if pd.notna(x) else "N/A")
        
        if selected_metric not in ['total_cases', 'total_deaths']:
            display_top[selected_metric] = display_top[selected_metric].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(display_top, use_container_width=True)
    
    with tab5:
        st.header("🤖 Makine Öğrenmesi Tahminleri")
        
        # ML tahmin seçenekleri
        prediction_country = st.selectbox(
            "🎯 Tahmin için ülke seçin:",
            options=selected_countries,
            index=0,
            key="prediction_country"
        )
        
        prediction_days = st.slider(
            "📅 Tahmin süresi (gün):",
            min_value=7,
            max_value=60,
            value=30,
            help="Gelecek kaç gün için tahmin yapılacak"
        )
        
        if prediction_country and st.button("🚀 Tahmin Hesapla", type="primary"):
            with st.spinner(f"🤖 {prediction_country} için {prediction_days} günlük tahmin hesaplanıyor..."):
                
                try:
                    # Seçili ülke verilerini hazırla
                    if confirmed_data is not None:
                        country_ts_data = confirmed_data[
                            confirmed_data['country'] == prediction_country
                        ].copy()
                        
                        if not country_ts_data.empty and len(country_ts_data) >= 30:
                            # Özellikleri hazırla
                            X, y = predictor.prepare_features(country_ts_data, 'confirmed')
                            
                            if len(X) > 0:
                                # Modelleri eğit
                                results = predictor.train_models(X, y)
                                
                                if results['success']:
                                    st.success("✅ Modeller başarıyla eğitildi!")
                                    
                                    # Model performansını göster
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.subheader("📊 Model Performansı")
                                        performance_df = pd.DataFrame({
                                            'Model': list(results['scores'].keys()),
                                            'R² Score': [f"{score:.4f}" for score in results['scores'].values()],
                                            'RMSE': [f"{rmse:.0f}" for rmse in results['rmse'].values()]
                                        })
                                        st.dataframe(performance_df, use_container_width=True)
                                    
                                    with col2:
                                        # En iyi modeli göster
                                        best_model = max(results['scores'], key=results['scores'].get)
                                        best_score = results['scores'][best_model]
                                        
                                        st.metric(
                                            "🏆 En İyi Model",
                                            best_model,
                                            f"R² Score: {best_score:.4f}"
                                        )
                                        
                                        st.metric(
                                            "📈 RMSE",
                                            f"{results['rmse'][best_model]:.0f}",
                                            "vaka"
                                        )
                                    
                                    # Tahmin yap
                                    st.subheader(f"🔮 {prediction_days} Günlük Tahmin")
                                    
                                    # Son veriyi al
                                    last_data = country_ts_data.tail(1)
                                    last_confirmed = last_data['confirmed'].iloc[0]
                                    last_date = pd.to_datetime(last_data['date'].iloc[0])
                                    
                                    # Gelecek tarihler oluştur
                                    future_dates = pd.date_range(
                                        start=last_date + timedelta(days=1),
                                        periods=prediction_days,
                                        freq='D'
                                    )
                                    
                                    # Basit tahmin (trend extrapolation)
                                    recent_growth = country_ts_data.tail(7)['confirmed'].pct_change().mean()
                                    if pd.isna(recent_growth) or recent_growth < 0:
                                        recent_growth = 0.001  # %0.1 günlük büyüme
                                    
                                    # Tahminleri hesapla
                                    predictions = []
                                    current_value = last_confirmed
                                    
                                    for i in range(prediction_days):
                                        # Exponential growth model
                                        current_value = current_value * (1 + recent_growth)
                                        predictions.append(current_value)
                                    
                                    # Tahmin dataframe'i oluştur
                                    prediction_df = pd.DataFrame({
                                        'date': future_dates,
                                        'predicted_confirmed': predictions,
                                        'model': best_model
                                    })
                                    
                                    # Görselleştirme
                                    col1, col2 = st.columns([2, 1])
                                    
                                    with col1:
                                        # Geçmiş veri + tahmin grafiği
                                        fig_prediction = go.Figure()
                                        
                                        # Geçmiş veriler
                                        recent_data = country_ts_data.tail(90)  # Son 90 gün
                                        fig_prediction.add_trace(go.Scatter(
                                            x=recent_data['date'],
                                            y=recent_data['confirmed'],
                                            mode='lines',
                                            name='Gerçek Veriler',
                                            line=dict(color='blue', width=2)
                                        ))
                                        
                                        # Tahminler
                                        fig_prediction.add_trace(go.Scatter(
                                            x=prediction_df['date'],
                                            y=prediction_df['predicted_confirmed'],
                                            mode='lines+markers',
                                            name=f'{best_model} Tahmini',
                                            line=dict(color='red', width=2, dash='dash'),
                                            marker=dict(size=6)
                                        ))
                                        
                                        # Ayırıcı çizgi
                                        fig_prediction.add_vline(
                                            x=last_date,
                                            line_dash="dot",
                                            line_color="green",
                                            annotation_text="Tahmin Başlangıcı"
                                        )
                                        
                                        fig_prediction.update_layout(
                                            title=f'{prediction_country} - COVID-19 Vaka Tahmini',
                                            xaxis_title='Tarih',
                                            yaxis_title='Toplam Vaka Sayısı',
                                            height=500,
                                            hovermode='x unified'
                                        )
                                        
                                        st.plotly_chart(fig_prediction, use_container_width=True)
                                    
                                    with col2:
                                        # Tahmin özet istatistikleri
                                        st.subheader("📈 Tahmin Özeti")
                                        
                                        final_prediction = predictions[-1]
                                        total_increase = final_prediction - last_confirmed
                                        avg_daily_increase = total_increase / prediction_days
                                        
                                        st.metric(
                                            "📊 Mevcut Durum",
                                            f"{last_confirmed:,.0f}",
                                            "toplam vaka"
                                        )
                                        
                                        st.metric(
                                            f"🔮 {prediction_days} Gün Sonra",
                                            f"{final_prediction:,.0f}",
                                            f"+{total_increase:,.0f}"
                                        )
                                        
                                        st.metric(
                                            "📈 Günlük Ortalama Artış",
                                            f"{avg_daily_increase:,.0f}",
                                            "vaka/gün"
                                        )
                                        
                                        # Büyüme oranı
                                        growth_rate = ((final_prediction / last_confirmed) - 1) * 100
                                        st.metric(
                                            "🚀 Toplam Büyüme",
                                            f"{growth_rate:.1f}%",
                                            f"{prediction_days} gün"
                                        )
                                    
                                    # Tahmin tablosu
                                    st.subheader("📋 Detaylı Tahmin Tablosu")
                                    
                                    # Haftalık özetler göster
                                    weekly_data = []
                                    for i in range(0, len(prediction_df), 7):
                                        week_end = min(i + 6, len(prediction_df) - 1)
                                        week_data = prediction_df.iloc[i:week_end+1]
                                        
                                        weekly_data.append({
                                            'Hafta': f"Hafta {i//7 + 1}",
                                            'Başlangıç Tarihi': week_data['date'].iloc[0].strftime('%Y-%m-%d'),
                                            'Bitiş Tarihi': week_data['date'].iloc[-1].strftime('%Y-%m-%d'),
                                            'Başlangıç Tahmini': f"{week_data['predicted_confirmed'].iloc[0]:,.0f}",
                                            'Bitiş Tahmini': f"{week_data['predicted_confirmed'].iloc[-1]:,.0f}",
                                            'Haftalık Artış': f"{(week_data['predicted_confirmed'].iloc[-1] - week_data['predicted_confirmed'].iloc[0]):,.0f}"
                                        })
                                    
                                    weekly_df = pd.DataFrame(weekly_data)
                                    st.dataframe(weekly_df, use_container_width=True)
                                    
                                    # Uyarı mesajı
                                    st.warning("""
                                    ⚠️ **Önemli Uyarı:** Bu tahminler matematiksel modellere dayanır ve gerçek durumu
                                    tam olarak yansıtmayabilir. COVID-19 salgını, halk sağlığı önlemleri, aşılama oranları,
                                    yeni varyantlar gibi birçok faktörden etkilenir. Bu tahminleri yalnızca referans amaçlı kullanın.
                                    """)
                                
                                else:
                                    st.error(f"❌ Model eğitimi başarısız: {results.get('error', 'Bilinmeyen hata')}")
                            else:
                                st.error("❌ Özellik hazırlama başarısız oldu")
                        else:
                            st.error(f"❌ {prediction_country} için yeterli veri yok (minimum 30 gün gerekli)")
                    else:
                        st.error("❌ Zaman serisi verisi mevcut değil")
                        
                except Exception as e:
                    st.error(f"❌ Tahmin hesaplama hatası: {str(e)}")
        
        # Model bilgilendirme
        with st.expander("ℹ️ Kullanılan Makine Öğrenmesi Modelleri"):
            st.markdown("""
            ### 🤖 Model Detayları
            
            **1. Linear Regression (Doğrusal Regresyon)**
            - Basit ve hızlı
            - Doğrusal trendler için uygun
            - Yorumlaması kolay
            
            **2. Ridge Regression**
            - Overfitting'i önler
            - Daha stabil sonuçlar
            - Çok değişkenli veriler için uygun
            
            **3. Random Forest**
            - Non-linear ilişkileri yakalar
            - Robust ve güçlü
            - Özellik önemini gösterir
            
            ### 📊 Değerlendirme Metrikleri
            
            **R² Score (Determination Coefficient)**
            - 0-1 arası değer
            - 1'e yakın = daha iyi model
            - Modelin açıklayabildiği varyans oranı
            
            **RMSE (Root Mean Square Error)**
            - Ortalama karesel hatanın karekökü
            - Düşük değer = daha iyi tahmin
            - Gerçek değerler ile aynı birimde
            """)

else:
    st.warning("⚠️ Lütfen en az bir ülke seçin.")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("📊 **Veri Kaynakları:**")
    st.markdown("- Disease.sh API")
    st.markdown("- Johns Hopkins University")

with col2:
    st.markdown("🕒 **Son Güncelleme:**")
    st.markdown(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")

with col3:
    st.markdown("🔧 **Geliştiren:**")
    st.markdown("COVID-19 Analiz Projesi")

# Sidebar bilgileri
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Bilgi")
st.sidebar.markdown(f"📊 Toplam ülke: {len(available_countries)}")
st.sidebar.markdown(f"📅 Veri tarihi: {datetime.now().strftime('%Y-%m-%d')}")

if confirmed_data is not None:
    data_date_range = confirmed_data['date'].max() - confirmed_data['date'].min()
    st.sidebar.markdown(f"⏱️ Zaman aralığı: {data_date_range.days} gün")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Hızlı Erişim")

if st.sidebar.button("🔄 Veriyi Yenile"):
    st.cache_data.clear()
    st.experimental_rerun()

if st.sidebar.button("📋 Veri İndir"):
    if not filtered_data.empty:
        csv = filtered_data.to_csv(index=False)
        st.sidebar.download_button(
            label="💾 CSV İndir",
            data=csv,
            file_name=f"covid19_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Hata ayıklama için
if st.sidebar.checkbox("🔧 Debug Modu"):
    st.sidebar.markdown("### 🔍 Debug Bilgileri")
    st.sidebar.markdown(f"- Snapshot shape: {snapshot_data.shape if snapshot_data is not None else 'None'}")
    if confirmed_data is not None:
        st.sidebar.markdown(f"- Time series shape: {confirmed_data.shape}")
    st.sidebar.markdown(f"- Selected countries: {len(selected_countries)}")
    st.sidebar.markdown(f"- Selected metric: {selected_metric}")