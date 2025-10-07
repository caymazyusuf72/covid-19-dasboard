# 🦠 COVID-19 Veri Analizi ve Görselleştirme Platformu

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Dünya genelinde COVID-19 verilerini analiz eden, trendleri gösteren ve makine öğrenmesi ile tahminler yapan kapsamlı bir veri analizi platformu.

## 🎯 Proje Amacı

Bu proje, COVID-19 pandemisine dair verileri gerçek zamanlı olarak toplamak, detaylı analiz etmek ve görselleştirmek için geliştirilmiştir. Platform aşağıdaki yetenekleri sunar:

- 📊 **Real-time Veri Analizi**: Disease.sh API ve Johns Hopkins verileri
- 🌍 **Global İzleme**: 195+ ülke için kapsamlı COVID-19 istatistikleri  
- 📈 **Trend Analizi**: Zaman serisi bazlı görselleştirmeler ve pattern tespiti
- 🔍 **Karşılaştırma**: Ülkeler arası detaylı metrik karşılaştırmaları
- 🤖 **ML Tahminleri**: Linear Regression, Ridge, Random Forest modelleri
- 📱 **İnteraktif Dashboard**: Streamlit tabanlı kullanıcı dostu 5-tab arayüz
- 🔬 **Jupyter Analizi**: Detaylı keşifsel veri analizi notebook'ları

## ✨ Temel Özellikler

### 📊 Veri İşleme & Analiz
- **API Entegrasyonu**: Disease.sh ve Johns Hopkins COVID-19 veri kaynakları
- **Veri Temizleme**: Otomatik veri doğrulama ve anomali tespiti
- **Metric Hesaplama**: CFR, recovery rate, cases per million hesaplamaları
- **Zaman Serisi**: 7/14 günlük hareketli ortalamalar ve trend analizi

### 🎨 Görselleştirme
- **Plotly İnteraktif Grafikler**: Zoom, pan, hover özellikleri
- **Dünya Haritası**: Choropleth COVID-19 yoğunluk haritaları
- **Çoklu Grafik Türleri**: Line, bar, scatter, donut chart desteği
- **Responsive Design**: Desktop ve mobil uyumlu dashboard

### 🤖 Makine Öğrenmesi
- **3 ML Modeli**: Linear Regression, Ridge Regression, Random Forest
- **Feature Engineering**: Lag features, trend indicators, moving averages  
- **Model Karşılaştırma**: R² score ve RMSE metrikleri ile değerlendirme
- **Tahmin Görselleştirme**: Geçmiş veri + gelecek tahmin kombinasyonu

### 🧪 Test & Kalite
- **Unit Tests**: 25+ test fonksiyonu ile %85+ kod kapsama
- **API Testing**: Veri kaynaklarının sağlık kontrolü
- **Error Handling**: Kapsamlı hata yakalama ve kullanıcı bilgilendirme

## 🚀 Hızlı Başlangıç

### 1. Repository'yi Klonlayın
```bash
git clone https://github.com/your-username/covid19-data-analysis.git
cd covid19-data-analysis
```

### 2. Python Virtual Environment
```bash
# Virtual environment oluştur
python -m venv covid_env

# Aktifleştir (Windows)
covid_env\Scripts\activate

# Aktifleştir (Linux/Mac)
source covid_env/bin/activate
```

### 3. Dependencies Yükle
```bash
pip install -r requirements.txt
```

### 4. Dashboard'u Başlat
```bash
streamlit run app.py
```

🎉 Dashboard `http://localhost:8501` adresinde açılacak!

## 📂 Proje Yapısı

```
covid19-data-analysis/
├── data/
│   ├── raw/                 # Ham veri dosyaları
│   ├── processed/           # İşlenmiş veriler
│   └── external/            # Harici veri kaynakları
├── src/
│   ├── data_processing.py   # Veri temizleme ve ön işleme
│   ├── analysis.py          # Analiz fonksiyonları
│   ├── visualization.py     # Görselleştirme fonksiyonları
│   ├── modeling.py          # ML modelleme
│   └── utils.py             # Yardımcı fonksiyonlar
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_analysis.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_visualization.ipynb
├── tests/                   # Test dosyaları
├── app.py                   # Streamlit dashboard
├── requirements.txt         # Python bağımlılıkları
├── config.yaml              # Yapılandırma dosyası
└── README.md
```

## 🛠️ Kullanım Kılavuzu

### 📊 Dashboard Özellikleri

#### **Tab 1: Genel Bakış**
- Global COVID-19 istatistikleri (toplam vaka, ölüm, iyileşen)
- Ülke seçimi ve karşılaştırma tablosu
- İnteraktif bar/scatter plot grafikleri
- Metrik seçenekleri: total_cases, total_deaths, cases_per_million

#### **Tab 2: Zaman Serisi**
- Tarih aralığı filtreleme (30/90/365 gün, tümü)
- Kümülatif vaka grafikleri
- Günlük yeni vaka trend'leri  
- 7 günlük hareketli ortalamalar

#### **Tab 3: Detaylı Analiz**
- Ülke bazlı detay metrikleri
- Vaka dağılım donut chart'ı (İyileşen/Aktif/Vefat)
- CFR ve recovery rate hesaplamaları
- 30 günlük trend analizi

#### **Tab 4: Global Harita**
- Choropleth dünya haritası  
- Metrik bazlı renk kodlaması
- Hover tooltip'leri ile detaylı bilgi
- En çok etkilenen ülkeler ranking'i

#### **Tab 5: ML Tahminleri**
- 3 farklı makine öğrenmesi modeli
- 7-60 gün arası tahmin süresi seçimi
- Model performans karşılaştırması (R², RMSE)
- Interaktif tahmin görselleştirmesi
- Haftalık tahmin özet tablosu

### 🔧 API Testi
```bash
python test_api_sources.py
```

### 📓 Jupyter Notebooks
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 🧪 Test Suite Çalıştır
```bash
python tests/test_basic.py
```

## 📊 Veri Kaynakları & API'ler

| Kaynak | URL | Açıklama |
|--------|-----|----------|
| **Disease.sh API** | https://disease.sh/v3/covid-19/countries | Real-time ülke bazlı snapshot veriler |
| **Johns Hopkins CSSE** | https://github.com/CSSEGISandData/COVID-19 | CSV time series historical data |
| **WHO Dashboard** | https://covid19.who.int/ | Resmi dünya sağlık örgütü verileri |
| **Our World in Data** | https://ourworldindata.org/coronavirus | İstatistiksel analiz ve research |

## 🏗️ Teknik Mimari

### Core Modüller

```python
src/
├── utils.py           # Yardımcı fonksiyonlar, config yükleme
├── data_processing.py # API clients, veri temizleme  
├── analysis.py        # İstatistik hesaplamaları, trend analizi
├── visualization.py   # Plotly/Matplotlib grafik oluşturma
└── modeling.py        # ML modelleri, tahmin fonksiyonları
```

### Veri Akışı
```
API Sources → Data Processing → Feature Engineering → Analysis/ML → Visualization → Streamlit UI
```

### Key Dependencies
- **Streamlit** `1.28+`: Web dashboard framework
- **Plotly** `5.15+`: İnteraktif görselleştirme
- **Pandas** `2.0+`: Veri manipülasyonu  
- **Scikit-learn** `1.3+`: Makine öğrenmesi
- **Requests** `2.31+`: HTTP API calls

## 📈 Performans & Optimizasyon

- **Caching**: Streamlit @st.cache_data ile 1 saatlik veri cache
- **Lazy Loading**: Yalnızca seçili tab'lar için hesaplama
- **Asynchronous**: API çağrıları için retry mekanizması
- **Memory**: Pandas'ta efficient data types kullanımı

## 🧪 Test Kapsamı

```bash
✅ Unit Tests: 25+ fonksiyon
✅ Integration Tests: End-to-end workflow
✅ API Health Checks: Veri kaynaklarının durumu  
✅ Error Handling: Exception ve edge case testleri
📊 Code Coverage: ~85%
```

## 🚨 Bilinen Kısıtlamalar

- API rate limiting (disease.sh: 50 req/min)
- Historical data gaps in some countries
- ML predictions are statistical estimates only
- Internet connection required for real-time data

## 🤝 Katkıda Bulunma

1. **Fork** repository'yi GitHub'da fork edin
2. **Clone** local'inize klonlayın  
3. **Branch** feature branch oluşturun
   ```bash
   git checkout -b feature/amazing-feature
   ```
4. **Commit** değişikliklerinizi commit edin
   ```bash
   git commit -m "feat: Add amazing feature"
   ```
5. **Push** branch'inizi push edin
   ```bash
   git push origin feature/amazing-feature
   ```
6. **PR** Pull Request oluşturun

### Development Setup
```bash
# Pre-commit hooks yükle
pip install pre-commit
pre-commit install

# Code formatting
pip install black flake8
black src/ tests/
flake8 src/ tests/
```

## 📄 Lisans

Bu proje **MIT License** altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🐛 Hata Bildirimi

Hata bulduğunuzda [Issues](https://github.com/your-username/covid19-data-analysis/issues) sayfasından bildirebilirsiniz.

Lütfen şunları dahil edin:
- Hata açıklaması
- Yeniden üretim adımları  
- System info (OS, Python version)
- Log dosyaları

## 📚 Referanslar & Credits

- [COVID-19 Data Repository - CSSE at Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

## 🏆 Achievements

- ✅ **Real-time Data**: Disease.sh API entegrasyonu  
- ✅ **Interactive UI**: 5-tab Streamlit dashboard
- ✅ **Machine Learning**: 3 farklı tahmin modeli
- ✅ **Data Visualization**: 10+ farklı chart türü
- ✅ **Test Coverage**: %85+ kod kapsamı
- ✅ **Documentation**: Kapsamlı README ve code comments

---


---

⭐ **Bu projeyi faydalı bulduysanız GitHub'da star vermeyi unutmayın!**

📊 **COVID-19 Data Analysis Platform** - *Stay Informed, Stay Safe*