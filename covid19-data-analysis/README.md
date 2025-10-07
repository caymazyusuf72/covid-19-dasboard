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

## 🚀 Kurulum

### Gereksinimler
- Python 3.8+
- pip

### Kurulum Adımları

```bash
# Repoyu klonlayın
git clone [repository-url]
cd covid19-data-analysis

# Sanal ortam oluşturun
python -m venv covid_env
source covid_env/bin/activate  # Windows: covid_env\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

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

## 🛠️ Kullanım

### Dashboard Çalıştırma
```bash
streamlit run app.py
```

### Jupyter Notebook'ları
```bash
jupyter notebook notebooks/
```

## 📊 Veri Kaynakları

- [John Hopkins University COVID-19 Data](https://github.com/CSSEGISandData/COVID-19)
- [WHO COVID-19 Dashboard](https://covid19.who.int/)
- [Our World in Data](https://ourworldindata.org/coronavirus)
- [Disease.sh API](https://disease.sh/)

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.



---

⭐ Bu projeyi beğendiyseniz star vermeyi unutmayın!