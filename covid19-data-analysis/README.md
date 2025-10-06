# COVID-19 Veri Analizi ve Görselleştirme Platformu

🦠 **COVID-19 Data Analysis & Visualization Platform**

Dünya genelinde COVID-19 verilerini analiz eden, trendleri gösteren ve tahminler yapan kapsamlı bir veri analizi uygulaması.

## 🎯 Proje Amacı

Bu proje, COVID-19 pandemisine dair verileri toplamak, analiz etmek ve görselleştirmek için geliştirilmiştir. Kullanıcılar:
- Dünya genelindeki COVID-19 verilerini takip edebilir
- Ülkeler arası karşılaştırma yapabilir
- Zaman serisi analizleri görüntüleyebilir
- Gelecek trendleri hakkında tahminler alabilir

## 📊 Özellikler

- 🌍 **Global Analiz**: Dünya genelinde COVID-19 istatistikleri
- 📈 **Trend Analizi**: Zaman serisi bazlı trend görselleştirmeleri
- 🔍 **Karşılaştırma**: Ülkeler arası detaylı karşılaştırmalar
- 🤖 **Tahmin Modelleri**: Makine öğrenmesi ile gelecek tahminleri
- 📱 **İnteraktif Dashboard**: Streamlit tabanlı kullanıcı dostu arayüz

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

## 📞 İletişim

- **Proje Sahibi**: [Your Name]
- **Email**: [your-email@example.com]
- **GitHub**: [your-github-profile]

---

⭐ Bu projeyi beğendiyseniz star vermeyi unutmayın!