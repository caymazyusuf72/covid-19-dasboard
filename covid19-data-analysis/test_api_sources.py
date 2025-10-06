"""
COVID-19 Veri Kaynakları Test Scripti
====================================

Bu script, COVID-19 projesi için kullanılacak veri kaynaklarını test eder.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import load_config, safe_request, setup_logging
import pandas as pd
from datetime import datetime


def test_disease_sh_api():
    """Disease.sh API'sini test eder."""
    print("\n🔗 Disease.sh API testi...")
    
    try:
        # Global özet
        response = safe_request("https://disease.sh/v3/covid-19/all")
        if response and response.status_code == 200:
            data = response.json()
            print(f"✅ Global veri başarıyla alındı")
            print(f"   Toplam vaka: {data.get('cases', 'N/A'):,}")
            print(f"   Toplam ölüm: {data.get('deaths', 'N/A'):,}")
            print(f"   Toplam iyileşen: {data.get('recovered', 'N/A'):,}")
            
            # Ülkeler verisi
            countries_response = safe_request("https://disease.sh/v3/covid-19/countries")
            if countries_response and countries_response.status_code == 200:
                countries_data = countries_response.json()
                print(f"✅ Ülkeler verisi alındı ({len(countries_data)} ülke)")
                
                # Türkiye verisi kontrolü
                turkey_data = next((c for c in countries_data if c['country'] == 'Turkey'), None)
                if turkey_data:
                    print(f"   🇹🇷 Türkiye - Vaka: {turkey_data.get('cases', 'N/A'):,}")
                
                return True
            else:
                print("❌ Ülkeler verisi alınamadı")
                return False
        else:
            print("❌ Global veri alınamadı")
            return False
            
    except Exception as e:
        print(f"❌ Disease.sh API hatası: {e}")
        return False


def test_covid19_api():
    """COVID19API.com'u test eder."""
    print("\n🔗 COVID19API.com testi...")
    
    try:
        response = safe_request("https://api.covid19api.com/summary")
        if response and response.status_code == 200:
            data = response.json()
            global_data = data.get('Global', {})
            print(f"✅ COVID19API verisi başarıyla alındı")
            print(f"   Yeni vaka: {global_data.get('NewConfirmed', 'N/A'):,}")
            print(f"   Yeni ölüm: {global_data.get('NewDeaths', 'N/A'):,}")
            
            countries = data.get('Countries', [])
            print(f"✅ {len(countries)} ülke verisi mevcut")
            
            # Türkiye verisi kontrolü
            turkey_data = next((c for c in countries if c.get('Country') == 'Turkey'), None)
            if turkey_data:
                print(f"   🇹🇷 Türkiye - Toplam Vaka: {turkey_data.get('TotalConfirmed', 'N/A'):,}")
            
            return True
        else:
            print("❌ COVID19API verisi alınamadı")
            return False
            
    except Exception as e:
        print(f"❌ COVID19API hatası: {e}")
        return False


def test_owid_data():
    """Our World in Data CSV'sini test eder."""
    print("\n🔗 Our World in Data testi...")
    
    try:
        # Sadece ilk birkaç satırı okuyarak test edelim
        url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
        response = safe_request(url)
        
        if response and response.status_code == 200:
            # CSV'yi pandas ile okumayı dene (sadece ilk 100 satır)
            from io import StringIO
            csv_content = StringIO(response.text)
            df = pd.read_csv(csv_content, nrows=100)
            
            print(f"✅ OWID CSV verisi başarıyla alındı")
            print(f"   Sütunlar: {len(df.columns)}")
            print(f"   İlk 100 satır: {len(df)} kayıt")
            print(f"   Sütun örnekleri: {', '.join(df.columns[:5])}...")
            
            # Türkiye verisi var mı kontrol et
            turkey_rows = df[df['location'] == 'Turkey']
            if not turkey_rows.empty:
                print(f"   🇹🇷 Türkiye verisi bulundu ({len(turkey_rows)} kayıt)")
            
            return True
        else:
            print("❌ OWID CSV verisi alınamadı")
            return False
            
    except Exception as e:
        print(f"❌ OWID veri hatası: {e}")
        return False


def test_johns_hopkins_data():
    """Johns Hopkins verilerini test eder."""
    print("\n🔗 Johns Hopkins GitHub verisi testi...")
    
    base_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series"
    files = [
        "time_series_covid19_confirmed_global.csv",
        "time_series_covid19_deaths_global.csv", 
        "time_series_covid19_recovered_global.csv"
    ]
    
    success_count = 0
    
    for file in files:
        try:
            url = f"{base_url}/{file}"
            response = safe_request(url, timeout=15)
            
            if response and response.status_code == 200:
                # İlk birkaç satırı oku
                from io import StringIO
                csv_content = StringIO(response.text)
                df = pd.read_csv(csv_content, nrows=50)
                
                print(f"✅ {file} - {len(df)} kayıt, {len(df.columns)} sütun")
                
                # Türkiye verisi kontrolü
                turkey_rows = df[df['Country/Region'] == 'Turkey']
                if not turkey_rows.empty:
                    print(f"   🇹🇷 Türkiye verisi mevcut")
                
                success_count += 1
            else:
                print(f"❌ {file} alınamadı")
                
        except Exception as e:
            print(f"❌ {file} hatası: {e}")
    
    return success_count == len(files)


def main():
    """Ana test fonksiyonu."""
    print("🦠 COVID-19 Veri Kaynakları Test Raporu")
    print("=" * 50)
    print(f"📅 Test Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test sonuçları
    results = {
        "Disease.sh API": test_disease_sh_api(),
        "COVID19API.com": test_covid19_api(),
        "Our World in Data": test_owid_data(),
        "Johns Hopkins": test_johns_hopkins_data()
    }
    
    print("\n" + "=" * 50)
    print("📊 TEST SONUÇLARI:")
    print("=" * 50)
    
    success_count = 0
    for source, success in results.items():
        status = "✅ BAŞARILI" if success else "❌ BAŞARISIZ"
        print(f"{source:<20} : {status}")
        if success:
            success_count += 1
    
    print(f"\n📈 Genel Başarı Oranı: {success_count}/{len(results)} (%{int(success_count/len(results)*100)})")
    
    if success_count >= 3:
        print("🎉 Çoğu veri kaynağı çalışıyor! Projeye devam edebiliriz.")
    elif success_count >= 2:
        print("⚠️  Bazı veri kaynakları çalışıyor. Dikkatli devam edelim.")
    else:
        print("🚨 Çoğu veri kaynağı çalışmıyor. İnternet bağlantısını kontrol edin.")
    
    return success_count >= 2


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚡ Test kullanıcı tarafından durduruldu.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Beklenmeyen hata: {e}")
        sys.exit(1)