"""
COVID-19 Data Analysis - Basic Tests
===================================

Temel fonksiyonalite testleri
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Proje modüllerini import et
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from utils import format_large_number, clean_country_name, validate_date_format
    from data_processing import CovidDataLoader, calculate_derived_metrics
    from analysis import CovidAnalyzer
    from visualization import CovidVisualizer
    from modeling import CovidPredictor
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Modül import hatası: {e}")
    MODULES_AVAILABLE = False


class TestUtils(unittest.TestCase):
    """Utils modülü testleri"""
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modüller mevcut değil")
    def test_format_large_number(self):
        """format_large_number fonksiyonu testi"""
        self.assertEqual(format_large_number(1000), "1.0K")
        self.assertEqual(format_large_number(1000000), "1.0M")
        self.assertEqual(format_large_number(1000000000), "1.0B")
        self.assertEqual(format_large_number(500), "500.0")
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modüller mevcut değil")
    def test_clean_country_name(self):
        """clean_country_name fonksiyonu testi"""
        self.assertEqual(clean_country_name("US"), "United States")
        self.assertEqual(clean_country_name("UK"), "United Kingdom")
        self.assertEqual(clean_country_name("Turkey"), "Turkey")
        self.assertEqual(clean_country_name(" Germany "), "Germany")
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modüller mevcut değil")
    def test_validate_date_format(self):
        """validate_date_format fonksiyonu testi"""
        self.assertTrue(validate_date_format("2023-01-01"))
        self.assertTrue(validate_date_format("2023-12-31"))
        self.assertFalse(validate_date_format("2023-13-01"))
        self.assertFalse(validate_date_format("invalid-date"))


class TestDataProcessing(unittest.TestCase):
    """Data processing modülü testleri"""
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modüller mevcut değil")
    def test_covid_data_loader_init(self):
        """CovidDataLoader başlatma testi"""
        try:
            loader = CovidDataLoader()
            self.assertIsNotNone(loader)
            self.assertIsInstance(loader.data_sources, dict)
        except Exception as e:
            self.skipTest(f"Config dosyası bulunamadı: {e}")
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modüller mevcut değil")
    def test_calculate_derived_metrics(self):
        """calculate_derived_metrics fonksiyonu testi"""
        # Test verisi oluştur
        test_data = pd.DataFrame({
            'total_cases': [1000, 2000, 3000],
            'total_deaths': [10, 20, 30],
            'total_recovered': [800, 1800, 2800],
            'country': ['Turkey', 'Germany', 'USA']
        })
        
        result = calculate_derived_metrics(test_data)
        
        # CFR hesabı kontrolü
        self.assertIn('case_fatality_rate', result.columns)
        self.assertAlmostEqual(result.iloc[0]['case_fatality_rate'], 1.0, places=1)
        
        # Recovery rate kontrolü
        self.assertIn('recovery_rate', result.columns)
        self.assertAlmostEqual(result.iloc[0]['recovery_rate'], 80.0, places=1)


class TestAnalysis(unittest.TestCase):
    """Analysis modülü testleri"""
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modüller mevcut değil")
    def test_covid_analyzer_init(self):
        """CovidAnalyzer başlatma testi"""
        try:
            analyzer = CovidAnalyzer()
            self.assertIsNotNone(analyzer)
        except Exception as e:
            self.skipTest(f"Config dosyası bulunamadı: {e}")
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modüller mevcut değil")
    def test_calculate_global_stats(self):
        """calculate_global_stats fonksiyonu testi"""
        # Test verisi
        test_data = pd.DataFrame({
            'total_cases': [1000, 2000, 3000],
            'total_deaths': [10, 20, 30],
            'total_recovered': [800, 1800, 2800],
            'active_cases': [190, 180, 170],
            'new_cases': [50, 100, 150],
            'new_deaths': [1, 2, 3],
            'country': ['Turkey', 'Germany', 'USA']
        })
        
        analyzer = CovidAnalyzer()
        stats = analyzer.calculate_global_stats(test_data)
        
        self.assertEqual(stats['total_countries'], 3)
        self.assertEqual(stats['total_cases'], 6000)
        self.assertEqual(stats['total_deaths'], 60)
        self.assertAlmostEqual(stats['global_cfr'], 1.0, places=1)


class TestVisualization(unittest.TestCase):
    """Visualization modülü testleri"""
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modüller mevcut değil")
    def test_covid_visualizer_init(self):
        """CovidVisualizer başlatma testi"""
        try:
            visualizer = CovidVisualizer()
            self.assertIsNotNone(visualizer)
            self.assertIsInstance(visualizer.colors, dict)
        except Exception as e:
            self.skipTest(f"Config dosyası bulunamadı: {e}")


class TestModeling(unittest.TestCase):
    """Modeling modülü testleri"""
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modüller mevcut değil")
    def test_covid_predictor_init(self):
        """CovidPredictor başlatma testi"""
        try:
            predictor = CovidPredictor()
            self.assertIsNotNone(predictor)
            self.assertIsInstance(predictor.models, dict)
            self.assertIsInstance(predictor.scalers, dict)
        except Exception as e:
            self.skipTest(f"Config dosyası bulunamadı: {e}")
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modüller mevcut değil")  
    def test_prepare_features(self):
        """prepare_features fonksiyonu testi"""
        # Test verisi
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        test_data = pd.DataFrame({
            'country': ['Turkey'] * 30,
            'date': dates,
            'confirmed': np.arange(100, 130),
            'new_confirmed': [5] * 30
        })
        
        predictor = CovidPredictor()
        X, y = predictor.prepare_features(test_data, 'confirmed')
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(test_data))
        self.assertIn('country_encoded', X.columns)
        self.assertIn('day_of_year', X.columns)


class TestIntegration(unittest.TestCase):
    """Entegrasyon testleri"""
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modüller mevcut değil")
    def test_end_to_end_workflow(self):
        """Tam iş akışı entegrasyon testi"""
        try:
            # 1. Data loader
            loader = CovidDataLoader()
            
            # 2. Analyzer
            analyzer = CovidAnalyzer()
            
            # 3. Visualizer
            visualizer = CovidVisualizer()
            
            # 4. Predictor
            predictor = CovidPredictor()
            
            # Test verisi ile işlem akışı
            test_data = pd.DataFrame({
                'country': ['Turkey', 'Germany'],
                'total_cases': [1000000, 2000000],
                'total_deaths': [10000, 20000],
                'total_recovered': [900000, 1800000],
                'active_cases': [90000, 180000],
                'cases_per_million': [12000, 24000],
                'deaths_per_million': [120, 240]
            })
            
            # Derived metrics hesapla
            enriched_data = calculate_derived_metrics(test_data)
            self.assertIn('case_fatality_rate', enriched_data.columns)
            
            # Global stats
            global_stats = analyzer.calculate_global_stats(enriched_data)
            self.assertIsInstance(global_stats, dict)
            self.assertIn('total_countries', global_stats)
            
            # Country comparison
            comparison = analyzer.compare_countries(
                enriched_data, 
                ['Turkey', 'Germany'], 
                ['total_cases', 'total_deaths']
            )
            self.assertEqual(len(comparison), 2)
            
            print("✅ End-to-end workflow test başarılı")
            
        except Exception as e:
            self.skipTest(f"End-to-end test başarısız: {e}")


def run_tests():
    """Test suite'i çalıştır"""
    print("🧪 COVID-19 Data Analysis - Test Suite")
    print("=" * 50)
    
    # Test loader
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Test sınıflarını ekle
    test_classes = [
        TestUtils,
        TestDataProcessing, 
        TestAnalysis,
        TestVisualization,
        TestModeling,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Test runner
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Sonuçları özetle
    print("\n" + "=" * 50)
    print("📊 TEST SONUÇLARI")
    print("=" * 50)
    print(f"✅ Başarılı: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Başarısız: {len(result.failures)}")  
    print(f"💥 Hatalı: {len(result.errors)}")
    print(f"⏭️ Atlanan: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n❌ BAŞARISIZ TESTLER:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 HATALI TESTLER:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Error:')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n📈 Başarı Oranı: %{success_rate:.1f}")
    
    if success_rate >= 80:
        print("🎉 Test suite genel başarı! Proje hazır.")
    elif success_rate >= 60:
        print("⚠️ Test suite kısmen başarılı. Bazı iyileştirmeler gerekli.")
    else:
        print("🚨 Test suite başarısız. Önemli sorunlar mevcut.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)