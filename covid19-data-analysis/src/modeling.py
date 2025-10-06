"""
COVID-19 Machine Learning Modeling Module
=========================================

Bu modül COVID-19 verilerini kullanarak basit makine öğrenmesi modelleri oluşturur.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import os
import sys

# Kendi modülümüzü import edelim
sys.path.append(os.path.dirname(__file__))
from utils import load_config, format_large_number

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class CovidPredictor:
    """COVID-19 tahmin modelleri için ana sınıf."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        CovidPredictor'ı başlatır.
        
        Args:
            config_path (str): Yapılandırma dosyasının yolu
        """
        try:
            self.config = load_config(config_path)
            self.ml_config = self.config.get('ml', {})
        except:
            self.ml_config = {
                'test_size': 0.2,
                'random_state': 42,
                'cross_validation_folds': 5
            }
        
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        
        logger.info("CovidPredictor başlatıldı")
    
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'confirmed') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Makine öğrenmesi için özellik mühendisliği yapar.
        
        Args:
            df (pd.DataFrame): Zaman serisi verisi
            target_column (str): Tahmin edilecek hedef sütun
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Özellikler ve hedef değişken
        """
        df = df.copy().sort_values(['country', 'date'])
        
        # Temel özellikler
        features_df = pd.DataFrame()
        
        # Ülke encoding
        le = LabelEncoder()
        features_df['country_encoded'] = le.fit_transform(df['country'])
        self.country_encoder = le
        
        # Zaman bazlı özellikler
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            features_df['day_of_year'] = df['date'].dt.dayofyear
            features_df['month'] = df['date'].dt.month
            features_df['week'] = df['date'].dt.isocalendar().week
            features_df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        
        # Lag features (gecikmeli özellikler)
        for country in df['country'].unique():
            country_mask = df['country'] == country
            country_data = df[country_mask].copy()
            
            if len(country_data) > 7:
                # 7 gün önceki değer
                lag_7 = country_data[target_column].shift(7)
                features_df.loc[country_mask, 'lag_7'] = lag_7
                
                # 14 gün önceki değer
                if len(country_data) > 14:
                    lag_14 = country_data[target_column].shift(14)
                    features_df.loc[country_mask, 'lag_14'] = lag_14
                
                # Hareketli ortalamalar
                ma_7 = country_data[target_column].rolling(window=7, min_periods=1).mean()
                features_df.loc[country_mask, 'ma_7'] = ma_7
                
                ma_14 = country_data[target_column].rolling(window=14, min_periods=1).mean()
                features_df.loc[country_mask, 'ma_14'] = ma_14
                
                # Trend (7 günlük değişim)
                trend = country_data[target_column].diff(7)
                features_df.loc[country_mask, 'trend_7'] = trend
        
        # Günlük yeni vakalar (eğer varsa)
        if f'new_{target_column}' in df.columns:
            features_df['new_cases'] = df[f'new_{target_column}']
        
        # Eksik değerleri doldur
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        # Hedef değişken
        target = df[target_column]
        
        return features_df, target
    
    
    def train_linear_regression(self, X: pd.DataFrame, y: pd.Series, 
                               test_size: Optional[float] = None) -> Dict[str, Union[float, object]]:
        """
        Linear Regression modeli eğitir.
        
        Args:
            X (pd.DataFrame): Özellikler
            y (pd.Series): Hedef değişken
            test_size (float, optional): Test seti oranı
            
        Returns:
            Dict: Model ve performans metrikleri
        """
        test_size = test_size or self.ml_config.get('test_size', 0.2)
        random_state = self.ml_config.get('random_state', 42)
        
        # Veri bölme
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Özellik ölçeklendirme
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model eğitimi
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Tahminler
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Performans metrikleri
        performance = {
            'model': model,
            'scaler': scaler,
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'feature_importance': dict(zip(X.columns, np.abs(model.coef_))),
            'X_test': X_test,
            'y_test': y_test,
            'y_test_pred': y_test_pred
        }
        
        # Cross validation
        cv_folds = self.ml_config.get('cross_validation_folds', 5)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')
        performance['cv_r2_mean'] = cv_scores.mean()
        performance['cv_r2_std'] = cv_scores.std()
        
        # Model kaydet
        self.models['linear_regression'] = model
        self.scalers['linear_regression'] = scaler
        self.model_performance['linear_regression'] = performance
        
        logger.info(f"Linear Regression eğitildi - Test R²: {performance['test_r2']:.3f}")
        
        return performance
    
    
    def train_ridge_regression(self, X: pd.DataFrame, y: pd.Series, 
                              alpha: float = 1.0, test_size: Optional[float] = None) -> Dict[str, Union[float, object]]:
        """
        Ridge Regression modeli eğitir.
        
        Args:
            X (pd.DataFrame): Özellikler
            y (pd.Series): Hedef değişken
            alpha (float): Regularization parametresi
            test_size (float, optional): Test seti oranı
            
        Returns:
            Dict: Model ve performans metrikleri
        """
        test_size = test_size or self.ml_config.get('test_size', 0.2)
        random_state = self.ml_config.get('random_state', 42)
        
        # Veri bölme
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Özellik ölçeklendirme
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model eğitimi
        model = Ridge(alpha=alpha, random_state=random_state)
        model.fit(X_train_scaled, y_train)
        
        # Tahminler
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Performans metrikleri
        performance = {
            'model': model,
            'scaler': scaler,
            'alpha': alpha,
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'feature_importance': dict(zip(X.columns, np.abs(model.coef_))),
            'X_test': X_test,
            'y_test': y_test,
            'y_test_pred': y_test_pred
        }
        
        # Cross validation
        cv_folds = self.ml_config.get('cross_validation_folds', 5)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')
        performance['cv_r2_mean'] = cv_scores.mean()
        performance['cv_r2_std'] = cv_scores.std()
        
        # Model kaydet
        self.models['ridge_regression'] = model
        self.scalers['ridge_regression'] = scaler
        self.model_performance['ridge_regression'] = performance
        
        logger.info(f"Ridge Regression eğitildi - Test R²: {performance['test_r2']:.3f}")
        
        return performance
    
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series, 
                           n_estimators: int = 100, test_size: Optional[float] = None) -> Dict[str, Union[float, object]]:
        """
        Random Forest modeli eğitir.
        
        Args:
            X (pd.DataFrame): Özellikler
            y (pd.Series): Hedef değişken
            n_estimators (int): Ağaç sayısı
            test_size (float, optional): Test seti oranı
            
        Returns:
            Dict: Model ve performans metrikleri
        """
        test_size = test_size or self.ml_config.get('test_size', 0.2)
        random_state = self.ml_config.get('random_state', 42)
        
        # Veri bölme
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Model eğitimi (Random Forest scaling gerektirmez)
        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Tahminler
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Performans metrikleri
        performance = {
            'model': model,
            'n_estimators': n_estimators,
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'feature_importance': dict(zip(X.columns, model.feature_importances_)),
            'X_test': X_test,
            'y_test': y_test,
            'y_test_pred': y_test_pred
        }
        
        # Cross validation
        cv_folds = self.ml_config.get('cross_validation_folds', 5)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
        performance['cv_r2_mean'] = cv_scores.mean()
        performance['cv_r2_std'] = cv_scores.std()
        
        # Model kaydet
        self.models['random_forest'] = model
        self.model_performance['random_forest'] = performance
        
        logger.info(f"Random Forest eğitildi - Test R²: {performance['test_r2']:.3f}")
        
        return performance
    
    
    def predict_future(self, model_name: str, X_recent: pd.DataFrame, days_ahead: int = 7) -> np.ndarray:
        """
        Gelecek tahminleri yapar.
        
        Args:
            model_name (str): Kullanılacak model adı
            X_recent (pd.DataFrame): Son dönem özellikleri  
            days_ahead (int): Kaç gün sonrası için tahmin
            
        Returns:
            np.ndarray: Tahmin değerleri
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' bulunamadı")
        
        model = self.models[model_name]
        
        # Scaling gerekiyorsa uygula
        if model_name in self.scalers:
            scaler = self.scalers[model_name]
            X_scaled = scaler.transform(X_recent)
            predictions = model.predict(X_scaled)
        else:
            predictions = model.predict(X_recent)
        
        return predictions
    
    
    def compare_models(self) -> pd.DataFrame:
        """
        Eğitilen modellerin performansını karşılaştırır.
        
        Returns:
            pd.DataFrame: Model karşılaştırma tablosu
        """
        if not self.model_performance:
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, performance in self.model_performance.items():
            comparison_data.append({
                'Model': model_name,
                'Test R²': performance.get('test_r2', 0),
                'Test MAE': performance.get('test_mae', 0),
                'Test RMSE': performance.get('test_rmse', 0),
                'CV R² Mean': performance.get('cv_r2_mean', 0),
                'CV R² Std': performance.get('cv_r2_std', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Test R²', ascending=False)
        
        return df
    
    
    def get_feature_importance(self, model_name: str, top_n: int = 10) -> pd.DataFrame:
        """
        Model özellik önemlerini döndürür.
        
        Args:
            model_name (str): Model adı
            top_n (int): En önemli N özellik
            
        Returns:
            pd.DataFrame: Özellik önemleri
        """
        if model_name not in self.model_performance:
            return pd.DataFrame()
        
        feature_importance = self.model_performance[model_name].get('feature_importance', {})
        
        if not feature_importance:
            return pd.DataFrame()
        
        # DataFrame'e dönüştür ve sırala
        importance_df = pd.DataFrame([
            {'Feature': feature, 'Importance': importance}
            for feature, importance in feature_importance.items()
        ])
        
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        return importance_df


def create_simple_prediction_model(time_series_df: pd.DataFrame, 
                                 country: str = 'Turkey', 
                                 target_column: str = 'confirmed') -> Dict:
    """
    Basit tahmin modeli oluşturur.
    
    Args:
        time_series_df (pd.DataFrame): Zaman serisi verisi
        country (str): Ülke adı
        target_column (str): Tahmin edilecek sütun
        
    Returns:
        Dict: Model sonuçları
    """
    # Belirli ülke verisi
    country_data = time_series_df[time_series_df['country'] == country].copy()
    
    if country_data.empty or len(country_data) < 30:
        return {'error': f'{country} için yeterli veri yok (minimum 30 gün gerekli)'}
    
    # Predictor'ı başlat
    predictor = CovidPredictor()
    
    # Özellik hazırlama
    X, y = predictor.prepare_features(country_data, target_column)
    
    if X.empty or len(X) < 20:
        return {'error': 'Özellik hazırlama başarısız'}
    
    results = {}
    
    # Linear Regression
    try:
        lr_results = predictor.train_linear_regression(X, y)
        results['linear_regression'] = lr_results
    except Exception as e:
        results['linear_regression'] = {'error': str(e)}
    
    # Ridge Regression
    try:
        ridge_results = predictor.train_ridge_regression(X, y, alpha=1.0)
        results['ridge_regression'] = ridge_results
    except Exception as e:
        results['ridge_regression'] = {'error': str(e)}
    
    # Random Forest (daha az ağaç sayısı - hızlı test için)
    try:
        rf_results = predictor.train_random_forest(X, y, n_estimators=50)
        results['random_forest'] = rf_results
    except Exception as e:
        results['random_forest'] = {'error': str(e)}
    
    # Model karşılaştırması
    comparison = predictor.compare_models()
    results['model_comparison'] = comparison
    
    # En iyi model
    if not comparison.empty:
        best_model = comparison.iloc[0]['Model']
        results['best_model'] = best_model
        results['best_performance'] = comparison.iloc[0].to_dict()
        
        # En iyi modelin özellik önemleri
        feature_importance = predictor.get_feature_importance(best_model)
        results['feature_importance'] = feature_importance
    
    results['country'] = country
    results['target_column'] = target_column
    results['data_points'] = len(X)
    
    return results


if __name__ == "__main__":
    # Test kodu
    print("🤖 COVID-19 Machine Learning Test")
    print("=" * 40)
    
    # Test verisi oluştur
    dates = pd.date_range('2020-01-22', periods=200, freq='D')
    test_data = []
    
    for i, date in enumerate(dates):
        # Basit sentetik veri
        base_cases = 1000 + i * 50 + np.random.randint(-100, 100)
        test_data.append({
            'country': 'Turkey',
            'date': date,
            'confirmed': max(0, base_cases),
            'new_confirmed': max(0, 50 + np.random.randint(-20, 20))
        })
    
    test_df = pd.DataFrame(test_data)
    
    print(f"📊 Test verisi: {len(test_df)} gün")
    print(f"📅 Tarih aralığı: {test_df['date'].min()} - {test_df['date'].max()}")
    
    # Model test et
    print("\n🤖 Basit tahmin modeli eğitiliyor...")
    results = create_simple_prediction_model(test_df, 'Turkey', 'confirmed')
    
    if 'error' not in results:
        print(f"✅ Model başarıyla eğitildi - {results['data_points']} veri noktası")
        
        # Model karşılaştırması
        if 'model_comparison' in results and not results['model_comparison'].empty:
            print("\n📊 Model Karşılaştırması:")
            comparison = results['model_comparison']
            for _, row in comparison.iterrows():
                print(f"   {row['Model']}: R² = {row['Test R²']:.3f}, MAE = {row['Test MAE']:.0f}")
        
        # En iyi model
        if 'best_model' in results:
            print(f"\n🏆 En iyi model: {results['best_model']}")
            print(f"   Test R²: {results['best_performance']['Test R²']:.3f}")
        
        # Özellik önemleri
        if 'feature_importance' in results and not results['feature_importance'].empty:
            print("\n🔍 En Önemli Özellikler:")
            for _, row in results['feature_importance'].head(5).iterrows():
                print(f"   {row['Feature']}: {row['Importance']:.3f}")
        
        print("\n🎉 Machine Learning test başarıyla tamamlandı!")
    
    else:
        print(f"❌ Model eğitimi başarısız: {results['error']}")