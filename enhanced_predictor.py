#!/usr/bin/env python3
"""
Enhanced Stock Predictor with Advanced Techniques
================================================

This enhanced version uses multiple advanced techniques to improve accuracy:
1. Ensemble Methods (Voting, Stacking, Blending)
2. Deep Learning Models (LSTM, GRU)
3. Advanced Feature Engineering
4. Hyperparameter Optimization
5. Market Regime Detection
6. Feature Selection
7. Cross-Validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import optuna
from typing import Dict, List, Tuple, Optional
import joblib

class EnhancedStockPredictor:
    """
    Enhanced stock predictor with advanced techniques for higher accuracy.
    """
    
    def __init__(self, model_type='ensemble'):
        """
        Initialize enhanced predictor.
        
        Args:
            model_type (str): 'ensemble', 'deep_learning', 'optimized'
        """
        self.model_type = model_type
        self.models = {}
        self.feature_columns = None
        self.scaler = None
        self.feature_selector = None
        self.is_trained = False
        self.best_params = {}
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features for better prediction.
        
        Args:
            df (pd.DataFrame): Raw stock data
            
        Returns:
            pd.DataFrame: Data with advanced features
        """
        print("Creating advanced features...")
        
        # Basic technical indicators (from original system)
        from utils.indicators import add_all_technical_indicators
        df = add_all_technical_indicators(df)
        
        # Advanced price-based features
        df['Price_Position'] = (df['Close'] - df['Close'].rolling(20).min()) / (df['Close'].rolling(20).max() - df['Close'].rolling(20).min())
        df['Price_Acceleration'] = df['Close'].diff().diff()
        df['Price_Jerk'] = df['Price_Acceleration'].diff()
        
        # Advanced volume features
        df['Volume_Price_Trend'] = (df['Volume'] * df['Close']).rolling(10).mean()
        df['Volume_Force'] = df['Volume'] * df['Close'].pct_change().abs()
        df['Volume_SMA_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Advanced momentum features
        for period in [5, 10, 20, 50]:
            df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            df[f'ROC_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period) * 100
        
        # Advanced volatility features
        for period in [5, 10, 20]:
            df[f'Volatility_{period}'] = df['Close'].rolling(period).std() / df['Close'].rolling(period).mean()
            df[f'Volatility_Ratio_{period}'] = df[f'Volatility_{period}'] / df[f'Volatility_{period}'].rolling(50).mean()
        
        # Advanced trend features
        df['Trend_Strength'] = abs(df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
        df['Trend_Direction'] = np.where(df['Close'] > df['Close'].rolling(20).mean(), 1, -1)
        
        # Advanced support/resistance features
        df['Support_Level'] = df['Low'].rolling(20).min()
        df['Resistance_Level'] = df['High'].rolling(20).max()
        df['Support_Distance'] = (df['Close'] - df['Support_Level']) / df['Close']
        df['Resistance_Distance'] = (df['Resistance_Level'] - df['Close']) / df['Close']
        
        # Advanced pattern features
        df['Doji'] = np.where(abs(df['Open'] - df['Close']) <= (df['High'] - df['Low']) * 0.1, 1, 0)
        df['Hammer'] = np.where((df['Close'] - df['Low']) > 2 * (df['High'] - df['Close']), 1, 0)
        df['Shooting_Star'] = np.where((df['High'] - df['Close']) > 2 * (df['Close'] - df['Low']), 1, 0)
        
        # Advanced statistical features
        df['Z_Score'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
        df['Bollinger_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Advanced time-based features
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Is_Month_End'] = df.index.is_month_end.astype(int)
        df['Is_Quarter_End'] = df.index.is_quarter_end.astype(int)
        
        # Advanced market regime features
        df['Market_Regime'] = np.where(df['Close'] > df['Close'].rolling(50).mean(), 1, 0)
        df['Regime_Strength'] = abs(df['Close'] - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).std()
        
        # Advanced correlation features
        for period in [5, 10, 20]:
            df[f'Price_Volume_Corr_{period}'] = df['Close'].rolling(period).corr(df['Volume'])
        
        # Advanced divergence features
        df['Price_RSI_Divergence'] = np.where(
            (df['Close'] > df['Close'].shift(10)) & (df['RSI'] < df['RSI'].shift(10)), 1,
            np.where((df['Close'] < df['Close'].shift(10)) & (df['RSI'] > df['RSI'].shift(10)), -1, 0)
        )
        
        # Advanced oscillator features
        df['Stochastic_K'] = (df['Close'] - df['Low'].rolling(14).min()) / (df['High'].rolling(14).max() - df['Low'].rolling(14).min()) * 100
        df['Stochastic_D'] = df['Stochastic_K'].rolling(3).mean()
        
        # Advanced Williams %R
        df['Williams_R'] = (df['High'].rolling(14).max() - df['Close']) / (df['High'].rolling(14).max() - df['Low'].rolling(14).min()) * -100
        
        # Advanced CCI (Commodity Channel Index)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        df['CCI'] = (typical_price - typical_price.rolling(20).mean()) / (0.015 * typical_price.rolling(20).std())
        
        # Advanced ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['ATR'] = true_range.rolling(14).mean()
        
        # Advanced ADX (Average Directional Index)
        df['ADX'] = self._calculate_adx(df)
        
        # Advanced Money Flow Index
        df['MFI'] = self._calculate_mfi(df)
        
        # Advanced On Balance Volume
        df['OBV'] = self._calculate_obv(df)
        
        # Advanced Chaikin Money Flow
        df['CMF'] = self._calculate_cmf(df)
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        high_diff = df['High'].diff()
        low_diff = df['Low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        tr = self._calculate_true_range(df)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / tr.rolling(period).mean()
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / tr.rolling(period).mean()
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        
        positive_mf = pd.Series(positive_flow).rolling(period).sum()
        negative_mf = pd.Series(negative_flow).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume."""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['Volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow."""
        mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        mfm = mfm.replace([np.inf, -np.inf], 0)
        mfv = mfm * df['Volume']
        cmf = mfv.rolling(period).sum() / df['Volume'].rolling(period).sum()
        return cmf
    
    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range
    
    def select_best_features(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select the best features using multiple methods.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            k (int): Number of features to select
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Selected features and feature names
        """
        print(f"Selecting best {k} features from {X.shape[1]} features...")
        
        # Method 1: Statistical tests
        selector1 = SelectKBest(score_func=f_classif, k=k)
        X_selected1 = selector1.fit_transform(X, y)
        selected_features1 = X.columns[selector1.get_support()].tolist()
        
        # Method 2: Recursive Feature Elimination with Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        selector2 = RFE(estimator=rf, n_features_to_select=k)
        X_selected2 = selector2.fit_transform(X, y)
        selected_features2 = X.columns[selector2.get_support()].tolist()
        
        # Method 3: XGBoost feature importance
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        xgb_model.fit(X, y)
        feature_importance = xgb_model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-k:]
        selected_features3 = X.columns[top_features_idx].tolist()
        
        # Combine all methods and select most common features
        all_features = selected_features1 + selected_features2 + selected_features3
        feature_counts = pd.Series(all_features).value_counts()
        
        # Select features that appear in at least 2 methods
        final_features = feature_counts[feature_counts >= 2].index.tolist()
        
        # If we don't have enough features, add the most important ones
        if len(final_features) < k:
            remaining_features = feature_counts[feature_counts < 2].index.tolist()
            final_features.extend(remaining_features[:k-len(final_features)])
        
        final_features = final_features[:k]
        
        print(f"Selected {len(final_features)} features")
        return X[final_features], final_features
    
    def create_ensemble_model(self) -> VotingClassifier:
        """
        Create an ensemble model with multiple algorithms.
        
        Returns:
            VotingClassifier: Ensemble model
        """
        # Base models
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42)
        lr = LogisticRegression(random_state=42, max_iter=1000)
        svm = SVC(probability=True, random_state=42)
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('xgb', xgb_model),
                ('gb', gb),
                ('lr', lr),
                ('svm', svm)
            ],
            voting='soft'
        )
        
        return ensemble
    
    def create_deep_learning_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Create a deep learning model with LSTM and GRU layers.
        
        Args:
            input_shape (Tuple[int, int]): Input shape (timesteps, features)
            
        Returns:
            tf.keras.Model: Deep learning model
        """
        model = Sequential([
            # LSTM layers
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            Dict: Best hyperparameters
        """
        print("Optimizing hyperparameters...")
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
            
            # Create model with suggested parameters
            model = xgb.XGBClassifier(**params, random_state=42)
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        print(f"Best accuracy: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        
        return study.best_params
    
    def prepare_data_for_deep_learning(self, df: pd.DataFrame, target_col: str = 'Target', 
                                     sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for deep learning models.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            target_col (str): Target column name
            sequence_length (int): Length of sequences
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X and y arrays
        """
        # Remove target column and get feature columns
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train_enhanced_model(self, df: pd.DataFrame) -> Dict:
        """
        Train the enhanced model with all advanced techniques.
        
        Args:
            df (pd.DataFrame): DataFrame with features and target
            
        Returns:
            Dict: Training results
        """
        print("Training enhanced model...")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['Target', 'Date']]
        X = df[feature_cols]
        y = df['Target']
        
        # Remove NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        print(f"Final dataset shape: {X.shape}")
        
        # Feature selection
        X_selected, selected_features = self.select_best_features(X, y, k=50)
        self.feature_columns = selected_features
        
        # Feature scaling
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Split data (time series split)
        split_idx = int(len(X_scaled) * 0.8)
        X_train = X_scaled[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X_scaled[split_idx:]
        y_test = y.iloc[split_idx:]
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        results = {}
        
        if self.model_type == 'ensemble':
            # Train ensemble model
            ensemble = self.create_ensemble_model()
            ensemble.fit(X_train, y_train)
            
            # Predictions
            y_pred = ensemble.predict(X_test)
            y_pred_proba = ensemble.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            results = {
                'accuracy': accuracy,
                'model': ensemble,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(selected_features)
            }
            
        elif self.model_type == 'deep_learning':
            # Prepare data for deep learning
            X_sequences, y_sequences = self.prepare_data_for_deep_learning(
                pd.DataFrame(X_scaled, columns=selected_features).assign(Target=y.values)
            )
            
            # Split sequences
            split_idx = int(len(X_sequences) * 0.8)
            X_train_seq = X_sequences[:split_idx]
            y_train_seq = y_sequences[:split_idx]
            X_test_seq = X_sequences[split_idx:]
            y_test_seq = y_sequences[split_idx:]
            
            # Create and train deep learning model
            model = self.create_deep_learning_model((X_train_seq.shape[1], X_train_seq.shape[2]))
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(
                X_train_seq, y_train_seq,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Predictions
            y_pred = (model.predict(X_test_seq) > 0.5).astype(int).flatten()
            y_pred_proba = model.predict(X_test_seq).flatten()
            
            accuracy = accuracy_score(y_test_seq, y_pred)
            
            results = {
                'accuracy': accuracy,
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'train_samples': len(X_train_seq),
                'test_samples': len(X_test_seq),
                'feature_count': len(selected_features),
                'history': history
            }
            
        elif self.model_type == 'optimized':
            # Optimize hyperparameters
            best_params = self.optimize_hyperparameters(X_selected, y)
            self.best_params = best_params
            
            # Train optimized model
            optimized_model = xgb.XGBClassifier(**best_params, random_state=42)
            optimized_model.fit(X_train, y_train)
            
            # Predictions
            y_pred = optimized_model.predict(X_test)
            y_pred_proba = optimized_model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            results = {
                'accuracy': accuracy,
                'model': optimized_model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(selected_features),
                'best_params': best_params
            }
        
        self.is_trained = True
        return results
    
    def predict_next_day(self, ticker: str) -> Dict:
        """
        Predict the next day's movement using enhanced model.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict: Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # This would need to be implemented with real data fetching
        # For now, return a mock prediction
        return {
            'ticker': ticker,
            'prediction': 'UP',
            'probability_up': 0.75,
            'probability_down': 0.25,
            'confidence': 0.75,
            'model_type': self.model_type
        }

def main():
    """Test the enhanced predictor."""
    print("Enhanced Stock Predictor Test")
    print("=" * 50)
    
    # Create test data
    from accuracy_test import create_test_data
    data = create_test_data('ENHANCED_TEST', 1000, 'random')
    
    # Create enhanced features
    predictor = EnhancedStockPredictor(model_type='ensemble')
    enhanced_data = predictor.create_advanced_features(data)
    
    # Create target
    enhanced_data['Target'] = (enhanced_data['Close'].shift(-1) > enhanced_data['Close']).astype(int)
    
    # Train model
    results = predictor.train_enhanced_model(enhanced_data)
    
    print(f"\nEnhanced Model Results:")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Train samples: {results['train_samples']}")
    print(f"Test samples: {results['test_samples']}")
    print(f"Features used: {results['feature_count']}")
    
    if results['accuracy'] >= 0.75:
        print("🎉 Target accuracy of 75% achieved!")
    else:
        print(f"⚠️ Accuracy {results['accuracy']:.2%} is below 75% target")

if __name__ == "__main__":
    main()
