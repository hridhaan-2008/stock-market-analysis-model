import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from utils.indicators import add_all_technical_indicators

class StockPredictor:
    """
    Stock price prediction model using machine learning.
    """
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize the predictor.
        
        Args:
            model_type (str): 'xgboost' or 'randomforest'
        """
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        
    def fetch_data(self, ticker: str, period: str = '2y') -> pd.DataFrame:
        """
        Fetch historical stock data using yfinance.
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Data period (e.g., '1y', '2y', '5y')
            
        Returns:
            pd.DataFrame: Historical stock data
        """
        try:
            print(f"Fetching data for {ticker}...")
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for {ticker}")
            
            print(f"Successfully fetched {len(data)} days of data")
            return data
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for the model.
        
        Args:
            df (pd.DataFrame): Raw stock data
            
        Returns:
            pd.DataFrame: Data with features and target
        """
        print("Preparing features...")
        
        # Add technical indicators
        df = add_all_technical_indicators(df)
        
        # Create binary target: 1 if next day's close > current close, 0 otherwise
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Additional features
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'Close_Rolling_Mean_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Close_Rolling_Std_{window}'] = df['Close'].rolling(window=window).std()
            df[f'Volume_Rolling_Mean_{window}'] = df['Volume'].rolling(window=window).mean()
        
        # Price momentum
        df['Price_Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Price_Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['Price_Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Volatility features
        df['Volatility_5'] = df['Price_Change'].rolling(window=5).std()
        df['Volatility_10'] = df['Price_Change'].rolling(window=10).std()
        df['Volatility_20'] = df['Price_Change'].rolling(window=20).std()
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """
        Select relevant features for the model.
        
        Args:
            df (pd.DataFrame): DataFrame with all features
            
        Returns:
            Tuple[pd.DataFrame, list]: Cleaned DataFrame and feature columns
        """
        # Define feature columns (exclude target and date-related columns)
        exclude_columns = ['Target', 'Date', 'Datetime']
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Remove columns with too many NaN values
        nan_threshold = 0.3
        valid_features = []
        
        for col in feature_columns:
            nan_ratio = df[col].isna().sum() / len(df)
            if nan_ratio < nan_threshold:
                valid_features.append(col)
        
        print(f"Selected {len(valid_features)} features out of {len(feature_columns)}")
        
        return df[valid_features + ['Target']], valid_features
    
    def train_model(self, df: pd.DataFrame) -> Dict:
        """
        Train the machine learning model.
        
        Args:
            df (pd.DataFrame): DataFrame with features and target
            
        Returns:
            Dict: Training results and metrics
        """
        print("Training model...")
        
        # Prepare features
        df_clean, self.feature_columns = self.select_features(df)
        
        # Remove rows with NaN values
        df_clean = df_clean.dropna()
        
        if len(df_clean) < 100:
            raise ValueError("Insufficient data for training (need at least 100 samples)")
        
        # Split data (80/20, no shuffle to maintain temporal order)
        split_idx = int(len(df_clean) * 0.8)
        train_data = df_clean.iloc[:split_idx]
        test_data = df_clean.iloc[split_idx:]
        
        X_train = train_data[self.feature_columns]
        y_train = train_data['Target']
        X_test = test_data[self.feature_columns]
        y_test = test_data['Target']
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Initialize and train model
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        else:  # randomforest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        else:
            feature_importance = {}
        
        results = {
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix,
            'feature_importance': feature_importance,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(self.feature_columns)
        }
        
        return results
    
    def predict_next_day(self, ticker: str) -> Dict:
        """
        Predict the probability of stock going up or down the next day.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict: Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Fetch recent data
        data = self.fetch_data(ticker, period='3mo')
        if data.empty:
            return {'error': 'Could not fetch data'}
        
        # Prepare features
        data = self.prepare_features(data)
        data_clean, _ = self.select_features(data)
        
        # Get the most recent data point
        latest_data = data_clean.iloc[-1:]
        
        if latest_data.empty:
            return {'error': 'Insufficient recent data'}
        
        # Make prediction
        X_latest = latest_data[self.feature_columns]
        prediction = self.model.predict(X_latest)[0]
        probabilities = self.model.predict_proba(X_latest)[0]
        
        # Get current price
        current_price = data['Close'].iloc[-1]
        
        result = {
            'ticker': ticker,
            'current_price': current_price,
            'prediction': 'UP' if prediction == 1 else 'DOWN',
            'probability_up': probabilities[1],
            'probability_down': probabilities[0],
            'confidence': max(probabilities),
            'timestamp': pd.Timestamp.now()
        }
        
        return result
    
    def print_results(self, results: Dict):
        """
        Print formatted training results.
        
        Args:
            results (Dict): Training results
        """
        print(f"\n{'='*60}")
        print("MODEL TRAINING RESULTS")
        print(f"{'='*60}")
        
        print(f"\nModel Type: {self.model_type.upper()}")
        print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"Training Samples: {results['train_samples']}")
        print(f"Test Samples: {results['test_samples']}")
        print(f"Features Used: {results['feature_count']}")
        
        print(f"\n{'='*40}")
        print("CLASSIFICATION REPORT:")
        print(f"{'='*40}")
        
        # Print classification report in a readable format
        report = results['classification_report']
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"\nClass {class_name}:")
                for metric, value in metrics.items():
                    if metric != 'support':
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {int(value)}")
        
        print(f"\n{'='*40}")
        print("TOP 10 FEATURE IMPORTANCE:")
        print(f"{'='*40}")
        
        feature_importance = results['feature_importance']
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
            print(f"{i+1:2d}. {feature}: {importance:.4f}")
        
        print(f"\n{'='*60}")
    
    def print_prediction(self, prediction: Dict):
        """
        Print formatted prediction results.
        
        Args:
            prediction (Dict): Prediction results
        """
        if 'error' in prediction:
            print(f"Error: {prediction['error']}")
            return
        
        print(f"\n{'='*60}")
        print(f"PREDICTION FOR {prediction['ticker']}")
        print(f"{'='*60}")
        
        print(f"Current Price: ${prediction['current_price']:.2f}")
        print(f"Prediction: {prediction['prediction']}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        
        print(f"\nProbabilities:")
        print(f"  UP:   {prediction['probability_up']:.2%}")
        print(f"  DOWN: {prediction['probability_down']:.2%}")
        
        # Add interpretation
        if prediction['confidence'] >= 0.7:
            confidence_level = "High"
        elif prediction['confidence'] >= 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        print(f"\nConfidence Level: {confidence_level}")
        
        if prediction['probability_up'] > 0.6:
            recommendation = "Consider buying"
        elif prediction['probability_down'] > 0.6:
            recommendation = "Consider selling"
        else:
            recommendation = "Hold/Neutral"
        
        print(f"Recommendation: {recommendation}")
        print(f"Timestamp: {prediction['timestamp']}")
        print(f"{'='*60}") 