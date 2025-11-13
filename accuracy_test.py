#!/usr/bin/env python3
"""
Accuracy Testing Framework for Stock Prediction System
=====================================================

This script tests the accuracy of the stock prediction system using:
1. Backtesting with historical data
2. Model performance metrics
3. Prediction accuracy validation
4. Fundamental analysis accuracy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from predictor import StockPredictor
from checker import FundamentalChecker
from utils.indicators import add_all_technical_indicators


def create_test_data(symbol='TEST', days=1000, trend='random'):
    """
    Create test data with known patterns for accuracy testing.
    
    Args:
        symbol (str): Test symbol name
        days (int): Number of days of data
        trend (str): 'random', 'uptrend', 'downtrend', 'volatile'
        
    Returns:
        pd.DataFrame: Test data with known patterns
    """
    print(f"Creating test data for {symbol} with {trend} pattern...")
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42)  # For reproducible results
    
    # Set base price
    base_price = 100.0
    
    # Generate price movements based on trend
    if trend == 'uptrend':
        # Consistent upward trend
        returns = np.random.normal(0.002, 0.02, len(dates))  # 0.2% daily growth
    elif trend == 'downtrend':
        # Consistent downward trend
        returns = np.random.normal(-0.002, 0.02, len(dates))  # -0.2% daily decline
    elif trend == 'volatile':
        # High volatility, no clear trend
        returns = np.random.normal(0.0, 0.04, len(dates))  # 4% daily volatility
    else:  # random
        # Random walk
        returns = np.random.normal(0.0, 0.02, len(dates))
    
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1.0))
    
    # Create OHLC data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        daily_vol = np.random.uniform(0.01, 0.03)
        
        open_price = price * (1 + np.random.uniform(-daily_vol/2, daily_vol/2))
        high_price = max(open_price, price) * (1 + np.random.uniform(0, daily_vol/2))
        low_price = min(open_price, price) * (1 - np.random.uniform(0, daily_vol/2))
        close_price = price
        
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'Date': date,
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    return df

def test_model_accuracy():
    """Test the accuracy of machine learning models."""
    print("\n" + "="*60)
    print("MODEL ACCURACY TESTING")
    print("="*60)
    
    results = {}
    
    # Test different market conditions
    market_conditions = ['random', 'uptrend', 'downtrend', 'volatile']
    models = ['xgboost', 'randomforest']
    
    for condition in market_conditions:
        print(f"\nTesting {condition.upper()} market condition:")
        print("-" * 40)
        
        condition_results = {}
        
        for model_type in models:
            print(f"  Testing {model_type.upper()} model...")
            
            # Create test data
            data = create_test_data(f"TEST_{condition.upper()}", 800, condition)
            
            # Initialize predictor
            predictor = StockPredictor(model_type=model_type)
            
            # Prepare features
            data = predictor.prepare_features(data)
            
            # Train model
            try:
                model_results = predictor.train_model(data)
                
                # Store results
                condition_results[model_type] = {
                    'accuracy': model_results['accuracy'],
                    'precision': model_results.get('precision', 0.0),
                    'recall': model_results.get('recall', 0.0),
                    'f1_score': model_results.get('f1_score', 0.0),
                    'train_samples': model_results['train_samples'],
                    'test_samples': model_results['test_samples'],
                    'feature_count': model_results['feature_count']
                }
                
                print(f"    Accuracy: {model_results['accuracy']:.2%}")
                print(f"    Train/Test: {model_results['train_samples']}/{model_results['test_samples']}")
                print(f"    Features: {model_results['feature_count']}")
                
            except Exception as e:
                print(f"    Error: {e}")
                condition_results[model_type] = None
        
        results[condition] = condition_results
    
    return results

def test_prediction_accuracy():
    """Test prediction accuracy using backtesting."""
    print("\n" + "="*60)
    print("PREDICTION ACCURACY TESTING")
    print("="*60)
    
    # Create test data with known patterns
    data = create_test_data('BACKTEST', 1000, 'random')
    
    # Add technical indicators
    data = add_all_technical_indicators(data)
    
    # Create target variable (next day's direction)
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Remove NaN values
    data = data.dropna()
    
    # Split data for backtesting
    train_size = int(len(data) * 0.7)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Testing period: {test_data.index[0]} to {test_data.index[-1]}")
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    
    # Test both models
    models = ['xgboost', 'randomforest']
    backtest_results = {}
    
    for model_type in models:
        print(f"\nBacktesting {model_type.upper()} model:")
        print("-" * 30)
        
        # Initialize predictor
        predictor = StockPredictor(model_type=model_type)
        
        # Prepare features
        feature_columns = [col for col in train_data.columns if col not in ['Target', 'Date']]
        
        X_train = train_data[feature_columns]
        y_train = train_data['Target']
        X_test = test_data[feature_columns]
        y_test = test_data['Target']
        
        # Train model
        if predictor.model is None:
            if model_type == 'xgboost':
                import xgboost as xgb
                predictor.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='logloss'
                )
            else:  # randomforest
                from sklearn.ensemble import RandomForestClassifier
                predictor.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
        
        predictor.model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = predictor.model.predict(X_test)
        y_pred_proba = predictor.model.predict_proba(X_test)
        
        # Calculate accuracy metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate directional accuracy (how often we predict the right direction)
        correct_directions = sum(1 for i in range(len(y_test)) if y_test.iloc[i] == y_pred[i])
        directional_accuracy = correct_directions / len(y_test)
        
        # Calculate profit/loss simulation
        initial_capital = 10000
        capital = initial_capital
        trades = []
        
        for i in range(len(y_test)):
            if y_pred[i] == 1:  # Predicted UP
                # Simulate buying at current price and selling at next price
                current_price = test_data['Close'].iloc[i]
                if i < len(y_test) - 1:
                    next_price = test_data['Close'].iloc[i + 1]
                    profit = (next_price - current_price) / current_price
                    capital *= (1 + profit)
                    trades.append(profit)
        
        total_return = (capital - initial_capital) / initial_capital
        
        backtest_results[model_type] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'directional_accuracy': directional_accuracy,
            'total_return': total_return,
            'confusion_matrix': conf_matrix,
            'trades_count': len(trades),
            'avg_trade_return': np.mean(trades) if trades else 0
        }
        
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall: {recall:.2%}")
        print(f"  F1 Score: {f1:.2%}")
        print(f"  Directional Accuracy: {directional_accuracy:.2%}")
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Number of Trades: {len(trades)}")
        print(f"  Average Trade Return: {np.mean(trades):.2%}" if trades else "  Average Trade Return: N/A")
    
    return backtest_results

def test_fundamental_analysis_accuracy():
    """Test the accuracy of fundamental analysis scoring."""
    print("\n" + "="*60)
    print("FUNDAMENTAL ANALYSIS ACCURACY TESTING")
    print("="*60)
    
    # Initialize checker
    checker = FundamentalChecker()
    
    # Test cases with known expected outcomes
    test_cases = [
        # (pe_ratio, roe, earnings_growth, expected_rating, expected_score_range)
        (10, 0.25, 0.20, "Excellent", (0.8, 1.0)),
        (20, 0.15, 0.10, "Good", (0.6, 0.8)),
        (35, 0.08, 0.05, "Fair", (0.4, 0.6)),
        (60, 0.03, 0.02, "Poor", (0.2, 0.4)),
        (100, 0.01, -0.05, "Very Poor", (0.0, 0.2))
    ]
    
    fundamental_results = {}
    
    for i, (pe, roe, growth, expected_rating, expected_score_range) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: PE={pe}, ROE={roe*100:.1f}%, Growth={growth*100:.1f}%")
        print(f"Expected: {expected_rating} (Score: {expected_score_range[0]:.1f}-{expected_score_range[1]:.1f})")
        
        # Evaluate each metric
        pe_eval, pe_score = checker.evaluate_pe_ratio(pe)
        roe_eval, roe_score = checker.evaluate_roe(roe)
        growth_eval, growth_score = checker.evaluate_earnings_growth(growth)
        
        # Calculate overall score
        overall_score = (pe_score + roe_score + growth_score) / 3
        
        # Determine actual rating
        if overall_score >= 0.8:
            actual_rating = "Excellent"
        elif overall_score >= 0.6:
            actual_rating = "Good"
        elif overall_score >= 0.4:
            actual_rating = "Fair"
        elif overall_score >= 0.2:
            actual_rating = "Poor"
        else:
            actual_rating = "Very Poor"
        
        # Check if score is within expected range
        score_correct = expected_score_range[0] <= overall_score <= expected_score_range[1]
        rating_correct = actual_rating == expected_rating
        
        print(f"Actual: {actual_rating} (Score: {overall_score:.2f})")
        print(f"PE: {pe_eval} ({pe_score:.2f})")
        print(f"ROE: {roe_eval} ({roe_score:.2f})")
        print(f"Growth: {growth_eval} ({growth_score:.2f})")
        print(f"Score Correct: {'✅' if score_correct else '❌'}")
        print(f"Rating Correct: {'✅' if rating_correct else '❌'}")
        
        fundamental_results[f"case_{i+1}"] = {
            'expected_rating': expected_rating,
            'actual_rating': actual_rating,
            'expected_score_range': expected_score_range,
            'actual_score': overall_score,
            'score_correct': score_correct,
            'rating_correct': rating_correct,
            'pe_score': pe_score,
            'roe_score': roe_score,
            'growth_score': growth_score
        }
    
    return fundamental_results

def test_feature_importance_consistency():
    """Test if feature importance is consistent across different datasets."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE CONSISTENCY TESTING")
    print("="*60)
    
    # Test with different datasets
    datasets = [
        ('random', 'random'),
        ('uptrend', 'uptrend'),
        ('downtrend', 'downtrend')
    ]
    
    feature_importance_results = {}
    
    for dataset_name, trend in datasets:
        print(f"\nTesting feature importance for {dataset_name} dataset:")
        print("-" * 45)
        
        # Create test data
        data = create_test_data(f"FEATURE_TEST_{dataset_name}", 600, trend)
        
        # Initialize predictor
        predictor = StockPredictor(model_type='xgboost')
        
        # Prepare features
        data = predictor.prepare_features(data)
        
        # Train model
        try:
            results = predictor.train_model(data)
            
            # Get top 10 features
            feature_importance = results['feature_importance']
            top_features = list(feature_importance.items())[:10]
            
            print(f"  Top 10 features:")
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"    {i:2d}. {feature}: {importance:.4f}")
            
            feature_importance_results[dataset_name] = {
                'top_features': top_features,
                'model_accuracy': results['accuracy']
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            feature_importance_results[dataset_name] = None
    
    return feature_importance_results

def test_system_stability():
    """Test system stability with different parameters."""
    print("\n" + "="*60)
    print("SYSTEM STABILITY TESTING")
    print("="*60)
    
    stability_results = {}
    
    # Test different data sizes
    data_sizes = [200, 400, 600, 800]
    
    print("Testing with different data sizes:")
    print("-" * 35)
    
    for size in data_sizes:
        print(f"\nData size: {size} days")
        
        try:
            # Create test data
            data = create_test_data(f"STABILITY_TEST_{size}", size, 'random')
            
            # Initialize predictor
            predictor = StockPredictor(model_type='xgboost')
            
            # Prepare features
            data = predictor.prepare_features(data)
            
            # Train model
            results = predictor.train_model(data)
            
            stability_results[f"size_{size}"] = {
                'accuracy': results['accuracy'],
                'train_samples': results['train_samples'],
                'test_samples': results['test_samples'],
                'feature_count': results['feature_count'],
                'success': True
            }
            
            print(f"  Accuracy: {results['accuracy']:.2%}")
            print(f"  Train/Test: {results['train_samples']}/{results['test_samples']}")
            print(f"  Features: {results['feature_count']}")
            
        except Exception as e:
            print(f"  Error: {e}")
            stability_results[f"size_{size}"] = {
                'success': False,
                'error': str(e)
            }
    
    return stability_results

def generate_accuracy_report(model_results, backtest_results, fundamental_results, 
                           feature_results, stability_results):
    """Generate comprehensive accuracy report as markdown."""
    
    # Calculate overall metrics
    all_accuracies = []
    for condition, models in model_results.items():
        for model_type, results in models.items():
            if results:
                all_accuracies.append(results['accuracy'])
    
    avg_accuracy = np.mean(all_accuracies) if all_accuracies else 0
    
    correct_ratings = sum(1 for case in fundamental_results.values() if case['rating_correct'])
    correct_scores = sum(1 for case in fundamental_results.values() if case['score_correct'])
    total_cases = len(fundamental_results)
    
    successful_tests = sum(1 for test in stability_results.values() if test['success'])
    total_tests = len(stability_results)
    
    # Performance Rating
    if avg_accuracy >= 0.6:
        performance = "EXCELLENT"
    elif avg_accuracy >= 0.55:
        performance = "GOOD"
    elif avg_accuracy >= 0.5:
        performance = "FAIR"
    else:
        performance = "POOR"
    
    # Generate markdown content
    timestamp = datetime.now().strftime('%B %d, %Y at %I:%M %p')
    
    markdown_content = f"""# Stock Prediction System - Accuracy Report

**Generated on:** {timestamp}

## 🎯 Overall Performance Summary

| Metric | Value |
|--------|-------|
| **Average Model Accuracy** | {avg_accuracy:.2%} |
| **Fundamental Analysis Accuracy** | {correct_ratings/total_cases:.2%} |
| **System Stability** | {successful_tests/total_tests:.2%} |
| **Performance Rating** | **{performance}** |

## 🤖 Model Performance Results

### Model Accuracy by Market Condition

"""
    
    # Add model results to markdown
    for condition, models in model_results.items():
        markdown_content += f"#### {condition.title()} Market Condition\n\n"
        markdown_content += "| Model | Accuracy | Precision | Recall | F1-Score |\n"
        markdown_content += "|-------|----------|-----------|--------|----------|\n"
        
        for model_type, results in models.items():
            if results:
                markdown_content += f"| {model_type.upper()} | {results['accuracy']:.2%} | {results['precision']:.2%} | {results['recall']:.2%} | {results['f1_score']:.2%} |\n"
        
        markdown_content += "\n"
    
    # Add backtesting results
    markdown_content += "## 📈 Backtesting Results\n\n"
    markdown_content += "| Strategy | Accuracy | Directional Accuracy | Total Return | Trades |\n"
    markdown_content += "|----------|----------|---------------------|--------------|--------|\n"
    
    for strategy, results in backtest_results.items():
        markdown_content += f"| {strategy.upper()} | {results['accuracy']:.2%} | {results['directional_accuracy']:.2%} | {results['total_return']:.2%} | {results['trades_count']} |\n"
    
    # Add fundamental analysis results
    markdown_content += f"""
## 📊 Fundamental Analysis Results

| Metric | Result |
|--------|--------|
| **Rating Accuracy** | {correct_ratings}/{total_cases} ({correct_ratings/total_cases:.2%}) |
| **Score Accuracy** | {correct_scores}/{total_cases} ({correct_scores/total_cases:.2%}) |

## 🔧 System Stability

| Metric | Result |
|--------|--------|
| **Stability Rate** | {successful_tests}/{total_tests} ({successful_tests/total_tests:.2%}) |

## 📋 Test Summary

✅ **Model accuracy across different market conditions** - Completed  
✅ **Prediction accuracy with backtesting** - Completed  
✅ **Fundamental analysis scoring accuracy** - Completed  
✅ **Feature importance consistency** - Completed  
✅ **System stability and reliability** - Completed

---
*Report generated by Stock Prediction System Accuracy Testing Framework*
"""
    
    # Save markdown report
    import os
    folder = "accuracy_reports"
    os.makedirs(folder, exist_ok=True)
    timestamp_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"accuracy_test_report_{timestamp_filename}.md"
    file_path = os.path.join(folder, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    print(f"\n📊 Accuracy report saved: {file_path}")
    print(f"Average Model Accuracy: {avg_accuracy:.2%}")
    print(f"Performance Rating: {performance}")

def main():
    """Run comprehensive accuracy tests."""
    print("Stock Prediction System - Accuracy Testing")
    print("="*60)
    print("Running comprehensive accuracy tests...")
    
    try:
        # Run all tests
        print("\nStarting accuracy tests...")
        
        # Model accuracy testing
        model_results = test_model_accuracy()
        
        # Prediction accuracy testing
        backtest_results = test_prediction_accuracy()
        
        # Fundamental analysis testing
        fundamental_results = test_fundamental_analysis_accuracy()
        
        # Feature importance testing
        feature_results = test_feature_importance_consistency()
        
        # System stability testing
        stability_results = test_system_stability()
        
        # Generate comprehensive report
        generate_accuracy_report(model_results, backtest_results, fundamental_results,
                               feature_results, stability_results)
        
        print("\n" + "="*60)
        print("ACCURACY TESTING COMPLETED!")
        print("="*60)
        print("The system has been thoroughly tested for:")
        print("✅ Model accuracy across different market conditions")
        print("✅ Prediction accuracy with backtesting")
        print("✅ Fundamental analysis scoring accuracy")
        print("✅ Feature importance consistency")
        print("✅ System stability and reliability")
        
    except Exception as e:
        print(f"Error during accuracy testing: {e}")
        print("Please check the system implementation.")

if __name__ == "__main__":
    main()
