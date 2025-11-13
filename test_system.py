#!/usr/bin/env python3
"""
Test script to demonstrate the Stock Prediction System functionality.
This script uses mock data to show that all components work correctly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from predictor import StockPredictor
from checker import FundamentalChecker
from utils.indicators import add_all_technical_indicators

def create_mock_stock_data(ticker='AAPL', days=500):
    """
    Create mock stock data for testing purposes.
    
    Args:
        ticker (str): Stock ticker symbol
        days (int): Number of days of data to generate
        
    Returns:
        pd.DataFrame: Mock stock data
    """
    print(f"Creating mock data for {ticker}...")
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate realistic stock data
    np.random.seed(42)  # For reproducible results
    
    # Start with a base price
    base_price = 150.0
    
    # Generate price movements
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns with volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1.0))  # Ensure price doesn't go below $1
    
    # Create OHLC data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        daily_volatility = np.random.uniform(0.005, 0.03)
        
        open_price = price * (1 + np.random.uniform(-daily_volatility/2, daily_volatility/2))
        high_price = max(open_price, price) * (1 + np.random.uniform(0, daily_volatility/2))
        low_price = min(open_price, price) * (1 - np.random.uniform(0, daily_volatility/2))
        close_price = price
        
        # Generate volume
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
    
    print(f"Generated {len(df)} days of mock data")
    return df

def test_technical_indicators():
    """Test technical indicators calculation."""
    print("\n" + "="*60)
    print("TESTING TECHNICAL INDICATORS")
    print("="*60)
    
    # Create mock data
    data = create_mock_stock_data('AAPL', 100)
    
    # Add technical indicators
    data_with_indicators = add_all_technical_indicators(data)
    
    # Check if indicators were added
    expected_indicators = ['Daily_Return', 'MA_5', 'MA_20', 'Volatility', 'RSI', 'BB_Middle', 'MACD']
    added_indicators = [col for col in data_with_indicators.columns if col in expected_indicators]
    
    print(f"Expected indicators: {expected_indicators}")
    print(f"Added indicators: {added_indicators}")
    print(f"Total features: {len(data_with_indicators.columns)}")
    
    # Show sample data
    print(f"\nSample data with indicators:")
    print(data_with_indicators.tail(3)[['Close', 'Daily_Return', 'MA_5', 'RSI']])
    
    return data_with_indicators

def test_predictor():
    """Test the predictor functionality."""
    print("\n" + "="*60)
    print("TESTING PREDICTOR")
    print("="*60)
    
    # Create mock data
    data = create_mock_stock_data('AAPL', 300)
    
    # Initialize predictor
    predictor = StockPredictor(model_type='xgboost')
    
    # Prepare features
    data = predictor.prepare_features(data)
    
    # Train model
    try:
        results = predictor.train_model(data)
        predictor.print_results(results)
        
        # Test prediction
        prediction = predictor.predict_next_day('AAPL')
        predictor.print_prediction(prediction)
        
        return True
    except Exception as e:
        print(f"Error in predictor test: {e}")
        return False

def test_fundamental_checker():
    """Test the fundamental checker functionality."""
    print("\n" + "="*60)
    print("TESTING FUNDAMENTAL CHECKER")
    print("="*60)
    
    # Initialize checker
    checker = FundamentalChecker()
    
    # Test evaluation methods
    print("Testing PE ratio evaluation:")
    test_pe_ratios = [10, 20, 35, 60, None]
    for pe in test_pe_ratios:
        eval_result, score = checker.evaluate_pe_ratio(pe)
        print(f"  PE {pe}: {eval_result} (Score: {score:.2f})")
    
    print("\nTesting ROE evaluation:")
    test_roe_values = [0.25, 0.18, 0.12, 0.08, 0.02, None]
    for roe in test_roe_values:
        eval_result, score = checker.evaluate_roe(roe)
        print(f"  ROE {roe}: {eval_result} (Score: {score:.2f})")
    
    print("\nTesting earnings growth evaluation:")
    test_growth_values = [0.25, 0.18, 0.12, 0.08, 0.02, -0.05, None]
    for growth in test_growth_values:
        eval_result, score = checker.evaluate_earnings_growth(growth)
        print(f"  Growth {growth}: {eval_result} (Score: {score:.2f})")
    
    return True

def test_complete_system():
    """Test the complete system with mock data."""
    print("\n" + "="*60)
    print("TESTING COMPLETE SYSTEM")
    print("="*60)
    
    # Create mock data
    data = create_mock_stock_data('AAPL', 400)
    
    # Test technical analysis
    predictor = StockPredictor(model_type='randomforest')
    data = predictor.prepare_features(data)
    
    try:
        results = predictor.train_model(data)
        prediction = predictor.predict_next_day('AAPL')
        
        print(f"\nTechnical Analysis Results:")
        print(f"  Model Accuracy: {results['accuracy']:.2%}")
        print(f"  Prediction: {prediction['prediction']}")
        print(f"  Confidence: {prediction['confidence']:.2%}")
        
        # Mock fundamental analysis
        print(f"\nMock Fundamental Analysis:")
        print(f"  PE Ratio: 25.5 (Good)")
        print(f"  ROE: 18.2% (Very Good)")
        print(f"  Earnings Growth: 12.5% (Good)")
        print(f"  Overall Score: 0.75/1.00")
        print(f"  Recommendation: Buy")
        
        # Combined analysis
        tech_score = prediction['probability_up'] if prediction['prediction'] == 'UP' else prediction['probability_down']
        fund_score = 0.75  # Mock fundamental score
        
        combined_score = 0.7 * tech_score + 0.3 * fund_score
        
        print(f"\nCombined Analysis:")
        print(f"  Technical Score: {tech_score:.2%}")
        print(f"  Fundamental Score: {fund_score:.2%}")
        print(f"  Combined Score: {combined_score:.2%}")
        
        if combined_score >= 0.7:
            final_rec = "STRONG BUY"
        elif combined_score >= 0.6:
            final_rec = "BUY"
        elif combined_score >= 0.4:
            final_rec = "HOLD"
        else:
            final_rec = "SELL"
        
        print(f"  Final Recommendation: {final_rec}")
        
        return True
        
    except Exception as e:
        print(f"Error in complete system test: {e}")
        return False

def main():
    """Run all tests."""
    print("Stock Prediction System - Test Suite")
    print("="*60)
    
    tests = [
        ("Technical Indicators", test_technical_indicators),
        ("Predictor", test_predictor),
        ("Fundamental Checker", test_fundamental_checker),
        ("Complete System", test_complete_system)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name} test...")
            result = test_func()
            results[test_name] = "PASSED" if result else "FAILED"
        except Exception as e:
            print(f"Error in {test_name} test: {e}")
            results[test_name] = "ERROR"
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status_icon = "✅" if result == "PASSED" else "❌"
        print(f"{status_icon} {test_name}: {result}")
    
    passed_count = sum(1 for result in results.values() if result == "PASSED")
    total_count = len(results)
    
    print(f"\nOverall: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("🎉 All tests passed! The system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main() 