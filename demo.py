#!/usr/bin/env python3
"""
Simple demo script for the Stock Prediction System.
This script demonstrates the core functionality without external API dependencies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from predictor import StockPredictor
from checker import FundamentalChecker
from utils.indicators import add_all_technical_indicators

def create_demo_data():
    """Create realistic demo stock data."""
    print("Creating demo stock data...")
    
    # Generate 2 years of daily data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create realistic price movements
    np.random.seed(42)
    base_price = 150.0
    
    # Generate price series with trend and volatility
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Slight upward trend
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 10.0))
    
    # Create OHLC data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        daily_vol = np.random.uniform(0.01, 0.04)
        
        open_price = price * (1 + np.random.uniform(-daily_vol/2, daily_vol/2))
        high_price = max(open_price, price) * (1 + np.random.uniform(0, daily_vol/2))
        low_price = min(open_price, price) * (1 - np.random.uniform(0, daily_vol/2))
        close_price = price
        
        volume = np.random.randint(5000000, 50000000)
        
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
    
    print(f"Generated {len(df)} days of demo data")
    return df

def demo_technical_analysis():
    """Demonstrate technical analysis capabilities."""
    print("\n" + "="*60)
    print("TECHNICAL ANALYSIS DEMO")
    print("="*60)
    
    # Create demo data
    data = create_demo_data()
    
    # Add technical indicators
    print("Adding technical indicators...")
    data_with_indicators = add_all_technical_indicators(data)
    
    # Show available indicators
    indicator_columns = [col for col in data_with_indicators.columns 
                        if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    
    print(f"\nTechnical indicators added: {len(indicator_columns)}")
    print("Key indicators:")
    for i, indicator in enumerate(indicator_columns[:10], 1):
        print(f"  {i:2d}. {indicator}")
    
    # Show sample data
    print(f"\nSample data with indicators:")
    sample_cols = ['Close', 'Daily_Return', 'MA_5', 'MA_20', 'RSI', 'Volatility']
    available_cols = [col for col in sample_cols if col in data_with_indicators.columns]
    print(data_with_indicators[available_cols].tail(3))
    
    return data_with_indicators

def demo_machine_learning():
    """Demonstrate machine learning prediction."""
    print("\n" + "="*60)
    print("MACHINE LEARNING DEMO")
    print("="*60)
    
    # Create demo data
    data = create_demo_data()
    
    # Initialize predictor
    predictor = StockPredictor(model_type='xgboost')
    
    # Prepare features
    print("Preparing features for machine learning...")
    data = predictor.prepare_features(data)
    
    # Train model
    print("Training XGBoost model...")
    results = predictor.train_model(data)
    
    # Show results
    print(f"\nModel Performance:")
    print(f"  Accuracy: {results['accuracy']:.2%}")
    print(f"  Training samples: {results['train_samples']}")
    print(f"  Test samples: {results['test_samples']}")
    print(f"  Features used: {results['feature_count']}")
    
    # Show top features
    print(f"\nTop 5 most important features:")
    feature_importance = results['feature_importance']
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:5], 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    return predictor, results

def demo_fundamental_analysis():
    """Demonstrate fundamental analysis capabilities."""
    print("\n" + "="*60)
    print("FUNDAMENTAL ANALYSIS DEMO")
    print("="*60)
    
    # Initialize checker
    checker = FundamentalChecker()
    
    # Demonstrate evaluation logic
    print("Fundamental analysis evaluation criteria:")
    
    print(f"\nPE Ratio Evaluation:")
    pe_examples = [12, 18, 28, 45, 75]
    for pe in pe_examples:
        eval_result, score = checker.evaluate_pe_ratio(pe)
        print(f"  PE {pe:2d}: {eval_result:12s} (Score: {score:.2f})")
    
    print(f"\nROE Evaluation:")
    roe_examples = [0.25, 0.18, 0.12, 0.08, 0.03]
    for roe in roe_examples:
        eval_result, score = checker.evaluate_roe(roe)
        print(f"  ROE {roe*100:5.1f}%: {eval_result:12s} (Score: {score:.2f})")
    
    print(f"\nEarnings Growth Evaluation:")
    growth_examples = [0.25, 0.15, 0.08, 0.03, -0.05]
    for growth in growth_examples:
        eval_result, score = checker.evaluate_earnings_growth(growth)
        print(f"  Growth {growth*100:5.1f}%: {eval_result:12s} (Score: {score:.2f})")
    
    # Show how overall scoring works
    print(f"\nOverall Scoring Example:")
    print(f"  PE Ratio (18): Good (0.80)")
    print(f"  ROE (18%): Very Good (0.90)")
    print(f"  Growth (15%): Very Good (0.90)")
    print(f"  Average Score: {(0.80 + 0.90 + 0.90) / 3:.2f}")
    print(f"  Recommendation: Buy")

def demo_combined_analysis():
    """Demonstrate combined technical and fundamental analysis."""
    print("\n" + "="*60)
    print("COMBINED ANALYSIS DEMO")
    print("="*60)
    
    # Technical analysis results (simulated)
    tech_prediction = "UP"
    tech_confidence = 0.65
    tech_prob_up = 0.65
    tech_prob_down = 0.35
    
    # Fundamental analysis results (simulated)
    fund_score = 0.75
    fund_rating = "Good"
    fund_recommendation = "Buy"
    
    print(f"Technical Analysis:")
    print(f"  Prediction: {tech_prediction}")
    print(f"  Confidence: {tech_confidence:.2%}")
    print(f"  Probability UP: {tech_prob_up:.2%}")
    print(f"  Probability DOWN: {tech_prob_down:.2%}")
    
    print(f"\nFundamental Analysis:")
    print(f"  Overall Score: {fund_score:.2f}/1.00")
    print(f"  Rating: {fund_rating}")
    print(f"  Recommendation: {fund_recommendation}")
    
    # Combined analysis
    tech_score = tech_prob_up if tech_prediction == "UP" else tech_prob_down
    combined_score = 0.7 * tech_score + 0.3 * fund_score
    
    print(f"\nCombined Analysis:")
    print(f"  Technical Weight: 70%")
    print(f"  Fundamental Weight: 30%")
    print(f"  Technical Score: {tech_score:.2%}")
    print(f"  Fundamental Score: {fund_score:.2%}")
    print(f"  Combined Score: {combined_score:.2%}")
    
    # Final recommendation
    if combined_score >= 0.7:
        final_rec = "STRONG BUY"
    elif combined_score >= 0.6:
        final_rec = "BUY"
    elif combined_score >= 0.4:
        final_rec = "HOLD"
    elif combined_score >= 0.3:
        final_rec = "SELL"
    else:
        final_rec = "STRONG SELL"
    
    print(f"  Final Recommendation: {final_rec}")

def main():
    """Run the complete demo."""
    print("Stock Market Prediction System - Demo")
    print("="*60)
    print("This demo shows the core functionality of the system")
    print("using simulated data (no external API calls required)")
    
    try:
        # Run all demos
        demo_technical_analysis()
        demo_machine_learning()
        demo_fundamental_analysis()
        demo_combined_analysis()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("The system demonstrates:")
        print("✅ Technical indicator calculation")
        print("✅ Machine learning model training")
        print("✅ Feature importance analysis")
        print("✅ Fundamental analysis logic")
        print("✅ Combined analysis approach")
        print("\nTo use with real data, run:")
        print("python3 main.py --ticker AAPL --fundamental")
        
    except Exception as e:
        print(f"Error in demo: {e}")
        print("Please check the installation and dependencies")

if __name__ == "__main__":
    main() 