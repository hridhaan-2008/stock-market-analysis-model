#!/usr/bin/env python3
"""
BSE Stock Market Prediction System - Demo
=========================================

This demo shows how the system works with BSE (Bombay Stock Exchange) stocks
using realistic Indian stock data patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from predictor import StockPredictor
from checker import FundamentalChecker
from utils.indicators import add_all_technical_indicators

def create_realistic_bse_data(symbol: str, company_name: str, days=500):
    """
    Create realistic BSE stock data based on Indian market patterns.
    
    Args:
        symbol (str): Stock symbol
        company_name (str): Company name
        days (int): Number of days of data
        
    Returns:
        pd.DataFrame: Realistic BSE stock data
    """
    print(f"Creating realistic BSE data for {symbol} ({company_name})...")
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Set realistic base prices for Indian stocks
    base_prices = {
        'RELIANCE': 2500,    # Reliance Industries
        'TCS': 3500,         # TCS
        'HDFCBANK': 1600,    # HDFC Bank
        'INFY': 1400,        # Infosys
        'ICICIBANK': 950,    # ICICI Bank
        'HINDUNILVR': 2400,  # HUL
        'ITC': 450,          # ITC
        'SBIN': 650,         # SBI
        'BHARTIARTL': 1100,  # Bharti Airtel
        'AXISBANK': 950,     # Axis Bank
        'KOTAKBANK': 1800,   # Kotak Bank
        'ASIANPAINT': 3200,  # Asian Paints
        'MARUTI': 9500,      # Maruti Suzuki
        'SUNPHARMA': 1200,   # Sun Pharma
        'TITAN': 3200,       # Titan
        'WIPRO': 450,        # Wipro
        'ULTRACEMCO': 8500,  # UltraTech Cement
        'TECHM': 1200,       # Tech Mahindra
        'NESTLEIND': 2400,   # Nestle India
        'POWERGRID': 280,    # Power Grid
        'BAJFINANCE': 6500,  # Bajaj Finance
        'NTPC': 180,         # NTPC
        'HCLTECH': 1100,     # HCL Tech
        'JSWSTEEL': 750,     # JSW Steel
        'ONGC': 180,         # ONGC
        'TATAMOTORS': 650,   # Tata Motors
        'ADANIENT': 2800,    # Adani Enterprises
        'ADANIPORTS': 1200,  # Adani Ports
        'COALINDIA': 450,    # Coal India
        'DRREDDY': 5500,     # Dr Reddy's
        'CIPLA': 1200,       # Cipla
        'EICHERMOT': 3500,   # Eicher Motors
        'HEROMOTOCO': 4500,  # Hero MotoCorp
        'DIVISLAB': 3800,    # Divi's Labs
        'SHREECEM': 25000,   # Shree Cement
        'BRITANNIA': 4800,   # Britannia
        'GRASIM': 1800,      # Grasim
        'BAJAJFINSV': 1600,  # Bajaj Finserv
        'HINDALCO': 450,     # Hindalco
        'TATASTEEL': 120,    # Tata Steel
        'UPL': 550,          # UPL
        'VEDL': 280,         # Vedanta
        'BPCL': 450,         # BPCL
        'IOC': 120,          # IOC
        'M&M': 1800,         # M&M
        'LT': 2800,          # L&T
        'INDUSINDBK': 1400,  # IndusInd Bank
        'SBILIFE': 1400,     # SBI Life
        'HDFC': 2800,        # HDFC
        'APOLLOHOSP': 5500,  # Apollo Hospitals
        'BAJAJ-AUTO': 4500,  # Bajaj Auto
        'TATACONSUM': 850,   # Tata Consumer
        'SBI': 650,          # SBI
    }
    
    base_price = base_prices.get(symbol, 1000)
    
    # Generate realistic Indian market patterns
    np.random.seed(hash(symbol) % 1000)  # Different seed for each stock
    
    # Indian market characteristics: higher volatility, rupee movements
    returns = np.random.normal(0.0008, 0.025, len(dates))  # Higher volatility than US
    
    # Add some sector-specific trends
    if 'BANK' in symbol:
        returns += np.random.normal(0.0002, 0.005, len(dates))  # Banking sector trend
    elif 'TECH' in symbol or symbol in ['TCS', 'INFY', 'WIPRO', 'HCLTECH']:
        returns += np.random.normal(0.0003, 0.008, len(dates))  # Tech sector trend
    elif 'PHARMA' in symbol or symbol in ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB']:
        returns += np.random.normal(0.0001, 0.012, len(dates))  # Pharma sector trend
    
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, base_price * 0.3))  # Don't go below 30% of base price
    
    # Create OHLC data with Indian market characteristics
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Indian markets have higher intraday volatility
        daily_vol = np.random.uniform(0.015, 0.045)  # 1.5% to 4.5% daily volatility
        
        open_price = price * (1 + np.random.uniform(-daily_vol/2, daily_vol/2))
        high_price = max(open_price, price) * (1 + np.random.uniform(0, daily_vol/2))
        low_price = min(open_price, price) * (1 - np.random.uniform(0, daily_vol/2))
        close_price = price
        
        # Indian stocks typically have higher volume
        volume = np.random.randint(1000000, 50000000)
        
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
    
    print(f"Generated {len(df)} days of realistic BSE data")
    print(f"Price range: ₹{df['Close'].min():.2f} - ₹{df['Close'].max():.2f}")
    print(f"Current price: ₹{df['Close'].iloc[-1]:.2f}")
    
    return df

def demo_bse_technical_analysis():
    """Demonstrate technical analysis for BSE stocks."""
    print("\n" + "="*60)
    print("BSE TECHNICAL ANALYSIS DEMO")
    print("="*60)
    
    # Test with multiple BSE stocks
    bse_stocks = [
        ('RELIANCE', 'Reliance Industries Ltd'),
        ('TCS', 'Tata Consultancy Services Ltd'),
        ('HDFCBANK', 'HDFC Bank Ltd'),
        ('INFY', 'Infosys Ltd'),
        ('ICICIBANK', 'ICICI Bank Ltd')
    ]
    
    for symbol, company_name in bse_stocks:
        print(f"\n--- Analyzing {symbol} ({company_name}) ---")
        
        # Create realistic data
        data = create_realistic_bse_data(symbol, company_name, 400)
        
        # Add technical indicators
        data_with_indicators = add_all_technical_indicators(data)
        
        # Show key indicators
        latest = data_with_indicators.iloc[-1]
        print(f"Current Price: ₹{latest['Close']:.2f}")
        print(f"5-day MA: ₹{latest['MA_5']:.2f}")
        print(f"20-day MA: ₹{latest['MA_20']:.2f}")
        print(f"RSI: {latest['RSI']:.2f}")
        print(f"Volatility: {latest['Volatility']:.4f}")
        
        # Simple trend analysis
        if latest['Close'] > latest['MA_20']:
            trend = "BULLISH"
        else:
            trend = "BEARISH"
        
        if latest['RSI'] > 70:
            rsi_signal = "OVERBOUGHT"
        elif latest['RSI'] < 30:
            rsi_signal = "OVERSOLD"
        else:
            rsi_signal = "NEUTRAL"
        
        print(f"Trend: {trend}")
        print(f"RSI Signal: {rsi_signal}")

def demo_bse_machine_learning():
    """Demonstrate machine learning for BSE stocks."""
    print("\n" + "="*60)
    print("BSE MACHINE LEARNING DEMO")
    print("="*60)
    
    # Test with a major BSE stock
    symbol = 'RELIANCE'
    company_name = 'Reliance Industries Ltd'
    
    print(f"Training ML model for {symbol} ({company_name})...")
    
    # Create realistic data
    data = create_realistic_bse_data(symbol, company_name, 500)
    
    # Initialize predictor
    predictor = StockPredictor(model_type='xgboost')
    
    # Prepare features
    data = predictor.prepare_features(data)
    
    # Train model
    results = predictor.train_model(data)
    
    # Show results
    print(f"\nModel Performance for {symbol}:")
    print(f"  Accuracy: {results['accuracy']:.2%}")
    print(f"  Training samples: {results['train_samples']}")
    print(f"  Test samples: {results['test_samples']}")
    print(f"  Features used: {results['feature_count']}")
    
    # Show top features
    print(f"\nTop 5 most important features for {symbol}:")
    feature_importance = results['feature_importance']
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:5], 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    # Make prediction
    prediction = predictor.predict_next_day(symbol)
    print(f"\nPrediction for {symbol}:")
    print(f"  Direction: {prediction['prediction']}")
    print(f"  Confidence: {prediction['confidence']:.2%}")
    print(f"  Probability UP: {prediction['probability_up']:.2%}")
    print(f"  Probability DOWN: {prediction['probability_down']:.2%}")

def demo_bse_fundamental_analysis():
    """Demonstrate fundamental analysis for BSE stocks."""
    print("\n" + "="*60)
    print("BSE FUNDAMENTAL ANALYSIS DEMO")
    print("="*60)
    
    # Initialize checker
    checker = FundamentalChecker()
    
    # Test with realistic Indian company metrics
    indian_companies = [
        ('RELIANCE', 25.5, 0.18, 0.15),    # (symbol, pe_ratio, roe, earnings_growth)
        ('TCS', 28.2, 0.35, 0.12),
        ('HDFCBANK', 18.5, 0.16, 0.08),
        ('INFY', 22.8, 0.28, 0.10),
        ('ICICIBANK', 15.2, 0.14, 0.06)
    ]
    
    print("Fundamental Analysis for Indian Companies:")
    print("-" * 50)
    
    for symbol, pe_ratio, roe, earnings_growth in indian_companies:
        print(f"\n{symbol}:")
        
        # Evaluate each metric
        pe_eval, pe_score = checker.evaluate_pe_ratio(pe_ratio)
        roe_eval, roe_score = checker.evaluate_roe(roe)
        growth_eval, growth_score = checker.evaluate_earnings_growth(earnings_growth)
        
        # Calculate overall score
        overall_score = (pe_score + roe_score + growth_score) / 3
        
        print(f"  PE Ratio: {pe_ratio:.1f} ({pe_eval})")
        print(f"  ROE: {roe*100:.1f}% ({roe_eval})")
        print(f"  Earnings Growth: {earnings_growth*100:.1f}% ({growth_eval})")
        print(f"  Overall Score: {overall_score:.2f}/1.00")
        
        # Determine recommendation
        if overall_score >= 0.8:
            recommendation = "STRONG BUY"
        elif overall_score >= 0.6:
            recommendation = "BUY"
        elif overall_score >= 0.4:
            recommendation = "HOLD"
        elif overall_score >= 0.2:
            recommendation = "SELL"
        else:
            recommendation = "STRONG SELL"
        
        print(f"  Recommendation: {recommendation}")

def demo_bse_combined_analysis():
    """Demonstrate combined analysis for BSE stocks."""
    print("\n" + "="*60)
    print("BSE COMBINED ANALYSIS DEMO")
    print("="*60)
    
    # Analyze multiple BSE stocks
    stocks = [
        ('RELIANCE', 'Reliance Industries Ltd'),
        ('TCS', 'Tata Consultancy Services Ltd'),
        ('HDFCBANK', 'HDFC Bank Ltd')
    ]
    
    for symbol, company_name in stocks:
        print(f"\n--- {symbol} ({company_name}) ---")
        
        # Simulate technical analysis results
        tech_prediction = np.random.choice(['UP', 'DOWN'])
        tech_confidence = np.random.uniform(0.55, 0.75)
        tech_prob_up = tech_confidence if tech_prediction == 'UP' else 1 - tech_confidence
        tech_prob_down = 1 - tech_prob_up
        
        # Simulate fundamental analysis results
        fund_score = np.random.uniform(0.4, 0.9)
        
        print(f"Technical Analysis:")
        print(f"  Prediction: {tech_prediction}")
        print(f"  Confidence: {tech_confidence:.2%}")
        print(f"  Probability UP: {tech_prob_up:.2%}")
        print(f"  Probability DOWN: {tech_prob_down:.2%}")
        
        print(f"Fundamental Analysis:")
        print(f"  Overall Score: {fund_score:.2f}/1.00")
        
        # Combined analysis
        tech_score = tech_prob_up if tech_prediction == 'UP' else tech_prob_down
        combined_score = 0.7 * tech_score + 0.3 * fund_score
        
        print(f"Combined Analysis:")
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
    """Run the complete BSE demo."""
    print("BSE Stock Market Prediction System - Demo")
    print("="*60)
    print("This demo shows how the system works with Indian BSE stocks")
    print("using realistic market patterns and data")
    
    try:
        # Run all demos
        demo_bse_technical_analysis()
        demo_bse_machine_learning()
        demo_bse_fundamental_analysis()
        demo_bse_combined_analysis()
        
        print("\n" + "="*60)
        print("BSE DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("The system demonstrates:")
        print("✅ BSE stock data handling")
        print("✅ Indian market patterns")
        print("✅ Technical analysis for Indian stocks")
        print("✅ Machine learning with BSE data")
        print("✅ Fundamental analysis for Indian companies")
        print("✅ Combined analysis approach")
        print("\nTo use with real BSE data, run:")
        print("python3 main_bse.py --ticker RELIANCE --fundamental")
        print("\nAvailable BSE symbols:")
        print("python3 main_bse.py --list-symbols")
        
    except Exception as e:
        print(f"Error in BSE demo: {e}")
        print("Please check the installation and dependencies")

if __name__ == "__main__":
    main()
