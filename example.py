#!/usr/bin/env python3
"""
Example script demonstrating how to use the Stock Prediction System programmatically.
"""

from predictor import StockPredictor
from checker import FundamentalChecker

def example_technical_analysis():
    """Example of technical analysis and prediction."""
    print("=== TECHNICAL ANALYSIS EXAMPLE ===")
    
    # Initialize predictor
    predictor = StockPredictor(model_type='xgboost')
    
    # Fetch and prepare data
    data = predictor.fetch_data('AAPL', period='1y')
    if not data.empty:
        data = predictor.prepare_features(data)
        
        # Train model
        results = predictor.train_model(data)
        predictor.print_results(results)
        
        # Make prediction
        prediction = predictor.predict_next_day('AAPL')
        predictor.print_prediction(prediction)
    else:
        print("Failed to fetch data")

def example_fundamental_analysis():
    """Example of fundamental analysis."""
    print("\n=== FUNDAMENTAL ANALYSIS EXAMPLE ===")
    
    # Initialize checker
    checker = FundamentalChecker()
    
    # Analyze multiple stocks
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    for ticker in stocks:
        print(f"\nAnalyzing {ticker}...")
        evaluation = checker.evaluate_stock(ticker)
        checker.print_evaluation(evaluation)

def example_combined_analysis():
    """Example of combined technical and fundamental analysis."""
    print("\n=== COMBINED ANALYSIS EXAMPLE ===")
    
    ticker = 'AAPL'
    
    # Technical analysis
    predictor = StockPredictor(model_type='randomforest')
    data = predictor.fetch_data(ticker, period='1y')
    
    if not data.empty:
        data = predictor.prepare_features(data)
        results = predictor.train_model(data)
        prediction = predictor.predict_next_day(ticker)
        
        # Fundamental analysis
        checker = FundamentalChecker()
        evaluation = checker.evaluate_stock(ticker)
        
        # Print combined results
        print(f"\nCombined Analysis for {ticker}:")
        print(f"Technical Prediction: {prediction['prediction']}")
        print(f"Technical Confidence: {prediction['confidence']:.2%}")
        print(f"Fundamental Rating: {evaluation['overall_rating']}")
        print(f"Fundamental Score: {evaluation['overall_score']:.2f}/1.00")
        
        # Combined recommendation
        if prediction['prediction'] == 'UP':
            tech_score = prediction['probability_up']
        else:
            tech_score = prediction['probability_down']
        
        fund_score = evaluation['overall_score']
        combined_score = 0.7 * tech_score + 0.3 * fund_score
        
        print(f"Combined Score: {combined_score:.2%}")
        
        if combined_score >= 0.7:
            recommendation = "STRONG BUY"
        elif combined_score >= 0.6:
            recommendation = "BUY"
        elif combined_score >= 0.4:
            recommendation = "HOLD"
        else:
            recommendation = "SELL"
        
        print(f"Final Recommendation: {recommendation}")

if __name__ == "__main__":
    print("Stock Prediction System - Example Usage")
    print("=" * 50)
    
    try:
        # Run examples
        example_technical_analysis()
        example_fundamental_analysis()
        example_combined_analysis()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have installed all dependencies:")
        print("pip install -r requirements.txt") 