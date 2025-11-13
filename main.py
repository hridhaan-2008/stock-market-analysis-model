#!/usr/bin/env python3
"""
Stock Market Prediction System
==============================

A comprehensive stock market prediction system that combines:
1. Technical analysis with machine learning
2. Fundamental analysis
3. Real-time predictions

Usage:
    python main.py --ticker AAPL --model xgboost
    python main.py --ticker MSFT --model randomforest --fundamental
"""

import argparse
import sys
import time
from typing import Dict, Optional

from predictor import StockPredictor
from checker import FundamentalChecker

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stock Market Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ticker AAPL
  python main.py --ticker MSFT --model randomforest
  python main.py --ticker GOOGL --fundamental
  python main.py --ticker TSLA --model xgboost --fundamental --period 5y
        """
    )
    
    parser.add_argument(
        '--ticker', 
        type=str, 
        default='AAPL',
        help='Stock ticker symbol (default: AAPL)'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['xgboost', 'randomforest'],
        default='xgboost',
        help='Machine learning model to use (default: xgboost)'
    )
    
    parser.add_argument(
        '--period', 
        type=str, 
        default='2y',
        help='Data period for training (default: 2y)'
    )
    
    parser.add_argument(
        '--fundamental', 
        action='store_true',
        help='Include fundamental analysis'
    )
    
    parser.add_argument(
        '--predict-only', 
        action='store_true',
        help='Skip training and only make prediction (requires pre-trained model)'
    )
    
    return parser.parse_args()

def run_technical_analysis(ticker: str, model_type: str, period: str, predict_only: bool = False) -> Optional[Dict]:
    """
    Run technical analysis and machine learning prediction.
    
    Args:
        ticker (str): Stock ticker symbol
        model_type (str): Type of model to use
        period (str): Data period for training
        predict_only (bool): Whether to skip training
        
    Returns:
        Optional[Dict]: Prediction results if predict_only is True
    """
    print(f"\n{'='*60}")
    print(f"TECHNICAL ANALYSIS & MACHINE LEARNING")
    print(f"{'='*60}")
    
    # Initialize predictor
    predictor = StockPredictor(model_type=model_type)
    
    if predict_only:
        # Try to load a pre-trained model (this would need to be implemented)
        print("Predict-only mode not implemented yet. Training model...")
        predict_only = False
    
    if not predict_only:
        # Fetch and prepare data
        data = predictor.fetch_data(ticker, period=period)
        if data.empty:
            print(f"Failed to fetch data for {ticker}")
            return None
        
        # Prepare features
        data = predictor.prepare_features(data)
        
        # Train model
        try:
            results = predictor.train_model(data)
            predictor.print_results(results)
        except Exception as e:
            print(f"Error training model: {e}")
            return None
    
    # Make prediction
    try:
        prediction = predictor.predict_next_day(ticker)
        predictor.print_prediction(prediction)
        return prediction
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def run_fundamental_analysis(ticker: str):
    """
    Run fundamental analysis.
    
    Args:
        ticker (str): Stock ticker symbol
    """
    print(f"\n{'='*60}")
    print(f"FUNDAMENTAL ANALYSIS")
    print(f"{'='*60}")
    
    # Initialize checker
    checker = FundamentalChecker()
    
    # Run evaluation
    try:
        evaluation = checker.evaluate_stock(ticker)
        checker.print_evaluation(evaluation)
        return evaluation
    except Exception as e:
        print(f"Error in fundamental analysis: {e}")
        return None

def print_summary(technical_result: Optional[Dict], fundamental_result: Optional[Dict]):
    """
    Print a summary combining technical and fundamental analysis.
    
    Args:
        technical_result (Optional[Dict]): Technical analysis results
        fundamental_result (Optional[Dict]): Fundamental analysis results
    """
    print(f"\n{'='*60}")
    print("COMBINED ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    if technical_result and 'error' not in technical_result:
        print(f"\nTechnical Analysis:")
        print(f"  Prediction: {technical_result['prediction']}")
        print(f"  Confidence: {technical_result['confidence']:.2%}")
        print(f"  Probability UP: {technical_result['probability_up']:.2%}")
        print(f"  Probability DOWN: {technical_result['probability_down']:.2%}")
    
    if fundamental_result:
        print(f"\nFundamental Analysis:")
        print(f"  Overall Score: {fundamental_result['overall_score']:.2f}/1.00")
        print(f"  Rating: {fundamental_result['overall_rating']}")
        print(f"  Recommendation: {fundamental_result['recommendation']}")
    
    # Combined recommendation
    print(f"\nCombined Recommendation:")
    
    if technical_result and fundamental_result and 'error' not in technical_result:
        tech_score = technical_result['probability_up'] if technical_result['prediction'] == 'UP' else technical_result['probability_down']
        fund_score = fundamental_result['overall_score']
        
        # Weighted average (70% technical, 30% fundamental)
        combined_score = 0.7 * tech_score + 0.3 * fund_score
        
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
        
        print(f"  Combined Score: {combined_score:.2%}")
        print(f"  Final Recommendation: {final_rec}")
    else:
        print("  Insufficient data for combined recommendation")
    
    print(f"{'='*60}")

def main():
    """Main function to orchestrate the stock prediction system."""
    args = parse_arguments()
    
    print(f"Stock Market Prediction System")
    print(f"Analyzing: {args.ticker.upper()}")
    print(f"Model: {args.model.upper()}")
    print(f"Period: {args.period}")
    
    technical_result = None
    fundamental_result = None
    
    try:
        # Run technical analysis
        technical_result = run_technical_analysis(
            args.ticker, 
            args.model, 
            args.period, 
            args.predict_only
        )
        
        # Run fundamental analysis if requested
        if args.fundamental:
            fundamental_result = run_fundamental_analysis(args.ticker)
        
        # Print combined summary
        print_summary(technical_result, fundamental_result)
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
    
    print(f"\nAnalysis completed successfully!")

if __name__ == "__main__":
    main() 