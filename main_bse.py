#!/usr/bin/env python3
"""
BSE Stock Market Prediction System
==================================

A comprehensive stock market prediction system specifically for BSE (Bombay Stock Exchange) stocks.
Uses real BSE data and provides predictions for Indian stocks.

Usage:
    python main_bse.py --ticker RELIANCE --fundamental
    python main_bse.py --ticker TCS --model randomforest
"""

import argparse
import sys
import time
import pandas as pd
from typing import Dict, Optional

from predictor import StockPredictor
from checker import FundamentalChecker
from bse_data_fetcher import BSEDataFetcher

class BSEModifiedPredictor(StockPredictor):
    """
    Modified predictor for BSE stocks
    """
    
    def __init__(self, model_type='xgboost'):
        super().__init__(model_type)
        self.bse_fetcher = BSEDataFetcher()
    
    def fetch_data(self, ticker: str, period: str = '2y') -> pd.DataFrame:
        """
        Fetch BSE stock data using multiple methods.
        
        Args:
            ticker (str): BSE stock symbol (e.g., 'RELIANCE', 'TCS')
            period (str): Data period (e.g., '1y', '2y', '5y')
            
        Returns:
            pd.DataFrame: Historical BSE stock data
        """
        print(f"Fetching BSE data for {ticker}...")
        
        # Get symbol info
        symbol_info = self.bse_fetcher.get_bse_symbol_info(ticker)
        if symbol_info:
            print(f"Company: {symbol_info['name']}")
            print(f"BSE Code: {symbol_info['code']}")
        
        # Try multiple data sources
        data_sources = [
            ('Yahoo Finance', self.bse_fetcher.fetch_bse_data_yahoo_finance),
            ('NSE India API', self.bse_fetcher.fetch_bse_data_nseindia),
        ]
        
        for source_name, fetch_func in data_sources:
            try:
                print(f"Trying {source_name}...")
                data = fetch_func(ticker, period)
                if not data.empty:
                    print(f"Successfully fetched {len(data)} days of data from {source_name}")
                    return data
            except Exception as e:
                print(f"Error with {source_name}: {e}")
                continue
        
        print(f"Could not fetch data for {ticker} from any source")
        return pd.DataFrame()

class BSEModifiedChecker(FundamentalChecker):
    """
    Modified fundamental checker for BSE stocks
    """
    
    def __init__(self):
        super().__init__()
        self.bse_fetcher = BSEDataFetcher()
    
    def get_financial_metrics(self, ticker: str) -> Dict:
        """
        Get key financial metrics for a BSE stock.
        
        Args:
            ticker (str): BSE stock symbol
            
        Returns:
            Dict: Dictionary containing financial metrics
        """
        print(f"Fetching fundamental data for {ticker}...")
        
        # Try BSE-specific fundamental data
        metrics = self.bse_fetcher.get_bse_fundamental_data(ticker)
        
        if metrics:
            return metrics
        
        # Fall back to original method
        return super().get_financial_metrics(ticker)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BSE Stock Market Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_bse.py --ticker RELIANCE --fundamental
  python main_bse.py --ticker TCS --model randomforest
  python main_bse.py --ticker HDFCBANK --period 5y --fundamental
  python main_bse.py --list-symbols
        """
    )
    
    parser.add_argument(
        '--ticker', 
        type=str, 
        default='RELIANCE',
        help='BSE stock symbol (default: RELIANCE)'
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
        '--list-symbols', 
        action='store_true',
        help='List available BSE symbols'
    )
    
    return parser.parse_args()

def list_bse_symbols():
    """List available BSE symbols."""
    fetcher = BSEDataFetcher()
    symbols = fetcher.list_available_bse_symbols()
    
    print("Available BSE Symbols:")
    print("=" * 50)
    
    for i, symbol in enumerate(symbols, 1):
        info = fetcher.get_bse_symbol_info(symbol)
        if info:
            print(f"{i:2d}. {symbol:12s} - {info['name']}")
        else:
            print(f"{i:2d}. {symbol:12s}")
    
    print(f"\nTotal: {len(symbols)} symbols available")
    print("\nUsage examples:")
    print("  python main_bse.py --ticker RELIANCE --fundamental")
    print("  python main_bse.py --ticker TCS --model randomforest")
    print("  python main_bse.py --ticker HDFCBANK --period 5y")

def run_technical_analysis(ticker: str, model_type: str, period: str) -> Optional[Dict]:
    """
    Run technical analysis and machine learning prediction for BSE stock.
    
    Args:
        ticker (str): BSE stock symbol
        model_type (str): Type of model to use
        period (str): Data period for training
        
    Returns:
        Optional[Dict]: Prediction results
    """
    print(f"\n{'='*60}")
    print(f"BSE TECHNICAL ANALYSIS & MACHINE LEARNING")
    print(f"{'='*60}")
    
    # Initialize BSE predictor
    predictor = BSEModifiedPredictor(model_type=model_type)
    
    # Fetch and prepare data
    data = predictor.fetch_data(ticker, period=period)
    if data.empty:
        print(f"Failed to fetch BSE data for {ticker}")
        return None
    
    # Prepare features
    data = predictor.prepare_features(data)
    
    # Train model
    try:
        results = predictor.train_model(data)
        predictor.print_results(results)
        
        # Make prediction
        prediction = predictor.predict_next_day(ticker)
        predictor.print_prediction(prediction)
        return prediction
        
    except Exception as e:
        print(f"Error in technical analysis: {e}")
        return None

def run_fundamental_analysis(ticker: str):
    """
    Run fundamental analysis for BSE stock.
    
    Args:
        ticker (str): BSE stock symbol
    """
    print(f"\n{'='*60}")
    print(f"BSE FUNDAMENTAL ANALYSIS")
    print(f"{'='*60}")
    
    # Initialize BSE checker
    checker = BSEModifiedChecker()
    
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
    print("BSE COMBINED ANALYSIS SUMMARY")
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
    """Main function to orchestrate the BSE stock prediction system."""
    args = parse_arguments()
    
    # Handle list symbols request
    if args.list_symbols:
        list_bse_symbols()
        return
    
    print(f"BSE Stock Market Prediction System")
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
            args.period
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
    
    print(f"\nBSE Analysis completed successfully!")

if __name__ == "__main__":
    main()
