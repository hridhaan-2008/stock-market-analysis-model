import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

class FundamentalChecker:
    """
    Class to evaluate fundamental metrics of stocks.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def get_stock_info(self, ticker: str) -> Optional[yf.Ticker]:
        """
        Get stock information using yfinance.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            yf.Ticker: Stock ticker object or None if error
        """
        try:
            stock = yf.Ticker(ticker)
            return stock
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def get_financial_metrics(self, ticker: str) -> Dict:
        """
        Get key financial metrics for a stock.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict: Dictionary containing financial metrics
        """
        stock = self.get_stock_info(ticker)
        if not stock:
            return {}
        
        metrics = {}
        
        try:
            # Get basic info
            info = stock.info
            
            # PE Ratio
            metrics['pe_ratio'] = info.get('trailingPE', None)
            metrics['forward_pe'] = info.get('forwardPE', None)
            
            # Return on Equity
            metrics['roe'] = info.get('returnOnEquity', None)
            
            # Earnings Growth
            metrics['earnings_growth'] = info.get('earningsQuarterlyGrowth', None)
            
            # Additional metrics
            metrics['market_cap'] = info.get('marketCap', None)
            metrics['price_to_book'] = info.get('priceToBook', None)
            metrics['debt_to_equity'] = info.get('debtToEquity', None)
            metrics['current_ratio'] = info.get('currentRatio', None)
            metrics['profit_margins'] = info.get('profitMargins', None)
            metrics['revenue_growth'] = info.get('revenueGrowth', None)
            
            # Dividend info
            metrics['dividend_yield'] = info.get('dividendYield', None)
            metrics['payout_ratio'] = info.get('payoutRatio', None)
            
        except Exception as e:
            print(f"Error getting financial metrics for {ticker}: {e}")
            return {}
        
        return metrics
    
    def evaluate_pe_ratio(self, pe_ratio: float) -> Tuple[str, float]:
        """
        Evaluate PE ratio.
        
        Args:
            pe_ratio (float): PE ratio value
            
        Returns:
            Tuple[str, float]: (evaluation, score)
        """
        if pe_ratio is None:
            return "No data", 0.0
        
        if pe_ratio < 0:
            return "Negative PE (Loss)", 0.0
        elif pe_ratio < 15:
            return "Excellent", 1.0
        elif pe_ratio < 25:
            return "Good", 0.8
        elif pe_ratio < 35:
            return "Fair", 0.6
        elif pe_ratio < 50:
            return "High", 0.4
        else:
            return "Very High", 0.2
    
    def evaluate_roe(self, roe: float) -> Tuple[str, float]:
        """
        Evaluate Return on Equity.
        
        Args:
            roe (float): ROE value
            
        Returns:
            Tuple[str, float]: (evaluation, score)
        """
        if roe is None:
            return "No data", 0.0
        
        roe_pct = roe * 100  # Convert to percentage
        
        if roe_pct > 20:
            return "Excellent", 1.0
        elif roe_pct > 15:
            return "Very Good", 0.9
        elif roe_pct > 10:
            return "Good", 0.8
        elif roe_pct > 5:
            return "Fair", 0.6
        elif roe_pct > 0:
            return "Poor", 0.3
        else:
            return "Negative", 0.0
    
    def evaluate_earnings_growth(self, growth: float) -> Tuple[str, float]:
        """
        Evaluate earnings growth.
        
        Args:
            growth (float): Earnings growth value
            
        Returns:
            Tuple[str, float]: (evaluation, score)
        """
        if growth is None:
            return "No data", 0.0
        
        growth_pct = growth * 100  # Convert to percentage
        
        if growth_pct > 20:
            return "Excellent", 1.0
        elif growth_pct > 15:
            return "Very Good", 0.9
        elif growth_pct > 10:
            return "Good", 0.8
        elif growth_pct > 5:
            return "Fair", 0.6
        elif growth_pct > 0:
            return "Slow", 0.4
        else:
            return "Declining", 0.1
    
    def evaluate_stock(self, ticker: str) -> Dict:
        """
        Comprehensive evaluation of a stock's fundamentals.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict: Comprehensive evaluation results
        """
        metrics = self.get_financial_metrics(ticker)
        
        if not metrics:
            return {
                'ticker': ticker,
                'overall_score': 0.0,
                'overall_rating': 'No Data Available',
                'recommendation': 'Cannot evaluate',
                'metrics': {},
                'evaluations': {}
            }
        
        # Evaluate each metric
        evaluations = {}
        
        # PE Ratio evaluation
        pe_eval, pe_score = self.evaluate_pe_ratio(metrics.get('pe_ratio'))
        evaluations['pe_ratio'] = {
            'value': metrics.get('pe_ratio'),
            'evaluation': pe_eval,
            'score': pe_score
        }
        
        # ROE evaluation
        roe_eval, roe_score = self.evaluate_roe(metrics.get('roe'))
        evaluations['roe'] = {
            'value': metrics.get('roe'),
            'evaluation': roe_eval,
            'score': roe_score
        }
        
        # Earnings Growth evaluation
        growth_eval, growth_score = self.evaluate_earnings_growth(metrics.get('earnings_growth'))
        evaluations['earnings_growth'] = {
            'value': metrics.get('earnings_growth'),
            'evaluation': growth_eval,
            'score': growth_score
        }
        
        # Calculate overall score (weighted average)
        scores = [pe_score, roe_score, growth_score]
        valid_scores = [s for s in scores if s > 0]
        
        if valid_scores:
            overall_score = sum(valid_scores) / len(valid_scores)
        else:
            overall_score = 0.0
        
        # Determine overall rating
        if overall_score >= 0.8:
            overall_rating = "Excellent"
            recommendation = "Strong Buy"
        elif overall_score >= 0.6:
            overall_rating = "Good"
            recommendation = "Buy"
        elif overall_score >= 0.4:
            overall_rating = "Fair"
            recommendation = "Hold"
        elif overall_score >= 0.2:
            overall_rating = "Poor"
            recommendation = "Sell"
        else:
            overall_rating = "Very Poor"
            recommendation = "Strong Sell"
        
        return {
            'ticker': ticker,
            'overall_score': overall_score,
            'overall_rating': overall_rating,
            'recommendation': recommendation,
            'metrics': metrics,
            'evaluations': evaluations
        }
    
    def print_evaluation(self, evaluation: Dict):
        """
        Print a formatted evaluation report.
        
        Args:
            evaluation (Dict): Evaluation results
        """
        print(f"\n{'='*60}")
        print(f"FUNDAMENTAL ANALYSIS: {evaluation['ticker']}")
        print(f"{'='*60}")
        
        print(f"\nOverall Score: {evaluation['overall_score']:.2f}/1.00")
        print(f"Overall Rating: {evaluation['overall_rating']}")
        print(f"Recommendation: {evaluation['recommendation']}")
        
        print(f"\n{'='*40}")
        print("METRIC EVALUATIONS:")
        print(f"{'='*40}")
        
        for metric, eval_data in evaluation['evaluations'].items():
            value = eval_data['value']
            if value is not None:
                if metric == 'pe_ratio':
                    display_value = f"{value:.2f}"
                elif metric == 'roe':
                    display_value = f"{value*100:.2f}%"
                elif metric == 'earnings_growth':
                    display_value = f"{value*100:.2f}%"
                else:
                    display_value = str(value)
            else:
                display_value = "N/A"
            
            print(f"{metric.replace('_', ' ').title()}: {display_value}")
            print(f"  Evaluation: {eval_data['evaluation']}")
            print(f"  Score: {eval_data['score']:.2f}/1.00")
            print()
        
        print(f"{'='*60}") 