#!/usr/bin/env python3
"""
BSE (Bombay Stock Exchange) Data Fetcher
Fetches real-time and historical data from BSE using various APIs
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

class BSEDataFetcher:
    """
    Fetches BSE stock data using various available APIs
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_bse_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Get BSE symbol information and mapping.
        
        Args:
            symbol (str): Stock symbol (e.g., 'RELIANCE', 'TCS')
            
        Returns:
            Optional[Dict]: Symbol information
        """
        # Common BSE symbols mapping
        bse_symbols = {
            'RELIANCE': {'code': '500325', 'name': 'Reliance Industries Ltd'},
            'TCS': {'code': '532540', 'name': 'Tata Consultancy Services Ltd'},
            'HDFCBANK': {'code': '500180', 'name': 'HDFC Bank Ltd'},
            'INFY': {'code': '500209', 'name': 'Infosys Ltd'},
            'ICICIBANK': {'code': '532174', 'name': 'ICICI Bank Ltd'},
            'HINDUNILVR': {'code': '500696', 'name': 'Hindustan Unilever Ltd'},
            'ITC': {'code': '500875', 'name': 'ITC Ltd'},
            'SBIN': {'code': '500112', 'name': 'State Bank of India'},
            'BHARTIARTL': {'code': '532454', 'name': 'Bharti Airtel Ltd'},
            'AXISBANK': {'code': '532215', 'name': 'Axis Bank Ltd'},
            'KOTAKBANK': {'code': '500247', 'name': 'Kotak Mahindra Bank Ltd'},
            'ASIANPAINT': {'code': '500820', 'name': 'Asian Paints Ltd'},
            'MARUTI': {'code': '532500', 'name': 'Maruti Suzuki India Ltd'},
            'SUNPHARMA': {'code': '524715', 'name': 'Sun Pharmaceutical Industries Ltd'},
            'TITAN': {'code': '500114', 'name': 'Titan Company Ltd'},
            'WIPRO': {'code': '507685', 'name': 'Wipro Ltd'},
            'ULTRACEMCO': {'code': '532538', 'name': 'UltraTech Cement Ltd'},
            'TECHM': {'code': '532755', 'name': 'Tech Mahindra Ltd'},
            'NESTLEIND': {'code': '500790', 'name': 'Nestle India Ltd'},
            'POWERGRID': {'code': '532898', 'name': 'Power Grid Corporation of India Ltd'},
            'BAJFINANCE': {'code': '500034', 'name': 'Bajaj Finance Ltd'},
            'NTPC': {'code': '532555', 'name': 'NTPC Ltd'},
            'HCLTECH': {'code': '532281', 'name': 'HCL Technologies Ltd'},
            'JSWSTEEL': {'code': '500228', 'name': 'JSW Steel Ltd'},
            'ONGC': {'code': '500312', 'name': 'Oil & Natural Gas Corporation Ltd'},
            'TATAMOTORS': {'code': '500570', 'name': 'Tata Motors Ltd'},
            'ADANIENT': {'code': '512599', 'name': 'Adani Enterprises Ltd'},
            'ADANIPORTS': {'code': '532921', 'name': 'Adani Ports & Special Economic Zone Ltd'},
            'COALINDIA': {'code': '533278', 'name': 'Coal India Ltd'},
            'DRREDDY': {'code': '500124', 'name': 'Dr. Reddy\'s Laboratories Ltd'},
            'CIPLA': {'code': '500087', 'name': 'Cipla Ltd'},
            'EICHERMOT': {'code': '505200', 'name': 'Eicher Motors Ltd'},
            'HEROMOTOCO': {'code': '500182', 'name': 'Hero MotoCorp Ltd'},
            'DIVISLAB': {'code': '532488', 'name': 'Divi\'s Laboratories Ltd'},
            'SHREECEM': {'code': '500387', 'name': 'Shree Cement Ltd'},
            'BRITANNIA': {'code': '500825', 'name': 'Britannia Industries Ltd'},
            'GRASIM': {'code': '500300', 'name': 'Grasim Industries Ltd'},
            'BAJAJFINSV': {'code': '532978', 'name': 'Bajaj Finserv Ltd'},
            'HINDALCO': {'code': '500440', 'name': 'Hindalco Industries Ltd'},
            'TATASTEEL': {'code': '500470', 'name': 'Tata Steel Ltd'},
            'UPL': {'code': '512070', 'name': 'UPL Ltd'},
            'VEDL': {'code': '500295', 'name': 'Vedanta Ltd'},
            'BPCL': {'code': '500547', 'name': 'Bharat Petroleum Corporation Ltd'},
            'IOC': {'code': '530965', 'name': 'Indian Oil Corporation Ltd'},
            'M&M': {'code': '500520', 'name': 'Mahindra & Mahindra Ltd'},
            'LT': {'code': '500510', 'name': 'Larsen & Toubro Ltd'},
            'INDUSINDBK': {'code': '532187', 'name': 'IndusInd Bank Ltd'},
            'SBILIFE': {'code': '540719', 'name': 'SBI Life Insurance Company Ltd'},
            'HDFC': {'code': '500010', 'name': 'Housing Development Finance Corporation Ltd'},
            'APOLLOHOSP': {'code': '508869', 'name': 'Apollo Hospitals Enterprise Ltd'},
            'BAJAJ-AUTO': {'code': '532977', 'name': 'Bajaj Auto Ltd'},
            'TATACONSUM': {'code': '500800', 'name': 'Tata Consumer Products Ltd'},
            'SBI': {'code': '500112', 'name': 'State Bank of India'},
            'WIPRO': {'code': '507685', 'name': 'Wipro Ltd'},
        }
        
        symbol_upper = symbol.upper()
        if symbol_upper in bse_symbols:
            return bse_symbols[symbol_upper]
        else:
            print(f"Symbol {symbol} not found in BSE mapping. Available symbols:")
            for sym in list(bse_symbols.keys())[:20]:  # Show first 20
                print(f"  {sym}")
            return None
    
    def fetch_bse_data_nseindia(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """
        Fetch BSE data using NSE India API (covers BSE stocks too)
        
        Args:
            symbol (str): Stock symbol
            period (str): Data period ('1y', '2y', '5y')
            
        Returns:
            pd.DataFrame: Historical stock data
        """
        try:
            # Convert period to days
            period_days = {
                '1y': 365,
                '2y': 730,
                '5y': 1825
            }.get(period, 365)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            # Format dates for API
            start_str = start_date.strftime('%d-%m-%Y')
            end_str = end_date.strftime('%d-%m-%Y')
            
            # NSE India API URL
            url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}&from={start_str}&to={end_str}"
            
            print(f"Fetching BSE data for {symbol} from {start_str} to {end_str}...")
            
            # Make request
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and data['data']:
                    # Convert to DataFrame
                    df_data = []
                    for item in data['data']:
                        df_data.append({
                            'Date': pd.to_datetime(item['date'], format='%d-%b-%Y'),
                            'Open': float(item['open']),
                            'High': float(item['high']),
                            'Low': float(item['low']),
                            'Close': float(item['close']),
                            'Volume': int(item['totalTradedVolume'])
                        })
                    
                    df = pd.DataFrame(df_data)
                    df.set_index('Date', inplace=True)
                    df.sort_index(inplace=True)
                    
                    print(f"Successfully fetched {len(df)} days of BSE data for {symbol}")
                    return df
                else:
                    print(f"No data found for {symbol}")
                    return pd.DataFrame()
            else:
                print(f"API request failed with status code: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching BSE data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_bse_data_alpha_vantage(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """
        Fetch BSE data using Alpha Vantage API (requires API key)
        
        Args:
            symbol (str): Stock symbol with .BSE suffix
            period (str): Data period
            
        Returns:
            pd.DataFrame: Historical stock data
        """
        try:
            # You would need to get a free API key from https://www.alphavantage.co/
            api_key = "demo"  # Replace with your actual API key
            
            # Add .BSE suffix for BSE stocks
            bse_symbol = f"{symbol}.BSE"
            
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={bse_symbol}&apikey={api_key}&outputsize=full"
            
            print(f"Fetching BSE data for {bse_symbol} using Alpha Vantage...")
            
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Time Series (Daily)' in data:
                    time_series = data['Time Series (Daily)']
                    
                    df_data = []
                    for date, values in time_series.items():
                        df_data.append({
                            'Date': pd.to_datetime(date),
                            'Open': float(values['1. open']),
                            'High': float(values['2. high']),
                            'Low': float(values['3. low']),
                            'Close': float(values['4. close']),
                            'Volume': int(values['5. volume'])
                        })
                    
                    df = pd.DataFrame(df_data)
                    df.set_index('Date', inplace=True)
                    df.sort_index(inplace=True)
                    
                    # Filter by period
                    if period == '1y':
                        df = df.last('365D')
                    elif period == '2y':
                        df = df.last('730D')
                    elif period == '5y':
                        df = df.last('1825D')
                    
                    print(f"Successfully fetched {len(df)} days of BSE data for {symbol}")
                    return df
                else:
                    print(f"No data found for {symbol}")
                    return pd.DataFrame()
            else:
                print(f"API request failed with status code: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching BSE data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_bse_data_yahoo_finance(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """
        Fetch BSE data using Yahoo Finance (some BSE stocks are available)
        
        Args:
            symbol (str): Stock symbol with .BO suffix for BSE
            period (str): Data period
            
        Returns:
            pd.DataFrame: Historical stock data
        """
        try:
            import yfinance as yf
            
            # Add .BO suffix for BSE stocks
            bse_symbol = f"{symbol}.BO"
            
            print(f"Fetching BSE data for {bse_symbol} using Yahoo Finance...")
            
            stock = yf.Ticker(bse_symbol)
            data = stock.history(period=period)
            
            if not data.empty:
                print(f"Successfully fetched {len(data)} days of BSE data for {symbol}")
                return data
            else:
                print(f"No data found for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching BSE data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_bse_fundamental_data(self, symbol: str) -> Dict:
        """
        Get fundamental data for BSE stocks
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: Fundamental metrics
        """
        try:
            # Try Yahoo Finance first
            import yfinance as yf
            
            bse_symbol = f"{symbol}.BO"
            stock = yf.Ticker(bse_symbol)
            info = stock.info
            
            metrics = {
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'roe': info.get('returnOnEquity', None),
                'earnings_growth': info.get('earningsQuarterlyGrowth', None),
                'market_cap': info.get('marketCap', None),
                'price_to_book': info.get('priceToBook', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'current_ratio': info.get('currentRatio', None),
                'profit_margins': info.get('profitMargins', None),
                'revenue_growth': info.get('revenueGrowth', None),
                'dividend_yield': info.get('dividendYield', None),
                'payout_ratio': info.get('payoutRatio', None)
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error getting fundamental data for {symbol}: {e}")
            return {}
    
    def list_available_bse_symbols(self) -> List[str]:
        """
        List available BSE symbols
        
        Returns:
            List[str]: List of available symbols
        """
        symbols = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR',
            'ITC', 'SBIN', 'BHARTIARTL', 'AXISBANK', 'KOTAKBANK', 'ASIANPAINT',
            'MARUTI', 'SUNPHARMA', 'TITAN', 'WIPRO', 'ULTRACEMCO', 'TECHM',
            'NESTLEIND', 'POWERGRID', 'BAJFINANCE', 'NTPC', 'HCLTECH',
            'JSWSTEEL', 'ONGC', 'TATAMOTORS', 'ADANIENT', 'ADANIPORTS',
            'COALINDIA', 'DRREDDY', 'CIPLA', 'EICHERMOT', 'HEROMOTOCO',
            'DIVISLAB', 'SHREECEM', 'BRITANNIA', 'GRASIM', 'BAJAJFINSV',
            'HINDALCO', 'TATASTEEL', 'UPL', 'VEDL', 'BPCL', 'IOC', 'M&M',
            'LT', 'INDUSINDBK', 'SBILIFE', 'HDFC', 'APOLLOHOSP', 'BAJAJ-AUTO',
            'TATACONSUM', 'SBI', 'WIPRO'
        ]
        
        return symbols

def main():
    """Test the BSE data fetcher"""
    fetcher = BSEDataFetcher()
    
    print("BSE Data Fetcher Test")
    print("=" * 50)
    
    # Test with a popular BSE stock
    symbol = 'RELIANCE'
    
    print(f"Testing with {symbol}...")
    
    # Get symbol info
    info = fetcher.get_bse_symbol_info(symbol)
    if info:
        print(f"Symbol: {symbol}")
        print(f"BSE Code: {info['code']}")
        print(f"Company: {info['name']}")
    
    # Try fetching data
    print(f"\nFetching data for {symbol}...")
    data = fetcher.fetch_bse_data_yahoo_finance(symbol, '1y')
    
    if not data.empty:
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Latest close price: ₹{data['Close'].iloc[-1]:.2f}")
        print(f"Sample data:")
        print(data.tail(3))
    else:
        print("No data available. Trying alternative methods...")
        
        # Try NSE India API
        data = fetcher.fetch_bse_data_nseindia(symbol, '1y')
        if not data.empty:
            print(f"Data fetched via NSE India API: {data.shape}")
        else:
            print("All methods failed. Please check your internet connection.")

if __name__ == "__main__":
    main()
