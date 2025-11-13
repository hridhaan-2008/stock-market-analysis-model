# Stock Market Prediction System

A comprehensive stock market prediction system that combines technical analysis with machine learning and fundamental analysis to provide stock price predictions and investment recommendations.

## Features

### 🚀 Technical Analysis & Machine Learning
- **Historical Data Fetching**: Uses `yfinance` to fetch real-time stock data
- **Feature Engineering**: 
  - Daily returns and price changes
  - Moving averages (5-day, 20-day)
  - Volatility indicators (10-day rolling standard deviation)
  - RSI, Bollinger Bands, MACD
  - Price momentum and lag features
- **Binary Classification**: Predicts if stock will go UP or DOWN the next day
- **Machine Learning Models**: 
  - XGBoost (default)
  - Random Forest
- **Model Evaluation**: Accuracy scores, classification reports, feature importance

### 📊 Fundamental Analysis
- **PE Ratio Analysis**: Evaluates price-to-earnings ratios
- **Return on Equity (ROE)**: Analyzes profitability efficiency
- **Earnings Growth**: Assesses quarterly earnings growth
- **Comprehensive Scoring**: Combines metrics for overall stock rating
- **Investment Recommendations**: Strong Buy to Strong Sell ratings

### 🔧 System Architecture
- **Modular Design**: Organized into separate modules for maintainability
- **Command Line Interface**: Easy-to-use CLI with various options
- **Error Handling**: Robust error handling and data validation
- **Extensible**: Easy to add new features and indicators

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd stock-price-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python main.py --ticker AAPL --help
   ```

## Usage

### Basic Usage

```bash
# Analyze Apple stock with XGBoost model
python main.py --ticker AAPL

# Use Random Forest model
python main.py --ticker MSFT --model randomforest

# Include fundamental analysis
python main.py --ticker GOOGL --fundamental

# Use different time period for training
python main.py --ticker TSLA --period 5y --fundamental
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ticker` | Stock ticker symbol | AAPL |
| `--model` | ML model (xgboost/randomforest) | xgboost |
| `--period` | Training data period | 2y |
| `--fundamental` | Include fundamental analysis | False |
| `--predict-only` | Skip training, predict only | False |

### Example Output

```
============================================================
TECHNICAL ANALYSIS & MACHINE LEARNING
============================================================

Fetching data for AAPL...
Successfully fetched 504 days of data
Preparing features...
Selected 45 features out of 52
Training model...
Training set: 403 samples
Test set: 101 samples

============================================================
MODEL TRAINING RESULTS
============================================================

Model Type: XGBOOST
Accuracy: 0.6238 (62.38%)
Training Samples: 403
Test Samples: 101
Features Used: 45

========================================
CLASSIFICATION REPORT:
========================================

Class 0:
  precision: 0.6000
  recall: 0.6000
  f1-score: 0.6000

Class 1:
  precision: 0.6429
  recall: 0.6429
  f1-score: 0.6429

========================================
TOP 10 FEATURE IMPORTANCE:
========================================
 1. RSI: 0.0892
 2. Volatility: 0.0856
 3. Price_Momentum_5: 0.0823
 4. MA_5: 0.0789
 5. Close_Lag_1: 0.0754
...

============================================================
PREDICTION FOR AAPL
============================================================
Current Price: $175.43
Prediction: UP
Confidence: 65.23%

Probabilities:
  UP:   65.23%
  DOWN: 34.77%

Confidence Level: Medium
Recommendation: Consider buying
Timestamp: 2024-01-15 14:30:25
============================================================
```

## Project Structure

```
stock-price-prediction/
├── main.py                 # Main orchestration script
├── predictor.py            # Machine learning prediction logic
├── checker.py              # Fundamental analysis logic
├── utils/
│   └── indicators.py       # Technical indicators
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Technical Details

### Feature Engineering

The system creates the following features:

**Basic Indicators:**
- Daily returns
- Moving averages (5, 20 days)
- Volatility (10-day rolling std)

**Advanced Indicators:**
- RSI (Relative Strength Index)
- Bollinger Bands
- MACD (Moving Average Convergence Divergence)

**Additional Features:**
- Price momentum (5, 10, 20 days)
- Volume changes and ratios
- Lag features (1, 2, 3, 5 days)
- Rolling statistics

### Model Training

1. **Data Preparation**: Fetches historical data and adds technical indicators
2. **Feature Selection**: Removes features with too many NaN values
3. **Target Creation**: Binary classification (1 if next day's close > current close)
4. **Train/Test Split**: 80/20 split without shuffling (maintains temporal order)
5. **Model Training**: XGBoost or Random Forest with optimized parameters
6. **Evaluation**: Accuracy, precision, recall, F1-score, feature importance

### Fundamental Analysis

**Metrics Evaluated:**
- **PE Ratio**: < 15 (Excellent), < 25 (Good), < 35 (Fair), etc.
- **ROE**: > 20% (Excellent), > 15% (Very Good), > 10% (Good), etc.
- **Earnings Growth**: > 20% (Excellent), > 15% (Very Good), > 10% (Good), etc.

**Scoring System:**
- Overall score calculated as average of individual metric scores
- Recommendations: Strong Buy, Buy, Hold, Sell, Strong Sell

## Limitations and Disclaimers

⚠️ **Important Disclaimers:**

1. **Not Financial Advice**: This system is for educational purposes only. Do not make investment decisions based solely on these predictions.

2. **Past Performance**: Historical performance does not guarantee future results.

3. **Market Volatility**: Stock markets are inherently unpredictable and subject to various external factors.

4. **Model Limitations**: 
   - Models are trained on historical data and may not capture future market changes
   - Accuracy varies by stock and market conditions
   - No model can predict market crashes or unexpected events

5. **Data Quality**: Depends on the quality and availability of data from Yahoo Finance.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- New technical indicators
- Additional fundamental metrics
- Model improvements
- Bug fixes
- Documentation improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **yfinance**: For providing stock data
- **scikit-learn**: For machine learning algorithms
- **XGBoost**: For gradient boosting implementation
- **pandas & numpy**: For data manipulation

## Support

If you encounter any issues or have questions:

1. Check the existing issues on GitHub
2. Create a new issue with detailed information
3. Include error messages and system information

---

**Remember**: Always do your own research and consider consulting with a financial advisor before making investment decisions. 