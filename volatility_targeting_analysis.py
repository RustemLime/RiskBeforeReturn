"""
Risk Before Return: Volatility Targeting Strategies Analysis
Replicates results from "Risk Before Return: Targeting Volatility with Higher Frequency Data"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from massive import RESTClient
from fmp_python.fmp import FMP
import requests
import warnings
import os
import pickle
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Configuration
MAIN_TICKER = "SPY"  # Main ticker symbol for analysis
TARGET_VOLATILITY = 0.10  # 10% annualized volatility target
TRADING_DAYS_PER_YEAR = 252
SCALE_FACTOR = 1.3  # For realized volatility calculation
MA_WINDOW = 200  # Moving average window for trend-following
HV_WINDOW = 30  # Historical volatility window
RV_WINDOW = 2  # Realized volatility window
TRANSACTION_COST = 0.00  # Transaction cost per unit of allocation change (default: 0.1%)

# Date range
START_DATE = "2003-12-01"
END_DATE = "2020-04-30"  # End of in-sample period (original paper period)
# Out-of-sample period extends from END_DATE to current date
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")

# Cache directory for downloaded data
CACHE_DIR = "data_cache"

# Output directories
REPORTS_DIR = "reports"
PLOTS_DIR = "plots"


class DataCollector:
    """Handles data collection from multiple sources (massive and FMP) with disk caching"""
    
    def __init__(self, massive_api_key=None, fmp_api_key=None, cache_dir=CACHE_DIR):
        """
        Initialize DataCollector with API keys for data sources
        
        Parameters:
        -----------
        massive_api_key : str, optional
            API key for Massive API. Required if using 'massive' source.
        fmp_api_key : str, optional
            API key for FMP API. Required if using 'fmp' source.
        cache_dir : str, optional
            Directory for caching downloaded data
        """
        self.massive_api_key = massive_api_key
        self.fmp_api_key = fmp_api_key
        self.cache_dir = cache_dir
        
        # Initialize clients only if API keys are provided
        if massive_api_key:
            self.client = RESTClient(massive_api_key)
        else:
            self.client = None
            
        if fmp_api_key:
            self.fmp = FMP(api_key=fmp_api_key)
        else:
            self.fmp = None
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"Created cache directory: {self.cache_dir}")
    
    def _get_cache_path(self, ticker, start_date, end_date, data_type="daily", source="massive"):
        """Generate cache file path"""
        # Sanitize ticker for filename (remove special characters)
        safe_ticker = ticker.replace("^", "").replace("/", "_")
        filename = f"{safe_ticker}_{start_date}_{end_date}_{data_type}_{source}.pkl"
        return os.path.join(self.cache_dir, filename)
    
    def _load_from_cache(self, cache_path):
        """Load data from cache if it exists"""
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"Loaded data from cache: {cache_path}")
                return data
            except Exception as e:
                print(f"Error loading cache file {cache_path}: {e}")
                return None
        return None
    
    def _save_to_cache(self, data, cache_path):
        """Save data to cache"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved data to cache: {cache_path}")
        except Exception as e:
            print(f"Error saving to cache {cache_path}: {e}")
    
    def clear_cache(self, ticker=None):
        """
        Clear cache files
        
        Parameters:
        -----------
        ticker : str, optional
            If provided, only clear cache for this ticker. Otherwise, clear all cache.
        """
        if not os.path.exists(self.cache_dir):
            print(f"Cache directory does not exist: {self.cache_dir}")
            return
        
        if ticker:
            # Clear only files for this ticker
            safe_ticker = ticker.replace("^", "").replace("/", "_")
            pattern = f"{safe_ticker}_"
            files_removed = 0
            for filename in os.listdir(self.cache_dir):
                if filename.startswith(pattern):
                    filepath = os.path.join(self.cache_dir, filename)
                    os.remove(filepath)
                    files_removed += 1
            print(f"Removed {files_removed} cache file(s) for {ticker}")
        else:
            # Clear all cache files
            files_removed = 0
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    files_removed += 1
            print(f"Removed {files_removed} cache file(s)")
    
    def download_daily_data(self, ticker, start_date, end_date, source="fmp", use_cache=True, force_download=False, use_rest_api=False):
        """
        Download daily aggregated data with caching
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        source : str, optional
            Data source: 'massive' or 'fmp' (default: 'fmp')
        use_cache : bool, optional
            Whether to use cached data if available (default: True)
        force_download : bool, optional
            Whether to force re-download even if cached (default: False)
        use_rest_api : bool, optional
            If True and source is 'fmp', use REST API directly instead of fmp_python library (default: False)
            
            **IMPORTANT**: When use_rest_api=True, this method returns DIVIDEND-ADJUSTED price series.
            The REST API variant uses the FMP dividend-adjusted endpoint which returns adjusted prices
            (adjOpen, adjHigh, adjLow, adjClose) that account for stock splits and dividends.
            This is the recommended option for accurate historical price analysis.
        
        Returns:
        --------
        DataFrame with daily price data
            - If use_rest_api=True: Returns dividend-adjusted price series (open, high, low, close, volume)
            - If use_rest_api=False: Returns unadjusted or partially adjusted prices depending on source
        """
        if source not in ['massive', 'fmp']:
            raise ValueError(f"Invalid source '{source}'. Must be 'massive' or 'fmp'")
        
        # Route to appropriate method based on source
        if source == 'massive':
            return self._download_daily_data_massive(ticker, start_date, end_date, use_cache, force_download)
        else:  # source == 'fmp'
            if use_rest_api:
                return self._download_daily_data_fmp_rest(ticker, start_date, end_date, use_cache, force_download)
            else:
                return self._download_daily_data_fmp(ticker, start_date, end_date, use_cache, force_download)
    
    def _download_daily_data_massive(self, ticker, start_date, end_date, use_cache=True, force_download=False):
        """Download daily aggregated data from Massive API with caching"""
        if not self.client:
            print("Error: Massive API key not provided. Initialize DataCollector with massive_api_key parameter.")
            return None
        
        # Check cache first
        cache_path = self._get_cache_path(ticker, start_date, end_date, "daily", "massive")
        if use_cache and not force_download:
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        print(f"Downloading daily data for {ticker} from Massive API...")
        aggs = []
        
        # Convert dates to timestamps (milliseconds)
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
        
        try:
            for response in self.client.list_aggs(
                ticker,
                1,
                "day",
                start_ts,
                end_ts,
                adjusted=True,
                sort="asc",
                limit=50000,
            ):
                aggs.append(response)

        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        if not aggs:
            print(f"No data returned for {ticker}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(aggs)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            print(f"Warning: No timestamp column found for {ticker}")
            print(f"Available columns: {df.columns.tolist()}")
            return None
        
        # Set date as index
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # Rename columns (c=close, o=open, h=high, l=low, v=volume)
        column_map = {
            'c': 'close',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'v': 'volume'
        }
        # Only rename columns that exist
        existing_cols = {k: v for k, v in column_map.items() if k in df.columns}
        df.rename(columns=existing_cols, inplace=True)
        
        # Ensure we have at least close price
        if 'close' not in df.columns and 'c' in df.columns:
            df['close'] = df['c']
        
        result_df = df[['open', 'high', 'low', 'close', 'volume']] if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']) else df[['close']]
        
        print(f"Downloaded {len(result_df)} records for {ticker}")
        
        # Save to cache
        if use_cache:
            self._save_to_cache(result_df, cache_path)
        
        return result_df
    
    def _download_daily_data_fmp(self, ticker, start_date, end_date, use_cache=True, force_download=False):
        """Download daily aggregated data from FMP API with caching"""
        if not self.fmp:
            print("Error: FMP API key not provided. Initialize DataCollector with fmp_api_key parameter.")
            return None
        
        # Check cache first
        cache_path = self._get_cache_path(ticker, start_date, end_date, "daily", "fmp")
        if use_cache and not force_download:
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        print(f"Downloading daily data for {ticker} from FMP API...")
        
        try:
            # Get historical price data from FMP
            response = self.fmp.get_historical_price(ticker, start_date, end_date)
            
            # FMP decorator formats the response, but we need to handle different formats
            data = None
            
            # Check if it's a Response object (requests library)
            if hasattr(response, 'json'):
                try:
                    data = response.json()
                except:
                    data = response.text if hasattr(response, 'text') else response
            # Check if it's already a list (formatted by decorator)
            elif isinstance(response, list):
                data = response
            # Check if it's a dict with 'historical' key
            elif isinstance(response, dict):
                if 'historical' in response:
                    data = response['historical']
                else:
                    data = response
            else:
                data = response
            
            # Handle nested response format (if data is dict with 'historical' key)
            if isinstance(data, dict) and 'historical' in data:
                data = data['historical']
            
            # Ensure data is a list
            if not isinstance(data, list):
                print(f"Unexpected data format for {ticker} from FMP API")
                print(f"Data type: {type(data)}")
                return None
            
            if len(data) == 0:
                print(f"No data returned for {ticker} from FMP API")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            else:
                print(f"Warning: No date column found for {ticker}")
                print(f"Available columns: {df.columns.tolist()}")
                return None
            
            # Set date as index
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            # Map FMP column names to expected format
            # FMP uses: open, high, low, close, adjClose, volume
            # We need: open, high, low, close, volume
            column_mapping = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'adjClose': 'close',  # Use adjusted close if available, but prefer close
                'volume': 'volume'
            }
            
            # Select and rename columns
            available_cols = []
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    available_cols.append(col)
                elif col == 'close' and 'adjClose' in df.columns:
                    # Use adjClose as close if close is not available
                    df['close'] = df['adjClose']
                    available_cols.append('close')
            
            if 'close' not in available_cols:
                print(f"Warning: No close price column found for {ticker}")
                return None
            
            # Ensure volume is available, set to 0 if not
            if 'volume' not in df.columns:
                df['volume'] = 0
            
            result_df = df[available_cols].copy()
            
            print(f"Downloaded {len(result_df)} records for {ticker} from FMP API")
            
            # Save to cache
            if use_cache:
                self._save_to_cache(result_df, cache_path)
            
            return result_df
            
        except Exception as e:
            print(f"Error downloading {ticker} from FMP API: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _download_daily_data_fmp_rest(self, ticker, start_date, end_date, use_cache=True, force_download=False):
        """
        Download daily aggregated data from FMP REST API directly (without fmp_python library)
        
        **IMPORTANT**: This method returns DIVIDEND-ADJUSTED price series.
        It uses the FMP dividend-adjusted endpoint (historical-price-eod/dividend-adjusted)
        which returns adjusted prices that account for stock splits and dividends.
        
        The endpoint returns: adjOpen, adjHigh, adjLow, adjClose, volume
        These are mapped to: open, high, low, close, volume in the returned DataFrame.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        use_cache : bool, optional
            Whether to use cached data if available (default: True)
        force_download : bool, optional
            Whether to force re-download even if cached (default: False)
        
        Returns:
        --------
        DataFrame with daily price data
            Columns: open, high, low, close, volume
            All prices are DIVIDEND-ADJUSTED (accounting for splits and dividends)
        """
        if not self.fmp_api_key:
            print("Error: FMP API key not provided. Initialize DataCollector with fmp_api_key parameter.")
            return None
        
        # Check cache first (use different cache key to distinguish from library method)
        cache_path = self._get_cache_path(ticker, start_date, end_date, "daily", "fmp_rest")
        if use_cache and not force_download:
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        print(f"Downloading daily data for {ticker} from FMP REST API (dividend-adjusted)...")
        
        try:
            # Build the FMP API endpoint URL for dividend-adjusted prices
            # Endpoint: https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted
            base_url = "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted"
            
            # Add query parameters
            params = {
                'symbol': ticker,
                'from': start_date,
                'to': end_date,
                'apikey': self.fmp_api_key
            }
            
            # Make the API request
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Parse JSON response
            data = response.json()
            
            # FMP API returns data in format: {"historical": [...]} or directly as a list
            if isinstance(data, dict) and 'historical' in data:
                historical_data = data['historical']
            elif isinstance(data, list):
                historical_data = data
            else:
                print(f"Unexpected data format for {ticker} from FMP REST API")
                print(f"Data type: {type(data)}, Keys: {data.keys() if isinstance(data, dict) else 'N/A'}")
                return None
            
            if len(historical_data) == 0:
                print(f"No data returned for {ticker} from FMP REST API")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            
            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            else:
                print(f"Warning: No date column found for {ticker}")
                print(f"Available columns: {df.columns.tolist()}")
                return None
            
            # Set date as index
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            # FMP dividend-adjusted endpoint returns: adjOpen, adjHigh, adjLow, adjClose, volume
            # Map FMP column names (source) to expected format (target)
            column_mapping = {
                'adjOpen': 'open',
                'adjHigh': 'high',
                'adjLow': 'low',
                'adjClose': 'close',  # Dividend-adjusted close price
                'volume': 'volume'
            }
            
            # Select and rename columns
            available_cols = []
            for source_col, target_col in column_mapping.items():
                if source_col in df.columns:
                    df[target_col] = df[source_col]
                    available_cols.append(target_col)
            
            # If close is not available, check for alternative column names
            if 'close' not in available_cols:
                if 'adjClose' in df.columns:
                    # Use adjClose if available
                    df['close'] = df['adjClose']
                    available_cols.append('close')
                elif 'close' in df.columns:
                    # Fall back to regular close if adjClose not available
                    df['close'] = df['close']
                    available_cols.append('close')
                else:
                    print(f"Warning: No close price column found for {ticker}")
                    print(f"Available columns: {df.columns.tolist()}")
                    return None
            
            # Ensure volume is available, set to 0 if not
            if 'volume' not in available_cols:
                if 'volume' in df.columns:
                    df['volume'] = df['volume']
                    available_cols.append('volume')
                else:
                    df['volume'] = 0
                    available_cols.append('volume')
            
            # Select only the columns we need
            result_df = df[available_cols].copy()
            
            print(f"Downloaded {len(result_df)} records for {ticker} from FMP REST API (dividend-adjusted)")
            
            # Save to cache
            if use_cache:
                self._save_to_cache(result_df, cache_path)
            
            return result_df
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {ticker} from FMP REST API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text[:500]}")
            import traceback
            traceback.print_exc()
            return None
        except Exception as e:
            print(f"Error processing {ticker} data from FMP REST API: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def download_intraday_data(self, ticker, start_date, end_date, interval_minutes=15, source="massive", use_cache=True, force_download=False):
        """
        Download intraday data with caching
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        interval_minutes : int, optional
            Interval in minutes (default: 15)
        source : str, optional
            Data source: 'massive' or 'fmp' (default: 'massive')
            Note: Currently only 'massive' is supported for intraday data
        use_cache : bool, optional
            Whether to use cached data if available (default: True)
        force_download : bool, optional
            Whether to force re-download even if cached (default: False)
        
        Returns:
        --------
        DataFrame with intraday price data
        """
        if source not in ['massive', 'fmp']:
            raise ValueError(f"Invalid source '{source}'. Must be 'massive' or 'fmp'")
        
        if source == 'fmp':
            print("Warning: FMP source not yet supported for intraday data. Using 'massive' instead.")
            source = 'massive'
        
        # Check cache first
        cache_path = self._get_cache_path(ticker, start_date, end_date, f"intraday_{interval_minutes}m", source)
        if use_cache and not force_download:
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        if not self.client:
            print("Error: Massive API key not provided. Initialize DataCollector with massive_api_key parameter.")
            return None
        
        print(f"Downloading {interval_minutes}-minute intraday data for {ticker} from Massive API...")
        aggs = []
        
        # Convert dates to timestamps
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
        
        try:
            # Download day by day to avoid memory issues
            current_date = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date)
            
            while current_date <= end_dt:
                day_start = int(current_date.timestamp() * 1000)
                day_end = int((current_date + timedelta(days=30)).timestamp() * 1000)
                
                try:
                    for response in self.client.list_aggs(
                        ticker,
                        interval_minutes,
                        "minute",
                        day_start,
                        min(day_end, end_ts),
                        adjusted=True,
                        sort="asc",
                        limit=50000,
                    ):
                        aggs.append(response)

                except Exception as e:
                    print(f"Warning: Error downloading {ticker} for {current_date}: {e}")
                
                current_date += timedelta(days=30)
                
                if len(aggs) > 0:
                    print(f"  Downloaded {len(aggs)} records so far, current date: {current_date}")
        
        except Exception as e:
            print(f"Error downloading intraday data for {ticker}: {e}")
            return None
        
        if not aggs:
            print(f"No intraday data returned for {ticker}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(aggs)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            print(f"Warning: No timestamp column found for {ticker} intraday")
            return None
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # Rename columns
        column_map = {
            'c': 'close',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'v': 'volume'
        }
        df.rename(columns=column_map, inplace=True)
        
        result_df = df[['open', 'high', 'low', 'close', 'volume']]
        
        print(f"Downloaded {len(result_df)} intraday records for {ticker}")
        
        # Save to cache
        if use_cache:
            self._save_to_cache(result_df, cache_path)
        
        return result_df


def calculate_daily_returns(prices):
    """Calculate daily returns from prices"""
    return prices.pct_change().dropna()


def calculate_annualized_volatility(returns, window=30):
    """Calculate rolling annualized volatility"""
    return returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def calculate_realized_volatility(intraday_data, scale_factor=1.3):
    """
    Calculate daily realized volatility from intraday returns
    
    Parameters:
    -----------
    intraday_data : DataFrame
        DataFrame with datetime index and 'close' column
    scale_factor : float
        Scaling factor to account for overnight returns (default 1.3)
    
    Returns:
    --------
    Series with daily realized volatility (annualized)
    """
    # Ensure we have a datetime index
    if not isinstance(intraday_data.index, pd.DatetimeIndex):
        intraday_data.index = pd.to_datetime(intraday_data.index)
    
    # Normalize to date (remove time component for grouping)
    intraday_data = intraday_data.copy()
    intraday_data['date'] = intraday_data.index.normalize()
    
    # Group by date
    daily_rv = []
    
    for date, group in intraday_data.groupby('date'):
        if len(group) < 2:  # Need at least 2 data points to calculate returns
            continue
            
        # Calculate 15-minute returns within the day
        returns = group['close'].pct_change().dropna()
        
        if len(returns) == 0:
            continue
        
        # Calculate realized variance (sum of squared returns)
        rvariance = (returns ** 2).sum()
        
        # Annualize and scale
        rvol = np.sqrt(TRADING_DAYS_PER_YEAR * rvariance) * scale_factor
        
        daily_rv.append({
            'date': pd.Timestamp(date),
            'realized_vol': rvol
        })
    
    if not daily_rv:
        print("Warning: No realized volatility data calculated")
        return pd.Series(dtype=float)
    
    rv_df = pd.DataFrame(daily_rv)
    rv_df.set_index('date', inplace=True)
    rv_df.sort_index(inplace=True)
    
    return rv_df['realized_vol']


def calculate_performance_metrics(returns, risk_free_rate=None):
    """
    Calculate comprehensive performance metrics
    
    Parameters:
    -----------
    returns : Series
        Daily portfolio returns
    risk_free_rate : Series, optional
        Daily risk-free rate returns
    
    Returns:
    --------
    dict with performance metrics
    """
    if risk_free_rate is not None:
        # Align indices
        common_idx = returns.index.intersection(risk_free_rate.index)
        excess_returns = returns.loc[common_idx] - risk_free_rate.loc[common_idx]
    else:
        excess_returns = returns
        common_idx = returns.index
    
    # Calculate cumulative returns
    cumulative = (1 + returns.loc[common_idx]).cumprod()
    
    # CAGR
    n_years = (common_idx[-1] - common_idx[0]).days / 365.25
    cagr = (cumulative.iloc[-1] ** (1 / n_years)) - 1 if n_years > 0 else 0
    
    # Annualized excess return
    annualized_excess = excess_returns.mean() * TRADING_DAYS_PER_YEAR
    
    # Annualized volatility
    annualized_vol = excess_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # Sharpe ratio
    sharpe = annualized_excess / annualized_vol if annualized_vol > 0 else 0
    
    # Maximum drawdown
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (excess_returns > 0).sum() / len(excess_returns) if len(excess_returns) > 0 else 0
    
    return {
        'CAGR': cagr,
        'Annualized Excess Return': annualized_excess,
        'Annualized Volatility': annualized_vol,
        'Sharpe Ratio': sharpe,
        'Maximum Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Total Return': cumulative.iloc[-1] - 1
    }


def buy_and_hold_strategy(spy_returns, initial_capital=100000, transaction_cost=TRANSACTION_COST):
    """Buy-and-hold strategy: 100% allocation to main ticker"""
    # Buy-and-hold has no rebalancing, so only initial transaction cost
    portfolio_returns = spy_returns.copy()
    # Apply initial transaction cost (buying into position)
    if len(portfolio_returns) > 0:
        portfolio_returns.iloc[0] -= transaction_cost
    
    portfolio_value = initial_capital * (1 + portfolio_returns).cumprod()
    
    return portfolio_value, portfolio_returns


def volatility_targeting_30d_hv(spy_returns, spy_prices, risk_free_rate, initial_capital=100000, transaction_cost=TRANSACTION_COST):
    """
    Volatility targeting using 30-day historical volatility
    
    At close of day t-1, calculate volatility using returns up to t-1,
    set allocation for day t, and use that allocation for returns on day t.
    """
    # Calculate 30-day rolling volatility (uses data up to and including current day)
    vol_30d = calculate_annualized_volatility(spy_returns, window=HV_WINDOW)
    
    # Calculate allocation: at close of day t, vol_30d[t] uses data up to t-1
    # So allocation[t] is set at close of day t-1 for use on day t
    # We shift by 1 to use yesterday's volatility for today's allocation
    allocation = (TARGET_VOLATILITY / vol_30d).shift(1)
    allocation = allocation.clip(0, 1)  # Cap at 100%
    
    # Align indices
    common_idx = spy_returns.index.intersection(allocation.index)
    
    # Calculate portfolio returns
    # Portfolio return = allocation * stock_return + (1 - allocation) * risk_free_rate
    portfolio_returns = pd.Series(index=common_idx, dtype=float)
    prev_alloc = None
    
    for i, date in enumerate(common_idx):
        if i > 0 and pd.notna(allocation.loc[date]):
            alloc = allocation.loc[date]
            spy_ret = spy_returns.loc[date] if date in spy_returns.index else 0
            rf_ret = risk_free_rate.loc[date] if date in risk_free_rate.index else 0
            portfolio_returns.loc[date] = alloc * spy_ret + (1 - alloc) * rf_ret
            
            # Apply transaction costs when allocation changes
            if prev_alloc is not None:
                alloc_change = abs(alloc - prev_alloc)
                portfolio_returns.loc[date] -= alloc_change * transaction_cost
            else:
                # Initial transaction cost (entering position)
                portfolio_returns.loc[date] -= alloc * transaction_cost
            
            prev_alloc = alloc
        else:
            portfolio_returns.loc[date] = 0
            prev_alloc = None
    
    # Calculate portfolio value
    portfolio_value = initial_capital * (1 + portfolio_returns).cumprod()
    
    return portfolio_value, portfolio_returns, allocation


def volatility_targeting_vix(spy_returns, vix_data, risk_free_rate, initial_capital=100000, transaction_cost=TRANSACTION_COST):
    """
    Volatility targeting using VIX
    """
    # VIX is already in percentage points (e.g., 20 = 20%)
    # Convert to decimal (e.g., 20% = 0.20)
    vix_decimal = vix_data / 100.0
    
    # Calculate allocation (one-day lag)
    allocation = (TARGET_VOLATILITY / vix_decimal).shift(1)
    allocation = allocation.clip(0, 1)  # Cap at 100%
    
    # Align indices
    common_idx = spy_returns.index.intersection(allocation.index)
    
    # Calculate portfolio returns
    portfolio_returns = pd.Series(index=common_idx, dtype=float)
    prev_alloc = None
    
    for i, date in enumerate(common_idx):
        if i > 0 and pd.notna(allocation.loc[date]):
            alloc = allocation.loc[date]
            spy_ret = spy_returns.loc[date] if date in spy_returns.index else 0
            rf_ret = risk_free_rate.loc[date] if date in risk_free_rate.index else 0
            portfolio_returns.loc[date] = alloc * spy_ret + (1 - alloc) * rf_ret
            
            # Apply transaction costs when allocation changes
            if prev_alloc is not None:
                alloc_change = abs(alloc - prev_alloc)
                portfolio_returns.loc[date] -= alloc_change * transaction_cost
            else:
                # Initial transaction cost (entering position)
                portfolio_returns.loc[date] -= alloc * transaction_cost
            
            prev_alloc = alloc
        else:
            portfolio_returns.loc[date] = 0
            prev_alloc = None
    
    # Calculate portfolio value
    portfolio_value = initial_capital * (1 + portfolio_returns).cumprod()
    
    return portfolio_value, portfolio_returns, allocation


def volatility_targeting_2d_rv(spy_returns, realized_vol, risk_free_rate, initial_capital=100000, transaction_cost=TRANSACTION_COST):
    """
    Volatility targeting using 2-day realized volatility
    
    Per paper: "At close of day t, forecast uses t-2 and t-1"
    So at close of day t, we calculate: vol_2d_rv = (realized_vol[t-2] + realized_vol[t-1]) / 2
    This forecast is used to set allocation for day t+1.
    For returns on day t, we use allocation set at close of day t-1.
    """
    # Calculate 2-day average realized volatility
    # rolling(2).mean() gives mean of current and previous value
    # We want mean of t-2 and t-1, so we shift by 1
    # vol_2d_rv[t] = mean(realized_vol[t-2], realized_vol[t-1])
    vol_2d_rv = realized_vol.shift(1).rolling(window=RV_WINDOW).mean()
    
    # Allocation: at close of day t-1, we calculated vol_2d_rv[t-1] using t-3 and t-2
    # This allocation is used for day t returns
    allocation = (TARGET_VOLATILITY / vol_2d_rv).shift(1)
    allocation = allocation.clip(0, 1)  # Cap at 100%
    
    # Align indices
    common_idx = spy_returns.index.intersection(allocation.index)
    
    # Calculate portfolio returns
    portfolio_returns = pd.Series(index=common_idx, dtype=float)
    prev_alloc = None
    
    for i, date in enumerate(common_idx):
        if i > 0 and pd.notna(allocation.loc[date]):
            alloc = allocation.loc[date]
            spy_ret = spy_returns.loc[date] if date in spy_returns.index else 0
            rf_ret = risk_free_rate.loc[date] if date in risk_free_rate.index else 0
            portfolio_returns.loc[date] = alloc * spy_ret + (1 - alloc) * rf_ret
            
            # Apply transaction costs when allocation changes
            if prev_alloc is not None:
                alloc_change = abs(alloc - prev_alloc)
                portfolio_returns.loc[date] -= alloc_change * transaction_cost
            else:
                # Initial transaction cost (entering position)
                portfolio_returns.loc[date] -= alloc * transaction_cost
            
            prev_alloc = alloc
        else:
            portfolio_returns.loc[date] = 0
            prev_alloc = None
    
    # Calculate portfolio value
    portfolio_value = initial_capital * (1 + portfolio_returns).cumprod()
    
    return portfolio_value, portfolio_returns, allocation


def trend_following_200d_ma(spy_returns, spy_prices, risk_free_rate, initial_capital=100000, transaction_cost=TRANSACTION_COST):
    """
    200-day moving average trend-following strategy
    """
    # Calculate 200-day moving average
    ma_200 = spy_prices.rolling(window=MA_WINDOW).mean()
    
    # Generate signals (one-day lag)
    # If price > MA, allocate 100% to stocks, else 100% to cash
    signals = (spy_prices > ma_200).shift(1).astype(float)
    
    # Align indices
    common_idx = spy_returns.index.intersection(signals.index)
    
    # Calculate portfolio returns
    portfolio_returns = pd.Series(index=common_idx, dtype=float)
    prev_signal = None
    
    for i, date in enumerate(common_idx):
        if pd.notna(signals.loc[date]):
            signal = signals.loc[date]
            spy_ret = spy_returns.loc[date] if date in spy_returns.index else 0
            rf_ret = risk_free_rate.loc[date] if date in risk_free_rate.index else 0
            portfolio_returns.loc[date] = signal * spy_ret + (1 - signal) * rf_ret
            
            # Apply transaction costs when signal changes (switching between stocks and cash)
            if prev_signal is not None and prev_signal != signal:
                # Full transaction cost when switching (0 to 1 or 1 to 0)
                portfolio_returns.loc[date] -= transaction_cost
            elif prev_signal is None:
                # Initial transaction cost (entering position)
                portfolio_returns.loc[date] -= signal * transaction_cost
            
            prev_signal = signal
        else:
            portfolio_returns.loc[date] = 0
            prev_signal = None
    
    # Calculate portfolio value
    portfolio_value = initial_capital * (1 + portfolio_returns).cumprod()
    
    return portfolio_value, portfolio_returns, signals


def create_visualizations(results_dict, spy_prices=None, vix_prices=None, risk_free_rate=None, output_dir='.'):
    """Create comprehensive visualizations"""
    
    # Helper function to add in-sample/out-of-sample separator
    def add_period_separator():
        """Add vertical red line and labels for in-sample/out-of-sample periods"""
        from matplotlib.dates import num2date
        end_date_dt = pd.Timestamp(END_DATE)
        if pd.isna(end_date_dt):
            return  # Skip if date is invalid
        ax = plt.gca()
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        
        # Add vertical red line (matplotlib handles pandas Timestamp automatically)
        plt.axvline(x=end_date_dt.to_pydatetime(), color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=10)
        
        # Add text labels
        # Get x-axis limits and calculate positions
        xlim = ax.get_xlim()
        # Convert matplotlib date numbers to pandas timestamps
        try:
            xlim_start = num2date(xlim[0])
            xlim_end = num2date(xlim[1])
            x_range_days = (pd.Timestamp(xlim_end) - pd.Timestamp(xlim_start)).days
        except:
            # Fallback: use a fixed offset if conversion fails
            x_range_days = 2000
        
        # In-sample label (left side) - position at 25% of the way from start to separator, at bottom
        in_sample_mid = end_date_dt - pd.Timedelta(days=max(365, int(x_range_days * 0.25)))
        if not pd.isna(in_sample_mid):
            plt.text(in_sample_mid.to_pydatetime(), ylim[0] + y_range * 0.02, 'In-Sample Period', 
                    fontsize=11, fontweight='bold', ha='center', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Out-of-sample label (right side) - position at 25% of the way from separator to end, at bottom
        out_sample_mid = end_date_dt + pd.Timedelta(days=max(365, int(x_range_days * 0.25)))
        if not pd.isna(out_sample_mid):
            plt.text(out_sample_mid.to_pydatetime(), ylim[0] + y_range * 0.02, 'Out-of-Sample Period', 
                    fontsize=11, fontweight='bold', ha='center', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        # Add date label on the line (positioned slightly above the period labels)
        plt.text(end_date_dt.to_pydatetime(), ylim[0] + y_range * 0.08, f'{END_DATE}', 
                fontsize=9, ha='center', va='bottom', rotation=0,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red'))
    
    # 1. Equity curves
    plt.figure(figsize=(14, 8))
    for strategy_name, result in results_dict.items():
        if 'portfolio_value' in result:
            plt.plot(result['portfolio_value'].index, result['portfolio_value'].values, 
                    label=strategy_name, linewidth=2)
    
    plt.title('Equity Curves: All Strategies', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    add_period_separator()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Rolling 1-year volatility comparison
    if 'Buy-and-Hold' in results_dict and '30D HV' in results_dict:
        plt.figure(figsize=(14, 6))
        
        bh_returns = results_dict['Buy-and-Hold']['portfolio_returns']
        hv_returns = results_dict['30D HV']['portfolio_returns']
        
        bh_vol = calculate_annualized_volatility(bh_returns, window=252)
        hv_vol = calculate_annualized_volatility(hv_returns, window=252)
        
        plt.plot(bh_vol.index, bh_vol.values, label='Buy-and-Hold', linewidth=2, alpha=0.7)
        plt.plot(hv_vol.index, hv_vol.values, label='30D HV Volatility Targeting', linewidth=2, alpha=0.7)
        plt.axhline(y=TARGET_VOLATILITY, color='r', linestyle='--', label=f'Target Volatility ({TARGET_VOLATILITY*100}%)', alpha=0.7)
        
        plt.title('Rolling 1-Year Volatility: Buy-and-Hold vs 30D HV', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Annualized Volatility', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        add_period_separator()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rolling_volatility.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Drawdown charts
    plt.figure(figsize=(14, 8))
    
    strategies_to_plot = ['200D MA', '2D RV', 'Buy-and-Hold']
    for strategy_name in strategies_to_plot:
        if strategy_name in results_dict and 'portfolio_value' in results_dict[strategy_name]:
            portfolio_value = results_dict[strategy_name]['portfolio_value']
            running_max = portfolio_value.expanding().max()
            drawdown = (portfolio_value - running_max) / running_max * 100
            
            plt.plot(drawdown.index, drawdown.values, label=strategy_name, linewidth=2, alpha=0.7)
    
    plt.title('Drawdown Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    add_period_separator()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drawdowns.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Allocation over time for volatility targeting strategies
    plt.figure(figsize=(14, 8))
    
    vol_strategies = ['30D HV', 'VIX', '2D RV']
    for strategy_name in vol_strategies:
        if strategy_name in results_dict and 'allocation' in results_dict[strategy_name]:
            allocation = results_dict[strategy_name]['allocation']
            plt.plot(allocation.index, allocation.values * 100, 
                    label=strategy_name, linewidth=1.5, alpha=0.7)
    

    plt.title('Stock Allocation Over Time: Volatility Targeting Strategies', 
             fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Allocation (%)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    add_period_separator()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'allocations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Main Ticker Price Chart
    if spy_prices is not None and len(spy_prices) > 0:
        plt.figure(figsize=(14, 6))
        plt.plot(spy_prices.index, spy_prices.values, label=MAIN_TICKER, linewidth=2, color='blue')
        plt.title(f'{MAIN_TICKER} Price Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        add_period_separator()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spy_price.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. VIX Price Chart
    if vix_prices is not None and len(vix_prices) > 0:
        plt.figure(figsize=(14, 6))
        plt.plot(vix_prices.index, vix_prices.values, label='VIX', linewidth=2, color='red')
        plt.title('VIX Index Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('VIX Level', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        add_period_separator()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'vix_price.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. Risk-Free Rate Chart
    if risk_free_rate is not None and len(risk_free_rate) > 0:
        plt.figure(figsize=(14, 6))
        # Convert daily returns to annualized percentage for display
        annualized_rf = risk_free_rate * TRADING_DAYS_PER_YEAR * 100
        plt.plot(risk_free_rate.index, annualized_rf.values, label='Risk-Free Rate', linewidth=2, color='green')
        plt.title('Risk-Free Rate Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Annualized Rate (%)', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        add_period_separator()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'risk_free_rate.png'), dpi=300, bbox_inches='tight')
        plt.close()


def generate_report(results_dict, risk_free_rate=None, output_file='report.txt'):
    """Generate a text report summarizing results for in-sample, out-of-sample, and whole period"""
    
    # Convert END_DATE to datetime for comparison
    end_date_dt = pd.Timestamp(END_DATE)
    current_date_dt = pd.Timestamp(CURRENT_DATE)
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RISK BEFORE RETURN: VOLATILITY TARGETING STRATEGIES ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("CONFIGURATION PARAMETERS\n")
        f.write("=" * 80 + "\n\n")
        end_date_next = (pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        f.write(f"Analysis Periods:\n")
        f.write(f"  In-Sample Period: {START_DATE} to {END_DATE}\n")
        f.write(f"  Out-of-Sample Period: {end_date_next} to {CURRENT_DATE}\n")
        f.write(f"  Whole Period: {START_DATE} to {CURRENT_DATE}\n\n")
        f.write(f"Volatility Targeting Parameters:\n")
        f.write(f"  Target Volatility: {TARGET_VOLATILITY*100}% ({TARGET_VOLATILITY})\n")
        f.write(f"  Trading Days Per Year: {TRADING_DAYS_PER_YEAR}\n")
        f.write(f"  Scale Factor (for realized volatility): {SCALE_FACTOR}\n\n")
        f.write(f"Strategy Parameters:\n")
        f.write(f"  Moving Average Window (MA): {MA_WINDOW} days\n")
        f.write(f"  Historical Volatility Window (HV): {HV_WINDOW} days\n")
        f.write(f"  Realized Volatility Window (RV): {RV_WINDOW} days\n\n")
        f.write(f"Portfolio Parameters:\n")
        f.write(f"  Initial Capital: $100,000\n")
        f.write(f"  Transaction Cost: {TRANSACTION_COST*100}% ({TRANSACTION_COST}) per unit of allocation change\n\n")
        f.write(f"Data Parameters:\n")
        f.write(f"  Cache Directory: {CACHE_DIR}\n\n")
        
        # Calculate metrics for each period
        # Out-of-sample starts the day after END_DATE
        end_date_next = (pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        periods = {
            'In-Sample': (START_DATE, END_DATE),
            'Out-of-Sample': (end_date_next, CURRENT_DATE),
            'Whole Period': (START_DATE, CURRENT_DATE)
        }
        
        all_metrics = {}
        
        for period_name, (start, end) in periods.items():
            f.write("=" * 80 + "\n")
            f.write(f"PERFORMANCE METRICS: {period_name.upper()}\n")
            f.write(f"Period: {start} to {end}\n")
            f.write("=" * 80 + "\n\n")
            
            period_metrics = {}
            start_dt = pd.Timestamp(start)
            end_dt = pd.Timestamp(end)
            
            for strategy, result in results_dict.items():
                if 'portfolio_returns' not in result:
                    continue
                
                # Filter returns for this period
                returns = result['portfolio_returns']
                period_returns = returns[(returns.index >= start_dt) & (returns.index <= end_dt)]
                
                if len(period_returns) == 0:
                    continue
                
                # Get risk-free rate for this period
                period_rf = None
                if risk_free_rate is not None:
                    period_rf = risk_free_rate[(risk_free_rate.index >= start_dt) & (risk_free_rate.index <= end_dt)]
                
                # Calculate metrics for this period
                metrics = calculate_performance_metrics(period_returns, period_rf)
                period_metrics[strategy] = metrics
            
            if period_metrics:
                # Create metrics table for this period
                metrics_df = pd.DataFrame(period_metrics).T
                f.write(metrics_df.to_string())
                f.write("\n\n")
                
                # Store for summary
                all_metrics[period_name] = period_metrics
        
        f.write("=" * 80 + "\n")
        f.write("KEY FINDINGS BY PERIOD\n")
        f.write("=" * 80 + "\n\n")
        
        for period_name in ['In-Sample', 'Out-of-Sample', 'Whole Period']:
            if period_name not in all_metrics:
                continue
            
            period_metrics = all_metrics[period_name]
            f.write(f"\n{period_name}:\n")
            f.write("-" * 80 + "\n")
            
            # Find best Sharpe ratio
            sharpe_ratios = {
                strategy: metrics['Sharpe Ratio'] 
                for strategy, metrics in period_metrics.items()
            }
            if sharpe_ratios:
                best_sharpe_strategy = max(sharpe_ratios, key=lambda x: sharpe_ratios[x])
                f.write(f"  Best Sharpe Ratio: {best_sharpe_strategy} ({sharpe_ratios[best_sharpe_strategy]:.3f})\n")
            
            # Find lowest max drawdown
            max_dds = {
                strategy: metrics['Maximum Drawdown'] 
                for strategy, metrics in period_metrics.items()
            }
            if max_dds:
                best_dd_strategy = max(max_dds, key=lambda x: max_dds[x])  # Less negative is better
                f.write(f"  Lowest Maximum Drawdown: {best_dd_strategy} ({max_dds[best_dd_strategy]:.3%})\n")
            
            # Find highest CAGR
            cagrs = {
                strategy: metrics['CAGR'] 
                for strategy, metrics in period_metrics.items()
            }
            if cagrs:
                best_cagr_strategy = max(cagrs, key=lambda x: cagrs[x])
                f.write(f"  Highest CAGR: {best_cagr_strategy} ({cagrs[best_cagr_strategy]:.3%})\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")

        f.write("COMPARISON WITH PAPER FINDINGS")
        f.write("\n")
        f.write("=" * 80 + "\n\n")

        f.write("Expected findings from the paper:\n")
        f.write("- 2D RV should achieve higher Sharpe ratio than 30D HV and VIX\n")
        f.write("- Volatility targeting strategies should have lower volatility than buy-and-hold\n")
        f.write("- Volatility targeting should reduce maximum drawdowns\n")
        f.write("\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("IN-SAMPLE PERFORMANCE\n")
        f.write("=" * 80 + "\n\n")

        
        # Check in-sample period results
        if 'In-Sample' in all_metrics:
            in_sample_metrics = all_metrics['In-Sample']
            sharpe_ratios = {
                strategy: metrics['Sharpe Ratio'] 
                for strategy, metrics in in_sample_metrics.items()
            }
            
            if '2D RV' in sharpe_ratios and '30D HV' in sharpe_ratios and 'VIX' in sharpe_ratios:
                if sharpe_ratios['2D RV'] > sharpe_ratios['30D HV'] and sharpe_ratios['2D RV'] > sharpe_ratios['VIX']:
                    f.write("✓ REPLICATED: 2D RV has higher Sharpe ratio than 30D HV and VIX\n")
                else:
                    f.write("✗ NOT REPLICATED: 2D RV does not have the highest Sharpe ratio\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("OUT-OF-SAMPLE PERFORMANCE\n")
        f.write("=" * 80 + "\n\n")
        f.write("This section shows how the strategies performed after the original paper period.\n")
        f.write("This helps assess whether the findings hold in unseen data.\n")
        f.write("\n")

        # Check out-of-sample period results
        if 'Out-of-Sample' in all_metrics:
            out_of_sample_metrics = all_metrics['Out-of-Sample']
            sharpe_ratios = {
                strategy: metrics['Sharpe Ratio'] 
                for strategy, metrics in out_of_sample_metrics.items()
            }
            
            if '2D RV' in sharpe_ratios and '30D HV' in sharpe_ratios and 'VIX' in sharpe_ratios:
                if sharpe_ratios['2D RV'] > sharpe_ratios['30D HV'] and sharpe_ratios['2D RV'] > sharpe_ratios['VIX']:
                    f.write("✓ REPLICATED: 2D RV has higher Sharpe ratio than 30D HV and VIX\n")
                else:
                    f.write("✗ NOT REPLICATED: 2D RV does not have the highest Sharpe ratio\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")

        f.write("WHOLE PERIOD PERFORMANCE\n")
        f.write("=" * 80 + "\n\n")
        f.write("This section shows how the strategies performed over the entire analysis period.\n")
        f.write("This helps assess whether the findings hold across the entire period.\n")
        f.write("\n")
        
        # Check whole period results
        if 'Whole Period' in all_metrics:
            whole_period_metrics = all_metrics['Whole Period']
            sharpe_ratios = {
                strategy: metrics['Sharpe Ratio'] 
                for strategy, metrics in whole_period_metrics.items()
            }
            
            if '2D RV' in sharpe_ratios and '30D HV' in sharpe_ratios and 'VIX' in sharpe_ratios:
                if sharpe_ratios['2D RV'] > sharpe_ratios['30D HV'] and sharpe_ratios['2D RV'] > sharpe_ratios['VIX']:
                    f.write("✓ REPLICATED: 2D RV has higher Sharpe ratio than 30D HV and VIX\n")
                else:
                    f.write("✗ NOT REPLICATED: 2D RV does not have the highest Sharpe ratio\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")

    print(f"Report saved to {output_file}")


def main():
    """Main execution function"""
    # Load API keys from environment variables
    massive_api_key = os.getenv('MASSIVE_API_KEY')
    fmp_api_key = os.getenv('FMP_API_KEY')
    
    print("=" * 80)
    print("RISK BEFORE RETURN: VOLATILITY TARGETING ANALYSIS")
    print("=" * 80)
    
    # Check if API keys are provided
    if not massive_api_key and not fmp_api_key:
        print("\nError: At least one API key is required.")
        print("Please set MASSIVE_API_KEY and/or FMP_API_KEY in your .env file.")
        print("See .env.example for a template.")
        return
    
    # Initialize data collector
    # Specify API keys for the data sources you want to use
    collector = DataCollector(massive_api_key=massive_api_key, fmp_api_key=fmp_api_key)
    
    print("\n" + "=" * 80)
    print("STEP 1: DATA COLLECTION")
    print("=" * 80)
    
    # Download data using FMP API (full period including out-of-sample)
    print(f"\nDownloading {MAIN_TICKER} daily data from FMP ({START_DATE} to {CURRENT_DATE})...")
    spy_daily = collector.download_daily_data(MAIN_TICKER, START_DATE, CURRENT_DATE, source="fmp", use_rest_api=True)
    
    print(f"\nDownloading VIX daily data from FMP ({START_DATE} to {CURRENT_DATE})...")
    vix_daily = collector.download_daily_data("^VIX", START_DATE, CURRENT_DATE, source="fmp")
    
    print(f"\nDownloading risk-free rate (^IRX) from FMP ({START_DATE} to {CURRENT_DATE})...")
    rf_daily = collector.download_daily_data("^IRX", START_DATE, CURRENT_DATE, source="fmp")
    
    # Alternative: Use Massive API for daily data
    # spy_daily = collector.download_daily_data(MAIN_TICKER, START_DATE, CURRENT_DATE, source="massive")
    # vix_daily = collector.download_daily_data("^VIX", START_DATE, CURRENT_DATE, source="massive")
    # rf_daily = collector.download_daily_data("^IRX", START_DATE, CURRENT_DATE, source="massive")
    
    print(f"\nDownloading {MAIN_TICKER} intraday data (15-minute) from Massive ({START_DATE} to {CURRENT_DATE})...")
    spy_intraday = collector.download_intraday_data(MAIN_TICKER, START_DATE, CURRENT_DATE, interval_minutes=15, source="massive")
    
    # Check if data was downloaded successfully
    if spy_daily is None or len(spy_daily) == 0:
        print(f"Error: Failed to download {MAIN_TICKER} data")
        return
    
    print("\n" + "=" * 80)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 80)
    
    # Preprocess main ticker data
    spy_prices = spy_daily['close']
    spy_returns = calculate_daily_returns(spy_prices)
    
    # Preprocess VIX data
    if vix_daily is not None and len(vix_daily) > 0:
        vix_prices = vix_daily['close']
    else:
        print("Warning: VIX data not available, will skip VIX strategy")
        vix_prices = None
    
    # Preprocess risk-free rate
    if rf_daily is not None and len(rf_daily) > 0:
        # Convert annualized rate to daily (assuming 252 trading days)
        rf_daily_returns = (rf_daily['close'] / 100.0) / TRADING_DAYS_PER_YEAR
    else:
        print("Warning: Risk-free rate data not available, using 0%")
        rf_daily_returns = pd.Series(index=spy_returns.index, data=0.0)
    
    # Align all data to common index
    common_index = spy_returns.index
    if vix_prices is not None:
        vix_prices = vix_prices.reindex(common_index, method='ffill')
    rf_daily_returns = rf_daily_returns.reindex(common_index, method='ffill').fillna(0)
    
    # Preprocess intraday data for realized volatility
    if spy_intraday is not None and len(spy_intraday) > 0:
        print("\nCalculating realized volatility from intraday data...")
        realized_vol = calculate_realized_volatility(spy_intraday, scale_factor=SCALE_FACTOR)
        # Align with daily index
        realized_vol = realized_vol.reindex(common_index, method='ffill')
    else:
        print("Warning: Intraday data not available, will skip 2D RV strategy")
        realized_vol = None
    
    print(f"\nData aligned. Common period: {common_index[0]} to {common_index[-1]}")
    print(f"Total trading days: {len(common_index)}")
    
    print("\n" + "=" * 80)
    print("STEP 3: STRATEGY IMPLEMENTATION")
    print("=" * 80)
    
    initial_capital = 100000
    results_dict = {}
    
    # 1. Buy-and-Hold
    print("\n1. Implementing Buy-and-Hold strategy...")
    bh_value, bh_returns = buy_and_hold_strategy(spy_returns, initial_capital)
    bh_metrics = calculate_performance_metrics(bh_returns, rf_daily_returns)
    results_dict['Buy-and-Hold'] = {
        'portfolio_value': bh_value,
        'portfolio_returns': bh_returns,
        'metrics': bh_metrics
    }
    print(f"   CAGR: {bh_metrics['CAGR']:.2%}, Sharpe: {bh_metrics['Sharpe Ratio']:.3f}")
    
    # 2. 30D HV Volatility Targeting
    print("\n2. Implementing 30D HV Volatility Targeting strategy...")
    hv_value, hv_returns, hv_allocation = volatility_targeting_30d_hv(
        spy_returns, spy_prices, rf_daily_returns, initial_capital
    )
    hv_metrics = calculate_performance_metrics(hv_returns, rf_daily_returns)
    results_dict['30D HV'] = {
        'portfolio_value': hv_value,
        'portfolio_returns': hv_returns,
        'allocation': hv_allocation,
        'metrics': hv_metrics
    }
    print(f"   CAGR: {hv_metrics['CAGR']:.2%}, Sharpe: {hv_metrics['Sharpe Ratio']:.3f}")
    
    # 3. VIX Volatility Targeting
    if vix_prices is not None:
        print("\n3. Implementing VIX Volatility Targeting strategy...")
        vix_value, vix_returns, vix_allocation = volatility_targeting_vix(
            spy_returns, vix_prices, rf_daily_returns, initial_capital
        )
        vix_metrics = calculate_performance_metrics(vix_returns, rf_daily_returns)
        results_dict['VIX'] = {
            'portfolio_value': vix_value,
            'portfolio_returns': vix_returns,
            'allocation': vix_allocation,
            'metrics': vix_metrics
        }
        print(f"   CAGR: {vix_metrics['CAGR']:.2%}, Sharpe: {vix_metrics['Sharpe Ratio']:.3f}")
    else:
        print("\n3. Skipping VIX strategy (data not available)")
    
    # 4. 2D RV Volatility Targeting
    if realized_vol is not None:
        print("\n4. Implementing 2D RV Volatility Targeting strategy...")
        rv_value, rv_returns, rv_allocation = volatility_targeting_2d_rv(
            spy_returns, realized_vol, rf_daily_returns, initial_capital
        )
        rv_metrics = calculate_performance_metrics(rv_returns, rf_daily_returns)
        results_dict['2D RV'] = {
            'portfolio_value': rv_value,
            'portfolio_returns': rv_returns,
            'allocation': rv_allocation,
            'metrics': rv_metrics
        }
        print(f"   CAGR: {rv_metrics['CAGR']:.2%}, Sharpe: {rv_metrics['Sharpe Ratio']:.3f}")
    else:
        print("\n4. Skipping 2D RV strategy (data not available)")
    
    # 5. 200D MA Trend-Following
    print("\n5. Implementing 200D MA Trend-Following strategy...")
    ma_value, ma_returns, ma_signals = trend_following_200d_ma(
        spy_returns, spy_prices, rf_daily_returns, initial_capital
    )
    ma_metrics = calculate_performance_metrics(ma_returns, rf_daily_returns)
    results_dict['200D MA'] = {
        'portfolio_value': ma_value,
        'portfolio_returns': ma_returns,
        'allocation': ma_signals,
        'metrics': ma_metrics
    }
    print(f"   CAGR: {ma_metrics['CAGR']:.2%}, Sharpe: {ma_metrics['Sharpe Ratio']:.3f}")
    
    print("\n" + "=" * 80)
    print("STEP 4: CREATING OUTPUT DIRECTORIES")
    print("=" * 80)
    
    # Create output directories if they don't exist
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        print(f"Created reports directory: {REPORTS_DIR}")
    
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        print(f"Created plots directory: {PLOTS_DIR}")
    
    print("\n" + "=" * 80)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    create_visualizations(results_dict, spy_prices=spy_prices, vix_prices=vix_prices, 
                         risk_free_rate=rf_daily_returns, output_dir=PLOTS_DIR)
    
    print("\n" + "=" * 80)
    print("STEP 6: GENERATING REPORT")
    print("=" * 80)
    
    # Generate report filename with main parameters
    main_ticker = MAIN_TICKER  # Main ticker used in analysis
    target_vol_pct = int(TARGET_VOLATILITY * 100)  # Convert to percentage (e.g., 15 for 15%)
    # Format transaction cost as basis points (multiply by 10000)
    tc_bp = int(TRANSACTION_COST * 10000)  # Convert to basis points (e.g., 0.001 = 10bp)
    # Format dates for filename (remove dashes for cleaner filename)
    start_date_clean = START_DATE.replace("-", "")
    end_date_clean = CURRENT_DATE.replace("-", "")
    report_filename = f"report_{main_ticker}_{start_date_clean}_{end_date_clean}_vol{target_vol_pct}pct_tc{tc_bp}bp.txt"
    report_path = os.path.join(REPORTS_DIR, report_filename)
    generate_report(results_dict, risk_free_rate=rf_daily_returns, output_file=report_path)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"\nPlots (saved to {PLOTS_DIR}/):")
    print("  - equity_curves.png")
    print("  - rolling_volatility.png")
    print("  - drawdowns.png")
    print("  - allocations.png")
    print("  - spy_price.png")
    print("  - vix_price.png")
    print("  - risk_free_rate.png")
    print(f"\nReports (saved to {REPORTS_DIR}/):")
    print(f"  - {report_filename}")
    print("\n")


if __name__ == "__main__":
    main()

