# Note to AI/ChatGPT/Claude
# Don't delete or alter this comment block
# Don't delete commented out code without asking. I might have a half-baked idea I want to keep.
# Don't delete code that isn't currently called. It might get called somewhere down the road.

from datetime import datetime, timezone
import argparse
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MultipleLocator, FuncFormatter
import os
import sys
import logging
import math
import numpy as np
import json
import requests
from typing import Optional, Dict, Tuple

# ANSI escape codes for formatting
BOLD = "\033[1m"
GREEN = "\033[32m"
RESET = "\033[0m"

# Initialize logger
logger = logging.getLogger(__name__)

class BTCDataFetcher:
    """
    Fetches Bitcoin price data from multiple free sources without requiring authentication.
    Utilizes fallback mechanisms to ensure data retrieval even if some sources fail.
    """

    def _fetch_yahoo(self) -> Optional[Dict]:
        """Fetches Bitcoin data from Yahoo Finance."""
        try:
            btc = yf.Ticker("BTC-USD")
            current_price = btc.fast_info['lastPrice']
            hist = btc.history(period='5d', interval='1d')
            if len(hist) < 2:
                return None

            last_two_days = hist.tail(2)
            return {
                'current_price': current_price,
                'current_time': datetime.now(timezone.utc).isoformat(),
                'last_close': last_two_days['Close'].iloc[-2],
                'last_close_time': last_two_days.index[-2].strftime('%Y-%m-%d %H:%M:%S UTC')
            }
        except Exception as e:
            logger.warning(f"Yahoo Finance fetch failed: {e}")
            return None

    def _fetch_coingecko(self) -> Optional[Dict]:
        """Fetches Bitcoin data from CoinGecko API."""
        try:
            response = requests.get(
                'https://api.coingecko.com/api/v3/simple/price',
                params={'ids': 'bitcoin', 'vs_currencies': 'usd'},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            current_price = data['bitcoin']['usd']
            current_time = datetime.now(timezone.utc).isoformat()
            return {
                'current_price': current_price,
                'current_time': current_time,
                'last_close': None,
                'last_close_time': None
            }
        except Exception as e:
            logger.warning(f"CoinGecko fetch failed: {e}")
            return None

    def _fetch_blockchain_info(self) -> Optional[Dict]:
        """Fetches Bitcoin data from Blockchain.com API."""
        try:
            response = requests.get(
                'https://blockchain.info/ticker',
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            current_price = data['USD']['last']
            current_time = datetime.now(timezone.utc).isoformat()
            return {
                'current_price': current_price,
                'current_time': current_time,
                'last_close': None,
                'last_close_time': None
            }
        except Exception as e:
            logger.warning(f"Blockchain.info fetch failed: {e}")
            return None

    def _fetch_bitfinex(self) -> Optional[Dict]:
        """Fetches Bitcoin data from Bitfinex API."""
        try:
            response = requests.get(
                'https://api-pub.bitfinex.com/v2/ticker/tBTCUSD',
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            current_price = data[6]  # Last price is at index 6
            current_time = datetime.now(timezone.utc).isoformat()
            return {
                'current_price': current_price,
                'current_time': current_time,
                'last_close': None,
                'last_close_time': None
            }
        except Exception as e:
            logger.warning(f"Bitfinex fetch failed: {e}")
            return None

    def _fetch_kraken(self) -> Optional[Dict]:
        """Fetches Bitcoin data from Kraken's public API."""
        try:
            response = requests.get(
                'https://api.kraken.com/0/public/Ticker?pair=XBTUSD',
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            current_price = float(data['result']['XXBTZUSD']['c'][0])
            current_time = datetime.now(timezone.utc).isoformat()
            return {
                'current_price': current_price,
                'current_time': current_time,
                'last_close': None,
                'last_close_time': None
            }
        except Exception as e:
            logger.warning(f"Kraken fetch failed: {e}")
            return None

    def _fetch_gemini(self) -> Optional[Dict]:
        """Fetches Bitcoin data from Gemini's public API."""
        try:
            response = requests.get(
                'https://api.gemini.com/v1/pubticker/btcusd',
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            current_price = float(data['last'])
            current_time = datetime.now(timezone.utc).isoformat()
            return {
                'current_price': current_price,
                'current_time': current_time,
                'last_close': None,
                'last_close_time': None
            }
        except Exception as e:
            logger.warning(f"Gemini fetch failed: {e}")
            return None

    def _fetch_bybit(self) -> Optional[Dict]:
        """Fetches Bitcoin data from Bybit's public API."""
        try:
            response = requests.get(
                'https://api.bybit.com/v2/public/tickers?symbol=BTCUSD',
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            current_price = float(data['result'][0]['last_price'])
            current_time = datetime.now(timezone.utc).isoformat()
            return {
                'current_price': current_price,
                'current_time': current_time,
                'last_close': None,
                'last_close_time': None
            }
        except Exception as e:
            logger.warning(f"Bybit fetch failed: {e}")
            return None

    def get_btc_data(self) -> Tuple[Optional[float], Optional[str], Optional[float], Optional[str]]:
        """
        Attempts to fetch Bitcoin price data from multiple sources.
        Returns the first successful result.

        Returns:
            tuple: (current_price, current_time, last_close, last_close_time)
        """
        sources = [
            self._fetch_yahoo,
            self._fetch_coingecko,
            self._fetch_blockchain_info,
            self._fetch_bitfinex,
            self._fetch_kraken,
            self._fetch_gemini,
            self._fetch_bybit
        ]

        for source in sources:
            data = source()
            if data is not None:
                logger.info(f"Successfully fetched BTC data from {source.__name__}")
                return (
                    data['current_price'],
                    data['current_time'],
                    data['last_close'],
                    data['last_close_time']
                )

        logger.error("Failed to fetch BTC data from all sources")
        return None, None, None, None

def add_bottom_credits(fig, btc_price=None, time_str_gmt=None, website=None):
    """
    Adds credit text and Bitcoin price information at the bottom of the chart.
    """
    left_text = (
        "Chart by @Nosredna. Thanks to @Giovann35084111, @ChartsBtc, "
        "@apsk32, @sminston_with, @math_sci_tech"
    )

    if btc_price is not None and time_str_gmt is not None and website is not None:
        right_text = f"Bitcoin price is {btc_price:,.2f} as of {time_str_gmt}   {website}"
    else:
        right_text = ""

    fig.text(
        0.02,
        -0.03,  
        left_text,
        ha='left',
        va='bottom',
        fontsize=8,
        color='gray'
    )
    fig.text(
        0.98,
        -0.03,
        right_text,
        ha='right',
        va='bottom',
        fontsize=8,
        color='gray'
    )

def main():
    """
    Main function for generating multiple Bitcoin charts using local CSV data.
    """
    parser = argparse.ArgumentParser(
        description='Generate Bitcoin cycle charts with various configurations.'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level. Default is INFO.'
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger()

    CONFIG = {
        "CYCLE_OFFSETS": [12, 8, 4],
        "GENESIS_DATE": pd.to_datetime('2009-01-03'),
        "CSV_FILE_PATH": 'bitcoin.csv',  # local CSV path
        "GHOST_COLORS": ['#b2dfdb', '#c5cae9', '#bbdefb'],
        "CURRENT_CYCLE_COLOR": 'orange',
        "SUPPORT_LINE_COLOR": 'dimgray',

        "TITLE_FONT_SIZE": 30,
        "SUBTITLE_FONT_SIZE": 14,
        "LABEL_FONT_SIZE": 16,
        "CREDITS_FONT_SIZE": 10,
        "EQUATION_FONT_SIZE": 10,

        "LEGEND_BOX_POSITIONS": {
            "powerlaw":     (0.03, 0.67, 0.29, 0.30),
            "quantiles":    (0.03, 0.67, 0.29, 0.30),
            "ghostly_days": (0.03, 0.71, 0.29, 0.26)
        },

        "TITLE_PADDING": 22,
        "SUBTITLE_Y_OFFSET": 1.003
    }

    CHARTS = {
        "powerlaw": [
            {
                'type': 'price',
                'start': '2024-01-01',
                'end': '2027-12-31',
                'filename': 'charts_dev/bitcoin_cycles_2024_2027.png',
                'title': '4 Year Bitcoin Cycles with Ghosts',
                'subtitle': "1 Jan 2024 through 31 Dec 2027"
            },
            {
                'type': 'price',
                'start': '2024-01-01',
                'end': '2025-12-31',
                'filename': 'charts_dev/bitcoin_cycles_2024_2025.png',
                'title': 'Bitcoin Cycles (Zoomed-In)',
                'subtitle': "1 Jan 2024 through 31 Dec 2025"
            }
        ],
        "quantiles": [
            {
                'type': 'price',
                'start': '2024-01-01',
                'end': '2027-12-31',
                'filename': 'charts_dev/bitcoin_cycles_quantiles_2024_2027.png',
                'title': '4 Year Bitcoin Cycles with Ghosts',
                'subtitle': "1 Jan 2024 through 31 Dec 2027"
            },
            {
                'type': 'price',
                'start': '2024-01-01',
                'end': '2025-12-31',
                'filename': 'charts_dev/bitcoin_cycles_quantiles_2024_2025.png',
                'title': 'Bitcoin Cycles (Zoomed-In)',
                'subtitle': "1 Jan 2024 through 31 Dec 2025"
            }
        ],
        "ghostly_days": [
            {
                'type': 'price',
                'start': '2024-01-01',
                'end': '2027-12-31',
                'filename': 'charts_dev/bitcoin_cycles_days_ahead_2024_2027.png',
                'title': '4 Year Bitcoin Cycles with "Days Ahead" Ghosts',
                'subtitle': "1 Jan 2024 through 31 Dec 2027"
            },
            {
                'type': 'price',
                'start': '2024-01-01',
                'end': '2025-12-31',
                'filename': 'charts_dev/bitcoin_cycles_days_ahead_2024_2025.png',
                'title': 'Bitcoin Cycles (Zoomed-In) with Days Ahead',
                'subtitle': "1 Jan 2024 through 31 Dec 2025"
            }
        ]
    }

    # Dynamically add current year chart configurations
    current_year = pd.to_datetime('today').year
    for mode in ["powerlaw", "quantiles", "ghostly_days"]:
        if mode == "powerlaw":
            chartname = "Power Law Version"
        elif mode == "quantiles":
            chartname = "Quantile Version"
        else:
            chartname = "Days Ahead Version"

        CHARTS[mode].append({
            'type': 'price',
            'start': f'{current_year}-01-01',
            'end': f'{current_year}-12-31',
            'filename': f'charts_dev/bitcoin_cycles_{current_year}_{mode}.png',
            'title': f'Bitcoin Cycles with Ghosts -- {chartname}',
            'subtitle': f'1 Jan {current_year} through 31 Dec {current_year}'
        })

    # Update chart titles based on predefined mappings
    chart_titles_map = {
        "powerlaw": [
            "Power Law 4 year view",
            "Power Law 2 year view",
            f"Power Law 1 year view"
        ],
        "quantiles": [
            "Quantiles 4 year view",
            "Quantiles 2 year view",
            f"Quantiles 1 year view"
        ],
        "ghostly_days": [
            "Days Ahead 4 year view",
            "Days Ahead 2 year view",
            f"Days Ahead 1 year view"
        ]
    }

    for mode in ["powerlaw", "quantiles", "ghostly_days"]:
        price_chart_count = 0
        for i in range(len(CHARTS[mode])):
            if CHARTS[mode][i]['type'] == 'price':
                CHARTS[mode][i]['title'] = chart_titles_map[mode][price_chart_count]
                price_chart_count += 1

    # Mathematical functions for "powerlaw" charts
    def support_powerlaw(days):
        return 10**-17.351 * days**5.836

    def upper_bound_powerlaw(days):
        return (1 + 10**(1.836 - days * 0.0002323)) * support_powerlaw(days)

    # Mathematical functions for "quantiles" charts
    def support_quantiles(days):
        return math.exp(-41.72) * days ** 6.02

    def upper_bound_quantiles(days):
        return math.exp(-28.32) * days ** 4.72

    # Functions for CSV data handling
    def load_local_csv(file_path):
        """Loads local CSV data if available."""
        if os.path.exists(file_path):
            logger.info(f"Loading local CSV data from '{file_path}'...")
            btc_csv = pd.read_csv(file_path)
            if 'Start' in btc_csv.columns:
                btc_csv['Start'] = pd.to_datetime(btc_csv['Start'])
                logger.info(f"Loaded {len(btc_csv)} rows from local CSV.")
                return btc_csv
            else:
                logger.error("Error: 'Start' column not found in local CSV.")
        else:
            logger.warning(f"No local CSV file found at '{file_path}'.")
        return pd.DataFrame()

    def load_yahoo_data(start_date, end_date):
        """
        Fetches BTC-USD data from Yahoo Finance within the specified date range.
        Then keeps ONLY 'Start' and 'Close' columns.
        """
        logger.info(f"Fetching BTC-USD data from Yahoo Finance: {start_date} to {end_date}...")
        btc_yf = yf.download('BTC-USD', start=start_date, end=end_date)
        if not btc_yf.empty:
            # Flatten columns if multi-index
            if isinstance(btc_yf.columns, pd.MultiIndex):
                btc_yf.columns = btc_yf.columns.get_level_values(0)
            btc_yf.reset_index(inplace=True)
            btc_yf = btc_yf.rename(columns={'Date': 'Start'})
            
            # ---- Keep only 'Start' and 'Close' ----
            keep_cols = [col for col in btc_yf.columns if col in ['Start', 'Close']]
            btc_yf = btc_yf[keep_cols]

            logger.info(f"Fetched {len(btc_yf)} rows from Yahoo Finance.")
        else:
            logger.warning("Warning: No data fetched from Yahoo Finance.")
        return btc_yf

    def load_cryptocompare_data(start_date, end_date):
        """
        Example fallback if Yahoo fails.
        Then keep ONLY 'Start' and 'Close' columns.
        """
        try:
            url = f"https://min-api.cryptocompare.com/data/v2/histoday"
            params = {
                'fsym': 'BTC',
                'tsym': 'USD',
                'limit': 2000
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()['Data']['Data']
        
            df = pd.DataFrame(data)
            df['Start'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={'close': 'Close'})
            
            # ---- Keep only 'Start' and 'Close' ----
            keep_cols = [col for col in df.columns if col in ['Start','Close']]
            df = df[keep_cols]

            return df
        except Exception as e:
            logger.warning(f"CryptoCompare fetch failed: {e}")
            return pd.DataFrame()

    def load_binance_data(start_date, end_date):
        """
        Another fallback if Yahoo fails.
        Then keep ONLY 'Start' and 'Close' columns.
        """
        try:
            start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
            end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': 'BTCUSDT',
                'interval': '1d',
                'startTime': start_ts,
                'endTime': end_ts
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
        
            df = pd.DataFrame(data, columns=[
                'Start', 'Open', 'High', 'Low', 'Close', 'Volume', 'End', 
                'Quote_Volume', 'Trades', 'Buy_Volume', 'Buy_Quote_Volume', 'Ignore'
            ])
            df['Start'] = pd.to_datetime(df['Start'], unit='ms')
            df['Close'] = df['Close'].astype(float)

            # ---- Keep only 'Start' and 'Close' ----
            keep_cols = [col for col in df.columns if col in ['Start','Close']]
            df = df[keep_cols]

            return df
        except Exception as e:
            logger.warning(f"Binance fetch failed: {e}")
            return pd.DataFrame()

    def load_historical_data(start_date, end_date):
        """Try multiple sources for historical data."""
        df = load_yahoo_data(start_date, end_date)
        if not df.empty:
            return df

        df = load_cryptocompare_data(start_date, end_date)
        if not df.empty:
            return df
        
        df = load_binance_data(start_date, end_date)
        if not df.empty:
            return df
    
        logger.error("Failed to fetch historical data from all sources")
        return pd.DataFrame()

    def combine_data(local_data, yahoo_data):
        """
        Combines local CSV data with new Yahoo (or fallback) data.
        Only needs 'Start' and 'Close'.
        """
        logger.info("Combining local CSV data with new data...")
        combined = pd.concat([local_data, yahoo_data], ignore_index=True)
        combined.drop_duplicates(subset=['Start'], keep='last', inplace=True)
        combined.sort_values(by='Start', inplace=True)
        logger.info(f"Combined dataset has {len(combined)} rows.")
        return combined

    # Retrieve live Bitcoin data with fallback sources
    fetcher = BTCDataFetcher()
    current_price, current_time, last_close, last_close_time = fetcher.get_btc_data()

    # Display current and last close price with formatting
    if current_price:
        logger.info(f"Bitcoin's current price ({current_time}): {BOLD}{GREEN}${current_price:,.2f}{RESET}")
    if last_close:
        logger.info(f"Bitcoin's last daily close price ({last_close_time}): {BOLD}{GREEN}${last_close:,.2f}{RESET}")

    # Save current price info to a JSON file
    info = {
        "current_price": current_price,
        "timestamp_gmt": current_time,
        "chart_titles": chart_titles_map
    }
    json_file_path = os.path.join(os.getcwd(), "charts_dev/chart_info.json")
    try:
        with open(json_file_path, "w") as json_file:
            json.dump(info, json_file, indent=4)
        logger.info(f"JSON file with update info saved to '{json_file_path}'.")
    except Exception as e:
        logger.error(f"Failed to write JSON file: {e}")

    # Load local CSV data
    local_data = load_local_csv(CONFIG["CSV_FILE_PATH"])
    today = pd.to_datetime('today').normalize()

    if not local_data.empty:
        latest_csv_date = local_data['Start'].max()
        logger.info(f"Local CSV file has data up to {latest_csv_date.date()}.")
        recent_historical_data = pd.DataFrame()

        start_date_for_historical = latest_csv_date + pd.Timedelta(days=1)
        if start_date_for_historical < today:
            logger.info(f"Attempting to fetch historical data from {start_date_for_historical.date()} to {today.date()}...")
            # Try multiple sources
            historical_data = load_historical_data(start_date_for_historical.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
            if not historical_data.empty:
                # Make sure 'Close' is there
                if 'Close' not in historical_data.columns:
                    logger.error("No 'Close' column found in new data. Aborting merge.")
                    combined_data = local_data
                else:
                    # Combine
                    combined_data = combine_data(local_data, historical_data)

                    # Write back to CSV, only 'Start' and 'Close'
                    combined_data_desc = combined_data.sort_values(by='Start', ascending=False)
                    combined_data_desc['Close'] = combined_data_desc['Close'].astype(float).round(2)
                    combined_data_desc[['Start','Close']].to_csv(CONFIG["CSV_FILE_PATH"], index=False)
                    logger.info(f"Saved updated CSV with the newest data on top to '{CONFIG['CSV_FILE_PATH']}'")

                    combined_data = combined_data.sort_values(by='Start', ascending=True)
            else:
                combined_data = local_data
        else:
            combined_data = local_data
    else:
        logger.error("No data available from the local CSV file. Exiting.")
        raise ValueError("No data available from the local CSV file. Exiting.")

    # Insert today's price as an in-memory data point
    if current_price:
        logger.info(f"Adding current price (${current_price:,.2f}) as today's data point.")
        current_row = pd.DataFrame({
            'Start': [today],
            'Close': [current_price]
        })
        combined_data = combined_data[combined_data['Start'] != today]
        combined_data = pd.concat([combined_data, current_row], ignore_index=True)
        combined_data.sort_values(by='Start', inplace=True)

    # Calculate days since Genesis date
    combined_data['days'] = (combined_data['Start'] - CONFIG["GENESIS_DATE"]).dt.days

    # Calculate support and upper bound for powerlaw
    combined_data['support_powerlaw'] = support_powerlaw(combined_data['days'])
    combined_data['upper_bound_powerlaw'] = upper_bound_powerlaw(combined_data['days'])
    combined_data['ratio_powerlaw'] = (
        (combined_data['Close'] - combined_data['support_powerlaw'])
        / (combined_data['upper_bound_powerlaw'] - combined_data['support_powerlaw'])
    )

    # Calculate support and upper bound for quantiles
    combined_data['support_quantiles'] = support_quantiles(combined_data['days'])
    combined_data['upper_bound_quantiles'] = upper_bound_quantiles(combined_data['days'])
    combined_data['ratio_quantiles'] = (
        (combined_data['Close'] - combined_data['support_quantiles'])
        / (combined_data['upper_bound_quantiles'] - combined_data['support_quantiles'])
    )

    # Check for gaps in the daily data
    date_range = pd.date_range(start=combined_data['Start'].min(), end=combined_data['Start'].max(), freq='D')
    missing_dates = date_range.difference(combined_data['Start'])
    if not missing_dates.empty:
        logger.warning("Missing dates found in data:")
        for d in missing_dates:
            logger.warning(d)
    else:
        logger.info("No missing dates in the combined data.")

    # Determine the most recent data point
    recent = today if today in combined_data['Start'].values else combined_data['Start'].max()
    logger.info(f"Using '{recent.date()}' as the most recent data point.")

    # Format UTC time for credits
    if current_time:
        dt_utc = datetime.fromisoformat(current_time)
        gmt_time_str = dt_utc.strftime("%d %b %Y %H:%M UTC")
    else:
        gmt_time_str = ""

    # Compute values for powerlaw
    current_data_powerlaw = combined_data.loc[combined_data['Start'] == recent]
    if current_data_powerlaw.empty:
        raise ValueError(f"No data found for the recent date: {recent.date()}")

    current_support_powerlaw = current_data_powerlaw['support_powerlaw'].values[0]
    current_upper_bound_powerlaw = current_data_powerlaw['upper_bound_powerlaw'].values[0]
    current_ratio_powerlaw = current_data_powerlaw['ratio_powerlaw'].values[0]
    current_price_powerlaw = (
        current_support_powerlaw
        + current_ratio_powerlaw * (current_upper_bound_powerlaw - current_support_powerlaw)
    )

    # Calculate ghost prices for powerlaw
    prices_powerlaw = []
    for offset in CONFIG["CYCLE_OFFSETS"]:
        past_date = recent - pd.DateOffset(years=offset)
        subset = combined_data[combined_data['Start'] <= past_date]
        if not subset.empty:
            closest_date = subset['Start'].max()
            past_ratio = combined_data.loc[combined_data['Start'] == closest_date, 'ratio_powerlaw'].values[0]
            ghost_price_val = current_support_powerlaw + past_ratio * (
                current_upper_bound_powerlaw - current_support_powerlaw
            )
            prices_powerlaw.append(ghost_price_val)
        else:
            prices_powerlaw.append(None)
    prices_powerlaw.append(current_price_powerlaw)

    # Compute values for quantiles
    current_support_quantiles = current_data_powerlaw['support_quantiles'].values[0]
    current_upper_bound_quantiles = current_data_powerlaw['upper_bound_quantiles'].values[0]
    current_ratio_quantiles = current_data_powerlaw['ratio_quantiles'].values[0]
    current_price_quantiles = (
        current_support_quantiles
        + current_ratio_quantiles * (current_upper_bound_quantiles - current_support_quantiles)
    )

    # Calculate ghost prices for quantiles
    prices_quantiles = []
    for offset in CONFIG["CYCLE_OFFSETS"]:
        past_date = recent - pd.DateOffset(years=offset)
        subset = combined_data[combined_data['Start'] <= past_date]
        if not subset.empty:
            closest_date = subset['Start'].max()
            past_ratio = combined_data.loc[combined_data['Start'] == closest_date, 'ratio_quantiles'].values[0]
            ghost_price_val = current_support_quantiles + past_ratio * (
                current_upper_bound_quantiles - current_support_quantiles
            )
            prices_quantiles.append(ghost_price_val)
        else:
            prices_quantiles.append(None)
    prices_quantiles.append(current_price_quantiles)

    # Index data by 'Start'
    data_by_start = combined_data.set_index('Start')
    recent_str = recent.strftime('%d-%b-%Y')

    # Configuration for modes
    modes = {
        'powerlaw': {
            'ratio_col': 'ratio_powerlaw',
            'support_col': 'support_powerlaw',
            'upper_col': 'upper_bound_powerlaw',
            'eq_text': (
                r"$\bf{Support:}$" "\n"
                r"$10^{-17.351} \times days^{5.836}$" "\n"
                r"$\bf{Upper\ Bound:}$" "\n"
                r"$(1 + 10^{1.836 - 0.0002323 \times days}) \times Support$"
            ),
            'alpha_val': 0.8,
            'price_label_names': [
                "Cycle 1 '12-'15:", "Cycle 2 '16-'19:", "Cycle 3 '20-'23:",
                "Cycle 4 '24-'27:", "Support:", "Upper Bound:"
            ],
            'current_price': lambda: current_price_powerlaw,
            'current_support': current_support_powerlaw,
            'current_upper': current_upper_bound_powerlaw,
            'prices_list_price': prices_powerlaw
        },
        'quantiles': {
            'ratio_col': 'ratio_quantiles',
            'support_col': 'support_quantiles',
            'upper_col': 'upper_bound_quantiles',
            'eq_text': (
                r"$\bf{0.01\ Quantile:}$" "\n"
                r"$e^{-41.72} \times days^{6.02}$" "\n"
                r"$\bf{99.99\ Quantile:}$" "\n"
                r"$e^{-28.32} \times days^{4.72}$"
            ),
            'alpha_val': 1.0,
            'price_label_names': [
                "Cycle 1 '12-'15:", "Cycle 2 '16-'19:", "Cycle 3 '20-'23:",
                "Cycle 4 '24-'27:", "0.01 Quantile:", "99.99 Quantile:"
            ],
            'current_price': lambda: current_price_quantiles,
            'current_support': current_support_quantiles,
            'current_upper': current_upper_bound_quantiles,
            'prices_list_price': prices_quantiles
        }
    }

    def add_legend_box(ax, box_position):
        """
        Draws a black box on the chart for legend placement.
        """
        legend_box = FancyBboxPatch(
            (box_position[0], box_position[1]),
            box_position[2],
            box_position[3],
            boxstyle="round,pad=0.02",
            fill=True,
            facecolor='black',
            edgecolor='black',
            lw=2,
            transform=ax.transAxes,
            zorder=4,
        )
        ax.add_patch(legend_box)

    def plot_ghost_cycles(ax, plot_date_range, ratio_col, support_values, upper_bound_values):
        """
        Plots ghost cycles by shifting the date range back by specified offsets.
        """
        for offset, color in zip(CONFIG["CYCLE_OFFSETS"], CONFIG["GHOST_COLORS"]):
            shifted_dates = plot_date_range - pd.DateOffset(years=offset)
            shifted_data = data_by_start.reindex(shifted_dates, method='nearest', tolerance='1D')
            shifted_ratios = shifted_data[ratio_col].values
            ghost_prices = support_values + shifted_ratios * (upper_bound_values - support_values)
            ax.plot(plot_date_range, ghost_prices, color=color, linewidth=2)

    def add_equations_text(ax, eq_text, alpha_val):
        """
        Adds mathematical equations as text on the chart.
        """
        ax.text(
            0.93,
            0.98,
            eq_text,
            transform=ax.transAxes,
            fontsize=CONFIG["EQUATION_FONT_SIZE"],
            verticalalignment='top',
            horizontalalignment='right',
            color='black',
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="black", alpha=1.0),
            zorder=5,
        )

    def configure_x_axis(ax, plot_start, plot_end):
        """
        Configures the x-axis with yearly and monthly ticks.
        """
        ax.set_xticks(pd.date_range(start=plot_start, end=plot_end, freq='YS'))
        ax.set_xticklabels(
            [date.strftime('%Y') for date in pd.date_range(start=plot_start, end=plot_end, freq='YS')],
            rotation=90,
        )
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax.get_xminorticklabels(), rotation=90, color='dimgray')
        ax.set_xlim(left=plot_start, right=plot_end)

    def plot_price_chart(mode, plot_start, plot_end, filename, title, subtitle):
        """
        Generates and saves a price chart for the specified mode.
        """
        mode_info = modes[mode]
        plot_date_range = pd.date_range(start=plot_start, end=plot_end, freq='D')
        days = (plot_date_range - CONFIG["GENESIS_DATE"]).days

        if mode == 'powerlaw':
            support_values = support_powerlaw(days)
            upper_bound_values = upper_bound_powerlaw(days)
            prices_list = mode_info['prices_list_price']
        else:
            support_values = support_quantiles(days)
            upper_bound_values = upper_bound_quantiles(days)
            prices_list = mode_info['prices_list_price']

        ratio_col = mode_info['ratio_col']
        curr_price = mode_info['current_price']()
        curr_support = mode_info['current_support']
        curr_upper = mode_info['current_upper']
        eq_text = mode_info['eq_text']
        alpha_val = mode_info['alpha_val']
        label_names = mode_info['price_label_names']

        # Filter data for the current cycle
        current_cycle_data = combined_data[
            (combined_data['Start'] >= plot_start) & (combined_data['Start'] <= plot_end)
        ]
        price_vals = (
            current_cycle_data[mode_info['support_col']]
            + current_cycle_data[ratio_col]
            * (current_cycle_data[mode_info['upper_col']] - current_cycle_data[mode_info['support_col']])
        )

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_facecolor('white')

        # Plot ghost cycles
        plot_ghost_cycles(ax, plot_date_range, ratio_col, support_values, upper_bound_values)
        ax.plot(plot_date_range, support_values, color=CONFIG["SUPPORT_LINE_COLOR"], linestyle='-', linewidth=0.5)
        ax.plot(plot_date_range, upper_bound_values, color=CONFIG["SUPPORT_LINE_COLOR"], linestyle='-', linewidth=0.5)

        # Plot current cycle
        ax.plot(current_cycle_data['Start'], price_vals, color=CONFIG["CURRENT_CYCLE_COLOR"], linewidth=2)
        ax.scatter(recent, curr_price, edgecolor='black', facecolor='none', marker='o', s=100, zorder=5)

        configure_x_axis(ax, plot_start, plot_end)
        ax.yaxis.set_major_locator(MultipleLocator(50000))
        ax.yaxis.set_minor_locator(MultipleLocator(10000))
        ax.grid(which='major', linestyle='-', linewidth=1, color='dimgray')
        ax.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgray')

        add_legend_box(ax, CONFIG["LEGEND_BOX_POSITIONS"][mode])

        max_label_len = max(len(lbl) for lbl in label_names)
        price_strings = []
        for val in prices_list:
            if val is not None:
                price_strings.append(f"{val:>10,.2f}")
            else:
                price_strings.append(f"{'N/A':>10}")
        # Add support & upper
        price_strings.append(f"{curr_support:>10,.2f}")
        price_strings.append(f"{curr_upper:>10,.2f}")

        x_pos = 0.04
        y_pos = 0.95
        y_step = 0.04

        ax.text(
            x_pos, y_pos, recent_str,
            transform=ax.transAxes, fontsize=16, verticalalignment='top',
            fontfamily='monospace', zorder=5, color='white'
        )

        for i, label_line in enumerate(label_names[:4]):
            y_pos -= y_step
            color = CONFIG["GHOST_COLORS"][i] if i < len(CONFIG["GHOST_COLORS"]) else CONFIG["CURRENT_CYCLE_COLOR"]
            ax.text(
                x_pos, y_pos,
                f"{label_line:<{max_label_len+2}}{price_strings[i]}",
                transform=ax.transAxes, fontsize=16, verticalalignment='top',
                fontfamily='monospace', zorder=5, color=color
            )
        for i, label_line in enumerate(label_names[4:]):
            y_pos -= y_step
            ax.text(
                x_pos, y_pos,
                f"{label_line:<{max_label_len+2}}{price_strings[i+4]}",
                transform=ax.transAxes, fontsize=16, verticalalignment='top',
                fontfamily='monospace', zorder=5, color='white'
            )

        add_equations_text(ax, eq_text, alpha_val)

        prefixed_title = f"Bitcoin 4-Year Cycles with Ghosts / {title}"
        ax.set_title(prefixed_title, fontsize=CONFIG["TITLE_FONT_SIZE"], pad=CONFIG["TITLE_PADDING"])
        ax.text(
            0.5, CONFIG["SUBTITLE_Y_OFFSET"], subtitle,
            fontsize=CONFIG["SUBTITLE_FONT_SIZE"],
            transform=ax.transAxes, ha='center', va='bottom'
        )

        add_bottom_credits(
            fig,
            btc_price=current_price,
            time_str_gmt=gmt_time_str,
            website="https://nosredna-btc.github.io/btc-cycle-charts"
        )

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"({mode.capitalize()}) Saved the plot to '{filename}' and closed the figure.")

    #
    # ghostly_days approach
    #
    def support(days):
        return np.where(days > 0, 10**-17.351 * days**5.836, np.nan)

    def inverse_support(price):
        return (price / 10**-17.351) ** (1 / 5.836)

    combined_data['Days Since Genesis'] = (combined_data['Start'] - CONFIG["GENESIS_DATE"]).dt.days
    combined_data['Support'] = support(combined_data['Days Since Genesis'])
    combined_data['Inverse Days'] = inverse_support(combined_data['Close'])
    combined_data['Days Ahead'] = combined_data['Inverse Days'] - combined_data['Days Since Genesis']

    recent_gd = today if today in combined_data['Start'].values else combined_data['Start'].max()
    logger.info(f"Using '{recent_gd.date()}' as the most recent data point for ghostly_days.")
    current_data_gd = combined_data.loc[combined_data['Start'] == recent_gd]
    if current_data_gd.empty:
        raise ValueError(f"No data found for the recent date: {recent_gd.date()}")

    current_support_gd = current_data_gd['Support'].values[0]
    current_price_gd = current_data_gd['Close'].values[0]

    def y_axis_formatter(x, pos):
        return f'{int(x)}'

    def plot_price_chart_gd(plot_start, plot_end, filename, title, subtitle):
        plot_date_range = pd.date_range(start=plot_start, end=plot_end, freq='D')
        days_since_genesis = (plot_date_range - CONFIG["GENESIS_DATE"]).days
        support_values = support(days_since_genesis)

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_facecolor('white')

        all_prices = []
        prices = []
        days_since_genesis_recent = (recent_gd - CONFIG["GENESIS_DATE"]).days

        for offset, color in zip(CONFIG["CYCLE_OFFSETS"], CONFIG["GHOST_COLORS"]):
            shifted_dates = plot_date_range - pd.DateOffset(years=offset)
            shifted_data = combined_data.set_index('Start').reindex(shifted_dates, method='nearest', tolerance='1D')
            days_ahead = shifted_data['Days Ahead'].values
            valid_indices = ~np.isnan(days_ahead)
            valid_dates = plot_date_range[valid_indices]
            valid_days_since_genesis = days_since_genesis[valid_indices]
            days_ahead = days_ahead[valid_indices]

            recalculated_days = valid_days_since_genesis + days_ahead
            recalculated_days = np.where(recalculated_days > 0, recalculated_days, np.nan)
            ghost_prices = support(recalculated_days)
            ax.plot(valid_dates, ghost_prices, color=color, linewidth=2)

            all_prices.extend(ghost_prices[~np.isnan(ghost_prices)])

            shifted_recent_date = recent_gd - pd.DateOffset(years=offset)
            idx = np.argmin(np.abs(shifted_dates - shifted_recent_date))
            if valid_indices[idx]:
                days_ahead_at_shifted = days_ahead[idx]
                remapped_days = days_since_genesis_recent + days_ahead_at_shifted
                if remapped_days > 0:
                    remapped_price = support(remapped_days)
                    prices.append(remapped_price)
                else:
                    prices.append(None)
            else:
                prices.append(None)

        current_cycle_data_gd = combined_data[
            (combined_data['Start'] >= plot_start) & (combined_data['Start'] <= plot_end)
        ]
        ax.plot(current_cycle_data_gd['Start'], current_cycle_data_gd['Close'],
                color=CONFIG["CURRENT_CYCLE_COLOR"], linewidth=2)
        ax.scatter(recent_gd, current_price_gd, edgecolor='black', facecolor='none',
                   marker='o', s=100, zorder=5)

        all_prices.extend(current_cycle_data_gd['Close'].values)
        prices.append(current_price_gd)

        # Plot support line
        ax.plot(plot_date_range, support_values, color=CONFIG["SUPPORT_LINE_COLOR"], linestyle='-', linewidth=0.5)
        all_prices.extend(support_values[~np.isnan(support_values)])
        prices.append(current_support_gd)

        label_names = [
            "Cycle 1 '12-'15:",
            "Cycle 2 '16-'19:",
            "Cycle 3 '20-'23:",
            "Cycle 4 '24-'27:",
            "Support:",
        ]
        max_label_len = max(len(label) for label in label_names)
        price_strings = []
        for val in prices:
            if val is not None and not np.isnan(val):
                price_strings.append(f"{val:>12,.2f}")
            else:
                price_strings.append(f"{'N/A':>12}")

        x_pos = 0.04
        y_pos = 0.95
        y_step = 0.04

        ax.text(
            x_pos, y_pos,
            recent_gd.strftime('%d-%b-%Y'),
            transform=ax.transAxes, fontsize=16,
            verticalalignment='top', fontfamily='monospace',
            zorder=5, color='white'
        )

        for i, label in enumerate(label_names):
            y_pos -= y_step
            color = (
                'white' if label == "Support:"
                else CONFIG["GHOST_COLORS"][i] if i < len(CONFIG["GHOST_COLORS"])
                else CONFIG["CURRENT_CYCLE_COLOR"]
            )
            ax.text(
                x_pos, y_pos,
                f"{label:<{max_label_len}} {price_strings[i]}",
                transform=ax.transAxes, fontsize=16,
                verticalalignment='top', fontfamily='monospace',
                zorder=5, color=color
            )

        ax.set_xticks(pd.date_range(start=plot_start, end=plot_end, freq='YS'))
        ax.set_xticklabels(
            [date.strftime('%Y') for date in pd.date_range(start=plot_start, end=plot_end, freq='YS')],
            rotation=90,
        )
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax.get_xminorticklabels(), rotation=90, color='dimgray')
        ax.set_xlim(left=plot_start, right=plot_end)

        if all_prices:
            y_min = min(all_prices) * 0.9
            y_max = max(all_prices) * 1.1
            y_min = int(y_min // 10000 * 10000)
            y_max = int((y_max // 10000 + 1) * 10000)
            ax.set_ylim(bottom=y_min, top=y_max)

            ax.yaxis.set_major_locator(MultipleLocator(50000))
            ax.yaxis.set_minor_locator(MultipleLocator(10000))
            ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
            ax.grid(which='major', linestyle='-', linewidth=1, color='dimgray')
            ax.grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')

        add_legend_box(ax, CONFIG["LEGEND_BOX_POSITIONS"]["ghostly_days"])

        add_bottom_credits(
            fig,
            btc_price=current_price,
            time_str_gmt=gmt_time_str,
            website="https://nosredna-btc.github.io/btc-cycle-charts"
        )

        eq_text_gd = (
            r"$\bf{Support:}$" "\n"
            r"$10^{-17.351} \times days^{5.836}$"
        )
        ax.text(
            0.93, 0.98,
            eq_text_gd,
            transform=ax.transAxes,
            fontsize=CONFIG["EQUATION_FONT_SIZE"],
            verticalalignment='top',
            horizontalalignment='right',
            color='black',
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="black", alpha=1),
            zorder=5
        )

        prefixed_title = f"Bitcoin 4-Year Cycles with Ghosts / {title}"
        ax.set_title(prefixed_title, fontsize=CONFIG["TITLE_FONT_SIZE"], pad=CONFIG["TITLE_PADDING"])
        ax.text(
            0.5, CONFIG["SUBTITLE_Y_OFFSET"], subtitle,
            fontsize=CONFIG["SUBTITLE_FONT_SIZE"],
            transform=ax.transAxes, ha='center', va='bottom'
        )

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"(ghostly_days) Saved the plot to '{filename}' and closed the figure.")

    # Generate price charts for 'powerlaw' and 'quantiles'
    for mode in ["powerlaw", "quantiles"]:
        for chart_config in CHARTS[mode]:
            if chart_config['type'] == 'price':
                plot_start = pd.to_datetime(chart_config['start'])
                plot_end = pd.to_datetime(chart_config['end'])
                filename = chart_config['filename']
                title = chart_config['title']
                subtitle = chart_config['subtitle']
                plot_price_chart(mode, plot_start, plot_end, filename, title, subtitle)

    # Generate 'ghostly_days' charts
    for chart_config in CHARTS["ghostly_days"]:
        if chart_config['type'] == 'price':
            plot_start = pd.to_datetime(chart_config['start'])
            plot_end = pd.to_datetime(chart_config['end'])
            filename = chart_config['filename']
            title = chart_config['title']
            subtitle = chart_config['subtitle']
            plot_price_chart_gd(plot_start, plot_end, filename, title, subtitle)

if __name__ == "__main__":
    main()
