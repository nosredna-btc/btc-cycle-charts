# Note to AI/ChatGPT/Claude
# Don't delete or alter this comment block
# Don't delete commented out code without asking. I might have a half-baked idea I want to keep.
# Don't delete code that isn't currently called. It miught get called somewhere down the road.

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

logger = logging.getLogger(__name__)

def add_bottom_credits(fig):
    """
    Adds a single-line credit at the bottom of each figure.
    Adjust its styling (position, color, font size/weight) here.
    """
    fig.text(
        0.5,       # horizontal center
        -0.02,     # vertical offset to push text below x-axis
        "Chart by @Nosredna. Thanks to @Giovann35084111, @ChartsBtc, @apsk32, @sminston_with, @math_sci_tech",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='normal',
        color='gray'
    )

def main():
    """
    Main entry point for generating Bitcoin cycle charts with various configurations.
    Fetches or loads BTC-USD data, computes relevant metrics, and plots multiple charts.
    """
    # -----------------------------
    # 1. Parse Command-Line Args
    # -----------------------------
    parser = argparse.ArgumentParser(
        description='Generate Bitcoin cycle charts with various configurations.'
    )

    parser.add_argument(
        '--generate-ratio-charts',
        action='store_true',
        help='Generate ratio charts. By default, ratio charts are not generated.'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level. Default is INFO.'
    )

    parser.add_argument(
        '--output-csv',
        action='store_true',
        help='Enable CSV output. By default, CSV output is off.'
    )

    args = parser.parse_args()

    # Configure Logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger()

    # --------------------------------------
    # 2. Define Configurations and Helpers
    # --------------------------------------
    CONFIG = {
        "CYCLE_OFFSETS": [12, 8, 4],
        "GENESIS_DATE": pd.to_datetime('2009-01-03'),
        "CSV_FILE_PATH": 'bitcoin.csv',
        "GHOST_COLORS": ['#b2dfdb', '#c5cae9', '#bbdefb'],
        "CURRENT_CYCLE_COLOR": 'orange',
        "CREDITS_BOX_COLOR": '#fde4d0',  # Unused but retained in config
        "SUPPORT_LINE_COLOR": 'dimgray',

        "TITLE_FONT_SIZE": 30,
        "SUBTITLE_FONT_SIZE": 14,
        "LABEL_FONT_SIZE": 16,
        "CREDITS_FONT_SIZE": 10,
        "EQUATION_FONT_SIZE": 10,

        # We create separate positions for each chart type so we can nudge them as needed.
        "LEGEND_BOX_POSITIONS": {
            "powerlaw":     (0.03, 0.67, 0.29, 0.30),
            "quantiles":    (0.03, 0.67, 0.29, 0.30),
            "ghostly_days": (0.03, 0.71, 0.29, 0.26)  # Slightly shorter height
        },

        # Credits box in config but not in use
        "CREDITS_BOX_POSITION": (0.62, 0.02, 0.37, 0.10),

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
            },
            {
                'type': 'ratio',
                'start': '2024-01-01',
                'end': '2027-12-31',
                'filename': 'charts_dev/bitcoin_ratios_2024_2027.png',
                'title': 'Bitcoin Ratios Over 4 Years',
                'subtitle': "1 Jan 2024 through 31 Dec 2027"
            },
            {
                'type': 'ratio',
                'start': '2024-01-01',
                'end': '2025-12-31',
                'filename': 'charts_dev/bitcoin_ratios_2024_2025.png',
                'title': 'Bitcoin Ratios (Zoomed-In)',
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
            },
            {
                'type': 'ratio',
                'start': '2024-01-01',
                'end': '2027-12-31',
                'filename': 'charts_dev/bitcoin_ratios_quantiles_2024_2027.png',
                'title': 'Bitcoin Ratios Over 4 Years',
                'subtitle': "1 Jan 2024 through 31 Dec 2027"
            },
            {
                'type': 'ratio',
                'start': '2024-01-01',
                'end': '2025-12-31',
                'filename': 'charts_dev/bitcoin_ratios_quantiles_2024_2025.png',
                'title': 'Bitcoin Ratios (Zoomed-In)',
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

    # Dynamically add a single-year chart config for the current year
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

    # ----------------------------------------------------------------
    # New lines to add the data-driven titles for the 9 price charts.
    # We insert them into the JSON and override the CHARTS dictionary.
    # ----------------------------------------------------------------
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

    # Override only the 'price' charts (ignore ratio charts)
    for mode in ["powerlaw", "quantiles", "ghostly_days"]:
        price_chart_count = 0
        for i in range(len(CHARTS[mode])):
            if CHARTS[mode][i]['type'] == 'price':
                CHARTS[mode][i]['title'] = chart_titles_map[mode][price_chart_count]
                price_chart_count += 1
    # ----------------------------------------------------------------

    # Power-law functions
    def support_powerlaw(days):
        return 10**-17.351 * days**5.836

    def upper_bound_powerlaw(days):
        return (1 + 10**(1.836 - days * 0.0002323)) * support_powerlaw(days)

    # Quantiles functions
    def support_quantiles(days):
        return math.exp(-41.72) * days ** 6.02

    def upper_bound_quantiles(days):
        return math.exp(-28.32) * days ** 4.72

    # ---------------------------------
    # 3. Data Loading / Merging Logic
    # ---------------------------------
    def load_local_csv(file_path):
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
        logger.info(f"Fetching BTC-USD data from Yahoo Finance: {start_date} to {end_date}...")
        # The line below is commented out, but we are not removing it per instructions:
        # btc_yf = yf.download('BTC-USD', start=start_date, end=end_date)
        btc_yf = yf.download('BTC-USD', start=start_date, end=end_date)
        if not btc_yf.empty:
            # Check if columns are multi-level
            if isinstance(btc_yf.columns, pd.MultiIndex):
                # Flatten multi-level columns and keep only the 'Close' values
                btc_yf.columns = btc_yf.columns.get_level_values(0)
            # Reset index as before
            btc_yf.reset_index(inplace=True)
            logger.info(f"Yahoo data after normalizing columns:\n{btc_yf.head()}")            
            logger.info(f"Raw Yahoo Finance data fetched:\n{btc_yf}")
            logger.info(f"DataFrame metadata: columns={btc_yf.columns}, index={btc_yf.index}")
            
            logger.info(f"Fetched {len(btc_yf)} rows from Yahoo Finance.")
            btc_yf.reset_index(inplace=True)
            btc_yf = btc_yf.rename(columns={'Date': 'Start'})
        else:
            logger.warning("Warning: No data fetched from Yahoo Finance.")
        return btc_yf

    def combine_data(local_data, yahoo_data):
        logger.info("Combining local CSV data with Yahoo Finance data...")
        combined = pd.concat([local_data, yahoo_data], ignore_index=True)
        combined.drop_duplicates(subset=['Start'], keep='last', inplace=True)
        combined.sort_values(by='Start', inplace=True)
        logger.info(f"Combined dataset has {len(combined)} rows.")
        return combined

    def get_btc_data():
        try:
            btc = yf.Ticker("BTC-USD")
            current_price = btc.fast_info['lastPrice']
            current_time = datetime.now(timezone.utc).isoformat()

            hist = btc.history(period='5d', interval='1d')
            if len(hist) < 2:
                print("Not enough historical data to fetch the last close price.")
                return current_price, current_time, None, None

            last_two_days = hist.tail(2)
            last_close = last_two_days['Close'].iloc[-2]
            last_close_time = last_two_days.index[-2].strftime('%Y-%m-%d %H:%M:%S UTC')
            return current_price, current_time, last_close, last_close_time

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None, None, None, None

    # ----------------------------------------
    # 4. Fetch/Load Data and Prepare Combined
    # ----------------------------------------
    current_price, current_time, last_close, last_close_time = get_btc_data()

    # Display to console
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    if current_price:
        print(f"Bitcoin's current price ({current_time}): {BOLD}{GREEN}${current_price:,.2f}{RESET}")
    if last_close:
        print(f"Bitcoin's last daily close price ({last_close_time}): {BOLD}{GREEN}${last_close:,.2f}{RESET}")

    # Write minimal info to JSON
    info = {
        "current_price": current_price,
        "timestamp_gmt": current_time,
        "chart_titles": chart_titles_map  # remain as-is in JSON (no prefix added here)
    }

    json_file_path = os.path.join(os.getcwd(), "charts_dev/chart_info.json")
    try:
        with open(json_file_path, "w") as json_file:
            json.dump(info, json_file, indent=4)
        logger.info(f"JSON file with update info saved to '{json_file_path}'.")
    except Exception as e:
        logger.error(f"Failed to write JSON file: {e}")

    local_data = load_local_csv(CONFIG["CSV_FILE_PATH"])
    today = pd.to_datetime('today').normalize()

    if not local_data.empty:
        latest_csv_date = local_data['Start'].max()
        logger.info(f"Local CSV file has data up to {latest_csv_date.date()}.")
        yahoo_data = pd.DataFrame()

        start_date_for_yahoo = latest_csv_date + pd.Timedelta(days=1)
        if start_date_for_yahoo < today:
            logger.info(f"Attempting to fetch Yahoo Finance data from {start_date_for_yahoo.date()} to {today.date()}...")
            yahoo_data = load_yahoo_data(start_date_for_yahoo, today)

        if not yahoo_data.empty:
            combined_data = combine_data(local_data, yahoo_data)
        else:
            combined_data = local_data
    else:
        logger.error("No data available from the local CSV file. Exiting.")
        raise ValueError("No data available from the local CSV file. Exiting.")

    if current_price:
        logger.info(f"Adding current price (${current_price:,.2f}) as today's data point.")
        current_row = pd.DataFrame({
            'Start': [today],
            'Open': [current_price],
            'High': [current_price],
            'Low': [current_price],
            'Close': [current_price],
            'Adj Close': [current_price],
            'Volume': [0]
        })
        combined_data = combined_data[combined_data['Start'].dt.date != today.date()]
        combined_data = pd.concat([combined_data, current_row], ignore_index=True)
        combined_data.sort_values(by='Start', inplace=True)

    combined_data['days'] = (combined_data['Start'] - CONFIG["GENESIS_DATE"]).dt.days

    # Powerlaw-based columns
    combined_data['support_powerlaw'] = support_powerlaw(combined_data['days'])
    combined_data['upper_bound_powerlaw'] = upper_bound_powerlaw(combined_data['days'])
    combined_data['ratio_powerlaw'] = (
        (combined_data['Close'] - combined_data['support_powerlaw']) /
        (combined_data['upper_bound_powerlaw'] - combined_data['support_powerlaw'])
    )

    # Quantiles-based columns
    combined_data['support_quantiles'] = support_quantiles(combined_data['days'])
    combined_data['upper_bound_quantiles'] = upper_bound_quantiles(combined_data['days'])
    combined_data['ratio_quantiles'] = (
        (combined_data['Close'] - combined_data['support_quantiles']) /
        (combined_data['upper_bound_quantiles'] - combined_data['support_quantiles'])
    )

    # Emit the first 5 rows for debugging
    logger.info("First 5 rows of the combined data:")
    logger.info(combined_data.head(5))
    # Emit the last 5 rows for debugging
    logger.info("Last 5 rows of the combined data:")
    logger.info(combined_data.tail(5))
    
    # Check for missing days
    date_range = pd.date_range(start=combined_data['Start'].min(), end=combined_data['Start'].max(), freq='D')
    missing_dates = date_range.difference(combined_data['Start'])
    if not missing_dates.empty:
        logger.warning("Missing dates found in data:")
        for d in missing_dates:
            logger.warning(d)
    else:
        logger.info("No missing dates in the combined data.")

    recent = today if today in combined_data['Start'].values else combined_data['Start'].max()
    logger.info(f"Using '{recent.date()}' as the most recent data point.")

    # ------------------------------------
    # 5. Prepare Calculations for Charts
    # ------------------------------------
    current_data_powerlaw = combined_data.loc[combined_data['Start'] == recent]
    if current_data_powerlaw.empty:
        raise ValueError(f"No data found for the recent date: {recent.date()}")

    current_support_powerlaw = current_data_powerlaw['support_powerlaw'].values[0]
    current_upper_bound_powerlaw = current_data_powerlaw['upper_bound_powerlaw'].values[0]
    current_ratio_powerlaw = current_data_powerlaw['ratio_powerlaw'].values[0]
    current_price_powerlaw = current_support_powerlaw + current_ratio_powerlaw * (
        current_upper_bound_powerlaw - current_support_powerlaw
    )

    prices_powerlaw = []
    for offset in CONFIG["CYCLE_OFFSETS"]:
        past_date = recent - pd.DateOffset(years=offset)
        subset = combined_data[combined_data['Start'] <= past_date]
        if not subset.empty:
            closest_date = subset['Start'].max()
            past_ratio = combined_data.loc[combined_data['Start'] == closest_date, 'ratio_powerlaw'].values[0]
            price = current_support_powerlaw + past_ratio * (
                current_upper_bound_powerlaw - current_support_powerlaw
            )
            prices_powerlaw.append(price)
        else:
            prices_powerlaw.append(None)
    prices_powerlaw.append(current_price_powerlaw)

    current_support_quantiles = current_data_powerlaw['support_quantiles'].values[0]
    current_upper_bound_quantiles = current_data_powerlaw['upper_bound_quantiles'].values[0]
    current_ratio_quantiles = current_data_powerlaw['ratio_quantiles'].values[0]
    current_price_quantiles = (
        current_support_quantiles + current_ratio_quantiles 
        * (current_upper_bound_quantiles - current_support_quantiles)
    )

    prices_quantiles = []
    for offset in CONFIG["CYCLE_OFFSETS"]:
        past_date = recent - pd.DateOffset(years=offset)
        subset = combined_data[combined_data['Start'] <= past_date]
        if not subset.empty:
            closest_date = subset['Start'].max()
            past_ratio = combined_data.loc[combined_data['Start'] == closest_date, 'ratio_quantiles'].values[0]
            price = current_support_quantiles + past_ratio * (
                current_upper_bound_quantiles - current_support_quantiles
            )
            prices_quantiles.append(price)
        else:
            prices_quantiles.append(None)
    prices_quantiles.append(current_price_quantiles)

    # Similarly gather ratios for ratio charts
    ratios_powerlaw = []
    for offset in CONFIG["CYCLE_OFFSETS"]:
        past_date = recent - pd.DateOffset(years=offset)
        subset = combined_data[combined_data['Start'] <= past_date]
        if not subset.empty:
            closest_date = subset['Start'].max()
            ratios_powerlaw.append(
                combined_data.loc[combined_data['Start'] == closest_date, 'ratio_powerlaw'].values[0]
            )
        else:
            ratios_powerlaw.append(None)
    ratios_powerlaw.append(current_ratio_powerlaw)

    ratios_quantiles = []
    for offset in CONFIG["CYCLE_OFFSETS"]:
        past_date = recent - pd.DateOffset(years=offset)
        subset = combined_data[combined_data['Start'] <= past_date]
        if not subset.empty:
            closest_date = subset['Start'].max()
            ratios_quantiles.append(
                combined_data.loc[combined_data['Start'] == closest_date, 'ratio_quantiles'].values[0]
            )
        else:
            ratios_quantiles.append(None)
    ratios_quantiles.append(current_ratio_quantiles)

    data_by_start = combined_data.set_index('Start')
    recent_str = recent.strftime('%d-%b-%Y')

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
            'ratio_label_names': [
                "Cycle 1 '12-'15:", "Cycle 2 '16-'19:", "Cycle 3 '20-'23:",
                "Cycle 4 '24-'27:", "Current Ratio:"
            ],
            'current_price': lambda: current_price_powerlaw,
            'current_support': current_support_powerlaw,
            'current_upper': current_upper_bound_powerlaw,
            'prices_list_price': prices_powerlaw,
            'ratios_list_ratio': ratios_powerlaw,
            'current_ratio': current_ratio_powerlaw
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
            'ratio_label_names': [
                "Cycle 1 '12-'15:", "Cycle 2 '16-'19:", "Cycle 3 '20-'23:",
                "Cycle 4 '24-'27:", "Current Ratio:"
            ],
            'current_price': lambda: current_price_quantiles,
            'current_support': current_support_quantiles,
            'current_upper': current_upper_bound_quantiles,
            'prices_list_price': prices_quantiles,
            'ratios_list_ratio': ratios_quantiles,
            'current_ratio': current_ratio_quantiles
        }
    }

    # ------------------------------------------------
    # 6. Functions to Build and Save Each Chart
    # ------------------------------------------------
    def add_credits_box(ax, credits_text):
        """Stub function kept for reference. Not actively drawing a box."""
        pass

    def add_legend_box(ax, box_position):
        """
        Adds a black legend box using the specified (x, y, width, height).
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
        for offset, color in zip(CONFIG["CYCLE_OFFSETS"], CONFIG["GHOST_COLORS"]):
            shifted_dates = plot_date_range - pd.DateOffset(years=offset)
            shifted_data = data_by_start.reindex(shifted_dates, method='nearest', tolerance='1D')
            shifted_ratios = shifted_data[ratio_col].values
            ghost_prices = support_values + shifted_ratios * (upper_bound_values - support_values)
            ax.plot(plot_date_range, ghost_prices, color=color, linewidth=2)

    def add_equations_text(ax, eq_text, alpha_val):
        ax.text(
            0.93,
            0.98,
            eq_text,
            transform=ax.transAxes,
            fontsize=CONFIG["EQUATION_FONT_SIZE"],
            verticalalignment='top',
            horizontalalignment='right',
            color='black',
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="black", alpha=alpha_val),
            zorder=5,
        )

    def configure_x_axis(ax, plot_start, plot_end):
        ax.set_xticks(pd.date_range(start=plot_start, end=plot_end, freq='YS'))
        ax.set_xticklabels(
            [date.strftime('%Y') for date in pd.date_range(start=plot_start, end=plot_end, freq='YS')],
            rotation=90,
        )
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax.get_xminorticklabels(), rotation=90, color='dimgray')
        ax.set_xlim(left=plot_start, right=plot_end)

    # ------------- PRICE CHART -------------
    def plot_price_chart(mode, plot_start, plot_end, filename, title, subtitle):
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

        # Ghost cycles
        plot_ghost_cycles(ax, plot_date_range, ratio_col, support_values, upper_bound_values)

        # Support & upper bound
        ax.plot(plot_date_range, support_values, color='dimgray', linestyle='-', linewidth=0.5)
        ax.plot(plot_date_range, upper_bound_values, color='dimgray', linestyle='-', linewidth=0.5)

        # Current cycle line & point
        ax.plot(current_cycle_data['Start'], price_vals, color=CONFIG["CURRENT_CYCLE_COLOR"], linewidth=2)
        ax.scatter(recent, curr_price, edgecolor='black', facecolor='none', marker='o', s=100, zorder=5)

        configure_x_axis(ax, plot_start, plot_end)
        ax.yaxis.set_major_locator(MultipleLocator(50000))
        ax.yaxis.set_minor_locator(MultipleLocator(10000))

        # Grid
        ax.grid(which='major', linestyle='-', linewidth=1, color='dimgray')
        ax.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgray')

        # Legend Box (use "powerlaw" or "quantiles" positions)
        add_legend_box(ax, CONFIG["LEGEND_BOX_POSITIONS"][mode])

        max_label_len = max(len(label) for label in label_names)
        price_strings = []
        for i in range(len(CONFIG["CYCLE_OFFSETS"]) + 1):
            val = prices_list[i]
            if val is not None:
                price_str = f"{val:>10,.2f}"
            else:
                price_str = f"{'N/A':>10}"
            price_strings.append(f"{label_names[i]:<{max_label_len + 2}}{price_str}")

        price_strings.append(f"{label_names[-2]:<{max_label_len + 2}}{curr_support:>10,.2f}")
        price_strings.append(f"{label_names[-1]:<{max_label_len + 2}}{curr_upper:>10,.2f}")

        x_pos = 0.04
        y_pos = 0.95
        y_step = 0.04

        ax.text(
            x_pos, y_pos, recent_str,
            transform=ax.transAxes, fontsize=16, verticalalignment='top',
            fontfamily='monospace', zorder=5, color='white'
        )

        # Ghost cycles lines
        for i, label_line in enumerate(price_strings[:4]):
            y_pos -= y_step
            color = CONFIG["GHOST_COLORS"][i] if i < len(CONFIG["GHOST_COLORS"]) else CONFIG["CURRENT_CYCLE_COLOR"]
            ax.text(
                x_pos, y_pos, label_line,
                transform=ax.transAxes, fontsize=16, verticalalignment='top',
                fontfamily='monospace', zorder=5, color=color
            )

        # Support & upper bound lines
        for label_line in price_strings[4:]:
            y_pos -= y_step
            ax.text(
                x_pos, y_pos, label_line,
                transform=ax.transAxes, fontsize=16, verticalalignment='top',
                fontfamily='monospace', zorder=5, color='white'
            )

        add_equations_text(ax, eq_text, alpha_val)

        # ------------------------------------------------
        # The single requested change:
        # Prefix the chart title with "Bitcoin 4-Year Cycles with Ghosts / "
        # but do NOT push that to JSON.
        # ------------------------------------------------
        prefixed_title = f"Bitcoin 4-Year Cycles with Ghosts / {title}"

        ax.set_title(prefixed_title, fontsize=CONFIG["TITLE_FONT_SIZE"], pad=CONFIG["TITLE_PADDING"])
        ax.text(
            0.5, CONFIG["SUBTITLE_Y_OFFSET"], subtitle,
            fontsize=CONFIG["SUBTITLE_FONT_SIZE"],
            transform=ax.transAxes,
            ha='center',
            va='bottom'
        )

        add_bottom_credits(fig)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"({mode.capitalize()}) Saved the plot to '{filename}' and closed the figure.")

    # ------------- RATIO CHART -------------
    def plot_ratio_chart(mode, plot_start, plot_end, filename, title, subtitle):
        mode_info = modes[mode]
        plot_date_range = pd.date_range(start=plot_start, end=plot_end, freq='D')

        ratio_col = mode_info['ratio_col']
        curr_ratio = mode_info['current_ratio']
        eq_text = mode_info['eq_text']
        alpha_val = mode_info['alpha_val']
        label_names = mode_info['ratio_label_names']
        ratios_list = mode_info['ratios_list_ratio']

        current_cycle_data = combined_data[
            (combined_data['Start'] >= plot_start) & (combined_data['Start'] <= plot_end)
        ]
        current_cycle_ratios = current_cycle_data[ratio_col]

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_facecolor('white')

        # Ghost cycles
        for offset, color in zip(CONFIG["CYCLE_OFFSETS"], CONFIG["GHOST_COLORS"]):
            shifted_dates = plot_date_range - pd.DateOffset(years=offset)
            shifted_data = data_by_start.reindex(shifted_dates, method='nearest', tolerance='1D')
            shifted_ratios = shifted_data[ratio_col].values
            ax.plot(plot_date_range, shifted_ratios, color=color, linewidth=2)

        # Current cycle ratio
        ax.plot(current_cycle_data['Start'], current_cycle_ratios, color=CONFIG["CURRENT_CYCLE_COLOR"], linewidth=2)
        ax.scatter(recent, curr_ratio, edgecolor='black', facecolor='none', marker='o', s=100, zorder=5)

        configure_x_axis(ax, plot_start, plot_end)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))

        # Grid + thick lines at 0 and 1
        ax.grid(which='major', linestyle='-', linewidth=1, color='dimgray')
        ax.grid(which='minor', linestyle='-', linewidth=0.5, color='lightgray')
        ax.axhline(0.0, color='black', linewidth=2)
        ax.axhline(1.0, color='black', linewidth=2)

        add_legend_box(ax, CONFIG["LEGEND_BOX_POSITIONS"][mode])

        max_label_len = max(len(label) for label in label_names)
        ratio_strings = []
        for i in range(len(CONFIG["CYCLE_OFFSETS"])):
            val = ratios_list[i]
            if val is not None:
                ratio_str = f"{val:.4f}"
            else:
                ratio_str = f"{'N/A':>10}"
            ratio_strings.append(f"{label_names[i]:<{max_label_len + 2}}{ratio_str}")

        ratio_strings.append(f"{label_names[4]:<{max_label_len + 2}}{curr_ratio:.4f}")

        x_pos = 0.04
        y_pos = 0.95
        y_step = 0.04

        ax.text(
            x_pos, y_pos, recent_str,
            transform=ax.transAxes, fontsize=16, verticalalignment='top',
            fontfamily='monospace', zorder=5, color='white'
        )

        for i, label_line in enumerate(ratio_strings):
            y_pos -= y_step
            color = CONFIG["GHOST_COLORS"][i] if i < len(CONFIG["GHOST_COLORS"]) else CONFIG["CURRENT_CYCLE_COLOR"]
            ax.text(
                x_pos, y_pos, label_line,
                transform=ax.transAxes, fontsize=16, verticalalignment='top',
                fontfamily='monospace', zorder=5, color=color
            )

        add_equations_text(ax, eq_text, alpha_val)
        # For ratio charts, no prefix is needed:
        ax.set_title(title, fontsize=CONFIG["TITLE_FONT_SIZE"], pad=CONFIG["TITLE_PADDING"])
        ax.text(
            0.5, CONFIG["SUBTITLE_Y_OFFSET"], subtitle,
            fontsize=CONFIG["SUBTITLE_FONT_SIZE"],
            transform=ax.transAxes,
            ha='center',
            va='bottom'
        )

        add_bottom_credits(fig)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"({mode.capitalize()} Ratio) Saved the plot to '{filename}' and closed the figure.")

    # --------------------------------------------
    # 7. Produce Price Charts (powerlaw/quantiles)
    # --------------------------------------------
    for mode in ["powerlaw", "quantiles"]:
        for chart_config in CHARTS[mode]:
            if chart_config['type'] == 'price':
                plot_start = pd.to_datetime(chart_config['start'])
                plot_end = pd.to_datetime(chart_config['end'])
                filename = chart_config['filename']
                title = chart_config['title']
                subtitle = chart_config['subtitle']
                plot_price_chart(mode, plot_start, plot_end, filename, title, subtitle)

    # Conditionally produce Ratio Charts
    if args.generate_ratio_charts:
        for mode in ["powerlaw", "quantiles"]:
            for chart_config in CHARTS[mode]:
                if chart_config['type'] == 'ratio':
                    plot_start = pd.to_datetime(chart_config['start'])
                    plot_end = pd.to_datetime(chart_config['end'])
                    filename = chart_config['filename']
                    title = chart_config['title']
                    subtitle = chart_config['subtitle']
                    plot_ratio_chart(mode, plot_start, plot_end, filename, title, subtitle)
    else:
        logger.info("Ratio charts generation skipped as per default settings.")

    # --------------------------------------------
    # 8. ghostly_days Computations and Chart
    # --------------------------------------------
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
        days_since_genesis_recent = (recent - CONFIG["GENESIS_DATE"]).days
        recent_gd_local = recent

        current_data_gd_local = combined_data.loc[combined_data['Start'] == recent_gd_local]
        if current_data_gd_local.empty:
            raise ValueError(f"No data found for the recent date: {recent_gd_local.date()}")

        # Ghost cycles
        for offset, color in zip(CONFIG["CYCLE_OFFSETS"], CONFIG["GHOST_COLORS"]):
            shifted_dates = plot_date_range - pd.DateOffset(years=offset)
            shifted_data = combined_data.set_index('Start').reindex(shifted_dates, method='nearest', tolerance='1D')
            days_ahead = shifted_data['Days Ahead'].values
            valid_indices = ~np.isnan(days_ahead)
            days_ahead = days_ahead[valid_indices]
            valid_dates = plot_date_range[valid_indices]
            valid_days_since_genesis = days_since_genesis[valid_indices]

            recalculated_days = valid_days_since_genesis + days_ahead
            recalculated_days = np.where(recalculated_days > 0, recalculated_days, np.nan)
            ghost_prices = support(recalculated_days)

            if len(ghost_prices) > 0:
                ax.plot(valid_dates, ghost_prices, color=color, linewidth=2)
                all_prices.extend(ghost_prices[~np.isnan(ghost_prices)])

                shifted_recent_date = recent_gd_local - pd.DateOffset(years=offset)
                idx = np.argmin(np.abs(shifted_dates - shifted_recent_date))
                if valid_indices[idx]:
                    days_ahead_at_shifted = days_ahead[idx]
                    remapped_days_since_genesis = days_since_genesis_recent + days_ahead_at_shifted
                    if remapped_days_since_genesis > 0:
                        remapped_price = support(remapped_days_since_genesis)
                        prices.append(remapped_price)
                    else:
                        prices.append(None)
                else:
                    prices.append(None)
            else:
                prices.append(None)

        current_cycle_data_gd = combined_data[
            (combined_data['Start'] >= plot_start) & (combined_data['Start'] <= plot_end)
        ]
        ax.plot(
            current_cycle_data_gd['Start'],
            current_cycle_data_gd['Close'],
            color=CONFIG["CURRENT_CYCLE_COLOR"],
            linewidth=2
        )
        ax.scatter(recent_gd_local, current_price_gd, edgecolor='black', facecolor='none', marker='o', s=100, zorder=5)
        all_prices.extend(current_cycle_data_gd['Close'].values)
        prices.append(current_price_gd)

        # Support line
        ax.plot(plot_date_range, support_values, color='dimgray', linestyle='-', linewidth=0.5)
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

        for p in prices:
            if p is not None and not np.isnan(p):
                price_strings.append(f"{p:>12,.2f}")
            else:
                price_strings.append(f"{'N/A':>12}")

        x_pos = 0.04
        y_pos = 0.95
        y_step = 0.04

        ax.text(
            x_pos, y_pos,
            recent_gd_local.strftime('%d-%b-%Y'),
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

        # Use a smaller legend box for ghostly_days
        add_legend_box(ax, CONFIG["LEGEND_BOX_POSITIONS"]["ghostly_days"])

        add_bottom_credits(fig)

        # Equations
        equations_text_gd = (
            r"$\bf{Support:}$" "\n"
            r"$10^{-17.351} \times days^{5.836}$"
        )
        ax.text(
            0.93, 0.98,
            equations_text_gd,
            transform=ax.transAxes,
            fontsize=CONFIG["EQUATION_FONT_SIZE"],
            verticalalignment='top',
            horizontalalignment='right',
            color='black',
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="black", alpha=1),
            zorder=5
        )

        # Again, prefix only the "price" chart's displayed title
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

    # Build ghostly_days charts
    for chart_config in CHARTS["ghostly_days"]:
        if chart_config['type'] == 'price':
            plot_start = pd.to_datetime(chart_config['start'])
            plot_end = pd.to_datetime(chart_config['end'])
            filename = chart_config['filename']
            title = chart_config['title']
            subtitle = chart_config['subtitle']
            plot_price_chart_gd(plot_start, plot_end, filename, title, subtitle)

    # --------------------------------
    # 9. Optionally Save Enhanced CSV
    # --------------------------------
    def save_enhanced_csv(combined_df, output_file_name):
        def support_func_pl(d):
            return 10**-17.351 * d**5.836

        def upper_bound_func_pl(d):
            return (1 + 10**(1.836 - d * 0.0002323)) * support_func_pl(d)

        def support_func_q(d):
            return math.exp(-41.72) * d**6.02

        def upper_bound_func_q(d):
            return math.exp(-28.32) * d**4.72

        combined_df['days'] = (combined_df['Start'] - CONFIG["GENESIS_DATE"]).dt.days

        # Compute ratios if missing
        if 'ratio_powerlaw' not in combined_df.columns:
            combined_df['support_powerlaw'] = support_func_pl(combined_df['days'])
            combined_df['upper_bound_powerlaw'] = upper_bound_func_pl(combined_df['days'])
            combined_df['ratio_powerlaw'] = (
                (combined_df['Close'] - combined_df['support_powerlaw'])
                / (combined_df['upper_bound_powerlaw'] - combined_df['support_powerlaw'])
            )

        if 'ratio_quantiles' not in combined_df.columns:
            combined_df['support_quantiles'] = support_func_q(combined_df['days'])
            combined_df['upper_bound_quantiles'] = upper_bound_func_q(combined_df['days'])
            combined_df['ratio_quantiles'] = (
                (combined_df['Close'] - combined_df['support_quantiles'])
                / (combined_df['upper_bound_quantiles'] - combined_df['support_quantiles'])
            )

        # Filter for years 2012–2023 for ghostly_days approach
        filtered_data = combined_df[
            (combined_df['Start'].dt.year >= 2012) & (combined_df['Start'].dt.year <= 2023)
        ].copy()

        CYCLE_OFFSETS = CONFIG["CYCLE_OFFSETS"]
        days_ahead_dict = {offset: [] for offset in CYCLE_OFFSETS}
        ghost_price_dict = {offset: [] for offset in CYCLE_OFFSETS}
        powerlaw_ghost_dict = {offset: [] for offset in CYCLE_OFFSETS}
        quantiles_ghost_dict = {offset: [] for offset in CYCLE_OFFSETS}

        def support_da(d):
            return 10**-17.351 * d**5.836

        for _, row in filtered_data.iterrows():
            for offset in CYCLE_OFFSETS:
                shifted_date = row['Start'] - pd.DateOffset(years=offset)
                shifted_row = combined_df.set_index('Start').reindex(
                    [shifted_date], method='nearest', tolerance='1D'
                )
                if not shifted_row.empty and 'Days Ahead' in shifted_row.columns:
                    days_ahead_val = shifted_row['Days Ahead'].values[0]
                else:
                    days_ahead_val = np.nan

                recalculated_days = row['Days Since Genesis'] + days_ahead_val
                ghost_price_val = support_da(recalculated_days) if recalculated_days > 0 else np.nan

                days_ahead_dict[offset].append(days_ahead_val)
                ghost_price_dict[offset].append(ghost_price_val)

            row_days = row['days']
            row_support_pl = support_func_pl(row_days)
            row_upper_pl = upper_bound_func_pl(row_days)
            diff_pl = row_upper_pl - row_support_pl

            row_support_q = support_func_q(row_days)
            row_upper_q = upper_bound_func_q(row_days)
            diff_q = row_upper_q - row_support_q

            for offset in CYCLE_OFFSETS:
                shifted_date = row['Start'] - pd.DateOffset(years=offset)
                subset = combined_df[combined_df['Start'] <= shifted_date]
                if not subset.empty:
                    closest_date = subset['Start'].max()
                    past_ratio_pl = subset.loc[subset['Start'] == closest_date, 'ratio_powerlaw'].values[0]
                    past_ratio_q  = subset.loc[subset['Start'] == closest_date, 'ratio_quantiles'].values[0]
                else:
                    past_ratio_pl = np.nan
                    past_ratio_q  = np.nan

                ghost_price_pl = row_support_pl + (past_ratio_pl * diff_pl) if not np.isnan(past_ratio_pl) else np.nan
                ghost_price_q  = row_support_q + (past_ratio_q  * diff_q ) if not np.isnan(past_ratio_q ) else np.nan

                powerlaw_ghost_dict[offset].append(ghost_price_pl)
                quantiles_ghost_dict[offset].append(ghost_price_q)

        for offset in CYCLE_OFFSETS:
            filtered_data[f'Days Ahead {offset}y'] = days_ahead_dict[offset]
            filtered_data[f'Ghost Price {offset}y'] = ghost_price_dict[offset]
            filtered_data[f'Ghost Price Powerlaw {offset}y'] = powerlaw_ghost_dict[offset]
            filtered_data[f'Ghost Price Quantiles {offset}y'] = quantiles_ghost_dict[offset]

        merged_cols = (
            ['Start']
            + [f'Days Ahead {off}y' for off in CYCLE_OFFSETS]
            + [f'Ghost Price {off}y' for off in CYCLE_OFFSETS]
            + [f'Ghost Price Powerlaw {off}y' for off in CYCLE_OFFSETS]
            + [f'Ghost Price Quantiles {off}y' for off in CYCLE_OFFSETS]
        )

        combined_merged = combined_df.merge(filtered_data[merged_cols], on='Start', how='left')

        output_file_path = os.path.join(os.getcwd(), output_file_name)
        combined_merged.to_csv(output_file_path, index=False)
        logging.info(f"(ghostly_days) Enhanced CSV file saved to {output_file_path}")

    if args.output_csv:
        save_enhanced_csv(combined_data, "enhanced_bitcoin_data.csv")
    else:
        logger.info("CSV output generation skipped as per default settings.")

if __name__ == "__main__":
    main()
