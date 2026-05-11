# Nicholas Christophides  nick.christophides@gmail.com

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from portfolio_optimization import optimal_portfolio
import time
import os


api_key = os.getenv('ALPACA_API_KEY')
api_secret = os.getenv('ALPACA_SECRET_KEY')


# Create a paper trading client
client = TradingClient(api_key, api_secret, paper=False)


# 1.) Get Current Portfolio Information and Determine the Weight Adjustments
# --------------------------------------------------------------------------
account = client.get_account()

# Get all open positions
current_positions = client.get_all_positions()

# Get total portfolio value
total_portfolio_value = float(account.equity)
cash_buffer = 0.99  # Leave buffer for trading costs/slippage and do not exceed your buying power
total_allocatable_cash = total_portfolio_value * cash_buffer

# Calculate how much we want in each asset after rebalance
target_asset_values = {}
for asset in optimal_portfolio.index:
    target_asset_values[asset] = optimal_portfolio.loc[asset, "Weight"] * total_allocatable_cash

# # Create holders for current positions and by how much to adjust positions
current_positions_data = {}  # The current info (market value, weight) of the positions
position_adjustments = {}  # The amounts to adjust each weight by position

# Extract the current portfolio information
for position in current_positions:
    current_positions_data[f"{position.symbol}"] = (position.market_value,
                                                    float(position.market_value) / total_portfolio_value)


# Standardize all tickers: replace '-' with '.' (e.g., BRK-B becomes BRK.B)
target_asset_values = {ticker.replace('-', '.'): val for ticker, val in target_asset_values.items()}

# Set of all the assets invested in currently or will be invested in
all_assets = set(target_asset_values.keys()).union(current_positions_data.keys())


# # 2.) Determine How Much of Each Asset to Buy/Sell, Separate Buy and Sell Orders
# # ------------------------------------------------------------------------------
buy_orders = {}
sell_orders = {}

for ticker in all_assets:
    target_val = target_asset_values.get(ticker, 0.0)
    current_val = float(current_positions_data.get(ticker, (0.0, 0))[0])

    dollar_position_adjustment = target_val - current_val

    # Threshold to avoid tiny trades (e.g., ignore trades under $20)
    if abs(dollar_position_adjustment) < 20.0:
        continue

    if dollar_position_adjustment > 0:
        buy_orders[ticker] = dollar_position_adjustment
    else:
        # Absolute value for the sell amount
        sell_orders[ticker] = abs(dollar_position_adjustment)


# 3.) Execute Buy and Sell Orders to Rebalance the Portfolio
# ----------------------------------------------------------


def create_order(ticker, action, dollar_amount):
    """
    Creates a buy or sell order and sends it to Alpaca to be filled
    :param ticker: the ticker of the stock to be bought/sold
    :param action: "buy" or "sell"
    :param dollar_amount: the dollar amount of the transaction
    """
    # Specify Buy or Sell
    order_side = None
    if action == "buy":
        order_side = OrderSide.BUY
    elif action == "sell":
        order_side = OrderSide.SELL

    market_order = MarketOrderRequest(
        symbol=ticker,
        notional=round(dollar_amount, 2),
        side=order_side,
        time_in_force=TimeInForce.DAY
    )

    client.submit_order(market_order)
    print(f"\nSubmitted order:\nTicker: {ticker}\nSide: {action}\nAmount: ${dollar_amount:.2f}"
          f"\nWeight: {100*dollar_amount/total_portfolio_value:.3f}%\n", "-"*80)


net_traded_dollars = 0  # Cumulative sum of dollars traded (selling -> negative; buying -> positive)

# Execute Sell Orders
for ticker in sell_orders.keys():
    dollar_amount = sell_orders[ticker]

    held_value = float(client.get_open_position(ticker).market_value) * .999
    final_sell_amount = min(dollar_amount, held_value)

    net_traded_dollars -= final_sell_amount
    create_order(ticker, "sell", final_sell_amount)

# Execute Buy Orders
for ticker in buy_orders.keys():
    dollar_amount = buy_orders[ticker]
    net_traded_dollars += dollar_amount
    create_order(ticker, "buy", dollar_amount)


time.sleep(10)  # Allow 10 seconds for the orders to fill


# 4.) Print Summary of Transactions after Re-balance
# --------------------------------------------------
account = client.get_account()
current_positions = client.get_all_positions()
total_portfolio_value = float(account.equity)

for position in current_positions:
    current_positions_data[f"{position.symbol}"] = (position.market_value,
                                                    float(position.market_value) / total_portfolio_value)

asset_weight_sum = 0

print(f"\n----- Summary -----\nTotal Amount Traded: ${net_traded_dollars:.2f}"
      f"\nTotal Portfolio Value: ${float(account.equity):.2f}")

print(f"\n----- New Portfolio Weights -----")
for asset in current_positions_data.keys():
    print(f"{asset}: {current_positions_data[asset][1]}")
    asset_weight_sum += current_positions_data[asset][1]

print(f"\nSum of Asset Weights: {asset_weight_sum}")
