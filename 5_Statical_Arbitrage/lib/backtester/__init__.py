import pandas as pd
from datetime import datetime, timedelta, time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

pio.templates.default = "plotly_dark"


class Backtester:
    def __init__(self, account_balance=10000, spread=0, commission=0):
        self.account_balance = account_balance
        self.commission = commission

        self.positions = dict()
        self.trades = dict()
        self.trade_id = 0

        self.historical_data = pd.DataFrame()
        self.current_time = None
        self.current_price = None

        self.on_bar = None

    def set_strategy(self, on_bar):
        self.on_bar = on_bar

    def set_historical_data(self, historical_data_df: pd.DataFrame):
        self.historical_data = historical_data_df

    def get_next_trade_id(self):
        self.trade_id += 1
        return self.trade_id

    def get_open_trade_ids(self):
        return list(self.positions)

    def get_num_open_positions(self):
        return len(list(self.positions))

    def open_trade(self, symbol, action, volume):
        next_trade_id = self.get_next_trade_id()

        self.positions[next_trade_id] = {
            'symbol': symbol,
            'action': action,
            'volume': volume,
            'open_time': self.current_time,
            'open_price': self.current_price,
        }

    def close_trade_by_id(self, trade_id):

        self.trades[trade_id] = self.positions[trade_id]
        self.trades[trade_id]['close_time'] = self.current_time
        self.trades[trade_id]['close_price'] = self.current_price

        profit = None
        if self.trades[trade_id]['action'] == 'buy':
            profit = (self.trades[trade_id]['close_price'] - self.trades[trade_id]['open_price']) * \
                     self.trades[trade_id]['volume']
        elif self.trades[trade_id]['action'] == 'sell':
            profit = (self.trades[trade_id]['open_price'] - self.trades[trade_id]['close_price']) * \
                     self.trades[trade_id]['volume']

        self.trades[trade_id]['profit'] = profit
        self.trades[trade_id]['commission'] = self.commission * self.trades[trade_id]['volume']
        self.trades[trade_id]['net_profit'] = self.trades[trade_id]['profit'] + self.trades[trade_id]['commission']

        del self.positions[trade_id]

    def close_trades(self, action='any'):
        trade_ids = self.get_open_trade_ids()

        for trade_id in trade_ids:
            if action == 'any':
                self.close_trade_by_id(trade_id)

            elif action == 'buy':
                if self.positions[trade_id]['action'] == 'buy':
                    self.close_trade_by_id(trade_id)

            elif action == 'sell':
                if self.positions[trade_id]['action'] == 'sell':
                    self.close_trade_by_id(trade_id)

    def run(self, strategy_params={}, price_column='spread'):
        backtest_time_start = datetime.now()

        for i, d in self.historical_data.iterrows():
            self.current_time = d['time']
            self.current_price = d[price_column]

            self.on_bar(self, d, strategy_params)

        self.trades_df = pd.DataFrame.from_dict(self.trades, orient='index')
        self.trades_df['profit_cumulative'] = self.trades_df['net_profit'].cumsum()

        backtest_time_end = datetime.now()

        print(f'Backtest finished - duration {backtest_time_end - backtest_time_start}')

    def evaluate(self):
        pass

    def visualize_backtest(self, indicators=[], price_column='spread', num_trades=None):

        fig = px.line(self.historical_data, x='time', y=[price_column] + indicators, height=600, width=1200,
                      title='Backtest Trades')

        for key in self.trades.keys():
            color = 'green' if self.trades[key]['profit'] > 0 else 'red'
            fig.add_shape(type="line",
                          x0=self.trades[key]['open_time'], y0=self.trades[key]['open_price'],
                          x1=self.trades[key]['close_time'],
                          y1=self.trades[key]['close_price'],
                          line=dict(
                              color=color,
                              width=5,
                          )
                          )

        return fig


def create_ohlc_fig(df, indicators=[]):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_width=[0.2, 0.7])

    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'], name='OHLC'),
                  row=1, col=1)

    for indicator in indicators:
        fig.add_trace(go.Scatter(x=df['time'], y=df[indicator], name=indicator))

    fig.add_trace(go.Bar(x=df['time'], y=df['volume'], showlegend=False), row=2, col=1)
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(height=600)

    return fig