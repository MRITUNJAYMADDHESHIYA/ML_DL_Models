import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"

from dash import Dash, dcc, html, Output, Input
import dash_bootstrap_components as dbc
import MetaTrader5 as mt5
from datetime import datetime, timedelta, time as dt_time


def get_ohlc(symbol):
    rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
    rates_df = pd.DataFrame(rates)
    rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')

    return rates_df

################################## Basic #############################
def create_basis_fig():
    ohlc1_df = get_ohlc(symbol1)
    ohlc2_df = get_ohlc(symbol2)

    compare_df = ohlc1_df[['time', 'close', 'spread']].merge(ohlc2_df[['time','close','spread']], on='time', suffixes=[f'_{symbol1}', f'_{symbol2}'])
    # converting bid prices to mid prices
    compare_df[f'close_{symbol1}'] = compare_df[f'close_{symbol1}'] + compare_df[f'spread_{symbol1}'] * 10 ** -mt5.symbol_info(symbol1).digits
    compare_df[f'close_{symbol2}'] = compare_df[f'close_{symbol2}'] + compare_df[f'spread_{symbol2}'] * 10 ** -mt5.symbol_info(symbol1).digits

    compare_df = compare_df[:-1]

    compare_df[f'close_{symbol1}'] = compare_df[f'close_{symbol1}']
    compare_df[f'close_{symbol2}'] = compare_df[f'close_{symbol2}'] * ratio

    compare_df['hour'] = compare_df['time'].dt.hour

    compare_df = compare_df.sort_values('time')

    compare_df['basis'] = compare_df[f'close_{symbol1}'] - compare_df[f'close_{symbol2}']

    compare_df['sma'] = compare_df[f'basis'].rolling(bollinger_period).mean()
    compare_df['std'] = compare_df[f'basis'].rolling(bollinger_period).std()

    compare_df['lower_band'] = compare_df['sma'] - num_std * compare_df['std']
    compare_df['upper_band'] = compare_df['sma'] + num_std * compare_df['std']

    print(compare_df)
    print('---\n')

    fig = px.line(compare_df, x='time', y=['basis', 'sma', 'lower_band', 'upper_band'], height= 800,
                  title=f'{symbol1} vs {symbol2} - Spread')

    return fig

##################################### Calculation Basic ###################################
def calculate_basis():
    # make sure symbols are available
    if not mt5.symbol_select(symbol1, True):
        print(f"Symbol {symbol1} not found or not visible in Market Watch")
        return None
    if not mt5.symbol_select(symbol2, True):
        print(f"Symbol {symbol2} not found or not visible in Market Watch")
        return None

    symbol_info1 = mt5.symbol_info(symbol1)
    symbol_info2 = mt5.symbol_info(symbol2)

    if symbol_info1 is None or symbol_info2 is None:
        print("Failed to get symbol info")
        return None

    symbol_price1 = symbol_info1._asdict()
    symbol_price2 = symbol_info2._asdict()

    # now your calculation logic
    mid_basis = (symbol_price1['ask'] / symbol_price2['ask'])
    return mid_basis



######################################## DashBoard ###########################################
app = Dash(__name__, title='Spread Dashboard', suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.DARKLY], update_title="")
app.layout = html.Div([
    dcc.Interval(id='interval-update', interval=60000),
    html.Div(id='page-content'),

    dcc.Interval(id='basis-update', interval=1000),
    html.Div(id='basis-content', style={'padding-top': '20px'}),
], style={'margin-left': '5%', 'margin-right': '5%', 'margin-top': '20px'})


@app.callback(Output('page-content', 'children'),
              Input('interval-update', 'n_intervals'),)
def update_page(interval):

    fig = create_basis_fig()
    return html.Div([
        dcc.Graph(figure=fig)
    ])


@app.callback(Output('basis-content', 'children'),
              Input('basis-update', 'n_intervals'),)
def update_page(interval):

    mid_basis = calculate_basis()

    return html.Div([
        html.H2(f'Current Spread: {mid_basis}', style={'textAlign': 'right'}),
    ])



######################################## Main ##############################
if __name__ == '__main__':

    symbol1   = 'US30m'
    symbol2   = 'USTECm'
    timeframe = mt5.TIMEFRAME_H1

    start_time = datetime.now() - timedelta(days=720)
    end_time   = datetime.now() + timedelta(hours=3)

    ratio            = 0.94
    bollinger_period = 200
    num_std          = 1.75



    login_number =  455544          
    password     = "**********8@$%^@"  
    server       = "2462-34634" 
    result = mt5.initialize(login=login_number, password=password, server=server)
    if result and mt5.account_info() and login_number == mt5.account_info().login:
        print('Connection with MT5 established')
    else:
        print('Connection failed:', mt5.last_error())
        mt5.shutdown()

    app.run(port=8051)