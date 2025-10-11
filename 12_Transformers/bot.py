##1.Running and MT5 on Real account

import os
import json
import time
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import ta
import MetaTrader5 as mt5

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

############################ Global variables ############################################
SYMBOL           = "BTCUSDm"           # change if your broker uses different symbol
TIMEFRAME        = mt5.TIMEFRAME_M5
LOOKBACK         = 60                  #60*5 = 300 minutes (5 hours) using this make the prediction
PREDICT_HORIZON  = 1                   #next 5 minute direction prediction
BATCH_SIZE       = 64                  #number of samples processed before updating weights.
EPOCHS           = 30                  #30 means the model will see the training data 30 times
MODEL_DIR        = "models"
STATE_FILE       = "bot_state.json"
os.makedirs(MODEL_DIR, exist_ok=True)  #create model directory if not exists

########################## Risk Factors ##################################################
LOT            = 0.01
SL_PIPS        = 8000      # stop loss in pips (for XAUUSD 1 pip typically = 0.01 on many brokers — verify)
TP_PIPS        = 8000
MAX_POS        = 1        # max concurrent positions
RISK_PER_TRADE = 0.01     # fraction of account balance — (used for position sizing if implemented)
RETRAIN_DAYS   = 7        # retrain frequency
POLL_INTERVAL  = 60 * 5   # 5 minutes
############################ Account Information #########################################
MT5_LOGIN    = 274242894
MT5_PASSWORD = "Mritunjay@76519"
MT5_SERVER   = "Exness-MT5Trial6"

############################# Device(cpu/gpu/cuda) ####################################
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
############################# Logging #########################################
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
############################ MT5 Connector (OOP) ##########################################
class MT5Connector:
    def __init__(self, login=None, password=None, server=None):
        self.login    = login
        self.password = password
        self.server   = server
        self._init_mt5()

    def _init_mt5(self):
        if mt5.initialize():
            logging.info("MT5 initialized (using terminal credentials).")
        else:
            logging.info("MT5 initialize failed; trying login if details provided.")
            if self.login and self.password and self.server:
                ok = mt5.initialize(login=self.login, password=self.password, server=self.server)
                if not ok:
                    raise RuntimeError("MT5 initialize/login failed.")
            else:
                raise RuntimeError("MT5 initialize failed and no login provided.")

    def shutdown(self):
        mt5.shutdown()

    def fetch_ohlc(self, symbol, timeframe, n_bars=2000):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if rates is None:
            raise RuntimeError(f"Failed to fetch rates for {symbol}")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def account_info(self):
        return mt5.account_info()

    def send_order(self, symbol, volume, order_type, price=None, sl=None, tp=None, deviation=20):
        if order_type == mt5.ORDER_TYPE_BUY:
            price = price or mt5.symbol_info_tick(symbol).ask
        else:
            price = price or mt5.symbol_info_tick(symbol).bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": int(order_type),
            "price": float(price),
            "deviation": deviation,
            "magic": 123456,
            "comment": "RNN_auto",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }
        if sl: request["sl"] = float(sl)
        if tp: request["tp"] = float(tp)

        result = mt5.order_send(request)
        return result

    def close_position(self, ticket):
        return mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL,
            "position": int(ticket),
            "type": mt5.ORDER_TYPE_SELL if mt5.positions_get(ticket=ticket)[0].type == 0 else mt5.ORDER_TYPE_BUY,
            "volume": mt5.positions_get(ticket=ticket)[0].volume,
            "symbol": mt5.positions_get(ticket=ticket)[0].symbol,
            "price": mt5.symbol_info_tick(mt5.positions_get(ticket=ticket)[0].symbol).bid
        })
    
############################ Feature Engineering ##########################################
class FeatureEngineer:
    def __init__(self, df=None):
        self.df = df
        self.feature_cols = ['open', 'high', 'low', 'close', 'tick_volume','rsi', 'ma_20', 'ma_20_slope', 'bb_high', 'bb_low']
        self.scaler = None

    def set_df(self, df):
        self.df = df

    def add_indicators(self):
        df                = self.df.copy()
        df['rsi']         = ta.momentum.rsi(df['close'], window=14)
        df['ma_20']       = df['close'].rolling(window=20).mean()
        df['ma_20_slope'] = df['ma_20'].diff()
        bollinger         = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high']     = bollinger.bollinger_hband()
        df['bb_low']      = bollinger.bollinger_lband()
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        self.df = df
        return df
    
    def scale_data(self):
        data = self.df[self.feature_cols].values
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)
        self.scaler = scaler
        return scaled, scaler

############################### Dataset Class #############################################
class ForexDataset(Dataset):
    def __init__(self, data, seq_length=LOOKBACK, pred_length=PREDICT_HORIZON, target_col_idx = 3): #prediction Close cloumn
        self.data           = data
        self.seq_length     = seq_length     #lookback period
        self.pred_length    = pred_length    #predicting  
        self.target_col_idx = target_col_idx #which col want to predict

    def __len__(self):
        # The maximum starting index is total_length - seq_length - prediction_length
        return len(self.data) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length] # Input sequence
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length, self.target_col_idx]  # Predicting 'Close' price
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32) #tensor provided to the transformer model
    

################## Transformer Model class ######################
class TransformerTimeSeries(nn.Module):
    def __init__(self, feature_size=10, 
                 num_layers=2, 
                 d_model=64,
                 nhead=8,
                 dim_feedforward=256,
                 dropout=0.1,
                 seq_length=30,
                 pred_length=1):
        super(TransformerTimeSeries, self).__init__()
        self.input_fc = nn.Linear(feature_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, pred_length)

    def forward(self, src):
        batch_size, seq_length, _ = src.shape
        src = self.input_fc(src)
        src = src + self.pos_embedding[:, :seq_length, :]
        src = src.permute(1, 0, 2)  
        encoded = self.transformer_encoder(src) 
        transformer_out = encoded[-1, :, :]  
        out = self.fc_out(transformer_out)
        return out
    

########################Training, Saving, Loading, Predicting##########################################
class ModelHandler:
    def __init__(self, model_path=os.path.join(MODEL_DIR, "transformer.pt"), feature_size=10, seq_length=LOOKBACK, pred_length=PREDICT_HORIZON, device=DEVICE):
        self.model_path = model_path
        self.device = device
        self.model = TransformerTimeSeries(feature_size=feature_size, seq_length=seq_length, pred_length=pred_length).to(device)

    def save(self):
        torch.save(self.model.state_dict(), self.model_path)
        logging.info(f"Model saved to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            logging.info(f"Model loaded from {self.model_path}")
            return True
        logging.info("No model file found; training required.")
        return False

    def train(self, scaled_data, target_col_idx=3, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-3, val_split=0.1):
        dataset = ForexDataset(scaled_data, seq_length=LOOKBACK, pred_length=PREDICT_HORIZON, target_col_idx=target_col_idx)
        total = len(dataset)
        if total < 10:
            logging.warning("Not enough samples to train.")
            return

        val_count = max(1, int(total * val_split))
        train_count = total - val_count
        train_set = torch.utils.data.Subset(dataset, list(range(0, train_count)))
        val_set = torch.utils.data.Subset(dataset, list(range(train_count, total)))

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        best_val = float('inf')
        for ep in range(epochs):
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)  # (B, pred_length)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # validation
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    out = self.model(xb)
                    val_losses.append(criterion(out, yb).item())

            mean_train = float(np.mean(train_losses)) if train_losses else 0.0
            mean_val = float(np.mean(val_losses)) if val_losses else 0.0
            logging.info(f"Epoch {ep+1}/{epochs} TrainLoss={mean_train:.6f} ValLoss={mean_val:.6f}")

            # save best
            if mean_val < best_val:
                best_val = mean_val
                self.save()

    def predict(self, last_seq_scaled):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(last_seq_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
            out = self.model(x).cpu().numpy().reshape(-1)
        return out
    
############################## The Bot Class #########################################
class MT5TransformerBot:
    def __init__(self, connector: MT5Connector, symbol: str = SYMBOL, seq_length: int = LOOKBACK, retrain_days: int = RETRAIN_DAYS,
                 model_path: str = os.path.join(MODEL_DIR, "transformer.pt"), lot=LOT, sl_pips=SL_PIPS, tp_pips=TP_PIPS, max_pos=MAX_POS, state_file=STATE_FILE):
        self.conn             = connector
        self.symbol           = symbol
        self.seq_length       = seq_length
        self.feature_engineer = FeatureEngineer()
        self.model_handler    = ModelHandler(model_path=model_path, feature_size=len(self.feature_engineer.feature_cols), seq_length=seq_length)
        self.retrain_interval = timedelta(days=retrain_days)
        self.lot              = lot
        self.sl_pips          = sl_pips
        self.tp_pips          = tp_pips
        self.max_pos          = max_pos
        self.state_file       = state_file
        self.state            = self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    st = json.load(f)
                    if 'last_retrain' in st and st['last_retrain'] is not None:
                        st['last_retrain'] = datetime.fromisoformat(st['last_retrain'])
                    return st
            except Exception:
                pass
        return {"last_retrain": None}

    def _save_state(self):
        st = dict(self.state)
        if st['last_retrain'] is not None:
            st['last_retrain'] = st['last_retrain'].isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(st, f)

    def fetch_and_prepare(self, n_bars=3000):
        df             = self.conn.fetch_ohlc(self.symbol, TIMEFRAME, n_bars)
        self.feature_engineer.set_df(df)
        df             = self.feature_engineer.add_indicators()
        scaled, scaler = self.feature_engineer.scale_data()
        return df, scaled, scaler

    def initial_train_if_needed(self):
        # load model if exists, otherwise train
        if not self.model_handler.load():
            logging.info("No saved model found. Running initial training.")
            df, scaled, scaler = self.fetch_and_prepare(n_bars=3000)
            target_idx = self.feature_engineer.feature_cols.index('close')
            self.model_handler.train(scaled, target_col_idx=target_idx, epochs=EPOCHS)
            self.model_handler.load()
            self.state['last_retrain'] = datetime.utcnow()
            self._save_state()

    def _pips_to_price_offset(self, pips):
        info = mt5.symbol_info(self.symbol)
        if info is None:
            point = 0.01
        else:
            point = info.point
        return pips * point

    def _position_count(self):
        positions = mt5.positions_get(symbol=self.symbol)
        return len(positions)

    def generate_signal(self):
        # fetch latest
        df, scaled, scaler = self.fetch_and_prepare(n_bars=max(1000, self.seq_length + 200))
        if len(scaled) < self.seq_length:
            logging.warning("Not enough data for prediction.")
            return "HOLD", None

        last_seq = scaled[-self.seq_length:]
        pred_scaled = self.model_handler.predict(last_seq)  # e.g. shape (1,)
        # inverse scale predicted close
        dummy = np.zeros((self.model_handler.model.fc_out.out_features, len(self.feature_engineer.feature_cols))) if False else np.zeros((PREDICT_HORIZON, len(self.feature_engineer.feature_cols)))
        # simpler: create a single-row dummy
        dummy = np.zeros((1, len(self.feature_engineer.feature_cols)))
        target_idx = self.feature_engineer.feature_cols.index('close')
        dummy[0, target_idx] = pred_scaled[0]
        pred_price = scaler.inverse_transform(dummy)[0, target_idx]
        last_close = df['close'].iloc[-1]

        logging.info(f"Prediction (price): {pred_price:.5f} last_close: {last_close:.5f}")

        rel_diff = (pred_price - last_close) / last_close
        threshold = 0.0003  # tweak per instrument/timeframe
        if rel_diff > threshold:
            return "BUY", {"pred_price": pred_price, "last_close": last_close}
        elif rel_diff < -threshold:
            return "SELL", {"pred_price": pred_price, "last_close": last_close}
        else:
            return "HOLD", {"pred_price": pred_price, "last_close": last_close}

    def execute_trade(self, side):
        # enforce position limits
        if self._position_count() >= self.max_pos:
            logging.info("Max positions reached, skipping trade.")
            return False

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logging.error("Failed to obtain tick for symbol.")
            return False

        if side == "BUY":
            price = tick.ask
            sl = price - self._pips_to_price_offset(self.sl_pips)
            tp = price + self._pips_to_price_offset(self.tp_pips)
            order_type = mt5.ORDER_TYPE_BUY
        elif side == "SELL":
            price = tick.bid
            sl = price + self._pips_to_price_offset(self.sl_pips)
            tp = price - self._pips_to_price_offset(self.tp_pips)
            order_type = mt5.ORDER_TYPE_SELL
        else:
            logging.info("No trade to execute (HOLD).")
            return False

        result = self.conn.send_order(self.symbol, self.lot, order_type, price=price, sl=sl, tp=tp)
        if result is None:
            logging.error("order_send returned None")
            return False

        # result is a structure with retcode and comment
        if getattr(result, "retcode", None) not in (0, mt5.TRADE_RETCODE_DONE):
            logging.error(f"Trade failed retcode={getattr(result, 'retcode', 'N/A')} comment={getattr(result, 'comment', '')}")
            return False

        logging.info(f"Trade executed: {side} price={price:.5f} volume={self.lot}")
        return True

    def maybe_retrain(self):
        last = self.state.get('last_retrain')
        now = datetime.utcnow()
        if last is None:
            need = True
        else:
            need = (now - last) >= self.retrain_interval

        if need:
            logging.info("Retraining model now (scheduled).")
            df, scaled, scaler = self.fetch_and_prepare(n_bars=4000)
            target_idx = self.feature_engineer.feature_cols.index('close')
            self.model_handler.train(scaled, target_col_idx=target_idx, epochs=max(3, EPOCHS // 4))
            # load best
            self.model_handler.load()
            self.state['last_retrain'] = datetime.utcnow()
            self._save_state()
            logging.info("Retrain completed and state saved.")
        else:
            next_retrain = last + self.retrain_interval if last else None
            logging.info(f"No retrain needed. Next retrain at {next_retrain} (UTC)")

    def start(self, poll_interval=POLL_INTERVAL):
        logging.info("Bot starting...")
        try:
            self.conn._init_mt5()
            # initial model readiness
            self.initial_train_if_needed()

            logging.info("Entering main loop.")
            while True:
                try:
                    signal, meta = self.generate_signal()
                    logging.info(f"Signal: {signal} meta={meta}")
                    if signal in ("BUY", "SELL"):
                        self.execute_trade(signal)
                    self.maybe_retrain()
                except Exception as e:
                    logging.exception("Error in loop: %s", str(e))
                time.sleep(poll_interval)

        finally:
            self.conn.shutdown()
            logging.info("Bot stopped.")


############################# Run the Main Function #########################################
if __name__ == "__main__":
    connector = MT5Connector(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    bot = MT5TransformerBot(connector   =connector,
                            symbol      =SYMBOL,
                            seq_length  =LOOKBACK,
                            retrain_days=RETRAIN_DAYS,
                            model_path  =os.path.join(MODEL_DIR, "transformer.pt"),
                            lot         =LOT,
                            sl_pips     =SL_PIPS,
                            tp_pips     =TP_PIPS,
                            max_pos     =MAX_POS,
                            state_file  =STATE_FILE)

    try:
        bot.start(poll_interval=POLL_INTERVAL)
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received — stopping bot.")
        connector.shutdown()