import os
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler


###################Data Preparation###################
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Datatime'] = pd.to_datetime(df['Datetime'], format='%d.%m.%Y %H:%M:%S.%f')
    df.sort_values('Datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

#################### indicator calculation ####################
def add_technical_indicators(df):
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    df['ma_20'] = df['Close'].rolling(window=20).mean()
    df['ma_20_slope'] = df['ma_20'].diff()
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()

    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

#################Scaled Arrays#################
def scale_data(df, feature_cols = None):

    if feature_cols is None:
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'rsi', 'ma_20', 'ma_20_slope', 'bb_high', 'bb_low']

    data = df[feature_cols].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler, feature_cols

################## DataSet variation ##################
class ForexDataset(Dataset):
    def __init__(self, data, seq_length=60, pred_length=1, feature_cols=4, target_col_idx = 3): #prediction Close cloumn
        self.data           = data
        self.seq_length     = seq_length     #lookback period
        self.pred_length    = pred_length    #predicting 
        self.feature_cols   = feature_cols   
        self.target_col_idx = target_col_idx #which col want to predict

    def __len__(self):
        # The maximum starting index is total_length - seq_length - prediction_length
        return len(self.data) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length] # Input sequence
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length, self.target_col_idx]  # Predicting 'Close' price
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32) #tensor provided to the transformer model
    
################## Transformer Model class ######################
# [Input: (B, 30, 9)]
#          |
#      [Linear: 9 → 64]
#          |
# [+Positional Embedding (1, 30, 64)]
#          |
#      [Transformer Encoder]
#      (2 Layers, 8 Heads, FF=256)
#          |
# [Output Linear: 64 → 1]
#          |
# [Predictions: (B, 30, 1)
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

        #Each feature vector(feature_size) into a d-model sized vector
        self.input_fc = nn.Linear(feature_size, d_model)
        #positional encoding(simple learnable or sinusoidal)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, d_model))

        #Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        #prediction_length 1(close price)
        self.fc_out = nn.Linear(d_model, pred_length)

    def forward(self, src):
        #src shape: (batch_size, seq_length, feature_size)
        batch_size, seq_length, _ = src.shape
        
        #Input linear layer(B, seq_length, d_model)
        src = self.input_fc(src)
        
        #Add positional encoding
        src = src + self.pos_embedding[:, :seq_length, :]
        
        #Transformer expects input shape: (seq_length, B, d_model)
        src = src.permute(1, 0, 2)  
        
        #Pass through Transformer Encoder
        encoded = self.transformer_encoder(src) 
        
        #Back to (B, seq_length, d_model)
        transformer_out = encoded[-1, :, :]  
        
        #Output layer to get predictions (B, seq_length, pred_length)
        out = self.fc_out(transformer_out)
        return out
    
################################ Train tansformer ##############################
def train_model(model, train_loader, val_loader=None, lr=1e-3, epochs=20, device='cpu'): #learning rate(lr)
    criterion = nn.MSELoss() #For regression on price
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)  # output shape: [batch_size, prediction_length]
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        mean_train_loss = np.mean(train_losses)

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    output_val = model(x_val)
                    loss_val = criterion(output_val, y_val)
                    val_losses.append(loss_val.item())
            mean_val_loss = np.mean(val_losses)
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {mean_train_loss:.6f}, Val Loss: {mean_val_loss:.6f}")
        else:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {mean_train_loss:.6f}")

    return model

######################### Evaluate the model #########################
def evaluate_model(model, test_loader, scaler, feature_cols, target_col_idx, 
                   window_width=10, start_index=0, pred_length=1, device='cpu'): #window_width: Number of points to plot for real vs. predicted prices.
    model.eval()
    real_prices = []
    predicted_prices = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)

            # Get model predictions
            predictions = model(x_batch).cpu().numpy()  # shape: [batch_size, pred_length]
            y_batch = y_batch.cpu().numpy()  # shape: [batch_size, pred_length]

            for i in range(len(predictions)):
                # Create dummy inputs for inverse scaling
                dummy_pred = np.zeros((pred_length, len(feature_cols)))
                dummy_pred[:, target_col_idx] = predictions[i]  # Assign predicted future prices

                dummy_real = np.zeros((pred_length, len(feature_cols)))
                dummy_real[:, target_col_idx] = y_batch[i]  # Assign real future prices

                # Inverse transform both predicted and actual prices
                pred_inversed = scaler.inverse_transform(dummy_pred)[:, target_col_idx]
                real_inversed = scaler.inverse_transform(dummy_real)[:, target_col_idx]

                # Store values
                predicted_prices.extend(pred_inversed)
                real_prices.extend(real_inversed)

    #convert lists to numpy arrays
    real_prices      = np.array(real_prices).flatten()
    predicted_prices = np.array(predicted_prices).flatten()

    # -------------------------
    # Compute Accuracy Metrics
    # -------------------------
    mse = np.mean((real_prices - predicted_prices) ** 2)
    mae = np.mean(np.abs(real_prices - predicted_prices))

    print(f"Model Evaluation:\n  - Mean Squared Error (MSE): {mse:.4f}")
    print(f"  - Mean Absolute Error (MAE): {mae:.4f}")

    # -------------------------
    # Adjust Start Index and Window Width for Plot
    # -------------------------
    if start_index < 0 or start_index >= len(real_prices):
        print(f"Warning: start_index {start_index} is out of bounds. Using 0 instead.")
        start_index = 0

    end_index = min(start_index + window_width * pred_length, len(real_prices))  # Adjust for multi-step forecasts

    # -------------------------
    # Plot Real vs. Predicted Prices
    # -------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(range(start_index, end_index), real_prices[start_index:end_index], 
             label="Real Close Prices", linestyle="dashed", marker='o')
    plt.plot(range(start_index, end_index), predicted_prices[start_index:end_index], 
             label="Predicted Close Prices", linestyle="-", marker='x')
    plt.title(f"Real vs. Predicted Close Prices (From index {start_index}, {window_width} Windows, {pred_length} Steps Each)")
    plt.xlabel("Time Steps")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()


################ Main function for runining the code ################
def main():
    file_path      = "C:/Users/Mritunjay Maddhesiya/OneDrive/Desktop/MT5/12_Transformers/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv"
    
    # Load and preprocess data
    df                                = load_data(file_path)
    df                                = add_technical_indicators(df)
    scaled_data, scaler, feature_cols = scale_data(df)
    target_col_idx                    = feature_cols.index('Close')


    # Split data into train, val, test sets (70%, 15%, 15%)
    seq_length   = 30
    pred_length  = 1    
    dataset      = ForexDataset(scaled_data, seq_length, pred_length, len(feature_cols), target_col_idx=target_col_idx)
    
    train_size = int(len(dataset) * 0.8)
    val_size   = int(len(dataset) * 0.1)
    test_size  = len(dataset) - train_size - val_size
    
    # !!! don't use this !!!! train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    # Perform sequential splitting (without shuffling)
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset   = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset  = torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset)))


    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------
    # 3. Create and Train Transformer Model
    # -------------------------
    model = TransformerTimeSeries(
        feature_size     =len(feature_cols),
        num_layers       =2,
        d_model          =64,
        nhead            =8,
        dim_feedforward  =256,
        dropout          =0.1,
        seq_length       =seq_length,
        pred_length      =pred_length
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trained_model = train_model(model, train_loader, val_loader, lr=1e-3, epochs=20, device=device)

    evaluate_model(trained_model, test_loader, scaler, feature_cols, target_col_idx, window_width=45, start_index=70, pred_length=1, device=device)

####### Run the main function ########
if __name__ == "__main__":
    main()