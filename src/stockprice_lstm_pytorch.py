import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Custom EarlyStopper class for better early stopping
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0, monitor='loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_metric = float('inf')
        self.best_state = None

    def early_stop(self, val_loss, val_mae, model):
        metric = val_loss if self.monitor == 'loss' else val_mae
        improved = False
        if metric < self.best_metric - self.min_delta:
            self.best_metric = metric
            self.counter = 0
            self.best_state = model.state_dict()
            improved = True
        else:
            self.counter += 1
        stop = self.counter >= self.patience
        return stop, improved

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define paths and parameters
script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets src/ dir
project_root = os.path.dirname(script_dir)  # Gets test-lstm/ root
data_dir = os.path.join(project_root, 'data/')  # Full path to data/
train_file = os.path.join(data_dir, 'training_data_spy_20250101_20250624.csv')
test_file = os.path.join(data_dir, 'testing_data_spy_20250625_20250710.csv')
export_dir = os.path.join(project_root, 'models')
os.makedirs(export_dir, exist_ok=True)
seq_length = 20
learning_rate = 0.0005
epochs = 1000
batch_size = 32

# Load data
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Define feature and target columns
# Features: columns 6 to 49 (0-based index 5 to 48)
feature_cols = train_df.columns[5:49]
target_col = 'Label_8'
num_features = len(feature_cols)

# Function to create sequences, respecting continuity
def create_sequences(df, seq_length):
    df = df.sort_values(['TradingDay', 'TradingMsOfDay']).reset_index(drop=True)
    features = df[feature_cols].values
    targets = df[target_col].values
    days = df['TradingDay'].values
    ms = df['TradingMsOfDay'].values
    
    X = []
    y = []
    
    i = 0
    while i <= len(df) - seq_length:
        # Check if the sequence is consecutive (same day, 60000 ms increments)
        is_consecutive = True
        current_day = days[i]
        current_ms = ms[i]
        
        for j in range(1, seq_length):
            if days[i + j] != current_day or ms[i + j] != current_ms + j * 60000:
                is_consecutive = False
                break
        
        if is_consecutive:
            seq = features[i:i + seq_length]
            label = targets[i + seq_length - 1]  # Label_8 of the last timestep
            X.append(seq)
            y.append(label)
            i += 1  # Sliding window, step by 1
        else:
            # Skip to the next potential start (after the gap)
            i += 1
    
    return np.array(X), np.array(y)

# Create sequences for train and test
X_train, y_train = create_sequences(train_df, seq_length)
X_test, y_test = create_sequences(test_df, seq_length)

# Print the size of x_train and y_train
print(f'Train sequences shape: {X_train.shape}, Train labels shape: {y_train.shape}')
print(f'Test sequences shape: {X_test.shape}, Test labels shape: {y_test.shape}')

# Print target stats
print(f'y_train mean: {np.mean(y_train):.4f}, std: {np.std(y_train):.4f}, min: {np.min(y_train):.4f}, max: {np.max(y_train):.4f}')
print(f'y_test mean: {np.mean(y_test):.4f}, std: {np.std(y_test):.4f}, min: {np.min(y_test):.4f}, max: {np.max(y_test):.4f}')

# Scale features (fit on train, apply to both)
scaler = MinMaxScaler()
X_train_reshaped = X_train.reshape(-1, num_features)
X_test_reshaped = X_test.reshape(-1, num_features)

scaler.fit(X_train_reshaped)
X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

# Save scaler params (same as before)
scaler_params = {
    "Min": scaler.data_min_.tolist(),
    "Max": scaler.data_max_.tolist()
}
with open(os.path.join(export_dir, 'scaler_params.json'), 'w') as f:
    json.dump(scaler_params, f)

# Scale targets
target_scaler = MinMaxScaler(feature_range=(-1, 1))
y_train_reshaped = y_train.reshape(-1, 1)
target_scaler.fit(y_train_reshaped)
y_train_scaled = target_scaler.transform(y_train_reshaped).flatten()
y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

# Save target scaler params
target_scaler_params = {
    "Min": target_scaler.data_min_.tolist(),
    "Max": target_scaler.data_max_.tolist()
}
with open(os.path.join(export_dir, 'target_scaler_params.json'), 'w') as f:
    json.dump(target_scaler_params, f)

# Custom Dataset class
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Add dimension for output

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create datasets and dataloaders
train_dataset = StockDataset(X_train_scaled, y_train_scaled)
test_dataset = StockDataset(X_test_scaled, y_test_scaled)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the LSTM model with dropout
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super(StockLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        # self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        # self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        # out = self.dropout1(out)
        out, _ = self.lstm2(out)
        # out = self.dropout2(out)
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Instantiate model, loss, optimizer
model = StockLSTM(num_features).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize early stopper
early_stopper = EarlyStopper(patience=1001, min_delta=0, monitor='loss')

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            mae = torch.mean(torch.abs(outputs - y_batch))
            val_mae += mae.item() * X_batch.size(0)
    
    val_loss /= len(test_loader.dataset)
    val_mae /= len(test_loader.dataset)
    
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
    
    # Early stopping check
    stop, improved = early_stopper.early_stop(val_loss, val_mae, model)
    if stop:
        print(f'Early stopping at epoch {epoch+1}')
        break

# Load best model state if available
if early_stopper.best_state is not None:
    model.load_state_dict(early_stopper.best_state)

model.eval()

# Evaluate on test set
test_loss = 0.0
test_mae = 0.0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item() * X_batch.size(0)
        mae = torch.mean(torch.abs(outputs - y_batch))
        test_mae += mae.item() * X_batch.size(0)

test_loss /= len(test_loader.dataset)
test_mae /= len(test_loader.dataset)
print(f'Test Loss (MSE): {test_loss}')
print(f'Test MAE: {test_mae}')

# Save the final model
torch.save(model.state_dict(), os.path.join(export_dir, 'lstm_stock_model.pth'))
print("Model saved as PyTorch state dict: lstm_stock_model.pth")

# To make predictions on train data
train_predictions = []
with torch.no_grad():
    for X_batch, _ in train_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        train_predictions.extend(outputs.cpu().numpy().flatten())

# Inverse transform train predictions
train_predictions = target_scaler.inverse_transform(np.array(train_predictions).reshape(-1, 1)).flatten()

train_results_df = pd.DataFrame({
    'Actual': y_train,
    'Predicted': train_predictions
})

# Save to CSV (in the project root; adjust path if needed)
train_output_file = os.path.join(project_root, 'train_predictions.csv')
train_results_df.to_csv(train_output_file, index=False)

print(f"Saved all {len(y_train)} train predictions and actuals to '{train_output_file}'")

# To make predictions on test data
predictions = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        predictions.extend(outputs.cpu().numpy().flatten())

# Inverse transform test predictions
predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Create DataFrame with all actuals and predictions
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions
})

# Save to CSV (in the project root; adjust path if needed)
output_file = os.path.join(project_root, 'test_predictions.csv')
results_df.to_csv(output_file, index=False)

print(f"Saved all {len(y_test)} test predictions and actuals to '{output_file}'")