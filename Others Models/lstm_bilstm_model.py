"""
LSTM & BiLSTM Model for Traffic Flow Prediction with Weather Integration
Excels at capturing temporal dependencies in time-series traffic data
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import json
import os

class LSTMTrafficPredictor:
    def __init__(self, input_len=12, output_len=12, use_bidirectional=True):
        """
        Args:
            input_len: Number of timesteps to look back
            output_len: Number of timesteps to predict
            use_bidirectional: Use BiLSTM instead of LSTM
        """
        self.input_len = input_len
        self.output_len = output_len
        self.use_bidirectional = use_bidirectional
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def load_data(self, csv_path, target_feature='speed'):
        """Load and preprocess traffic data with weather features"""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract features
        feature_cols = ['speed', 'volume', 'temp', 'precipitation', 
                       'wind_speed', 'humidity', 'visibility', 'pressure']
        
        # Group by sensor and timestamp
        sensors = df['sensor_id'].unique()
        print(f"Number of sensors: {len(sensors)}")
        
        # Prepare sequences for each sensor
        X_all, y_all = [], []
        
        for sensor_id in sensors:
            sensor_data = df[df['sensor_id'] == sensor_id].sort_values('timestamp')
            values = sensor_data[feature_cols].values
            
            # Create sequences
            for i in range(len(values) - self.input_len - self.output_len + 1):
                X_all.append(values[i:i+self.input_len])
                # Target: predict the target feature
                target_idx = feature_cols.index(target_feature)
                y_all.append(values[i+self.input_len:i+self.input_len+self.output_len, target_idx])
        
        X = np.array(X_all)
        y = np.array(y_all)
        
        print(f"Data shape - X: {X.shape}, y: {y.shape}")
        return X, y, feature_cols
    
    def prepare_data(self, X, y, train_split=0.7, val_split=0.15):
        """Split and normalize data"""
        n_samples = len(X)
        train_size = int(n_samples * train_split)
        val_size = int(n_samples * val_split)
        
        # Split data
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        # Normalize features
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_reshaped)
        
        X_train = self.scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def build_model(self, input_shape, lstm_units=[128, 64], dropout=0.2):
        """Build LSTM or BiLSTM model"""
        inputs = keras.Input(shape=input_shape)
        x = inputs
        
        # Stack LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            
            if self.use_bidirectional:
                x = layers.Bidirectional(
                    layers.LSTM(units, return_sequences=return_sequences, dropout=dropout)
                )(x)
            else:
                x = layers.LSTM(units, return_sequences=return_sequences, dropout=dropout)(x)
            
            x = layers.Dropout(dropout)(x)
        
        # Dense layers for prediction
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(self.output_len)(x)
        
        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        print(f"\n{'BiLSTM' if self.use_bidirectional else 'LSTM'} Model Summary:")
        model.summary()
        return model
    
    def train(self, train_data, val_data, epochs=100, batch_size=32):
        """Train the model"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                f'best_{"bilstm" if self.use_bidirectional else "lstm"}_model.h5',
                monitor='val_loss', save_best_only=True
            )
        ]
        
        print("\nTraining model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, test_data):
        """Evaluate model on test data"""
        X_test, y_test = test_data
        
        print("\nEvaluating model...")
        test_loss, test_mae, test_mse = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Predictions
        y_pred = self.model.predict(X_test, verbose=0)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test.flatten(), y_pred.flatten())
        
        # MAPE
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        
        print(f"\nTest Results:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return metrics, y_pred
    
    def plot_results(self, y_true, y_pred, save_path='lstm_results.png'):
        """Plot training history and predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training history
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.history.history['mae'], label='Train MAE')
        axes[0, 1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('Model MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Predictions vs Actual (first timestep)
        axes[1, 0].scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5)
        axes[1, 0].plot([y_true[:, 0].min(), y_true[:, 0].max()], 
                        [y_true[:, 0].min(), y_true[:, 0].max()], 'r--', lw=2)
        axes[1, 0].set_title('Predictions vs Actual (t+1)')
        axes[1, 0].set_xlabel('Actual')
        axes[1, 0].set_ylabel('Predicted')
        axes[1, 0].grid(True)
        
        # Time series plot
        sample_idx = 100
        axes[1, 1].plot(y_true[sample_idx], label='Actual', marker='o')
        axes[1, 1].plot(y_pred[sample_idx], label='Predicted', marker='s')
        axes[1, 1].set_title('Sample Prediction Sequence')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Speed')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nResults saved to {save_path}")
        plt.close()


def main():
    """Main training pipeline"""
    # Configuration
    DATA_PATH = '/home/arnab/Desktop/TraffCal/newdatset/merged_unscaled.csv'
    USE_BILSTM = True  # Set to False for LSTM
    
    # Initialize predictor
    predictor = LSTMTrafficPredictor(
        input_len=12,
        output_len=12,
        use_bidirectional=USE_BILSTM
    )
    
    # Load data
    X, y, feature_cols = predictor.load_data(DATA_PATH, target_feature='speed')
    
    # Prepare data
    train_data, val_data, test_data = predictor.prepare_data(X, y)
    
    # Build model
    predictor.build_model(
        input_shape=(predictor.input_len, len(feature_cols)),
        lstm_units=[128, 64],
        dropout=0.2
    )
    
    # Train model
    predictor.train(train_data, val_data, epochs=100, batch_size=64)
    
    # Evaluate
    metrics, y_pred = predictor.evaluate(test_data)
    
    # Plot results
    predictor.plot_results(
        test_data[1], 
        y_pred,
        save_path=f'{"bilstm" if USE_BILSTM else "lstm"}_results.png'
    )
    
    # Save metrics
    with open(f'{"bilstm" if USE_BILSTM else "lstm"}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\n✅ Training completed successfully!")


if __name__ == '__main__':
    main()
