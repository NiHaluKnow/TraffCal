"""
Convolutional Neural Network (CNN) Model for Traffic Flow Prediction
Used to analyze spatial features of traffic, often combined with LSTM
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

class CNNTrafficPredictor:
    def __init__(self, input_len=12, output_len=12):
        """
        Args:
            input_len: Number of timesteps to look back
            output_len: Number of timesteps to predict
        """
        self.input_len = input_len
        self.output_len = output_len
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def load_data(self, csv_path, target_feature='speed'):
        """Load and preprocess traffic data"""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        feature_cols = ['speed', 'volume', 'temp', 'precipitation', 
                       'wind_speed', 'humidity', 'visibility', 'pressure']
        
        sensors = df['sensor_id'].unique()
        print(f"Number of sensors: {len(sensors)}")
        
        X_all, y_all = [], []
        
        for sensor_id in sensors:
            sensor_data = df[df['sensor_id'] == sensor_id].sort_values('timestamp')
            values = sensor_data[feature_cols].values
            
            # Create sequences
            for i in range(len(values) - self.input_len - self.output_len + 1):
                X_all.append(values[i:i+self.input_len])
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
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        # Normalize
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_reshaped)
        
        X_train = self.scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def build_model(self, input_shape, filters=[64, 128, 64], kernel_size=3, dropout=0.3):
        """Build 1D CNN model for time series"""
        inputs = keras.Input(shape=input_shape)
        x = inputs
        
        # Stack Conv1D layers
        for i, num_filters in enumerate(filters):
            x = layers.Conv1D(
                filters=num_filters,
                kernel_size=kernel_size,
                padding='same',
                activation='relu'
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout)(x)
            
            # Add max pooling every 2 layers (if sequence is long enough)
            if i % 2 == 1 and x.shape[1] > 2:
                x = layers.MaxPooling1D(pool_size=2)(x)
        
        # Global pooling
        x = layers.GlobalMaxPooling1D()(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        
        # Output layer
        outputs = layers.Dense(self.output_len)(x)
        
        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        print("\nCNN Model Summary:")
        model.summary()
        return model
    
    def train(self, train_data, val_data, epochs=100, batch_size=64):
        """Train the CNN model"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'best_cnn_model.h5', monitor='val_loss', save_best_only=True
            )
        ]
        
        print("\nTraining CNN model...")
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
        """Evaluate model"""
        X_test, y_test = test_data
        
        print("\nEvaluating CNN model...")
        y_pred = self.model.predict(X_test, verbose=0)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test.flatten(), y_pred.flatten())
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
    
    def plot_results(self, y_true, y_pred, save_path='cnn_results.png'):
        """Plot results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training history
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('CNN Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.history.history['mae'], label='Train MAE')
        axes[0, 1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('CNN Model MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Predictions vs Actual
        axes[1, 0].scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5)
        axes[1, 0].plot([y_true[:, 0].min(), y_true[:, 0].max()], 
                        [y_true[:, 0].min(), y_true[:, 0].max()], 'r--', lw=2)
        axes[1, 0].set_title('Predictions vs Actual (t+1)')
        axes[1, 0].set_xlabel('Actual Speed')
        axes[1, 0].set_ylabel('Predicted Speed')
        axes[1, 0].grid(True)
        
        # Sample sequence
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
    DATA_PATH = '/home/arnab/Desktop/TraffCal/newdatset/merged_unscaled.csv'
    
    predictor = CNNTrafficPredictor(input_len=12, output_len=12)
    
    X, y, feature_cols = predictor.load_data(DATA_PATH, target_feature='speed')
    
    train_data, val_data, test_data = predictor.prepare_data(X, y)
    
    predictor.build_model(
        input_shape=(predictor.input_len, len(feature_cols)),
        filters=[64, 128, 64],
        kernel_size=3,
        dropout=0.3
    )
    
    predictor.train(train_data, val_data, epochs=100, batch_size=64)
    
    metrics, y_pred = predictor.evaluate(test_data)
    
    predictor.plot_results(test_data[1], y_pred, save_path='cnn_results.png')
    
    with open('cnn_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\n✅ CNN Training completed!")


if __name__ == '__main__':
    main()
