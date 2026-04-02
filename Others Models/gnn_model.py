"""
Graph Neural Network (GNN) Model for Traffic Flow Prediction
Designed for spatial data - handles road network topology and node dependencies
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
from scipy.spatial.distance import pdist, squareform

class GraphConvolution(layers.Layer):
    """Custom Graph Convolution Layer"""
    def __init__(self, units, activation='relu', **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        
    def build(self, input_shape):
        # input_shape: [(batch, nodes, features), (nodes, nodes)]
        feature_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        
    def call(self, inputs):
        features, adjacency = inputs
        # features: (batch, nodes, features)
        # adjacency: (nodes, nodes)
        
        # Graph convolution: A * X * W
        support = tf.matmul(features, self.kernel)
        output = tf.matmul(adjacency, support)
        output = output + self.bias
        return self.activation(output)
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'activation': self.activation})
        return config


class GNNTrafficPredictor:
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
        self.adjacency_matrix = None
        
    def load_data(self, csv_path, target_feature='speed'):
        """Load and preprocess traffic data"""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        feature_cols = ['speed', 'volume', 'temp', 'precipitation', 
                       'wind_speed', 'humidity', 'visibility', 'pressure']
        
        # Get unique sensors
        sensors = sorted(df['sensor_id'].unique())
        num_nodes = len(sensors)
        print(f"Number of sensors (nodes): {num_nodes}")
        
        # Pivot data: create matrix [time, nodes, features]
        data_matrix = []
        timestamps = sorted(df['timestamp'].unique())
        
        for ts in timestamps:
            ts_data = df[df['timestamp'] == ts].set_index('sensor_id')
            # Ensure all sensors present
            node_features = []
            for sensor in sensors:
                if sensor in ts_data.index:
                    node_features.append(ts_data.loc[sensor, feature_cols].values)
                else:
                    node_features.append(np.zeros(len(feature_cols)))
            data_matrix.append(node_features)
        
        data_matrix = np.array(data_matrix)  # Shape: (time, nodes, features)
        print(f"Data matrix shape: {data_matrix.shape}")
        
        # Create sequences
        X_all, y_all = [], []
        target_idx = feature_cols.index(target_feature)
        
        for i in range(len(data_matrix) - self.input_len - self.output_len + 1):
            X_all.append(data_matrix[i:i+self.input_len])
            y_all.append(data_matrix[i+self.input_len:i+self.input_len+self.output_len, :, target_idx])
        
        X = np.array(X_all)  # (samples, time_in, nodes, features)
        y = np.array(y_all)  # (samples, time_out, nodes)
        
        print(f"Data shape - X: {X.shape}, y: {y.shape}")
        return X, y, feature_cols, num_nodes, sensors
    
    def create_adjacency_matrix(self, num_nodes, method='distance'):
        """
        Create adjacency matrix for road network
        Methods: 'distance', 'knn', 'fully_connected'
        """
        if method == 'fully_connected':
            # Simple fully connected graph
            adj = np.ones((num_nodes, num_nodes))
            np.fill_diagonal(adj, 0)
        elif method == 'knn':
            # K-nearest neighbors (simulate with random for demo)
            # In practice, use actual geographic coordinates
            k = min(10, num_nodes - 1)
            adj = np.zeros((num_nodes, num_nodes))
            for i in range(num_nodes):
                neighbors = np.random.choice(num_nodes, k, replace=False)
                adj[i, neighbors] = 1
            adj = (adj + adj.T) > 0  # Make symmetric
            adj = adj.astype(float)
        else:  # distance-based (simulated)
            # Simulate sensor locations and compute distance-based adjacency
            np.random.seed(42)
            locations = np.random.rand(num_nodes, 2) * 100
            distances = squareform(pdist(locations))
            
            # Gaussian kernel
            sigma = np.std(distances)
            adj = np.exp(-distances**2 / (2 * sigma**2))
            np.fill_diagonal(adj, 0)
            
            # Threshold
            threshold = np.percentile(adj, 75)
            adj[adj < threshold] = 0
        
        # Normalize adjacency matrix (symmetric normalization)
        adj = adj + np.eye(num_nodes)  # Add self-loops
        degree = np.sum(adj, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        adj_normalized = D_inv_sqrt @ adj @ D_inv_sqrt
        
        self.adjacency_matrix = adj_normalized.astype(np.float32)
        print(f"Adjacency matrix shape: {self.adjacency_matrix.shape}")
        return self.adjacency_matrix
    
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
        original_shape = X_train.shape
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(X_train_flat)
        
        X_train = self.scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(original_shape)
        X_val = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def build_model(self, input_shape, num_nodes, gcn_units=[64, 32], dropout=0.3):
        """Build GNN model with temporal and spatial components"""
        # Input: (batch, time, nodes, features)
        feature_input = keras.Input(shape=input_shape, name='features')
        adjacency_input = keras.Input(shape=(num_nodes, num_nodes), name='adjacency')
        
        # Process each timestep with GNN
        outputs = []
        for t in range(self.input_len):
            # Extract features at timestep t
            x_t = layers.Lambda(lambda x: x[:, t, :, :])(feature_input)
            
            # Apply GCN layers
            for units in gcn_units:
                x_t = GraphConvolution(units, activation='relu')([x_t, adjacency_input])
                x_t = layers.Dropout(dropout)(x_t)
            
            outputs.append(x_t)
        
        # Stack temporal features
        x = layers.Lambda(lambda x: tf.stack(x, axis=1))(outputs)
        
        # Temporal processing with LSTM
        x = layers.Reshape((self.input_len, num_nodes * gcn_units[-1]))(x)
        x = layers.LSTM(128, return_sequences=False)(x)
        x = layers.Dropout(dropout)(x)
        
        # Output layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(num_nodes * self.output_len)(x)
        outputs = layers.Reshape((self.output_len, num_nodes))(x)
        
        model = keras.Model(inputs=[feature_input, adjacency_input], outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        print("\nGNN Model Summary:")
        model.summary()
        return model
    
    def train(self, train_data, val_data, epochs=100, batch_size=32):
        """Train the GNN model"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Prepare adjacency matrix for batching
        adj_train = np.tile(self.adjacency_matrix[np.newaxis, :, :], (len(X_train), 1, 1))
        adj_val = np.tile(self.adjacency_matrix[np.newaxis, :, :], (len(X_val), 1, 1))
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'best_gnn_model.h5', monitor='val_loss', save_best_only=True
            )
        ]
        
        print("\nTraining GNN model...")
        self.history = self.model.fit(
            [X_train, adj_train], y_train,
            validation_data=([X_val, adj_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, test_data):
        """Evaluate model"""
        X_test, y_test = test_data
        adj_test = np.tile(self.adjacency_matrix[np.newaxis, :, :], (len(X_test), 1, 1))
        
        print("\nEvaluating GNN model...")
        y_pred = self.model.predict([X_test, adj_test], verbose=0)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test.flatten(), y_pred.flatten())
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
        
        print(f"\nTest Results:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return metrics, y_pred
    
    def plot_results(self, y_true, y_pred, save_path='gnn_results.png'):
        """Plot results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('GNN Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.history.history['mae'], label='Train MAE')
        axes[0, 1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('GNN Model MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Average across nodes for visualization
        y_true_avg = y_true.mean(axis=2)[:, 0]
        y_pred_avg = y_pred.mean(axis=2)[:, 0]
        
        axes[1, 0].scatter(y_true_avg, y_pred_avg, alpha=0.5)
        axes[1, 0].plot([y_true_avg.min(), y_true_avg.max()], 
                        [y_true_avg.min(), y_true_avg.max()], 'r--', lw=2)
        axes[1, 0].set_title('Predictions vs Actual (Avg across nodes)')
        axes[1, 0].set_xlabel('Actual')
        axes[1, 0].set_ylabel('Predicted')
        axes[1, 0].grid(True)
        
        # Sample node prediction
        sample_idx, node_idx = 50, 0
        axes[1, 1].plot(y_true[sample_idx, :, node_idx], label='Actual', marker='o')
        axes[1, 1].plot(y_pred[sample_idx, :, node_idx], label='Predicted', marker='s')
        axes[1, 1].set_title(f'Sample Prediction (Node {node_idx})')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Speed')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nResults saved to {save_path}")
        plt.close()


def main():
    DATA_PATH = '/home/arnab/Desktop/TraffCal/newdatset/merged_unscaled.csv'
    
    predictor = GNNTrafficPredictor(input_len=12, output_len=12)
    
    X, y, feature_cols, num_nodes, sensors = predictor.load_data(DATA_PATH)
    
    predictor.create_adjacency_matrix(num_nodes, method='distance')
    
    train_data, val_data, test_data = predictor.prepare_data(X, y)
    
    predictor.build_model(
        input_shape=(predictor.input_len, num_nodes, len(feature_cols)),
        num_nodes=num_nodes,
        gcn_units=[64, 32],
        dropout=0.3
    )
    
    predictor.train(train_data, val_data, epochs=100, batch_size=16)
    
    metrics, y_pred = predictor.evaluate(test_data)
    
    predictor.plot_results(test_data[1], y_pred, save_path='gnn_results.png')
    
    with open('gnn_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\n✅ GNN Training completed!")


if __name__ == '__main__':
    main()
