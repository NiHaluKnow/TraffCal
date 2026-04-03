"""
Random Forest & XGBoost Models for Traffic Flow Prediction
Ensemble methods highly effective for structured data with >95% accuracy
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import json
import pickle
from datetime import datetime

class TreeBasedTrafficPredictor:
    def __init__(self, model_type='xgboost', input_len=12, output_len=12):
        """
        Args:
            model_type: 'random_forest' or 'xgboost'
            input_len: Number of timesteps to look back
            output_len: Number of timesteps to predict
        """
        self.model_type = model_type
        self.input_len = input_len
        self.output_len = output_len
        self.models = []  # One model per output timestep
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def load_data(self, csv_path, target_feature='speed'):
        """Load and prepare data"""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Feature columns
        feature_cols = ['speed', 'volume', 'temp', 'precipitation', 
                       'wind_speed', 'humidity', 'visibility', 'pressure',
                       'hour', 'day_of_week', 'month', 'is_weekend']
        
        sensors = df['sensor_id'].unique()
        print(f"Number of sensors: {len(sensors)}")
        
        X_all, y_all = [], []
        
        for sensor_id in sensors:
            sensor_data = df[df['sensor_id'] == sensor_id].sort_values('timestamp')
            values = sensor_data[feature_cols].values
            
            # Create flattened sequences for tree models
            for i in range(len(values) - self.input_len - self.output_len + 1):
                # Flatten input sequence
                X_sequence = values[i:i+self.input_len].flatten()
                X_all.append(X_sequence)
                
                # Target: multiple timesteps
                target_idx = feature_cols.index(target_feature)
                y_sequence = values[i+self.input_len:i+self.input_len+self.output_len, target_idx]
                y_all.append(y_sequence)
        
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
        
        # Normalize features
        self.scaler_X.fit(X_train)
        X_train = self.scaler_X.transform(X_train)
        X_val = self.scaler_X.transform(X_val)
        X_test = self.scaler_X.transform(X_test)
        
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def build_and_train_models(self, train_data, val_data):
        """Build and train separate models for each prediction horizon"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        self.models = []
        
        print(f"\nTraining {self.model_type} models...")
        
        for i in range(self.output_len):
            print(f"\nTraining model for timestep t+{i+1}...")
            
            if self.model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    n_jobs=-1,
                    random_state=42,
                    verbose=0
                )
            elif self.model_type == 'xgboost':
                model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    n_jobs=-1,
                    random_state=42,
                    verbosity=0
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Train on this output timestep
            model.fit(
                X_train, y_train[:, i],
                eval_set=[(X_val, y_val[:, i])],
                verbose=False
            )
            
            self.models.append(model)
            
            # Validation score
            val_pred = model.predict(X_val)
            val_mae = mean_absolute_error(y_val[:, i], val_pred)
            print(f"  Validation MAE: {val_mae:.4f}")
        
        print(f"\n✅ All {self.output_len} models trained!")
    
    def predict(self, X):
        """Make predictions using all models"""
        predictions = []
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def evaluate(self, test_data):
        """Evaluate models"""
        X_test, y_test = test_data
        
        print("\nEvaluating models...")
        y_pred = self.predict(X_test)
        
        # Overall metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test.flatten(), y_pred.flatten())
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        # Per-timestep metrics
        timestep_metrics = []
        for i in range(self.output_len):
            t_mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            t_rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            timestep_metrics.append({'timestep': i+1, 'MAE': t_mae, 'RMSE': t_rmse})
        
        metrics = {
            'Overall': {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            },
            'Per_Timestep': timestep_metrics
        }
        
        print(f"\nOverall Test Results:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return metrics, y_pred
    
    def get_feature_importance(self, feature_names, top_k=20):
        """Get feature importance from first model"""
        if self.model_type == 'xgboost':
            importance = self.models[0].feature_importances_
        else:
            importance = self.models[0].feature_importances_
        
        # Create feature names for flattened input
        expanded_names = []
        for t in range(self.input_len):
            for fname in feature_names:
                expanded_names.append(f"{fname}_t-{self.input_len-t}")
        
        feature_imp = pd.DataFrame({
            'feature': expanded_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_imp.head(top_k)
    
    def plot_results(self, y_true, y_pred, feature_importance, save_path='tree_results.png'):
        """Plot results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Feature importance
        top_features = feature_importance.head(15)
        axes[0, 0].barh(range(len(top_features)), top_features['importance'])
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features['feature'], fontsize=8)
        axes[0, 0].set_xlabel('Importance')
        axes[0, 0].set_title('Top 15 Feature Importances')
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(True, axis='x')
        
        # Per-timestep MAE
        timestep_mae = [mean_absolute_error(y_true[:, i], y_pred[:, i]) 
                       for i in range(self.output_len)]
        axes[0, 1].plot(range(1, self.output_len+1), timestep_mae, marker='o')
        axes[0, 1].set_xlabel('Prediction Horizon')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('MAE by Prediction Horizon')
        axes[0, 1].grid(True)
        
        # Predictions vs Actual (t+1)
        axes[1, 0].scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5)
        axes[1, 0].plot([y_true[:, 0].min(), y_true[:, 0].max()], 
                        [y_true[:, 0].min(), y_true[:, 0].max()], 'r--', lw=2)
        axes[1, 0].set_title('Predictions vs Actual (t+1)')
        axes[1, 0].set_xlabel('Actual Speed')
        axes[1, 0].set_ylabel('Predicted Speed')
        axes[1, 0].grid(True)
        
        # Sample prediction sequence
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
    
    def save_models(self, path_prefix):
        """Save trained models"""
        for i, model in enumerate(self.models):
            filename = f"{path_prefix}_t{i+1}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
        
        # Save scalers
        with open(f"{path_prefix}_scaler_X.pkl", 'wb') as f:
            pickle.dump(self.scaler_X, f)
        
        print(f"Models saved with prefix: {path_prefix}")


def main():
    """Main training pipeline"""
    DATA_PATH = '/home/arnab/Desktop/TraffCal/newdatset/merged_unscaled.csv'
    MODEL_TYPE = 'xgboost'  # or 'random_forest'
    
    predictor = TreeBasedTrafficPredictor(
        model_type=MODEL_TYPE,
        input_len=12,
        output_len=12
    )
    
    X, y, feature_cols = predictor.load_data(DATA_PATH, target_feature='speed')
    
    train_data, val_data, test_data = predictor.prepare_data(X, y)
    
    predictor.build_and_train_models(train_data, val_data)
    
    metrics, y_pred = predictor.evaluate(test_data)
    
    feature_importance = predictor.get_feature_importance(feature_cols)
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    predictor.plot_results(
        test_data[1], 
        y_pred,
        feature_importance,
        save_path=f'{MODEL_TYPE}_results.png'
    )
    
    with open(f'{MODEL_TYPE}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    predictor.save_models(f'{MODEL_TYPE}_model')
    
    print(f"\n✅ {MODEL_TYPE.upper()} Training completed!")


if __name__ == '__main__':
    main()
