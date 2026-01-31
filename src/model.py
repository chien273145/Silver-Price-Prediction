"""
LSTM/GRU Model for Silver Price Prediction
Multi-layer LSTM architecture with dropout for 7-day price forecasting.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
from typing import Tuple, Optional


class SilverPriceLSTM:
    """
    LSTM-based model for silver price prediction.
    Predicts 7 days of future prices based on historical data.
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 n_features: int = 1,
                 prediction_days: int = 7,
                 model_type: str = 'lstm'):
        """
        Initialize the LSTM model.
        
        Args:
            sequence_length: Number of past days to use for prediction
            n_features: Number of input features
            prediction_days: Number of days to predict
            model_type: Type of RNN layer ('lstm' or 'gru')
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.prediction_days = prediction_days
        self.model_type = model_type.lower()
        self.model = None
        self.history = None
        
    def build_model(self, 
                    units: Tuple[int, ...] = (128, 64, 32),
                    dropout_rate: float = 0.2,
                    learning_rate: float = 0.001) -> Sequential:
        """
        Build the LSTM/GRU model architecture.
        
        Args:
            units: Tuple of units for each layer
            dropout_rate: Dropout rate between layers
            learning_rate: Learning rate for Adam optimizer
            
        Returns:
            Compiled Keras model
        """
        # Select RNN layer type
        RNNLayer = LSTM if self.model_type == 'lstm' else GRU
        
        self.model = Sequential([
            # Input layer
            Input(shape=(self.sequence_length, self.n_features)),
            
            # First RNN layer
            RNNLayer(units[0], return_sequences=True, name=f'{self.model_type}_1'),
            Dropout(dropout_rate, name='dropout_1'),
            
            # Second RNN layer
            RNNLayer(units[1], return_sequences=True, name=f'{self.model_type}_2'),
            Dropout(dropout_rate, name='dropout_2'),
            
            # Third RNN layer
            RNNLayer(units[2], return_sequences=False, name=f'{self.model_type}_3'),
            Dropout(dropout_rate, name='dropout_3'),
            
            # Dense layers
            Dense(32, activation='relu', name='dense_1'),
            Dense(16, activation='relu', name='dense_2'),
            
            # Output layer - predict 7 days
            Dense(self.prediction_days, name='output')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        print(f"✓ Built {self.model_type.upper()} model")
        self.model.summary()
        
        return self.model
    
    def build_bidirectional_model(self,
                                   units: Tuple[int, ...] = (128, 64, 32),
                                   dropout_rate: float = 0.2,
                                   learning_rate: float = 0.001) -> Sequential:
        """
        Build a bidirectional LSTM/GRU model for better sequence understanding.
        
        Args:
            units: Tuple of units for each layer
            dropout_rate: Dropout rate between layers
            learning_rate: Learning rate for Adam optimizer
            
        Returns:
            Compiled Keras model
        """
        RNNLayer = LSTM if self.model_type == 'lstm' else GRU
        
        self.model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),
            
            # Bidirectional layers
            Bidirectional(RNNLayer(units[0], return_sequences=True), name='bi_1'),
            Dropout(dropout_rate),
            
            Bidirectional(RNNLayer(units[1], return_sequences=True), name='bi_2'),
            Dropout(dropout_rate),
            
            RNNLayer(units[2], return_sequences=False, name=f'{self.model_type}_3'),
            Dropout(dropout_rate),
            
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.prediction_days)
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print(f"✓ Built Bidirectional {self.model_type.upper()} model")
        self.model.summary()
        
        return self.model
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              patience: int = 15,
              model_path: Optional[str] = None) -> dict:
        """
        Train the model with early stopping and model checkpointing.
        
        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data
            y_val: Validation target data
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            patience: Patience for early stopping
            model_path: Path to save the best model
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Add model checkpoint if path provided
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        print(f"\n🚀 Starting training...")
        print(f"   Epochs: {epochs}, Batch size: {batch_size}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}\n")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n✓ Training complete!")
        print(f"  Final training loss: {self.history.history['loss'][-1]:.6f}")
        print(f"  Final validation loss: {self.history.history['val_loss'][-1]:.6f}")
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input data for prediction
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not available. Train or load a model first.")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test input data
            y_test: Test target data
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not available.")
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        
        # R² score
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }
        
        print(f"\n📊 Model Evaluation:")
        print(f"   MSE:  {mse:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   MAE:  {mae:.6f}")
        print(f"   R²:   {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        
        return metrics
    
    def save(self, filepath: str):
        """Save the model to a file."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a model from a file."""
        self.model = load_model(filepath)
        print(f"✓ Model loaded from {filepath}")
        return self.model
    
    def get_model_summary(self) -> str:
        """Get model summary as string."""
        if self.model is None:
            return "No model built."
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)


def create_model(sequence_length: int = 60,
                 n_features: int = 1,
                 prediction_days: int = 7,
                 model_type: str = 'lstm',
                 bidirectional: bool = False) -> SilverPriceLSTM:
    """
    Factory function to create and build a model.
    
    Args:
        sequence_length: Number of past days for input
        n_features: Number of input features
        prediction_days: Number of days to predict
        model_type: 'lstm' or 'gru'
        bidirectional: Whether to use bidirectional layers
        
    Returns:
        Built SilverPriceLSTM model
    """
    model = SilverPriceLSTM(
        sequence_length=sequence_length,
        n_features=n_features,
        prediction_days=prediction_days,
        model_type=model_type
    )
    
    if bidirectional:
        model.build_bidirectional_model()
    else:
        model.build_model()
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model(
        sequence_length=60,
        n_features=1,
        prediction_days=7,
        model_type='lstm',
        bidirectional=False
    )
