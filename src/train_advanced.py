"""
Advanced Training Script for Silver Price Prediction
With improved LSTM architecture and feature engineering
"""

import os
import sys
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.advanced_processor import AdvancedSilverDataProcessor

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Input, 
    BatchNormalization, Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def build_advanced_model(
    input_shape: tuple,
    prediction_days: int = 7,
    units: tuple = (128, 64, 32),
    dropout_rate: float = 0.2,
    use_bidirectional: bool = True,
    learning_rate: float = 0.001
) -> Sequential:
    """
    Build an advanced LSTM model with optional bidirectional layers.
    """
    model = Sequential()
    
    # Input layer
    model.add(Input(shape=input_shape))
    
    # First LSTM layer
    if use_bidirectional:
        model.add(Bidirectional(LSTM(units[0], return_sequences=True), name='bilstm_1'))
    else:
        model.add(LSTM(units[0], return_sequences=True, name='lstm_1'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer
    if use_bidirectional:
        model.add(Bidirectional(LSTM(units[1], return_sequences=True), name='bilstm_2'))
    else:
        model.add(LSTM(units[1], return_sequences=True, name='lstm_2'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Third LSTM layer
    if use_bidirectional:
        model.add(Bidirectional(LSTM(units[2], return_sequences=False), name='bilstm_3'))
    else:
        model.add(LSTM(units[2], return_sequences=False, name='lstm_3'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Dense layers
    model.add(Dense(64, activation='relu', name='dense_1'))
    model.add(Dropout(dropout_rate / 2))
    model.add(Dense(32, activation='relu', name='dense_2'))
    model.add(Dense(prediction_days, name='output'))
    
    # Compile
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def train_advanced_model(
    data_path: str,
    model_save_path: str,
    scaler_save_path: str,
    sequence_length: int = 60,
    prediction_days: int = 7,
    epochs: int = 100,
    batch_size: int = 32,
    use_all_features: bool = True,
    use_bidirectional: bool = True
):
    """
    Train the advanced silver price prediction model.
    """
    print("\n" + "=" * 60)
    print("🥈 ADVANCED SILVER PRICE PREDICTION MODEL")
    print("=" * 60)
    print(f"📁 Data: {data_path}")
    print(f"🧠 Model: {'Bidirectional ' if use_bidirectional else ''}LSTM")
    print(f"📊 Features: {'All technical indicators' if use_all_features else 'Price only'}")
    print(f"⏱️ Sequence length: {sequence_length} days")
    print(f"🔮 Prediction horizon: {prediction_days} days")
    
    # Step 1: Load and process data
    print("\n" + "=" * 60)
    print("STEP 1: DATA PROCESSING")
    print("=" * 60)
    
    processor = AdvancedSilverDataProcessor(
        sequence_length=sequence_length,
        prediction_days=prediction_days
    )
    
    processor.load_investing_data(data_path)
    processor.add_technical_indicators()
    processor.clean_data()
    
    # Prepare features
    X, y = processor.prepare_features(use_all_features=use_all_features)
    
    # Split data
    splits = processor.split_data(X, y, train_ratio=0.8, val_ratio=0.1)
    
    # Step 2: Build model
    print("\n" + "=" * 60)
    print("STEP 2: MODEL BUILDING")
    print("=" * 60)
    
    input_shape = (sequence_length, X.shape[2])
    print(f"\n📐 Input shape: {input_shape}")
    
    model = build_advanced_model(
        input_shape=input_shape,
        prediction_days=prediction_days,
        units=(128, 64, 32),
        dropout_rate=0.2,
        use_bidirectional=use_bidirectional,
        learning_rate=0.001
    )
    
    print("\n✓ Model built successfully")
    model.summary()
    
    # Step 3: Training
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING")
    print("=" * 60)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {epochs}, Batch size: {batch_size}")
    print(f"   Training samples: {len(splits['X_train'])}")
    print(f"   Validation samples: {len(splits['X_val'])}")
    
    history = model.fit(
        splits['X_train'], splits['y_train'],
        validation_data=(splits['X_val'], splits['y_val']),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 4: Evaluation
    print("\n" + "=" * 60)
    print("STEP 4: EVALUATION")
    print("=" * 60)
    
    # Make predictions on test set
    y_pred = model.predict(splits['X_test'], verbose=0)
    y_true = splits['y_test']
    
    # Inverse transform to get actual prices
    y_pred_prices = []
    y_true_prices = []
    
    for i in range(len(y_pred)):
        pred_prices = processor.inverse_transform_price(y_pred[i])
        true_prices = processor.inverse_transform_price(y_true[i])
        y_pred_prices.extend(pred_prices)
        y_true_prices.extend(true_prices)
    
    y_pred_prices = np.array(y_pred_prices)
    y_true_prices = np.array(y_true_prices)
    
    # Calculate metrics
    mse = mean_squared_error(y_true_prices, y_pred_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_prices, y_pred_prices)
    r2 = r2_score(y_true_prices, y_pred_prices)
    mape = np.mean(np.abs((y_true_prices - y_pred_prices) / y_true_prices)) * 100
    
    print(f"\n📊 Test Results:")
    print(f"   • RMSE: ${rmse:.4f}")
    print(f"   • MAE:  ${mae:.4f}")
    print(f"   • R²:   {r2:.4f}")
    print(f"   • MAPE: {mape:.2f}%")
    
    # Step 5: Save model and scaler
    print("\n" + "=" * 60)
    print("STEP 5: SAVING")
    print("=" * 60)
    
    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"✓ Model saved to {model_save_path}")
    
    # Save scaler
    processor.save_scaler(scaler_save_path)
    
    # Export standard format data
    standard_data_path = os.path.join(os.path.dirname(data_path), 'silver_price.csv')
    processor.export_to_standard_format(standard_data_path)
    
    # Save training info
    info_path = os.path.join(os.path.dirname(model_save_path), 'training_info.json')
    training_info = {
        'timestamp': datetime.now().isoformat(),
        'data_path': data_path,
        'model_path': model_save_path,
        'sequence_length': sequence_length,
        'prediction_days': prediction_days,
        'epochs_trained': len(history.history['loss']),
        'features_used': len(processor.feature_columns),
        'feature_names': processor.feature_columns,
        'use_bidirectional': use_bidirectional,
        'use_all_features': use_all_features,
        'metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'mse': float(mse)
        },
        'data_info': {
            'total_records': len(processor.data),
            'date_range': {
                'start': processor.data['date'].min().strftime('%Y-%m-%d'),
                'end': processor.data['date'].max().strftime('%Y-%m-%d')
            },
            'price_range': {
                'min': float(processor.data['price'].min()),
                'max': float(processor.data['price'].max())
            }
        }
    }
    
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"✓ Training info saved to {info_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Final Results:")
    print(f"   • Test RMSE: ${rmse:.4f}")
    print(f"   • Test MAE:  ${mae:.4f}")
    print(f"   • Test R²:   {r2:.4f} {'⭐ Excellent!' if r2 > 0.9 else '✅ Good!' if r2 > 0.8 else ''}")
    print(f"   • Test MAPE: {mape:.2f}%")
    print(f"\n💾 Model saved to: {model_save_path}")
    print(f"📁 Scaler saved to: {scaler_save_path}")
    
    return model, processor, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Advanced Silver Price Model')
    parser.add_argument('--data', type=str, default=None, help='Path to Investing.com CSV')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--sequence', type=int, default=60, help='Sequence length')
    parser.add_argument('--simple', action='store_true', help='Use simple features (price only)')
    parser.add_argument('--no_bidirectional', action='store_true', help='Disable bidirectional LSTM')
    
    args = parser.parse_args()
    
    # Set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.data:
        data_path = args.data
    else:
        data_path = os.path.join(base_dir, 'dataset', 'dataset_silver.csv')
    
    model_path = os.path.join(base_dir, 'models', 'silver_lstm_model.h5')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    
    # Train
    train_advanced_model(
        data_path=data_path,
        model_save_path=model_path,
        scaler_save_path=scaler_path,
        sequence_length=args.sequence,
        prediction_days=7,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_all_features=not args.simple,
        use_bidirectional=not args.no_bidirectional
    )
