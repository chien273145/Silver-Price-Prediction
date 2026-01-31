"""
Training Script for Silver Price Prediction Model
Trains LSTM/GRU model on historical silver price data.
"""

import os
import sys
import argparse
import numpy as np
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import SilverDataProcessor
from src.model import SilverPriceLSTM, create_model


def train_model(
    data_path: str = None,
    model_save_path: str = None,
    scaler_save_path: str = None,
    sequence_length: int = 60,
    prediction_days: int = 7,
    epochs: int = 100,
    batch_size: int = 32,
    model_type: str = 'lstm',
    bidirectional: bool = False
) -> dict:
    """
    Train the silver price prediction model.
    
    Args:
        data_path: Path to the CSV data file
        model_save_path: Path to save the trained model
        scaler_save_path: Path to save the scaler
        sequence_length: Number of past days to use
        prediction_days: Number of future days to predict
        epochs: Maximum training epochs
        batch_size: Training batch size
        model_type: 'lstm' or 'gru'
        bidirectional: Use bidirectional layers
        
    Returns:
        Training results including metrics
    """
    # Set default paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if data_path is None:
        data_path = os.path.join(base_dir, 'dataset', 'silver_price.csv')
    
    if model_save_path is None:
        model_save_path = os.path.join(base_dir, 'models', 'silver_lstm_model.h5')
    
    if scaler_save_path is None:
        scaler_save_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    
    print("=" * 60)
    print("🥈 SILVER PRICE PREDICTION - MODEL TRAINING")
    print("=" * 60)
    print(f"\n📁 Data path: {data_path}")
    print(f"💾 Model will be saved to: {model_save_path}")
    print(f"📊 Sequence length: {sequence_length} days")
    print(f"🔮 Prediction horizon: {prediction_days} days")
    print(f"🧠 Model type: {model_type.upper()}")
    if bidirectional:
        print(f"   ↔️  Bidirectional: Yes")
    print()
    
    # Step 1: Load and process data
    print("=" * 60)
    print("STEP 1: DATA PROCESSING")
    print("=" * 60)
    
    processor = SilverDataProcessor(
        sequence_length=sequence_length,
        prediction_days=prediction_days
    )
    
    # Load data
    processor.load_data(data_path)
    
    # Clean data
    processor.clean_data()
    
    # Prepare data (simple - price only for better generalization)
    X, y = processor.prepare_simple_data()
    
    # Split data
    splits = processor.split_data(X, y, train_ratio=0.8, val_ratio=0.1)
    
    # Save scaler
    processor.save_scaler(scaler_save_path)
    
    # Step 2: Build model
    print("\n" + "=" * 60)
    print("STEP 2: MODEL BUILDING")
    print("=" * 60)
    
    model = SilverPriceLSTM(
        sequence_length=sequence_length,
        n_features=1,  # Using only price
        prediction_days=prediction_days,
        model_type=model_type
    )
    
    if bidirectional:
        model.build_bidirectional_model()
    else:
        model.build_model()
    
    # Step 3: Train model
    print("\n" + "=" * 60)
    print("STEP 3: MODEL TRAINING")
    print("=" * 60)
    
    history = model.train(
        X_train=splits['X_train'],
        y_train=splits['y_train'],
        X_val=splits['X_val'],
        y_val=splits['y_val'],
        epochs=epochs,
        batch_size=batch_size,
        patience=15,
        model_path=model_save_path
    )
    
    # Step 4: Evaluate model
    print("\n" + "=" * 60)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 60)
    
    metrics = model.evaluate(splits['X_test'], splits['y_test'])
    
    # Step 5: Save final model
    print("\n" + "=" * 60)
    print("STEP 5: SAVING MODEL")
    print("=" * 60)
    
    model.save(model_save_path)
    
    # Save training info
    training_info = {
        'training_date': datetime.now().isoformat(),
        'data_path': data_path,
        'sequence_length': sequence_length,
        'prediction_days': prediction_days,
        'model_type': model_type,
        'bidirectional': bidirectional,
        'epochs_trained': len(history['loss']),
        'final_train_loss': float(history['loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'test_metrics': metrics,
        'data_stats': {
            'total_samples': len(X),
            'train_samples': len(splits['X_train']),
            'val_samples': len(splits['X_val']),
            'test_samples': len(splits['X_test'])
        }
    }
    
    # Save training info
    info_path = os.path.join(os.path.dirname(model_save_path), 'training_info.json')
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"✓ Training info saved to {info_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Final Results:")
    print(f"   • Test RMSE: ${metrics['rmse']:.4f}")
    print(f"   • Test MAE:  ${metrics['mae']:.4f}")
    print(f"   • Test R²:   {metrics['r2']:.4f}")
    print(f"   • Test MAPE: {metrics['mape']:.2f}%")
    print(f"\n💾 Model saved to: {model_save_path}")
    print(f"📁 Scaler saved to: {scaler_save_path}")
    
    return training_info


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description='Train Silver Price Prediction Model')
    
    parser.add_argument('--data', type=str, default=None,
                        help='Path to CSV data file')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to save trained model')
    parser.add_argument('--scaler-path', type=str, default=None,
                        help='Path to save scaler')
    parser.add_argument('--sequence-length', type=int, default=60,
                        help='Number of past days to use for prediction')
    parser.add_argument('--prediction-days', type=int, default=7,
                        help='Number of future days to predict')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--model-type', type=str, default='lstm',
                        choices=['lstm', 'gru'],
                        help='Type of RNN layer')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Use bidirectional layers')
    parser.add_argument('--evaluate', action='store_true',
                        help='Only evaluate existing model')
    
    args = parser.parse_args()
    
    if args.evaluate:
        # Evaluation mode
        print("Evaluation mode - loading existing model...")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = args.model_path or os.path.join(base_dir, 'models', 'silver_lstm_model.h5')
        scaler_path = args.scaler_path or os.path.join(base_dir, 'models', 'scaler.pkl')
        data_path = args.data or os.path.join(base_dir, 'dataset', 'silver_price.csv')
        
        # Load processor and data
        processor = SilverDataProcessor(
            sequence_length=args.sequence_length,
            prediction_days=args.prediction_days
        )
        processor.load_data(data_path)
        processor.clean_data()
        X, y = processor.prepare_simple_data()
        splits = processor.split_data(X, y)
        
        # Load and evaluate model
        model = SilverPriceLSTM(
            sequence_length=args.sequence_length,
            n_features=1,
            prediction_days=args.prediction_days
        )
        model.load(model_path)
        metrics = model.evaluate(splits['X_test'], splits['y_test'])
        
    else:
        # Training mode
        train_model(
            data_path=args.data,
            model_save_path=args.model_path,
            scaler_save_path=args.scaler_path,
            sequence_length=args.sequence_length,
            prediction_days=args.prediction_days,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_type=args.model_type,
            bidirectional=args.bidirectional
        )


if __name__ == "__main__":
    main()
