"""
FastAPI Backend for Silver Price Prediction
Provides REST API endpoints for predictions, historical data, and real-time updates.
Uses Ridge Regression model (primary) or LSTM (fallback).
"""

import os
import sys
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.unified_predictor import UnifiedPredictor
from backend.realtime_data import RealTimeDataFetcher


# Global instances
predictor: Optional[UnifiedPredictor] = None
data_fetcher: Optional[RealTimeDataFetcher] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global predictor, data_fetcher
    
    print("🚀 Starting Silver Price Prediction API...")
    
    # Initialize predictor with Ridge Regression (best performing model)
    # Ridge is much lighter than LSTM - no TensorFlow required!
    try:
        predictor = UnifiedPredictor(model_type='ridge')
        predictor.load()
        print("✓ Ridge Regression model loaded (R²=0.96, MAPE=3.37%)")
    except Exception as e:
        print(f"⚠️ Could not load Ridge model: {e}")
        print("   Please train the model first: python src/train_ridge.py")
        predictor = None
    
    # Initialize data fetcher
    data_fetcher = RealTimeDataFetcher(cache_duration_minutes=5)
    print("✓ Real-time data fetcher initialized")
    
    yield
    
    # Cleanup
    print("👋 Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Silver Price Prediction API",
    description="API dự đoán giá bạc 7 ngày sử dụng Ridge Regression với độ chính xác R²=0.96",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class PredictionResponse(BaseModel):
    """Response model for predictions."""
    success: bool
    timestamp: str
    currency: str
    unit: str
    exchange_rate: Optional[float]
    last_known: dict
    predictions: list
    summary: dict


class HistoricalResponse(BaseModel):
    """Response model for historical data."""
    success: bool
    currency: str
    unit: str
    exchange_rate: Optional[float]
    count: int
    data: list


class RealTimeResponse(BaseModel):
    """Response model for real-time data."""
    success: bool
    timestamp: str
    silver_price: dict
    exchange_rate: dict
    price_vnd: Optional[float]


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    success: bool
    model_info: dict


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint - serve frontend or API info."""
    frontend_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'frontend', 'index.html'
    )
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    
    return {
        "message": "Silver Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/api/predict": "Dự đoán giá 7 ngày tới",
            "/api/historical": "Dữ liệu lịch sử",
            "/api/realtime": "Giá thời gian thực",
            "/api/metrics": "Thông tin mô hình AI",
            "/docs": "API Documentation"
        }
    }


@app.get("/api/predict", response_model=PredictionResponse)
async def predict(
    currency: str = Query("VND", description="Currency: VND or USD"),
    exchange_rate: Optional[float] = Query(None, description="Custom USD/VND rate")
):
    """
    Dự đoán giá bạc 7 ngày tới.
    
    - **currency**: VND hoặc USD
    - **exchange_rate**: Tỷ giá USD/VND tùy chỉnh (mặc định lấy từ API)
    """
    global predictor, data_fetcher
    
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using: python src/train.py"
        )
    
    try:
        # Update exchange rate if needed
        if currency.upper() == "VND":
            if exchange_rate:
                predictor.set_exchange_rate(exchange_rate)
            else:
                # Get live exchange rate
                rate_data = data_fetcher.get_usd_vnd_rate()
                if rate_data['rate']:
                    predictor.set_exchange_rate(rate_data['rate'])
        
        # Make prediction
        result = predictor.predict(in_vnd=(currency.upper() == "VND"))
        
        return PredictionResponse(
            success=True,
            timestamp=result['timestamp'],
            currency=result['currency'],
            unit=result['unit'],
            exchange_rate=result['exchange_rate'],
            last_known=result['last_known'],
            predictions=result['predictions'],
            summary=result['summary']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/historical", response_model=HistoricalResponse)
async def get_historical(
    days: int = Query(30, ge=1, le=365, description="Number of days"),
    currency: str = Query("VND", description="Currency: VND or USD")
):
    """
    Lấy dữ liệu giá lịch sử.
    
    - **days**: Số ngày lịch sử (1-365)
    - **currency**: VND hoặc USD
    """
    global predictor, data_fetcher
    
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Predictor not initialized"
        )
    
    try:
        # Update exchange rate for VND
        if currency.upper() == "VND":
            rate_data = data_fetcher.get_usd_vnd_rate()
            if rate_data['rate']:
                predictor.set_exchange_rate(rate_data['rate'])
        
        result = predictor.get_historical_data(
            days=days,
            in_vnd=(currency.upper() == "VND")
        )
        
        return HistoricalResponse(
            success=True,
            currency=result['currency'],
            unit=result['unit'],
            exchange_rate=result['exchange_rate'],
            count=len(result['data']),
            data=result['data']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/realtime", response_model=RealTimeResponse)
async def get_realtime():
    """
    Lấy giá bạc và tỷ giá thời gian thực.
    """
    global data_fetcher, predictor
    
    try:
        # Get silver price
        silver = data_fetcher.get_silver_price()
        
        # Get exchange rate
        rate_data = data_fetcher.get_usd_vnd_rate()
        
        # Calculate VND price
        price_vnd = None
        if silver['price'] and rate_data['rate']:
            # Convert to VND per lượng
            troy_ounce_to_luong = 1.20565
            price_vnd = silver['price'] * troy_ounce_to_luong * rate_data['rate']
        
        return RealTimeResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            silver_price=silver,
            exchange_rate=rate_data,
            price_vnd=price_vnd
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Lấy thông tin về mô hình AI đã train.
    """
    global predictor
    
    if predictor is None:
        return ModelInfoResponse(
            success=False,
            model_info={"error": "Model not loaded"}
        )
    
    try:
        info = predictor.get_model_info()
        return ModelInfoResponse(
            success=True,
            model_info=info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/update")
async def update_data():
    """
    Cập nhật dữ liệu giá mới nhất từ API vào file CSV.
    """
    global data_fetcher
    
    try:
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'dataset', 'silver_price.csv'
        )
        
        success = data_fetcher.update_csv_with_latest(csv_path)
        
        return {
            "success": success,
            "message": "Data updated successfully" if success else "No new data available",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    global predictor, data_fetcher
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "model_loaded": predictor is not None,
            "data_fetcher": data_fetcher is not None
        }
    }


# Mount static files for frontend
frontend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'frontend'
)
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🥈 SILVER PRICE PREDICTION API")
    print("=" * 60)
    print("\n📍 Starting server at http://localhost:8000")
    print("📚 API docs at http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
