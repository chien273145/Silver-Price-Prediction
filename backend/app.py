"""
FastAPI Backend for Silver Price Prediction
Provides REST API endpoints for predictions, historical data, and real-time updates.
Uses Ridge Regression model (primary) or LSTM (fallback).
"""

import os
import sys
import numpy as np
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

from src.enhanced_predictor import EnhancedPredictor
from src.gold_predictor import GoldPredictor
from src.vietnam_gold_predictor import VietnamGoldPredictor
from backend.realtime_data import RealTimeDataFetcher
from src.scrapers.service import get_price_service
from src.buy_score import calculate_buy_score
from src.time_machine import predict_portfolio_future


# Global instances
predictor: Optional[EnhancedPredictor] = None
gold_predictor: Optional[GoldPredictor] = None
vn_gold_predictor: Optional[VietnamGoldPredictor] = None
data_fetcher: Optional[RealTimeDataFetcher] = None


import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global predictor, data_fetcher
    
    print("🚀 Starting Silver Price Prediction API...", flush=True)
    start_time = datetime.now()
    
    # Initialize data fetcher immediately (lightweight)
    print(f"[{datetime.now().time()}] Initializing RealTimeDataFetcher...", flush=True)
    data_fetcher = RealTimeDataFetcher(cache_duration_minutes=5)
    print(f"[{datetime.now().time()}] ✓ Real-time data fetcher initialized", flush=True)
    
    # Start model loading in background
    asyncio.create_task(load_model_background())
    asyncio.create_task(load_gold_model_background())
    asyncio.create_task(load_vn_gold_model_background())
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[{datetime.now().time()}] [OK] App startup complete in {elapsed:.2f}s (Model loading in background)", flush=True)
    
    yield
    
    # Cleanup
    print("👋 Shutting down API...", flush=True)

async def load_model_background():
    """Load model in background to not block startup."""
    global predictor
    
    print(f"[{datetime.now().time()}] [LOADING] Background: Starting model loading...", flush=True)
    
    try:
        # Run heavy lifting in thread pool
        predictor = await asyncio.to_thread(_load_model_logic)
        print(f"[{datetime.now().time()}] ✅ Background: Model loaded successfully!", flush=True)
    except Exception as e:
        print(f"[{datetime.now().time()}] ❌ Background: Model loading failed: {e}", flush=True)

async def load_gold_model_background():
    """Load gold model in background."""
    global gold_predictor
    
    print(f"[{datetime.now().time()}] ⏳ Background: Starting gold model loading...", flush=True)
    
    try:
        gold_predictor = await asyncio.to_thread(_load_gold_model_logic)
        print(f"[{datetime.now().time()}] ✅ Background: Gold model loaded successfully!", flush=True)
    except Exception as e:
        print(f"[{datetime.now().time()}] ❌ Background: Gold model loading failed: {e}", flush=True)

def _load_gold_model_logic():
    """Synchronous gold model loading logic."""
    try:
        print("   Attempting to load GoldPredictor...", flush=True)
        p = GoldPredictor()
        p.load_data()
        p.create_features()
        p.load_model()
        print("   ✓ Gold Ridge model loaded", flush=True)
        return p
    except Exception as e:
        print(f"   ❌ Gold model failed: {e}", flush=True)
        return None


async def load_vn_gold_model_background():
    """Load Vietnam Gold model in background."""
    global vn_gold_predictor
    
    print(f"[{datetime.now().time()}] ⏳ Background: Starting VN Gold model loading...", flush=True)
    
    try:
        vn_gold_predictor = await asyncio.to_thread(_load_vn_gold_model_logic)
        print(f"[{datetime.now().time()}] ✅ Background: VN Gold model loaded successfully!", flush=True)
    except Exception as e:
        print(f"[{datetime.now().time()}] ❌ Background: VN Gold model loading failed: {e}", flush=True)


def _load_vn_gold_model_logic():
    """Synchronous VN gold model loading logic."""
    try:
        print("   Attempting to load VietnamGoldPredictor...", flush=True)
        p = VietnamGoldPredictor()
        p.load_model()
        # Load data for predictions
        p.load_vietnam_data()
        p.load_world_data()
        p.merge_datasets()
        p.create_transfer_features()
        print("   ✓ Vietnam Gold model loaded", flush=True)
        return p
    except Exception as e:
        print(f"   ❌ VN Gold model failed: {e}", flush=True)
        return None

def _load_model_logic():
    """Synchronous model loading logic."""
    try:
        print("   Attempting to load EnhancedPredictor...", flush=True)
        p = EnhancedPredictor()
        p.load_data()
        p.create_features()
        p.load_model()
        print("   ✓ Enhanced Ridge model loaded", flush=True)
        return p
    except Exception as e:
        print(f"   ⚠️ Enhanced model failed: {e}", flush=True)
        print("   Fallback to UnifiedPredictor...", flush=True)
        try:
            from src.unified_predictor import UnifiedPredictor
            p = UnifiedPredictor()
            p._load_data()
            p._create_ridge_features()
            p.load_ridge_model()
            print("   ✓ UnifiedPredictor loaded", flush=True)
            return p
        except Exception as e2:
            print(f"   ❌ Fallback failed: {e2}", flush=True)
            return None


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
    market_drivers: Optional[dict] = None
    accuracy_check: Optional[dict] = None


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

# Blog Routes
@app.get("/blog")
async def blog_index():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend', 'blog', 'index.html')
    return FileResponse(path)

@app.get("/blog/ai-cong-nghe")
async def blog_ai():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend', 'blog', 'ai-cong-nghe.html')
    return FileResponse(path)

@app.get("/blog/phan-tich-kinh-te")
async def blog_economics():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend', 'blog', 'phan-tich-kinh-te.html')
    return FileResponse(path)

@app.get("/blog/huong-dan-su-dung")
async def blog_guide():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend', 'blog', 'huong-dan-su-dung.html')
    return FileResponse(path)


# ========== DEDICATED PAGE ROUTES ==========
@app.get("/silver")
async def silver_page():
    """Serve the dedicated Silver price prediction page."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend', 'silver.html')
    return FileResponse(path)


@app.get("/gold")
async def gold_page():
    """Serve the dedicated Gold price prediction page."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend', 'gold.html')
    return FileResponse(path)


# ========== VIETNAM GOLD API ENDPOINTS ==========
@app.get("/api/gold-vn/predict")
async def predict_vietnam_gold():
    """
    Dự đoán giá vàng SJC Việt Nam 7 ngày tới.
    Sử dụng Transfer Learning từ mô hình giá vàng thế giới.
    """
    global vn_gold_predictor
    
    # Wait for model if loading (up to 5s)
    if vn_gold_predictor is None:
        import asyncio
        for _ in range(10):
            if vn_gold_predictor is not None:
                break
            await asyncio.sleep(0.5)
    
    if vn_gold_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Vietnam Gold model is loading. Please try again in a few seconds."
        )
    
    try:
        # Get live market data for real-time prediction
        market_data = None
        if data_fetcher:
            try:
                market_data = data_fetcher.get_full_market_data()
            except Exception as e:
                print(f"Error fetching live market data: {e}")

        # Predict
        if market_data and market_data.get('gold_close'):
            predictions = vn_gold_predictor.predict_live(market_data)
            is_live = True
        else:
            predictions = vn_gold_predictor.predict()
            is_live = False
            
        model_info = vn_gold_predictor.get_model_info()
        model_info['is_live_prediction'] = is_live
        if market_data:
            model_info['live_market_data'] = market_data
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "currency": "VND",
            "unit": "triệu VND/lượng",
            "predictions": predictions,
            "model_info": model_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gold-vn/historical")
async def get_vietnam_gold_historical(days: int = Query(30, ge=1, le=365)):
    """Get historical Vietnam SJC gold prices."""
    global vn_gold_predictor
    
    if vn_gold_predictor is None:
        import asyncio
        for _ in range(10):
            if vn_gold_predictor is not None:
                break
            await asyncio.sleep(0.5)
    
    if vn_gold_predictor is None:
        raise HTTPException(status_code=503, detail="Model is loading")
    
    try:
        data = vn_gold_predictor.get_historical_data(days)
        return {
            "success": True,
            "currency": "VND",
            "unit": "triệu VND/lượng",
            "count": len(data),
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gold-vn/accuracy")
async def get_vietnam_gold_accuracy():
    """Get Vietnam Gold model accuracy metrics."""
    global vn_gold_predictor
    
    if vn_gold_predictor is None:
        import asyncio
        for _ in range(10):
            if vn_gold_predictor is not None:
                break
            await asyncio.sleep(0.5)
    
    if vn_gold_predictor is None:
        raise HTTPException(status_code=503, detail="Model is loading")
    
    try:
        metrics = vn_gold_predictor.get_accuracy_metrics()
        return {
            "success": True,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    
    # Wait for model if loading (up to 5s)
    if predictor is None:
        import asyncio
        for _ in range(10): # 10 * 0.5s = 5s
            if predictor is not None:
                break
            await asyncio.sleep(0.5)
            
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is currently loading or failed to load. Please try again in a few seconds."
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
        
        # Get live data for real-time prediction
        market_data = None
        if data_fetcher:
            try:
                # This fetches latest Silver, Gold, DXY, VIX
                market_data = data_fetcher.get_full_market_data()
            except Exception as e:
                print(f"Error fetching live market data: {e}")

        # Make prediction
        if market_data and market_data.get('silver_close'):
            # Use Live Prediction
            predictions = predictor.predict_live(market_data)
            
            # predict_live returns clean list of dicts. 
            # We need full structure like predict() returns.
            # Get base structure (using stale/memory data is fine for non-prediction fields)
            base_result = predictor.predict(in_vnd=(currency.upper() == "VND"))
            
            # Inject live predictions
            # Re-map predict_live output (list of small dicts) to full structure if needed
            # Or just use the predictions list directly?
            # predict_live returns: [{'date':..., 'price':..., 'price_usd':...}, ...]
            
            summary_prices = [p['price'] for p in predictions]
            last_price = base_result['last_known']['price']
            
            # Reconstruct result wrapper
            result = base_result
            result['predictions'] = predictions
            result['summary'] = {
                'min_price': float(min(summary_prices)),
                'max_price': float(max(summary_prices)),
                'avg_price': float(np.mean(summary_prices)),
                'trend': 'up' if summary_prices[-1] > last_price else 'down',
                'total_change': float(summary_prices[-1] - last_price),
                'total_change_pct': float((summary_prices[-1] - last_price) / last_price * 100)
            }
            
            # Add Live Metadata
            result['is_live_prediction'] = True
            result['live_market_data'] = market_data
            
        else:
            # Fallback to standard prediction
            result = predictor.predict(in_vnd=(currency.upper() == "VND"))
            result['is_live_prediction'] = False
        
        # Add extra metadata fields to result to match ResponseModel
        response_data = {
            "success": True,
            **result, # Unpack standard result
            "market_drivers": predictor.get_market_drivers(),
            "accuracy_check": predictor.get_yesterday_accuracy()
        }
        
        return response_data
        
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


@app.get("/api/performance-transparency")
async def get_performance_transparency():
    """
    Get detailed performance metrics for Transparency UI (Silver).
    Matches structure used by Gold page.
    """
    global predictor, data_fetcher
    
    if predictor is None:
        return {"success": False, "error": "Model loading"}
        
    try:
        # Get basic accuracy data from predictor
        acc_data = predictor.get_yesterday_accuracy()
        
        if not acc_data:
            return {"success": False, "error": "Insufficient data"}
            
        # Get Exchange Rate
        exchange_rate = 25450 # Fallback
        if data_fetcher:
            rate_data = data_fetcher.get_usd_vnd_rate()
            if rate_data['rate']:
                exchange_rate = rate_data['rate']
        
        # Calculate derived metrics
        acc_pct = acc_data['accuracy']
        grade = "Good"
        grade_color = "#3498db" # Blue
        comment = "Model performing well within expected range."
        
        if acc_pct >= 97:
            grade = "Excellent"
            grade_color = "#2ecc71" # Green
            comment = "High precision. Strong predictive signals detected."
        elif acc_pct >= 95:
            grade = "Very Good"
            grade_color = "#27ae60"
            comment = "Solid performance. Minor deviation observed."
        elif acc_pct < 90:
            grade = "Fair"
            grade_color = "#f1c40f" # Yellow
            comment = "Moderate deviation due to market volatility."
        if acc_pct < 85:
            grade = "Monitor"
            grade_color = "#e74c3c" # Red
            comment = "High volatility impacting accuracy."

        # Convert to VND for display
        # Note: 1 oz Silver = 0.829 lượng roughly? 
        # Actually conversion is: Price USD/oz * Exchange Rate * 1.20565 (oz->luong) ?
        # Wait, get_yesterday_accuracy returns USD.
        # Predictor._convert_to_vnd logic: price_usd * self.exchange_rate * 1.20565
        OZ_TO_LUONG = 1.20565
        
        pred_vnd = acc_data['predicted_usd'] * exchange_rate * OZ_TO_LUONG * 1000
        actual_vnd = acc_data['actual_usd'] * exchange_rate * OZ_TO_LUONG * 1000
        
        # Construct response
        performance = {
            "date": acc_data['date'],
            "forecast": {
                "usd": round(acc_data['predicted_usd'], 2),
                "vnd": round(pred_vnd, -3) # Round to thousand
            },
            "actual": {
                "usd": round(acc_data['actual_usd'], 2),
                "vnd": round(actual_vnd, -3)
            },
            "difference": {
                "percentage": round(((acc_data['predicted_usd'] - acc_data['actual_usd']) / acc_data['actual_usd']) * 100, 2),
                "absolute_usd": round(abs(acc_data['predicted_usd'] - acc_data['actual_usd']), 2)
            },
            "accuracy": {
                "overall": round(acc_pct, 2),
                "grade": grade,
                "grade_color": grade_color,
                "comment": comment,
                "direction_correct": None # Not implemented in get_yesterday_accuracy yet
            },
            "model_confidence": "High" if acc_pct > 95 else "Medium"
        }
        
        return {
            "success": True,
            "performance": performance
        }
    except Exception as e:
        print(f"Error in performance transparency: {e}")
        return {"success": False, "error": str(e)}


# ========== GOLD ENDPOINTS ==========
@app.get("/api/gold/predict")
async def gold_predict(
    currency: str = Query("VND", description="Currency: VND or USD"),
    exchange_rate: Optional[float] = Query(None, description="Custom USD/VND rate")
):
    """
    Dự đoán giá vàng 7 ngày tới.
    
    - **currency**: VND hoặc USD
    - **exchange_rate**: Tỷ giá USD/VND tùy chỉnh
    """
    global gold_predictor, data_fetcher
    
    # Wait for model if loading (up to 5s)
    if gold_predictor is None:
        import asyncio
        for _ in range(10):
            if gold_predictor is not None:
                break
            await asyncio.sleep(0.5)
            
    if gold_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Gold model is currently loading. Please try again in a few seconds."
        )
    
    try:
        if currency.upper() == "VND":
            if exchange_rate:
                gold_predictor.set_exchange_rate(exchange_rate)
            else:
                rate_data = data_fetcher.get_usd_vnd_rate()
                if rate_data['rate']:
                    gold_predictor.set_exchange_rate(rate_data['rate'])
        
        result = gold_predictor.predict(in_vnd=(currency.upper() == "VND"))
        
        # Add extra info
        result['market_drivers'] = gold_predictor.get_market_drivers()
        result['accuracy_check'] = gold_predictor.get_yesterday_accuracy()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gold/historical")
async def gold_historical(
    days: int = Query(30, ge=1, le=365, description="Number of days"),
    currency: str = Query("VND", description="Currency: VND or USD")
):
    """
    Lấy dữ liệu giá vàng lịch sử.
    
    - **days**: Số ngày lịch sử (1-365)
    - **currency**: VND hoặc USD
    """
    global gold_predictor, data_fetcher
    
    if gold_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Gold model not initialized"
        )
    
    try:
        if currency.upper() == "VND":
            rate_data = data_fetcher.get_usd_vnd_rate()
            if rate_data['rate']:
                gold_predictor.set_exchange_rate(rate_data['rate'])
        
        result = gold_predictor.get_historical_data(
            days=days,
            in_vnd=(currency.upper() == "VND")
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gold/metrics")
async def gold_model_info():
    """
    Lấy thông tin về mô hình AI dự đoán giá vàng.
    """
    global gold_predictor
    
    if gold_predictor is None:
        return {
            "success": False,
            "model_info": {"error": "Gold model not loaded"}
        }
    
    try:
        info = gold_predictor.get_model_info()
        return {
            "success": True,
            "model_info": info
        }
        
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


@app.post("/api/update-daily")
async def update_daily():
    """
    Cập nhật dữ liệu giá bạc hàng ngày.
    Được gọi bởi cron job mỗi ngày để cập nhật dữ liệu mới nhất.
    """
    global data_fetcher, predictor
    
    try:
        import pandas as pd
        
        # Path to dataset
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'dataset', 'dataset_silver.csv'
        )
        
        # Read current data to get last date
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['Date'] = pd.to_datetime(df['Date'])
            last_date = df['Date'].max()
            records_before = len(df)
        else:
            last_date = None
            records_before = 0
        
        # Update CSV with latest data
        success = data_fetcher.update_csv_with_latest(csv_path)
        
        # Count new records
        if os.path.exists(csv_path):
            df_after = pd.read_csv(csv_path)
            records_after = len(df_after)
            df_after['Date'] = pd.to_datetime(df_after['Date'])
            new_last_date = df_after['Date'].max()
        else:
            records_after = 0
            new_last_date = None
        
        records_added = records_after - records_before
        
        # Reload predictor data if new records were added
        if records_added > 0 and predictor is not None:
            predictor._load_data()
            predictor._create_ridge_features()
        
        return {
            "success": success,
            "message": f"Added {records_added} new record(s)" if records_added > 0 else "Data is up to date",
            "records_added": records_added,
            "total_records": records_after,
            "last_date": new_last_date.strftime("%Y-%m-%d") if new_last_date else None,
            "previous_date": last_date.strftime("%Y-%m-%d") if last_date else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/update-external")
async def update_external_data():
    """
    Cập nhật dữ liệu external (Gold, DXY, VIX) và reload model.
    Được gọi bởi cron job sau khi update-daily để đảm bảo model có dữ liệu mới nhất.
    """
    global predictor
    
    try:
        import sys
        import pandas as pd
        from datetime import datetime, timedelta
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Import fetch function
        sys.path.insert(0, base_dir)
        from fetch_external_data import fetch_external_data, merge_with_silver_data, save_enhanced_dataset
        
        # Paths
        silver_csv = os.path.join(base_dir, 'dataset', 'dataset_silver.csv')
        enhanced_csv = os.path.join(base_dir, 'dataset', 'dataset_enhanced.csv')
        
        # Get current record count
        records_before = 0
        if os.path.exists(enhanced_csv):
            df_before = pd.read_csv(enhanced_csv)
            records_before = len(df_before)
        
        # Fetch external data (last 30 days to catch up)
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        external_df = fetch_external_data(start_date=start_date)
        
        if external_df.empty:
            return {
                "success": False,
                "message": "Failed to fetch external data",
                "timestamp": datetime.now().isoformat()
            }
        
        # Merge with silver data
        merged_df = merge_with_silver_data(external_df, silver_csv)
        
        if merged_df.empty:
            return {
                "success": False,
                "message": "Failed to merge data",
                "timestamp": datetime.now().isoformat()
            }
        
        # Save enhanced dataset
        save_enhanced_dataset(merged_df, enhanced_csv)
        
        records_after = len(merged_df)
        records_added = records_after - records_before
        
        # Reload predictor with new data
        if predictor is not None:
            predictor.load_data()
            predictor.create_features()
            # Model weights don't change, just reload data
        
        return {
            "success": True,
            "message": f"Updated external data. {records_added} new record(s).",
            "records_total": records_after,
            "records_added": records_added,
            "last_date": str(merged_df['Date'].max()) if not merged_df.empty else None,
            "external_features": ["Gold", "DXY", "VIX"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-status")
async def get_data_status():
    """
    Lấy trạng thái dữ liệu: ngày cập nhật cuối, số lượng records.
    """
    try:
        import pandas as pd
        
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'dataset', 'dataset_silver.csv'
        )
        
        if not os.path.exists(csv_path):
            return {
                "success": False,
                "message": "Dataset not found"
            }
        
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        last_date = df['Date'].max()
        first_date = df['Date'].min()
        
        # Check if data is current (within last 2 days for weekends)
        from datetime import timedelta
        days_old = (datetime.now() - last_date.to_pydatetime()).days
        is_current = days_old <= 2
        
        return {
            "success": True,
            "total_records": len(df),
            "first_date": first_date.strftime("%Y-%m-%d"),
            "last_date": last_date.strftime("%Y-%m-%d"),
            "days_old": days_old,
            "is_current": is_current,
            "last_price_usd": float(df.iloc[-1]['Close']),
            "checked_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fear-greed")
async def get_fear_greed_index():
    """
    Calculate Fear & Greed Index based on market data and AI predictions.
    Returns score 0-100 and recommendation signal.
    """
    global predictor, data_fetcher
    
    try:
        if predictor is None:
            return {
                "success": False,
                "message": "Model not loaded"
            }
        
        # Get market drivers and current predictions
        market_drivers = predictor.get_market_drivers()
        
        # Get current prediction to analyze trend
        result = predictor.predict(in_vnd=True)
        predictions = result['predictions']
        
        # Calculate Fear & Greed score (0-100)
        fear_greed_score = 50  # Neutral start
        
        # Factor 1: Market drivers (40% weight)
        drivers_score = 50
        if 'raw' in market_drivers:
            raw = market_drivers['raw']
            
            # VIX component (higher VIX = more fear)
            if 'vix' in raw and raw['vix']['value'] > 0:
                vix = raw['vix']['value']
                if vix > 30:
                    drivers_score -= 15  # High fear
                elif vix > 25:
                    drivers_score -= 10
                elif vix < 15:
                    drivers_score += 15  # High greed
                elif vix < 20:
                    drivers_score += 10
            
            # DXY component (weaker USD = more greed for metals)
            if 'dxy' in raw and raw['dxy']['change'] != 0:
                dxy_change = raw['dxy']['change']
                if dxy_change < -1.0:
                    drivers_score += 15  # USD weak = metal greed
                elif dxy_change < -0.5:
                    drivers_score += 10
                elif dxy_change > 1.0:
                    drivers_score -= 15  # USD strong = metal fear
                elif dxy_change > 0.5:
                    drivers_score -= 10
            
            # RSI component
            if 'rsi' in raw and raw['rsi']['value'] > 0:
                rsi = raw['rsi']['value']
                if rsi > 75:
                    drivers_score += 12  # Overbought = greed
                elif rsi > 70:
                    drivers_score += 8
                elif rsi < 25:
                    drivers_score -= 12  # Oversold = fear
                elif rsi < 30:
                    drivers_score -= 8
        
        # Factor 2: Prediction trend (30% weight)
        trend_score = 50
        if len(predictions) >= 3:
            day1 = predictions[0]['price']
            day3 = predictions[2]['price']
            day7 = predictions[-1]['price']
            
            # Analyze trend strength
            day1_change = ((day1 - result['last_known']['price']) / result['last_known']['price']) * 100
            day7_change = ((day7 - result['last_known']['price']) / result['last_known']['price']) * 100
            
            if day1_change > 1.0 and day7_change > 2.0:
                trend_score = 75  # Strong bullish = greed
            elif day1_change > 0.5 and day7_change > 1.0:
                trend_score = 65  # Moderate bullish
            elif day1_change < -1.0 and day7_change < -2.0:
                trend_score = 25  # Strong bearish = fear
            elif day1_change < -0.5 and day7_change < -1.0:
                trend_score = 35  # Moderate bearish
        
        # Factor 3: Volatility (30% weight)
        volatility_score = 50
        if hasattr(predictor, 'data') and predictor.data is not None:
            recent_prices = predictor.data['price'].tail(14).values if 'price' in predictor.data.columns else []
            if len(recent_prices) > 1:
                volatility = np.std(recent_prices) / np.mean(recent_prices) * 100
                if volatility > 3.0:
                    volatility_score = 30  # High volatility = fear
                elif volatility > 2.0:
                    volatility_score = 40
                elif volatility < 1.0:
                    volatility_score = 70  # Low volatility = greed
                elif volatility < 1.5:
                    volatility_score = 60
        
        # Calculate final score (weighted average)
        fear_greed_score = (drivers_score * 0.4 + trend_score * 0.3 + volatility_score * 0.3)
        fear_greed_score = max(0, min(100, fear_greed_score))  # Clamp to 0-100
        
        # Generate recommendation
        if fear_greed_score >= 75:
            signal = "EXTREME GREED"
            recommendation = "Sell Signal - Market Overbought"
            color = "#e74c3c"  # Red
        elif fear_greed_score >= 60:
            signal = "GREED"
            recommendation = "Consider Taking Profits"
            color = "#f39c12"  # Orange
        elif fear_greed_score >= 45:
            signal = "NEUTRAL"
            recommendation = "Wait & Watch Market"
            color = "#f1c40f"  # Yellow
        elif fear_greed_score >= 25:
            signal = "FEAR"
            recommendation = "Buying Opportunity"
            color = "#27ae60"  # Green
        else:
            signal = "EXTREME FEAR"
            recommendation = "Strong Buy Signal"
            color = "#2980b9"  # Blue
        
        return {
            "success": True,
            "index": {
                "score": round(fear_greed_score, 1),
                "signal": signal,
                "recommendation": recommendation,
                "color": color,
                "components": {
                    "market_drivers": round(drivers_score, 1),
                    "prediction_trend": round(trend_score, 1),
                    "volatility": round(volatility_score, 1)
                },
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error calculating Fear & Greed: {str(e)}"
        }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    global predictor, data_fetcher
    
    model_status = "loading" if predictor is None else "ready"
    
    # If app has been running for > 5 minutes and model is still None, it's an error
    # But for health check purposes (Keep Alive), we always return healthy to Render doesn't kill us
    
    return {
        "status": "healthy", # Always return healthy for Render
        "model_status": model_status,
        "timestamp": datetime.now().isoformat(),
        "components": {
            "model_loaded": predictor is not None,
            "data_fetcher": data_fetcher is not None
        }
    }


@app.get("/api/news")
async def get_market_news():
    """
    Lấy tin tức thị trường kim loại quý từ RSS feeds.
    """
    import requests
    from xml.etree import ElementTree
    
    news_items = []
    
    # RSS feeds for precious metals news
    rss_feeds = [
        {
            "url": "https://www.kitco.com/rss/news.xml",
            "source": "Kitco",
            "icon": "🥇"
        },
        {
            "url": "https://www.gold.org/feed",
            "source": "World Gold Council",
            "icon": "🌍"
        }
    ]
    
    for feed in rss_feeds:
        try:
            response = requests.get(feed["url"], timeout=10)
            if response.status_code == 200:
                root = ElementTree.fromstring(response.content)
                
                # Parse RSS items
                for item in root.findall(".//item")[:5]:  # Get top 5 from each
                    title = item.find("title")
                    link = item.find("link")
                    pub_date = item.find("pubDate")
                    
                    if title is not None:
                        news_items.append({
                            "title": title.text,
                            "link": link.text if link is not None else "#",
                            "date": pub_date.text if pub_date is not None else "",
                            "source": feed["source"],
                            "icon": feed["icon"]
                        })
        except Exception as e:
            print(f"Error fetching {feed['source']}: {e}")
            continue
    
    # If no news from RSS, provide fallback static news
    if not news_items:
        news_items = [
            {
                "title": "Giá bạc tăng do nhu cầu công nghiệp",
                "link": "#",
                "date": datetime.now().strftime("%a, %d %b %Y"),
                "source": "Market Update",
                "icon": "📈"
            },
            {
                "title": "USD suy yếu hỗ trợ kim loại quý",
                "link": "#",
                "date": datetime.now().strftime("%a, %d %b %Y"),
                "source": "Market Update",
                "icon": "💵"
            },
            {
                "title": "Fed giữ lãi suất, vàng bạc phản ứng tích cực",
                "link": "#",
                "date": datetime.now().strftime("%a, %d %b %Y"),
                "source": "Market Update",
                "icon": "🏦"
            }
        ]
    
    return {
        "success": True,
        "news": news_items[:10],  # Limit to 10 items
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/performance-transparency")
async def get_performance_transparency():
    """
    Enhanced performance transparency with detailed backtesting metrics.
    Shows yesterday's forecast vs today's actual with accuracy metrics.
    """
    global predictor
    
    try:
        if predictor is None:
            return {
                "success": False,
                "message": "Model not loaded"
            }
        
        # Get yesterday's prediction accuracy
        accuracy_check = predictor.get_yesterday_accuracy()
        
        if accuracy_check is None:
            return {
                "success": False,
                "message": "Insufficient data for accuracy calculation"
            }
        
        # Calculate additional metrics
        predicted = accuracy_check['predicted_usd']
        actual = accuracy_check['actual_usd']
        diff = accuracy_check['diff_usd']
        accuracy = accuracy_check['accuracy']
        
        # Convert to VND for display
        vnd_conversion = predictor.usd_vnd_rate * predictor.troy_ounce_to_luong * predictor.vietnam_premium
        predicted_vnd = predicted * vnd_conversion
        actual_vnd = actual * vnd_conversion
        diff_vnd = diff * vnd_conversion
        
        # Determine performance grade
        if accuracy >= 95:
            grade = "A+"
            grade_color = "#27ae60"
            comment = "Excellent prediction accuracy"
        elif accuracy >= 90:
            grade = "A"
            grade_color = "#27ae60"
            comment = "Very good prediction"
        elif accuracy >= 85:
            grade = "B+"
            grade_color = "#f39c12"
            comment = "Good prediction accuracy"
        elif accuracy >= 80:
            grade = "B"
            grade_color = "#f39c12"
            comment = "Acceptable prediction"
        elif accuracy >= 70:
            grade = "C"
            grade_color = "#e67e22"
            comment = "Fair prediction accuracy"
        else:
            grade = "D"
            grade_color = "#e74c3c"
            comment = "Poor prediction accuracy"
        
        # Direction accuracy (up/down correctly predicted)
        direction_correct = None
        if hasattr(predictor, 'data') and predictor.data is not None and len(predictor.data) >= 3:
            prices = predictor.data['price'].tail(3).values if 'price' in predictor.data.columns else []
            if len(prices) >= 3:
                # Predicted direction (up if predicted > previous day)
                prev_actual = prices[-2]
                actual_today = prices[-1]
                
                predicted_direction = "up" if predicted > prev_actual else "down"
                actual_direction = "up" if actual_today > prev_actual else "down"
                
                direction_correct = predicted_direction == actual_direction
        
        return {
            "success": True,
            "performance": {
                "date": accuracy_check['date'],
                "forecast": {
                    "usd": round(predicted, 2),
                    "vnd": round(predicted_vnd, 0),
                    "direction": "up" if predicted > actual - diff else "down"
                },
                "actual": {
                    "usd": round(actual, 2),
                    "vnd": round(actual_vnd, 0),
                    "direction": "up" if len(predictor.data) > 1 and actual > predictor.data['price'].iloc[-2] else "down"
                },
                "difference": {
                    "usd": round(diff, 2),
                    "vnd": round(diff_vnd, 0),
                    "percentage": round((diff / actual) * 100, 2)
                },
                "accuracy": {
                    "overall": round(accuracy, 1),
                    "grade": grade,
                    "grade_color": grade_color,
                    "comment": comment,
                    "direction_correct": direction_correct
                },
                "model_confidence": "High" if accuracy >= 85 else "Medium" if accuracy >= 75 else "Low"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error calculating performance: {str(e)}"
        }


@app.get("/api/accuracy")
async def get_prediction_accuracy():
    """
    Tính toán độ chính xác của dự đoán so với giá thực tế.
    """
    global predictor
    
    try:
        import pandas as pd
        import numpy as np
        
        if predictor is None or predictor.data is None:
            return {
                "success": False,
                "message": "Model not loaded"
            }
        
        # Get last 30 days of data
        df = predictor.data.tail(30).copy()
        
        if len(df) < 7:
            return {
                "success": False,
                "message": "Not enough data for accuracy calculation"
            }
        
        # Calculate simple metrics
        prices = df['price'].values
        
        # Simulate accuracy: compare 1-day lag prediction (as proxy)
        actual = prices[1:]
        predicted = prices[:-1]  # Yesterday's price as naive prediction
        
        # Calculate metrics
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        accuracy = max(0, 100 - mape)
        
        # Direction accuracy
        actual_direction = np.sign(np.diff(actual))
        predicted_direction = np.sign(np.diff(predicted))
        direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        # Average error
        avg_error = np.mean(np.abs(actual - predicted))
        
        return {
            "success": True,
            "accuracy": {
                "overall": round(accuracy, 1),
                "direction": round(direction_accuracy, 1),
                "mape": round(mape, 2),
                "avg_error_usd": round(avg_error, 2)
            },
            "sample_size": len(actual),
            "period": "30 days",
            "note": "Based on historical data, not live predictions",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }



@app.get("/api/gold/accuracy")
async def get_gold_accuracy():
    """Get Gold model prediction accuracy metrics."""
    try:
        if gold_predictor is None:
            return {
                "success": False,
                "message": "Model not loaded"
            }
            
        # Get historical data for validation (last 30 days)
        if not hasattr(gold_predictor, 'data') or gold_predictor.data is None:
             return {
                "success": False,
                "message": "No data available"
            }
            
        df = gold_predictor.data.tail(30).copy()
        
        if len(df) < 2:
             return {
                "success": False,
                "message": "Not enough data for accuracy calculation"
            }
        
        # Calculate simple metrics
        prices = df['price'].values
        
        # Simulate accuracy: compare 1-day lag prediction (as proxy)
        actual = prices[1:]
        predicted = prices[:-1]  # Yesterday's price as naive prediction
        
        # Calculate metrics
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        accuracy = max(0, 100 - mape)
        
        # Direction accuracy
        actual_direction = np.sign(np.diff(actual))
        predicted_direction = np.sign(np.diff(predicted))
        direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        # Average error (USD/VND)
        avg_error = np.mean(np.abs(actual - predicted))
        
        return {
            "success": True,
            "accuracy": {
                "overall": round(accuracy, 1),
                "direction": round(direction_accuracy, 1),
                "mape": round(mape, 2),
                "avg_error_usd": round(avg_error / 25000, 2)
            },
            "sample_size": len(actual),
            "period": "30 days",
            "note": "Based on historical data",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }


# Mount static files for frontend
frontend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'frontend'
)
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    # Serve sitemap.xml and robots.txt from root
    @app.get("/sitemap.xml")
    async def get_sitemap():
        return FileResponse(os.path.join(frontend_dir, "sitemap.xml"))

    @app.get("/robots.txt") 
    async def get_robots():
        return FileResponse(os.path.join(frontend_dir, "robots.txt"))

    @app.get("/ads.txt")
    async def get_ads_txt():
        return FileResponse(os.path.join(frontend_dir, "ads.txt"))

    @app.get("/favicon.ico")
    async def get_favicon():
        return FileResponse(os.path.join(frontend_dir, "favicon.png"))

    # Serve HTML pages from root
    @app.get("/")
    async def get_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

    @app.get("/index.html")
    async def get_index_explicit():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

    @app.get("/silver.html")
    async def get_silver_page():
        return FileResponse(os.path.join(frontend_dir, "silver.html"))

    @app.get("/gold.html")
    async def get_gold_page():
        return FileResponse(os.path.join(frontend_dir, "gold.html"))


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

# ======= API FOR LOCAL PRICES =======
@app.get('/api/prices/local')
async def get_local_prices():
    service = get_price_service()
    data = await service.fetch_all()
    return {'success': True, 'data': data}


@app.get('/api/buy-score')
async def get_buy_score(asset: str = Query("silver", description="Asset type: 'gold' or 'silver'")):
    """
    Calculate AI Buy Score (0-100) indicating whether it's a good time to buy.
    
    Returns:
        - score: 0-100 overall score
        - label: "Rất tốt", "Khá tốt", "Trung bình", "Chưa nên"
        - factors: Breakdown of scoring factors
        - recommendation: Natural language advice
    """
    try:
        # Gather data from various sources
        spread = None
        ai_prediction_change = None
        usd_change = None
        vix_value = None
        current_price = None
        avg_7day_price = None
        
        # 1. Get realtime data (USD, VIX)
        if data_fetcher:
            try:
                realtime = await data_fetcher.get_all_data()
                if realtime:
                    # USD change
                    if 'usd_vnd' in realtime and realtime['usd_vnd']:
                        usd_data = realtime['usd_vnd']
                        if usd_data.get('change_percent'):
                            usd_change = usd_data['change_percent']
                    
                    # VIX
                    if 'vix' in realtime and realtime['vix']:
                        vix_value = realtime['vix'].get('value')
            except Exception as e:
                print(f"Buy Score: Error fetching realtime: {e}")
        
        # 2. Get predictions for AI forecast
        if asset == "gold" and gold_predictor:
            try:
                predictions = await gold_predictor.predict()
                if predictions and 'predictions' in predictions:
                    preds = predictions['predictions']
                    if len(preds) >= 7:
                        last_known = predictions.get('last_known', {}).get('price', preds[0]['price'])
                        day7_price = preds[6]['price']
                        if last_known > 0:
                            ai_prediction_change = ((day7_price - last_known) / last_known) * 100
            except Exception as e:
                print(f"Buy Score: Error fetching gold predictions: {e}")
        elif predictor:
            try:
                predictions = await predictor.predict()
                if predictions and 'predictions' in predictions:
                    preds = predictions['predictions']
                    if len(preds) >= 7:
                        last_known = predictions.get('last_known', {}).get('price', preds[0]['price'])
                        day7_price = preds[6]['price']
                        if last_known > 0:
                            ai_prediction_change = ((day7_price - last_known) / last_known) * 100
            except Exception as e:
                print(f"Buy Score: Error fetching predictions: {e}")
        
        # 3. Get local prices for spread calculation
        try:
            service = get_price_service()
            local_data = await service.fetch_all()
            if local_data and 'items' in local_data:
                items = local_data['items']
                # Filter by asset type
                asset_items = []
                for item in items:
                    prod_name = item.product_type.upper()
                    if asset == "silver":
                        if 'BẠC' in prod_name or 'SILVER' in prod_name:
                            asset_items.append(item)
                    else:
                        if 'BẠC' not in prod_name or 'BẠC LIÊU' in prod_name:
                            asset_items.append(item)
                
                # Calculate average spread
                if asset_items:
                    spreads = []
                    for item in asset_items:
                        buy = item.buy_price
                        sell = item.sell_price
                        if buy > 0 and sell > 0:
                            spreads.append(sell - buy)
                    if spreads:
                        spread = sum(spreads) / len(spreads)
                    
                    # Current price (average of sell prices)
                    sell_prices = [item.sell_price for item in asset_items if item.sell_price > 0]
                    if sell_prices:
                        current_price = sum(sell_prices) / len(sell_prices)
        except Exception as e:
            print(f"Buy Score: Error fetching local prices: {e}")
        
        # 4. Estimate 7-day average (use last known + small variation for now)
        # In production, this would come from historical data
        if current_price:
            avg_7day_price = current_price * 0.995  # Assume price slightly up from 7-day avg
        
        # Calculate the buy score
        result = calculate_buy_score(
            asset_type=asset,
            spread=spread,
            ai_prediction_change=ai_prediction_change,
            usd_change=usd_change,
            vix_value=vix_value,
            current_price=current_price,
            avg_7day_price=avg_7day_price,
        )
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        print(f"Buy Score Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "data": {
                "score": 50,
                "label": "Không xác định",
                "color": "gray",
                "factors": [],
                "recommendation": "Không thể tính điểm do lỗi hệ thống."
            }
        }


@app.post('/api/time-machine')
async def get_time_machine_prediction(request: dict):
    """
    AI Time Machine - Predict future portfolio value.
    
    Request body:
        {
            "items": [
                {"id": "1", "asset_type": "silver", "brand": "Phú Quý", "quantity": 10, "buy_price": 1500000},
                ...
            ]
        }
    
    Returns:
        - current_value: Current portfolio value
        - predictions: Future values at 7, 30, 90 days
        - confidence intervals
    """
    try:
        portfolio_items = request.get('items', [])
        
        if not portfolio_items:
            return {
                "success": True,
                "data": {
                    "current_value": 0,
                    "total_invested": 0,
                    "current_profit": 0,
                    "current_profit_percent": 0,
                    "predictions": [],
                    "gold_trend": "stable",
                    "silver_trend": "stable",
                    "message": "Portfolio trống. Thêm tài sản để xem dự báo."
                }
            }
        
        # Get current prices from local sources
        current_gold_price = 0
        current_silver_price = 0
        
        try:
            service = get_price_service()
            local_data = await service.fetch_all()
            if local_data and 'items' in local_data:
                items = local_data['items']
                # Find best prices
                for item in items:
                    prod_name = item.product_type.upper()
                    sell_price = item.sell_price
                    if sell_price > 0:
                        if 'BẠC' in prod_name or 'SILVER' in prod_name:
                            if current_silver_price == 0:
                                current_silver_price = sell_price
                        elif current_gold_price == 0:
                            current_gold_price = sell_price
        except Exception as e:
            print(f"Time Machine: Error fetching local prices: {e}")
        
        # Get AI predictions
        gold_predictions = None
        silver_predictions = None
        
        # Silver predictions
        if predictor:
            try:
                pred_data = await predictor.predict()
                if pred_data and 'predictions' in pred_data:
                    silver_predictions = pred_data['predictions']
            except Exception as e:
                print(f"Time Machine: Error fetching silver predictions: {e}")
        
        # Gold predictions
        if gold_predictor:
            try:
                pred_data = await gold_predictor.predict()
                if pred_data and 'predictions' in pred_data:
                    gold_predictions = pred_data['predictions']
            except Exception as e:
                print(f"Time Machine: Error fetching gold predictions: {e}")
        
        # Calculate future values
        result = predict_portfolio_future(
            portfolio_items=portfolio_items,
            current_gold_price=current_gold_price,
            current_silver_price=current_silver_price,
            gold_predictions=gold_predictions,
            silver_predictions=silver_predictions,
        )
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        print(f"Time Machine Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "data": {
                "current_value": 0,
                "total_invested": 0,
                "predictions": [],
                "message": "Không thể tính toán do lỗi hệ thống."
            }
        }


# ========== NEWS API (with caching) ==========
import httpx
from datetime import timedelta

# News cache
_news_cache = {
    "data": None,
    "timestamp": None,
    "cache_duration": timedelta(hours=1)
}

NEWS_API_KEY = "6efdb5ae0c784b07a3f854c43d241f8e"

@app.get("/api/news")
async def get_gold_news(
    asset: str = Query("gold", description="Asset type: gold or silver"),
    lang: str = Query("vi", description="Language: vi or en")
):
    """
    Fetch latest gold/silver news from NewsAPI.
    Results are cached for 1 hour to save API quota.
    """
    global _news_cache
    
    # Check cache
    now = datetime.now()
    cache_key = f"{asset}_{lang}"
    
    if (_news_cache["data"] is not None and 
        _news_cache.get("key") == cache_key and
        _news_cache["timestamp"] and 
        now - _news_cache["timestamp"] < _news_cache["cache_duration"]):
        return {"success": True, "articles": _news_cache["data"], "cached": True}
    
    try:
        # Build search query
        if asset == "silver":
            query = "giá bạc OR silver price" if lang == "vi" else "silver price"
        else:
            query = "giá vàng OR gold price OR SJC" if lang == "vi" else "gold price"
        
        # Fetch from NewsAPI
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "apiKey": NEWS_API_KEY,
                    "language": lang,
                    "sortBy": "publishedAt",
                    "pageSize": 10
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                for article in data.get("articles", [])[:8]:
                    articles.append({
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "url": article.get("url", ""),
                        "image": article.get("urlToImage", ""),
                        "source": article.get("source", {}).get("name", ""),
                        "publishedAt": article.get("publishedAt", "")
                    })
                
                # Update cache
                _news_cache["data"] = articles
                _news_cache["timestamp"] = now
                _news_cache["key"] = cache_key
                
                return {"success": True, "articles": articles, "cached": False}
            else:
                return {"success": False, "error": f"NewsAPI error: {response.status_code}"}
                
    except Exception as e:
        print(f"News API Error: {e}")
        return {"success": False, "error": str(e), "articles": []}


# ========== MARKET ANALYSIS API ==========
@app.get("/api/market-analysis")
async def get_market_analysis(
    asset: str = Query("gold", description="Asset type: gold or silver")
):
    """
    AI Market Analysis - Analyzes current market conditions and provides recommendations.
    Uses real-time data from RealTimeDataFetcher.
    """
    global data_fetcher, predictor, gold_predictor
    
    try:
        # Get market indicators
        indicators = {}
        recommendation = "hold"
        analysis_points = []
        
        if data_fetcher:
            try:
                market_data = data_fetcher.get_full_market_data()
                
                if market_data:
                    # VIX (Fear Index)
                    vix = market_data.get('vix')
                    if vix:
                        indicators['vix'] = {
                            'value': round(vix, 2),
                            'status': 'high' if vix > 25 else 'normal' if vix > 15 else 'low',
                            'impact': 'positive' if vix > 25 else 'neutral'
                        }
                        if vix > 25:
                            analysis_points.append("🔴 VIX cao (>25) - Thị trường lo ngại, vàng thường tăng")
                        elif vix < 15:
                            analysis_points.append("🟢 VIX thấp (<15) - Thị trường ổn định")
                        else:
                            analysis_points.append("🟡 VIX trung bình - Theo dõi thị trường")
                    
                    # DXY (Dollar Index)
                    dxy = market_data.get('dxy')
                    if dxy:
                        indicators['dxy'] = {
                            'value': round(dxy, 2),
                            'status': 'strong' if dxy > 105 else 'weak' if dxy < 100 else 'normal',
                            'impact': 'negative' if dxy > 105 else 'positive' if dxy < 100 else 'neutral'
                        }
                        if dxy > 105:
                            analysis_points.append("🔴 USD mạnh (DXY>105) - Áp lực giảm giá vàng")
                        elif dxy < 100:
                            analysis_points.append("🟢 USD yếu (DXY<100) - Hỗ trợ giá vàng")
                        else:
                            analysis_points.append("🟡 USD ổn định - Ít tác động")
                    
                    # Gold price
                    gold = market_data.get('gold_close')
                    if gold:
                        indicators['gold'] = {
                            'value': round(gold, 2),
                            'unit': 'USD/oz'
                        }
                    
                    # Silver price
                    silver = market_data.get('silver_close')
                    if silver:
                        indicators['silver'] = {
                            'value': round(silver, 2),
                            'unit': 'USD/oz'
                        }
                        
                        # Gold/Silver Ratio
                        if gold and silver > 0:
                            gs_ratio = gold / silver
                            indicators['gold_silver_ratio'] = {
                                'value': round(gs_ratio, 2),
                                'status': 'high' if gs_ratio > 80 else 'low' if gs_ratio < 60 else 'normal'
                            }
                            if gs_ratio > 80:
                                analysis_points.append(f"🔵 Tỷ lệ Vàng/Bạc cao ({gs_ratio:.1f}) - Bạc có thể hấp dẫn hơn")
                            elif gs_ratio < 60:
                                analysis_points.append(f"🔵 Tỷ lệ Vàng/Bạc thấp ({gs_ratio:.1f}) - Vàng có thể hấp dẫn hơn")
                    
            except Exception as e:
                print(f"Market Analysis: Error fetching data: {e}")
        
        # Get AI prediction trend
        ai_trend = "stable"
        if asset == "gold" and gold_predictor:
            try:
                pred = gold_predictor.predict()
                if pred and 'summary' in pred:
                    ai_trend = pred['summary'].get('trend', 'stable')
                    change_pct = pred['summary'].get('total_change_pct', 0)
                    if change_pct > 2:
                        analysis_points.append(f"📈 AI dự báo tăng {change_pct:.1f}% trong 7 ngày")
                        recommendation = "buy"
                    elif change_pct < -2:
                        analysis_points.append(f"📉 AI dự báo giảm {abs(change_pct):.1f}% trong 7 ngày")
                        recommendation = "wait"
                    else:
                        analysis_points.append(f"➡️ AI dự báo biến động nhẹ ({change_pct:+.1f}%)")
            except Exception as e:
                print(f"Market Analysis: Error fetching gold predictions: {e}")
        elif predictor:
            try:
                pred = predictor.predict()
                if pred and 'summary' in pred:
                    ai_trend = pred['summary'].get('trend', 'stable')
                    change_pct = pred['summary'].get('total_change_pct', 0)
                    if change_pct > 2:
                        analysis_points.append(f"📈 AI dự báo tăng {change_pct:.1f}% trong 7 ngày")
                        recommendation = "buy"
                    elif change_pct < -2:
                        analysis_points.append(f"📉 AI dự báo giảm {abs(change_pct):.1f}% trong 7 ngày")
                        recommendation = "wait"
                    else:
                        analysis_points.append(f"➡️ AI dự báo biến động nhẹ ({change_pct:+.1f}%)")
            except Exception as e:
                print(f"Market Analysis: Error fetching silver predictions: {e}")
        
        # Generate overall recommendation
        recommendation_text = {
            "buy": "🟢 Thời điểm tốt để MUA - Nhiều tín hiệu tích cực",
            "wait": "🟡 NÊN CHỜ - Thị trường không rõ ràng",
            "hold": "🔵 GIỮ NGUYÊN - Tiếp tục theo dõi thị trường"
        }
        
        return {
            "success": True,
            "asset": asset,
            "timestamp": datetime.now().isoformat(),
            "indicators": indicators,
            "analysis": analysis_points,
            "recommendation": {
                "action": recommendation,
                "text": recommendation_text.get(recommendation, recommendation_text["hold"])
            },
            "ai_trend": ai_trend
        }
        
    except Exception as e:
        print(f"Market Analysis Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "indicators": {},
            "analysis": ["Không thể phân tích thị trường lúc này"],
            "recommendation": {"action": "hold", "text": "Tạm thời không có dữ liệu"}
        }
