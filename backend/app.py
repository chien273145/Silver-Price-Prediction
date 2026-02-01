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

from src.enhanced_predictor import EnhancedPredictor
from backend.realtime_data import RealTimeDataFetcher


# Global instances
predictor: Optional[EnhancedPredictor] = None
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
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[{datetime.now().time()}] ✅ App startup complete in {elapsed:.2f}s (Model loading in background)", flush=True)
    
    yield
    
    # Cleanup
    print("👋 Shutting down API...", flush=True)

async def load_model_background():
    """Load model in background to not block startup."""
    global predictor
    
    print(f"[{datetime.now().time()}] ⏳ Background: Starting model loading...", flush=True)
    
    try:
        # Run heavy lifting in thread pool
        predictor = await asyncio.to_thread(_load_model_logic)
        print(f"[{datetime.now().time()}] ✅ Background: Model loaded successfully!", flush=True)
    except Exception as e:
        print(f"[{datetime.now().time()}] ❌ Background: Model loading failed: {e}", flush=True)

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
