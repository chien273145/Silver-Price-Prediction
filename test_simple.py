"""
Minimal server test without emoji issues
"""
import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create FastAPI app
app = FastAPI(title="Silver Price Test API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Test API is working!"}

@app.get("/test-fear-greed")
async def test_fear_greed():
    """Test Fear & Greed endpoint with mock data"""
    return {
        "success": True,
        "index": {
            "score": 65.5,
            "signal": "GREED",
            "recommendation": "Consider Taking Profits",
            "color": "#f39c12",
            "components": {
                "market_drivers": 60.0,
                "prediction_trend": 75.0,
                "volatility": 62.0
            },
            "timestamp": "2026-02-02T08:35:00"
        }
    }

@app.get("/test-performance")
async def test_performance():
    """Test Performance endpoint with mock data"""
    return {
        "success": True,
        "performance": {
            "date": "2026-02-01",
            "forecast": {
                "vnd": 1850000,
                "direction": "up"
            },
            "actual": {
                "vnd": 1820000,
                "direction": "up"
            },
            "difference": {
                "vnd": -30000,
                "percentage": -1.64
            },
            "accuracy": {
                "overall": 95.2,
                "grade": "A+",
                "grade_color": "#27ae60",
                "comment": "Excellent prediction accuracy",
                "direction_correct": True
            },
            "model_confidence": "High"
        },
        "timestamp": "2026-02-02T08:35:00"
    }

if __name__ == "__main__":
    print("=" * 50)
    print("TEST SERVER")
    print("=" * 50)
    print("Server running at: http://localhost:8001")
    print("Test endpoints:")
    print("  - http://localhost:8001/test-fear-greed")
    print("  - http://localhost:8001/test-performance")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )