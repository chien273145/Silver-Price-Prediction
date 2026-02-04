"""
Simple test script to run the server without emoji issues
"""
import os
import sys
import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Silver Price Prediction API...")
    print("=" * 60)
    print("Server will be available at: http://localhost:8000")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Start server
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to avoid issues
        log_level="info"
    )