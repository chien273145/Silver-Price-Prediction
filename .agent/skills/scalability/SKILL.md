---
name: Scalability Architect
description: Expert in designing high-concurrency systems, distributed caching, and database scaling for millions of users.
---

# Scalability Architect Skill

## 🚀 Philosophy
"Scale horizontally, cache aggressively."
For a system serving millions, single-server solutions (SQLite/CSV/In-memory) are dead ends. We must design for statelessness and distributed consistency.

## 🏗️ Architecture for Millions

### 1. Database Layer (The Foundation)
- **Problem**: CSV/SQLite locks under write load.
- **Solution**: 
  - **Primary DB**: PostgreSQL (Managed) for transactional data (Users, Subscriptions).
  - **Time-Series DB**: TimescaleDB or InfluxDB for storing price history (Tick data).
  - **Read Replicas**: Separate Write (updates from Yahoo) vs Read (user queries) connections.

### 2. Caching Layer (The Speed)
- **Problem**: Python in-memory cache is per-process and vanishes on restart.
- **Solution**: **Redis** (Cluster Mode).
  - **Hot Data**: Store "Current Price" in Redis with 10s TTL.
  - **API Response Caching**: Cache full JSON responses for `/api/predict` in Redis for 1 minute.
  - **Result**: 99.9% of user traffic hits Redis, not the DB or Python logic.

### 3. Serving Layer (The Distribution)
- **Problem**: Serving static HTML/JS from Python/Uvicorn burns CPU.
- **Solution**: **CDN** (Cloudflare/AWS CloudFront).
  - Serve `frontend/` (HTML, CSS, JS) directly from Edge locations.
  - Only `/api/*` requests hit the backend.

### 4. Real-time Updates (The Push)
- **Problem**: Polling (Client asking server every 5s) x 1M users = 200k req/sec = DDoS.
- **Solution**: **WebSockets / Server-Sent Events (SSE)**.
  - Use a dedicated Push Service (Pusher, PubNub) or a separate Go/Node.js WebSocket microservice.
  - Push price update ONCE to the channel, and the service broadcasts to 1M subscribers.

## 🛡️ Operational Safeguards
- **Rate Limiting**: Token Bucket algorithm (via Redis) to ban abusive IPs.
- **Load Balancing**: NGINX or Application Load Balancer (ALB) to distribute traffic across 10+ Backend Instances.
- **Auto-Scaling**: Kubernetes HPA (Horizontal Pod Autoscaler) based on CPU/Memory/Request count.

## 🛠️ Implementation Roadmap
1.  **Phase 1**: Move CSV -> PostgreSQL.
2.  **Phase 2**: Add Redis for caching API responses.
3.  **Phase 3**: CDN for Frontend.
4.  **Phase 4**: WebSocket implementation.
