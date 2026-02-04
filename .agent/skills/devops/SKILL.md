---
name: DevOps Architect
description: Expert in deployment automation, CI/CD, and operational stability for Render.com.
---

# DevOps Architect Skill

## 🏗️ Philosophy
"Deploy with confidence, sleep with peace."
We prioritize automation and environment parity. If it works on Local, it MUST work on Production.

## ☁️ Render.com Deployment Standards

### 1. Configuration (`render.yaml`)
- **Infrastructure as Code**: All deployment settings must be in `render.yaml`.
- **Environment Variables**: NEVER hardcode secrets. Use `envVars` in YAML or Dashboard.
- **Start Command**: Must be robust (e.g., `gunicorn` or `uvicorn` with workers).

### 2. Dependency Management
- **requirements.txt**: Keep it lightweight. Split extensive ML libraries if possible (e.g., use CPU versions for `torch`/`tensorflow` if GPU not needed).
- **Python Version**: Explicitly set in `.python-version` (e.g., 3.9.18).

### 3. Operational Resilience
- **Health Checks**: Implement `/health` endpoint that checks DB/External API connectivity.
- **Keep-Alive**: For free tier, use `cron-job.org` or internal cron to wake up services.
- **Logging**: Structure logs to be readable in Render dashboard (JSON preferred, but structured text ok).

### 4. CI/CD Workflow
- **Pre-deploy**: Run minimal test suite (lint + unit tests).
- **Deploy**: Push to `main` triggers auto-deploy (if configured).
- **Post-deploy**: Smoke test the `/api/predict` endpoint.

## 🛡️ Security Best Practices
- **CORS**: Restrict `allow_origins` in Production.
- **Rate Limiting**: Implement basic rate limiting if abuse is detected.
- **Input Sanitization**: Trust no user input, validate with Pydantic.

## 🚀 Troubleshooting Playbook
1. **502 Bad Gateway**: Check memory usage (OOM) or startup timeout.
2. **Module Not Found**: Check `requirements.txt` vs local environment.
3. **Slow Response**: Check external API latency (Yahoo Finance). Add caching.
