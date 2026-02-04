---
name: Backend Quality Guardian
description: Ensures Python backend reliability through strict code quality standards, type safety, and robust error handling.
---

# Backend Quality Guardian Skill

## 🛡️ Philosophy
"Catch bugs at design time, not runtime."
Reliability comes from predictability. We enforce strict typing, comprehensive error handling, and clean architecture to ensure the prediction API never fails silently.

## 🐍 Python Best Practices

### 1. Type Hinting (Strict)
- **Rule**: All function signatures MUST have type hints.
- **Rule**: Use `typing.Optional`, `typing.List`, `typing.Dict` explicitly.
- **Example**:
  ```python
  def get_prediction(days: int) -> Dict[str, Any]:
      ...
  ```

### 2. Error Handling (Graceful)
- **Global**: Never let the app crash with a 500 Internal Server Error without a meaningful trace.
- **FastAPI**: Use `HTTPException` with specific status codes (404, 400, 503).
- **Try/Except**: Wrap external API calls (Yahoo Finance) in specific try/except blocks.
  ```python
  try:
      data = yf.download(...)
  except requests.exceptions.Timeout:
      logger.error("Yahoo Finance timeout")
      raise HTTPException(status_code=503, detail="Data source unavailable")
  ```

### 3. Code Structure
- **Service Layer**: Business logic goes into `src/`, NOT directly in `app.py` routes.
- **Pydantic**: Use Pydantic models for ALL request/response validation.
- **Constants**: No magic numbers. Use `config.py` or uppercase constants.

### 4. Testing Strategy
- **Unit Tests**: Test individual predictor methods.
- **Integration**: Test API endpoints using `TestClient`.
- **Mocking**: Mock Yahoo Finance calls during tests to avoid network dependency.

## 🛠️ Tools & Configs
- **Linter**: `flake8` (max-line-length=100)
- **Formatter**: `black` or `autopep8`
- **Import Sort**: `isort`

## 🚀 Workflow
1. **Refactor**: When touching file, apply typing and clean imports.
2. **Validate**: Run local tests before confirming execution.
3. **Log**: Ensure critical paths have `print()` or `logging` statements for Render logs.
