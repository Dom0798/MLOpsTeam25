import uvicorn

uvicorn.run("api.app:app", host="0.0.0.0", port=5000, log_level="info")