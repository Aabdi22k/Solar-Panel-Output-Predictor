import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main.api.app:app",
        host="0.0.0.0",
        port=8501,
        reload=False,
        app_dir="src",
    )