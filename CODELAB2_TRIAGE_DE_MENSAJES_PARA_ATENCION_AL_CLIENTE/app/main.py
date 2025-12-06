import os
from fastapi import FastAPI

from app.api.routes.triage import router as triage_router

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.include_router(triage_router, prefix= "/api/triage", tags = ["Triage"])


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)