from fastapi import FastAPI
from app.middlewares.cors import setup_cors
from app.routes.legacy_routes import router as legacy_router

app = FastAPI(title="Vigilora AML API", version="2.0")

# 1. Setup Middlewares
setup_cors(app)

# 2. Include Legacy Routes
app.include_router(legacy_router)
