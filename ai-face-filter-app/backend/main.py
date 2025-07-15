from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import route correctly using relative import
from backend.routes import filter  # No need to use `backend.routes` here

app = FastAPI()

# CORS Middleware for allowing frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register router
app.include_router(filter.router, prefix="/filter")
