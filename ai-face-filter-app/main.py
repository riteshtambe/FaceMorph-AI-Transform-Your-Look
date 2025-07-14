from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import filter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(filter.router, prefix="/filter")
