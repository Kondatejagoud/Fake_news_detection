import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.db.session import engine, Base
from app.api.routes import health, analyze

# 1. Initialize logging
setup_logging()
logger.info("Initializing Hybrid Multimodal Fake Content Detection Backend")

# 2. Database migrations / Table creation
try:
    logger.info("Auto-creating database tables if they do not exist...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified.")
except Exception as e:
    logger.error(f"Error checking/creating database tables: {e}")

# 3. Create FastAPI app instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API for URL scraping, NLP, Image Forensics, and Deepfake detection.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 4. Configure CORS Middleware
# Allows React dev server and production frontend to connect
origins = []
if isinstance(settings.ALLOWED_CORS_ORIGINS, list):
    origins = settings.ALLOWED_CORS_ORIGINS
else:
    origins = [o.strip() for o in settings.ALLOWED_CORS_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Include API routers
app.include_router(health.router, prefix=settings.API_PREFIX, tags=["Health"])
app.include_router(analyze.router, prefix=settings.API_PREFIX, tags=["Analyze"])

# 6. Serves static files for Single Page Application (SPA) if compiled
# Look for a local static/ folder (production build location)
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(static_dir):
    # Fallback to frontend build directory if structure is nested
    static_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "frontend", "dist"))

if os.path.exists(static_dir):
    logger.info(f"Mounting production static assets from: {static_dir}")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    logger.warning(
        f"Static files directory not found at {static_dir}. "
        "Frontend assets will not be served directly by FastAPI. "
        "Ensure React is running independently via Vite dev server."
    )
    
    # Simple friendly default response for root URL when no frontend is compiled yet
    @app.get("/", response_class=HTMLResponse)
    def index():
        return """
        <html>
            <head>
                <title>Hybrid Fake Content Detection API</title>
                <style>
                    body { font-family: sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; background: #0f172a; color: #f8fafc; }
                    h1 { color: #38bdf8; }
                    a { color: #f43f5e; text-decoration: none; font-weight: bold; }
                </style>
            </head>
            <body>
                <h1>Fake Content Detector Backend API</h1>
                <p>Status: Running successfully.</p>
                <p>Interactive docs available at <a href="/docs">/docs</a>.</p>
            </body>
        </html>
        """
