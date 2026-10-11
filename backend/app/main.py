from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.api.routes_prediction import router as prediction_router
from app.api.routes_training import router as training_router
from app.api.routes_dataset import router as dataset_router
from app.api.routes_metrics import router as metrics_router
from app.ml.prediction import prediction_engine
from app.utils.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("==================================================")
    logger.info(" Initializing Sentiment Analysis Model Comparison API ")
    logger.info("==================================================")
    logger.info("Dataset: Sentiment140")
    logger.info("Models: Logistic Regression (Baseline), Random Forest (Bagging), AdaBoost (Boosting)")
    logger.info("Feature Extraction: TF-IDF Vectorizer")
    
    loaded = prediction_engine.load_artifacts()
    if loaded:
        logger.info("Loaded Artifacts:")
        logger.info("  ✓ TF-IDF Vectorizer")
        logger.info("  ✓ Logistic Regression")
        logger.info("  ✓ Random Forest")
        logger.info("  ✓ AdaBoost")
        logger.info("  ✓ Model Metadata")
        logger.info(f"  Selected Model: {prediction_engine.selected_model_name}")
    else:
        logger.info("No pre-trained model artifacts found on startup. Models must be trained via API or training script.")
    yield
    # Shutdown logic
    logger.info("Shutting down API service.")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Full-stack Machine Learning & Ensemble Learning (Logistic Regression vs Random Forest vs AdaBoost) Sentiment Analysis Application",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(prediction_router)
app.include_router(training_router)
app.include_router(dataset_router)
app.include_router(metrics_router)

@app.get("/", tags=["Health"])
def root():
    return {
        "title": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs": "/docs",
        "message": "Welcome to Sentiment Analysis and Model Comparison API"
    }

@app.get("/api/health", tags=["Health"])
def health_check():
    is_trained = prediction_engine.is_loaded
    return {
        "status": "healthy",
        "models_loaded": is_trained,
        "vectorizer_loaded": is_trained,
        "metadata_loaded": is_trained,
        "active_model": prediction_engine.selected_model_name if is_trained else None,
        "models_available": ["Logistic Regression", "Random Forest", "AdaBoost"]
    }

