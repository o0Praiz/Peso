#!/usr/bin/env python3
"""
Peso API Server
Complete RESTful API implementation for the Peso data warehouse
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid
import secrets

from fastapi import FastAPI, Depends, HTTPException, Security, status, Request, BackgroundTasks
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import jwt
from passlib.context import CryptContext

# Import Peso modules
from peso_integration_module import PesoIntegration
from peso_visualization_module import VisualizationEngine

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("peso-api")

# Create API app
app = FastAPI(
    title="Peso API",
    description="RESTful API for the Peso Marketing Data Warehouse",
    version="0.6.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Peso Integration
peso_integration = None

# Security configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
JWT_SECRET = secrets.token_hex(32)  # Generate random secret
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = 30  # minutes

# Pydantic models for request/response validation
class DatasetCreate(BaseModel):
    name: str = Field(..., description="Name of the dataset")
    data_type: str = Field(..., description="Type of dataset (human, non-human, mixed)")
    synthetic_count: int = Field(100, description="Number of synthetic records to generate")

class DatasetEnrich(BaseModel):
    dataset_id: int = Field(..., description="ID of the dataset to enrich")
    ai_tool: str = Field("openai", description="AI tool to use for enrichment")

class MLModelTrain(BaseModel):
    dataset_id: int = Field(..., description="ID of the dataset")
    target_column: str = Field(..., description="Column to predict")
    model_type: str = Field("classifier", description="Type of model (classifier/regressor)")
    features: Optional[List[str]] = Field(None, description="List of feature columns")

class PredictionRequest(BaseModel):
    model_name: str = Field(..., description="Name of the trained model")
    data: List[Dict[str, Any]] = Field(..., description="Data to make predictions on")

class QueryRequest(BaseModel):
    dataset_id: Optional[int] = Field(None, description="Dataset ID to query")
    filters: Dict[str, Any] = Field({}, description="Filter criteria")
    group_by: Optional[str] = Field(None, description="Field to group by")
    aggregations: Optional[List[Dict[str, str]]] = Field(None, description="Aggregation specifications")

class VisualizationRequest(BaseModel):
    dataset_id: int = Field(..., description="ID of the dataset")
    chart_type: str = Field(..., description="Type of chart to generate")
    columns: List[str] = Field(..., description="Columns to include in visualization")
    parameters: Dict[str, Any] = Field({}, description="Additional parameters for visualization")
    version: Optional[int] = Field(None, description="Dataset version")

class DashboardRequest(BaseModel):
    dataset_id: int = Field(..., description="ID of the dataset")
    version: Optional[int] = Field(None, description="Dataset version")

class UserCreate(BaseModel):
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    password: str = Field(..., description="Password")
    role: str = Field("user", description="User role (user/admin)")

class User(BaseModel):
    username: str
    email: str
    role: str
    disabled: bool = False

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class TaskResponse(BaseModel):
    task_id: str
    message: str
    status: str

# User management functions
def get_user(db, username: str):
    """Get user from database"""
    # In a production environment, this would query a user database
    # For this example, we use a simple dictionary
    users_db = {
        "admin": {
            "username": "admin",
            "email": "admin@example.com",
            "role": "admin",
            "disabled": False,
            "hashed_password": pwd_context.hash("adminpassword")
        }
    }
    
    if username in users_db:
        user_dict = users_db[username]
        return UserInDB(**user_dict)
    return None

def verify_password(plain_password, hashed_password):
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Generate password hash"""
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str):
    """Authenticate user credentials"""
    user = get_user(None, username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[int] = None):
    """Create JWT token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + datetime.timedelta(minutes=expires_delta or JWT_EXPIRATION)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = get_user(None, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Get current active user"""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def get_api_key(api_key: str = Security(api_key_header)):
    """Validate API key"""
    config = getattr(peso_integration, "config", {})
    api_keys = config.get("api", {}).get("api_keys", [])
    
    if not api_keys or api_key in api_keys:
        return api_key
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
    )

def require_admin(current_user: User = Depends(get_current_active_user)):
    """Verify user has admin role"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized"
        )
    return current_user

def verify_auth():
    """Check which authentication method to use based on config"""
    config = getattr(peso_integration, "config", {})
    auth_method = config.get("api", {}).get("auth_method", "api_key")
    require_auth = config.get("api", {}).get("require_auth", True)
    
    if not require_auth:
        return []
    
    if auth_method == "oauth2":
        return [Depends(get_current_active_user)]
    else:
        return [Depends(get_api_key)]

@app.on_event("startup")
async def startup_event():
    """Initialize Peso integration on startup"""
    global peso_integration
    
    try:
        # Initialize integration
        peso_integration = PesoIntegration()
        
        # Create visualization output directory if it doesn't exist
        config = getattr(peso_integration, "config", {})
        viz_dir = config.get("visualization", {}).get("output_dir", "visualizations")
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # Initialize task store for background tasks
        peso_integration.task_store = {}
            
        logger.info("Peso API initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Peso API: {e}")
        raise

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and issue JWT token"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return current_user

@app.post("/users", response_model=User)
async def create_user(user: UserCreate, admin: User = Depends(require_admin)):
    """Create a new user (admin only)"""
    # In a production environment, this would add a user to the database
    return {
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "disabled": False
    }

@app.get("/", dependencies=verify_auth())
async def root():
    """API root endpoint"""
    return {
        "name": "Peso API",
        "version": "0.6.0",
        "status": "online",
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/datasets/{dataset_id}", dependencies=verify_auth())
async def delete_dataset(dataset_id: int, current_user: User = Depends(get_current_active_user)):
    """Delete a dataset (requires authentication)"""
    try:
        # Check if dataset exists
        dataset = peso_integration.db.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found",
            )
        
        # Implement dataset deletion
        # This would be a method in your database module
        success = peso_integration.db.delete_dataset(dataset_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete dataset",
            )
        
        return {"message": f"Dataset {dataset_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

@app.get("/health")
async def health_check():
    """Health check endpoint (no auth required)"""
    if peso_integration is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized",
        )
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/datasets", dependencies=verify_auth())
async def list_datasets(name: Optional[str] = None, type: Optional[str] = None):
    """List all datasets with optional filtering"""
    filters = {}
    
    if name:
        filters["name_contains"] = name
    
    if type:
        filters["type"] = type
    
    try:
        datasets = peso_integration.query_engine.query_datasets(filters)
        return {
            "count": len(datasets),
            "datasets": datasets
        }
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

@app.get("/tasks/{task_id}", dependencies=verify_auth())
async def get_task_status(task_id: str):
    """Get the status of a background task"""
    task_store = getattr(peso_integration, "task_store", {})
    
    if task_id not in task_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID {task_id} not found",
        )
    
    return task_store[task_id]

@app.get("/datasets/{dataset_id}", dependencies=verify_auth())
async def get_dataset(dataset_id: int, version: Optional[int] = None):
    """Get a specific dataset"""
    try:
        dataset = peso_integration.db.get_dataset(dataset_id, version)
        
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found",
            )
        
        return dataset
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

@app.post("/datasets", dependencies=verify_auth())
async def create_dataset(dataset: DatasetCreate):
    """Create a new dataset"""
    try:
        dataset_id = peso_integration.collect_and_process_data(
            name=dataset.name,
            data_type=dataset.data_type,
            synthetic_count=dataset.synthetic_count
        )
        
        if dataset_id < 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create dataset",
            )
        
        return {
            "dataset_id": dataset_id,
            "message": f"Created dataset {dataset.name} with {dataset.synthetic_count} records"
        }
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

@app.post("/datasets/enrich", dependencies=verify_auth(), response_model=TaskResponse)
async def enrich_dataset(request: DatasetEnrich, background_tasks: BackgroundTasks):
    """Enrich a dataset using AI"""
    # Check if dataset exists
    dataset = peso_integration.db.get_dataset(request.dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {request.dataset_id} not found",
        )
    
    # Run enrichment in background task
    task_id = str(uuid.uuid4())
    
    background_tasks.add_task(
        enrich_dataset_task,
        request.dataset_id,
        request.ai_tool,
        task_id
    )
    
    return {
        "task_id": task_id,
        "message": f"Started enrichment of dataset {request.dataset_id} using {request.ai_tool}",
        "status": "processing"
    }

async def enrich_dataset_task(dataset_id: int, ai_tool: str, task_id: str):
    """Background task for dataset enrichment"""
    try:
        success = peso_integration.enrich_dataset_with_ai(dataset_id, ai_tool)
        
        # Store task status for later retrieval
        task_store = getattr(peso_integration, "task_store", {})
        task_store[task_id] = {
            "type": "enrich",
            "dataset_id": dataset_id,
            "ai_tool": ai_tool,
            "status": "completed" if success else "failed",
            "completed_at": datetime.now().isoformat()
        }
        
        logger.info(f"Enrichment task {task_id} completed with status: {success}")
    except Exception as e:
        # Store error information
        task_store = getattr(peso_integration, "task_store", {})
        task_store[task_id] = {
            "type": "enrich",
            "dataset_id": dataset_id,
            "ai_tool": ai_tool,
            "status": "error",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        }
        logger.error(f"Error in enrichment task {task_id}: {e}")