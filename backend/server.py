from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json
import io
import base64
from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Pydantic Models
class CustomerData(BaseModel):
    customer_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    age: float
    annual_income: float
    spending_score: float
    gender: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now())

class CustomerDataCreate(BaseModel):
    age: float
    annual_income: float
    spending_score: float
    gender: Optional[str] = None

class ClusteringRequest(BaseModel):
    algorithm: str  # 'kmeans' or 'hierarchical'
    n_clusters: int
    features: List[str]  # ['age', 'annual_income', 'spending_score']

class ClusterResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    algorithm: str
    n_clusters: int
    features: List[str]
    silhouette_score: float
    cluster_centers: Optional[List[List[float]]] = None
    cluster_labels: List[int]
    inertia: Optional[float] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now())

class CustomerSegment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    cluster_id: int
    label: str
    description: str
    customer_count: int
    avg_age: float
    avg_income: float
    avg_spending: float
    business_recommendations: List[str]
    created_at: datetime = Field(default_factory=lambda: datetime.now())

class ElbowAnalysis(BaseModel):
    k_values: List[int]
    inertias: List[float]
    silhouette_scores: List[float]

# Helper Functions
def prepare_for_mongo(data):
    """Convert numpy types to Python native types for MongoDB storage"""
    if isinstance(data, dict):
        return {k: prepare_for_mongo(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [prepare_for_mongo(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)
    else:
        return data

async def get_ai_insights(segment_data: Dict[str, Any]) -> List[str]:
    """Generate AI-powered business recommendations for customer segments"""
    try:
        # Initialize LLM chat with system message
        chat = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id=f"segment-analysis-{uuid.uuid4()}",
            system_message="You are a senior business analyst specializing in customer segmentation and marketing strategy. Provide actionable, specific business recommendations based on customer segment data."
        ).with_model("openai", "gpt-4o-mini")
        
        # Create detailed prompt with segment characteristics
        prompt = f"""
        Analyze this customer segment and provide 3-5 specific, actionable business recommendations:
        
        Segment Profile:
        - Label: {segment_data.get('label', 'Unknown')}
        - Customer Count: {segment_data.get('customer_count', 0)} customers
        - Average Age: {segment_data.get('avg_age', 0):.1f} years
        - Average Annual Income: ₹{segment_data.get('avg_income', 0):,.0f}
        - Average Spending Score: {segment_data.get('avg_spending', 0):.1f}/100
        
        Focus on:
        1. Marketing strategies tailored to this segment
        2. Product/service recommendations
        3. Pricing strategies
        4. Communication channels
        5. Retention/acquisition tactics
        
        Provide each recommendation as a complete sentence, actionable and specific to this segment's characteristics.
        """
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        
        # Parse the response into a list of recommendations
        recommendations = []
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and len(line) > 20:
                # Remove bullet points or numbering
                clean_line = line.lstrip('•-*1234567890. ')
                if clean_line:
                    recommendations.append(clean_line)
        
        return recommendations[:5] if recommendations else [
            "Develop targeted marketing campaigns for this customer segment",
            "Consider personalized product recommendations",
            "Implement segment-specific pricing strategies"
        ]
        
    except Exception as e:
        logging.error(f"Error generating AI insights: {e}")
        return [
            "Develop targeted marketing campaigns for this customer segment",
            "Consider personalized product recommendations based on spending patterns",
            "Implement segment-specific communication strategies"
        ]

# Sample data generation
def generate_sample_data() -> pd.DataFrame:
    """Generate sample customer data similar to Mall Customer Segmentation dataset"""
    np.random.seed(42)
    n_customers = 200
    
    # Generate demographic data
    ages = np.random.normal(35, 12, n_customers)
    ages = np.clip(ages, 18, 70)
    
    # Generate income data (correlated with age)
    base_income = 30000 + (ages - 18) * 1000 + np.random.normal(0, 15000, n_customers)
    annual_income = np.clip(base_income, 15000, 137000)
    
    # Generate spending scores (influenced by income and age)
    income_factor = (annual_income - annual_income.min()) / (annual_income.max() - annual_income.min())
    age_factor = 1 - (ages - 18) / (70 - 18)  # Younger people tend to spend more
    spending_scores = 20 + income_factor * 40 + age_factor * 30 + np.random.normal(0, 10, n_customers)
    spending_scores = np.clip(spending_scores, 1, 100)
    
    # Generate gender
    genders = np.random.choice(['Male', 'Female'], n_customers)
    
    df = pd.DataFrame({
        'Age': ages.round(0).astype(int),
        'Annual_Income': annual_income.round(0).astype(int),
        'Spending_Score': spending_scores.round(0).astype(int),
        'Gender': genders
    })
    
    return df

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Customer Segmentation & Targeting API"}

# Data Management Routes
@api_router.post("/data/sample")
async def generate_sample_customer_data():
    """Generate and store sample customer data"""
    try:
        # Clear existing data
        await db.customers.delete_many({})
        
        # Generate sample data
        df = generate_sample_data()
        
        # Convert to records and store in database
        customers = []
        for _, row in df.iterrows():
            customer = CustomerData(
                age=float(row['Age']),
                annual_income=float(row['Annual_Income']),
                spending_score=float(row['Spending_Score']),
                gender=str(row['Gender'])
            )
            customers.append(customer.dict())
        
        # Insert into database
        await db.customers.insert_many(customers)
        
        return {
            "message": f"Generated and stored {len(customers)} sample customers",
            "count": len(customers)
        }
    except Exception as e:
        logging.error(f"Error generating sample data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/data/customers", response_model=List[CustomerData])
async def get_customers():
    """Get all customer data"""
    try:
        customers = await db.customers.find().to_list(1000)
        return [CustomerData(**customer) for customer in customers]
    except Exception as e:
        logging.error(f"Error fetching customers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/data/stats")
async def get_data_statistics():
    """Get basic statistics about the customer data"""
    try:
        customers = await db.customers.find().to_list(1000)
        if not customers:
            return {"message": "No customer data found"}
        
        df = pd.DataFrame(customers)
        
        stats = {
            "total_customers": len(df),
            "age_stats": {
                "mean": float(df['age'].mean()),
                "std": float(df['age'].std()),
                "min": float(df['age'].min()),
                "max": float(df['age'].max())
            },
            "income_stats": {
                "mean": float(df['annual_income'].mean()),
                "std": float(df['annual_income'].std()),
                "min": float(df['annual_income'].min()),
                "max": float(df['annual_income'].max())
            },
            "spending_stats": {
                "mean": float(df['spending_score'].mean()),
                "std": float(df['spending_score'].std()),
                "min": float(df['spending_score'].min()),
                "max": float(df['spending_score'].max())
            },
            "gender_distribution": df['gender'].value_counts().to_dict() if 'gender' in df.columns else {}
        }
        
        return stats
    except Exception as e:
        logging.error(f"Error calculating statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Clustering Analysis Routes
@api_router.post("/analysis/elbow")
async def elbow_analysis(features: List[str] = ["age", "annual_income", "spending_score"]):
    """Perform elbow analysis to find optimal number of clusters"""
    try:
        customers = await db.customers.find().to_list(1000)
        if not customers:
            raise HTTPException(status_code=404, detail="No customer data found")
        
        df = pd.DataFrame(customers)
        
        # Prepare feature data
        X = df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform elbow analysis
        k_range = range(2, 11)
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(float(kmeans.inertia_))
            silhouette_scores.append(float(silhouette_score(X_scaled, kmeans.labels_)))
        
        result = ElbowAnalysis(
            k_values=list(k_range),
            inertias=inertias,
            silhouette_scores=silhouette_scores
        )
        
        return result
    except Exception as e:
        logging.error(f"Error in elbow analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analysis/cluster", response_model=ClusterResult)
async def perform_clustering(request: ClusteringRequest):
    """Perform customer clustering analysis"""
    try:
        customers = await db.customers.find().to_list(1000)
        if not customers:
            raise HTTPException(status_code=404, detail="No customer data found")
        
        df = pd.DataFrame(customers)
        
        # Prepare feature data
        X = df[request.features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        if request.algorithm == 'kmeans':
            model = KMeans(n_clusters=request.n_clusters, random_state=42, n_init=10)
            model.fit(X_scaled)
            cluster_centers = model.cluster_centers_.tolist()
            inertia = float(model.inertia_)
        elif request.algorithm == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=request.n_clusters)
            model.fit(X_scaled)
            cluster_centers = None
            inertia = None
        else:
            raise HTTPException(status_code=400, detail="Invalid algorithm. Use 'kmeans' or 'hierarchical'")
        
        labels = model.labels_.tolist()
        sil_score = float(silhouette_score(X_scaled, model.labels_))
        
        # Create result
        result = ClusterResult(
            algorithm=request.algorithm,
            n_clusters=request.n_clusters,
            features=request.features,
            silhouette_score=sil_score,
            cluster_centers=cluster_centers,
            cluster_labels=labels,
            inertia=inertia
        )
        result_dict = prepare_for_mongo(result.dict())
        
        # Store result in database
        await db.cluster_results.insert_one(result_dict)
        
        # Update customer records with cluster labels
        for i, customer in enumerate(customers):
            await db.customers.update_one(
                {"customer_id": customer["customer_id"]},
                {"$set": {"cluster_id": int(labels[i])}}
            )
        
        return result
    except Exception as e:
        logging.error(f"Error performing clustering: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/analysis/segments", response_model=List[CustomerSegment])
async def get_customer_segments():
    """Get customer segments with AI-powered business recommendations"""
    try:
        customers = await db.customers.find({"cluster_id": {"$exists": True}}).to_list(1000)
        if not customers:
            raise HTTPException(status_code=404, detail="No clustered customer data found")
        
        df = pd.DataFrame(customers)
        segments = []
        
        # Group by cluster
        for cluster_id in df['cluster_id'].unique():
            cluster_data = df[df['cluster_id'] == cluster_id]
            
            # Calculate cluster statistics
            avg_age = float(cluster_data['age'].mean())
            avg_income = float(cluster_data['annual_income'].mean())
            avg_spending = float(cluster_data['spending_score'].mean())
            customer_count = len(cluster_data)
            
            # Generate cluster label based on characteristics
            if avg_income > df['annual_income'].mean() and avg_spending > df['spending_score'].mean():
                label = "High Value Customers"
                description = "High income, high spending customers with premium preferences"
            elif avg_income < df['annual_income'].mean() and avg_spending < df['spending_score'].mean():
                label = "Budget Conscious"
                description = "Price-sensitive customers with lower income and spending"
            elif avg_income > df['annual_income'].mean() and avg_spending < df['spending_score'].mean():
                label = "Careful Spenders"
                description = "High income but conservative spending behavior"
            elif avg_income < df['annual_income'].mean() and avg_spending > df['spending_score'].mean():
                label = "Aspirational Shoppers"
                description = "Lower income but high spending, possibly credit-driven purchases"
            else:
                label = f"Segment {cluster_id + 1}"
                description = "Mid-range customer segment with balanced characteristics"
            
            # Prepare data for AI insights
            segment_data = {
                "cluster_id": int(cluster_id),
                "label": label,
                "customer_count": customer_count,
                "avg_age": avg_age,
                "avg_income": avg_income,
                "avg_spending": avg_spending
            }
            
            # Generate AI-powered recommendations
            recommendations = await get_ai_insights(segment_data)
            
            segment = CustomerSegment(
                cluster_id=int(cluster_id),
                label=label,
                description=description,
                customer_count=customer_count,
                avg_age=avg_age,
                avg_income=avg_income,
                avg_spending=avg_spending,
                business_recommendations=recommendations
            )
            segments.append(segment)
        
        # Store segments in database
        await db.segments.delete_many({})  # Clear old segments
        for segment in segments:
            await db.segments.insert_one(segment.dict())
        
        return segments
    except Exception as e:
        logging.error(f"Error generating segments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/analysis/cluster-data/{cluster_id}")
async def get_cluster_customers(cluster_id: int):
    """Get all customers in a specific cluster"""
    try:
        customers = await db.customers.find({"cluster_id": cluster_id}).to_list(1000)
        return {"cluster_id": cluster_id, "customers": customers, "count": len(customers)}
    except Exception as e:
        logging.error(f"Error fetching cluster customers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()