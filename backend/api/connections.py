"""
Database and Redis connections
"""
import os
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as aioredis
from typing import Optional

# MongoDB client
mongodb_client: Optional[AsyncIOMotorClient] = None
db = None

# Redis client
redis_client: Optional[aioredis.Redis] = None
redis_pubsub = None


async def init_connections():
    """Initialize MongoDB and Redis connections"""
    global mongodb_client, db, redis_client, redis_pubsub
    
    try:
        # MongoDB
        mongodb_url = os.getenv(
            'MONGODB_URL',
            'mongodb://lydlr:lydlr_password@mongodb:27017/lydlr_db?authSource=admin'
        )
        mongodb_client = AsyncIOMotorClient(mongodb_url)
        db = mongodb_client.lydlr_db
        
        # Test MongoDB connection
        await mongodb_client.admin.command('ping')
        
        # Redis
        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        redis_client = aioredis.from_url(redis_url, decode_responses=True)
        redis_pubsub = redis_client.pubsub()
        
        # Test Redis connection
        await redis_client.ping()
        
        print("✅ Connected to MongoDB and Redis")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        db = None
        raise


async def close_connections():
    """Close all connections"""
    global mongodb_client, redis_client, redis_pubsub
    
    if redis_pubsub:
        await redis_pubsub.unsubscribe()
        await redis_pubsub.close()
    
    if redis_client:
        await redis_client.close()
    
    if mongodb_client:
        mongodb_client.close()

