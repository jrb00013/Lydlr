#!/usr/bin/env python3
"""
Helper script to get node configuration from MongoDB/Redis
Used by launch_lydlr_system.sh to dynamically determine which nodes to start
"""
import os
import sys
import json
import asyncio

try:
    from motor.motor_asyncio import AsyncIOMotorClient
except ImportError:
    # If motor is not available, return empty result
    print(json.dumps({"nodes": [], "config": None}))
    sys.exit(0)

# MongoDB connection
MONGODB_URL = os.getenv(
    'MONGODB_URL',
    'mongodb://lydlr:lydlr_password@mongodb:27017/lydlr_db?authSource=admin'
)

async def get_nodes():
    """Get list of nodes from MongoDB"""
    try:
        client = AsyncIOMotorClient(MONGODB_URL)
        db = client.lydlr_db
        
        # Get all nodes
        nodes = await db.nodes.find({}, {"node_id": 1, "status": 1}).to_list(100)
        node_ids = [node['node_id'] for node in nodes]
        
        # Also check system configuration for target node count
        config = await db.system_config.find_one({"type": "node_configuration"})
        
        client.close()
        
        return {
            "nodes": node_ids,
            "config": config if config else None
        }
    except Exception as e:
        print(f"Error getting nodes: {e}", file=sys.stderr)
        return {"nodes": [], "config": None}

if __name__ == "__main__":
    result = asyncio.run(get_nodes())
    print(json.dumps(result))

