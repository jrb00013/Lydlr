"""
Metrics and statistics views
"""
import json
import logging
from datetime import datetime
from rest_framework.response import Response
from rest_framework import status

from backend.api.views.base import AsyncAPIView, ensure_db_connection
from backend.api.serializers import CompressionMetricsSerializer, SystemStatsSerializer
from backend.api.connections import redis_client
from backend.api.redis_pubsub import publish_message

logger = logging.getLogger(__name__)


class MetricsView(AsyncAPIView):
    """Get and store metrics"""
    
    async def get(self, request):
        """Get metrics"""
        node_id = request.query_params.get('node_id')
        limit = int(request.query_params.get('limit', 100))
        db = await ensure_db_connection()
        query = {"node_id": node_id} if node_id else {}
        metrics = await db.metrics.find(query).sort("timestamp", -1).limit(limit).to_list(limit)
        serializer = CompressionMetricsSerializer(metrics, many=True)
        return Response(serializer.data)
    
    async def post(self, request):
        """Store metrics"""
        db = await ensure_db_connection()
        serializer = CompressionMetricsSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        await db.metrics.insert_one(serializer.validated_data)
        
        # Cache in Redis for real-time access
        await redis_client.setex(
            f"metrics:{serializer.validated_data['node_id']}:latest",
            60,  # Expire after 60 seconds
            json.dumps(serializer.validated_data)
        )
        
        # Publish metrics update via Redis Pub/Sub
        await publish_message('metrics_update', serializer.validated_data)
        
        return Response({"status": "success"})


class SystemStatsView(AsyncAPIView):
    """Get overall system statistics"""
    
    async def get(self, request):
        """Get stats"""
        db = await ensure_db_connection()
        total_nodes = await db.nodes.count_documents({})
        active_nodes = await db.nodes.count_documents({"status": "active"})
        
        # Get recent metrics
        recent_metrics = await db.metrics.find().sort("timestamp", -1).limit(100).to_list(100)
        
        if recent_metrics:
            avg_compression = sum(m.get("compression_ratio", 0) for m in recent_metrics) / len(recent_metrics)
            avg_latency = sum(m.get("latency_ms", 0) for m in recent_metrics) / len(recent_metrics)
            avg_quality = sum(m.get("quality_score", 0) for m in recent_metrics) / len(recent_metrics)
        else:
            avg_compression = avg_latency = avg_quality = 0
        
        stats = {
            "total_nodes": total_nodes,
            "active_nodes": active_nodes,
            "avg_compression_ratio": avg_compression,
            "avg_latency_ms": avg_latency,
            "avg_quality_score": avg_quality,
            "timestamp": datetime.utcnow()
        }
        
        serializer = SystemStatsSerializer(stats)
        return Response(serializer.data)

