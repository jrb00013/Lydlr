"""
Node-related views
"""
import json
import os
import asyncio
import logging
from datetime import datetime
from rest_framework.response import Response
from rest_framework import status

from backend.api.views.base import AsyncAPIView, ensure_db_connection
from backend.api.node_manager import (
    start_node, stop_node, restart_node, get_node_status,
    get_node_logs, deploy_model_to_node
)
from backend.api.ros2_metrics_collector import (
    start_metrics_collector, stop_metrics_collector
)
from backend.api.serializers import (
    NodeStatusSerializer, NodeConfigSerializer,
    NodeCreateSerializer, NodeConfigurationSerializer
)
from backend.api.redis_pubsub import publish_message

logger = logging.getLogger(__name__)


class NodeListView(AsyncAPIView):
    """List all nodes"""
    
    async def get(self, request):
        """Get all nodes"""
        db = await ensure_db_connection()
        nodes = await db.nodes.find().to_list(100)
        
        # Enrich with process status and latest metrics
        loop = asyncio.get_event_loop()
        for node in nodes:
            node_id = node.get('node_id')
            
            # Get process status
            process_status = await loop.run_in_executor(
                None, get_node_status, node_id
            )
            if process_status.get('status') == 'running':
                node['process_info'] = process_status
                node['status'] = 'running'
                
                # Start metrics collector if not already running
                try:
                    loop.run_in_executor(
                        None, start_metrics_collector, node_id,
                        os.getenv('API_URL', 'http://localhost:8000')
                    )
                except Exception as e:
                    logger.warning(f"Failed to start metrics collector for {node_id}: {e}")
            elif node.get('status') != 'running':
                node['status'] = process_status.get('status', node.get('status', 'stopped'))
            
            # Get latest metrics from database
            latest_metrics = await db.metrics.find_one(
                {"node_id": node_id},
                sort=[("timestamp", -1)]
            )
            if latest_metrics:
                node['compression_ratio'] = latest_metrics.get('compression_ratio', 0.0)
                node['latency_ms'] = latest_metrics.get('latency_ms', 0.0)
                node['quality_score'] = latest_metrics.get('quality_score', 0.0)
                node['bandwidth_estimate'] = latest_metrics.get('bandwidth_estimate', 0.0)
                node['compression_level'] = latest_metrics.get('compression_level', 0.0)
        
        serializer = NodeStatusSerializer(nodes, many=True)
        return Response(serializer.data)


class NodeDetailView(AsyncAPIView):
    """Get specific node"""
    
    async def get(self, request, node_id):
        """Get node"""
        db = await ensure_db_connection()
        node = await db.nodes.find_one({"node_id": node_id})
        if not node:
            return Response(
                {"detail": "Node not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        serializer = NodeStatusSerializer(node)
        return Response(serializer.data)
    
    async def post(self, request, node_id):
        """Update node configuration"""
        db = await ensure_db_connection()
        serializer = NodeConfigSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        result = await db.nodes.update_one(
            {"node_id": node_id},
            {"$set": {
                "config": serializer.validated_data,
                "updated_at": datetime.utcnow()
            }}
        )
        
        if result.matched_count == 0:
            return Response(
                {"detail": "Node not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Publish update via Redis Pub/Sub
        await publish_message('node_config_update', {
            "node_id": node_id,
            "config": serializer.validated_data
        })
        
        return Response({"status": "success"})


class NodeControlView(AsyncAPIView):
    """Control node commands (start, stop, restart) and status/logs"""
    
    async def post(self, request, node_id, action):
        """Execute control command (start, stop, restart)"""
        valid_actions = ['start', 'stop', 'restart']
        if action not in valid_actions:
            return Response(
                {"detail": f"Invalid action for POST. Must be one of: {valid_actions}. Use GET for 'status' or 'logs'"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Check if node exists in database
        db = await ensure_db_connection()
        node = await db.nodes.find_one({"node_id": node_id})
        if not node:
            return Response(
                {"detail": f"Node {node_id} not found. Please create the node first."},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Get optional model version for start/restart
        try:
            body = json.loads(request.body) if request.body else {}
            model_version = body.get('model_version') or node.get('model_version')
        except (json.JSONDecodeError, AttributeError):
            model_version = node.get('model_version')
        
        try:
            # Run node manager functions in executor since they're sync
            loop = asyncio.get_event_loop()
            
            if action == 'start':
                result = await loop.run_in_executor(None, start_node, node_id, model_version)
            elif action == 'stop':
                result = await loop.run_in_executor(None, stop_node, node_id)
            elif action == 'restart':
                result = await loop.run_in_executor(None, restart_node, node_id, model_version)
            
            # Update database status
            status_value = "running" if result.get("status") in ["started", "already_running"] else "stopped"
            update_data = {
                "status": status_value,
                "updated_at": datetime.utcnow(),
                "process_info": result
            }
            
            # Update model_version if it was provided
            if model_version and action in ['start', 'restart']:
                update_data["model_version"] = model_version
            
            await db.nodes.update_one(
                {"node_id": node_id},
                {"$set": update_data}
            )
            
            # Start/stop metrics collector based on action
            api_url = os.getenv('API_URL', 'http://localhost:8000')
            if action == 'start' and status_value == "running":
                # Start metrics collector in background
                loop.run_in_executor(None, start_metrics_collector, node_id, api_url)
            elif action == 'stop':
                # Stop metrics collector
                loop.run_in_executor(None, stop_metrics_collector, node_id)
        
            # Publish command via Redis Pub/Sub
            try:
               await publish_message('node_command', {
                    "node_id": node_id,
                    "command": action,
                    "result": result
                })
            except Exception as pub_error:
                # Log but don't fail if Redis is unavailable
                logger.warning(f"Failed to publish node command to Redis: {pub_error}")
            
            if result.get("status") == "error":
                error_msg = result.get("error", "Unknown error")
                return Response(
                    {"detail": error_msg, **result},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            return Response(result)
        
        except Exception as e:
            logger.error(f"Failed to {action} node {node_id}: {str(e)}", exc_info=True)
            return Response(
                {"detail": f"Failed to {action} node: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    async def get(self, request, node_id, action):
        """Get node status or logs"""
        loop = asyncio.get_event_loop()
        
        valid_get_actions = ['status', 'logs']
        if action not in valid_get_actions:
            return Response(
                {"detail": f"Invalid action for GET. Must be one of: {valid_get_actions}. Use POST for 'start', 'stop', or 'restart'"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if action == 'status':
            result = await loop.run_in_executor(None, get_node_status, node_id)
            return Response(result)
        elif action == 'logs':
            lines = int(request.query_params.get('lines', 100))
            logs = await loop.run_in_executor(None, get_node_logs, node_id, lines)
            return Response({
                "node_id": node_id,
                "logs": logs,
                "line_count": len(logs)
            })


class NodeDeployView(AsyncAPIView):
    """Deploy model to a specific node"""
    
    async def post(self, request, node_id):
        """Deploy model to node"""
        model_version = request.data.get('model_version')
        if not model_version:
            return Response(
                {"detail": "model_version is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        db = await ensure_db_connection()
        
        # Verify node exists
        node = await db.nodes.find_one({"node_id": node_id})
        if not node:
            return Response(
                {"detail": "Node not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Check if node is running
        loop = asyncio.get_event_loop()
        process_status = await loop.run_in_executor(None, get_node_status, node_id)
        if process_status.get("status") != "running":
            return Response(
                {"detail": f"Node {node_id} is not running. Please start the node first."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Deploy model (run in executor)
        success = await loop.run_in_executor(None, deploy_model_to_node, node_id, model_version)
        
        if success:
            # Update node record
            await db.nodes.update_one(
                {"node_id": node_id},
                {"$set": {
                    "model_version": model_version,
                    "updated_at": datetime.utcnow()
                }}
            )
            
            # Publish deployment event
            await publish_message('node_deploy', {
                "node_id": node_id,
                "model_version": model_version,
                "status": "success"
            })
            
            return Response({
                "status": "success",
                "message": f"Model {model_version} deployed to {node_id}",
                "node_id": node_id,
                "model_version": model_version
            })
        else:
            return Response(
                {"detail": "Failed to deploy model. Check ROS2 connection and node status."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class NodeCreateView(AsyncAPIView):
    """Create a new node dynamically"""
    
    async def post(self, request):
        """Create node"""
        try:
            db = await ensure_db_connection()
        except Exception as e:
            return Response(
                {"detail": f"Database connection failed: {str(e)}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        # Safety check - ensure db is not None
        if db is None:
            return Response(
                {"detail": "Database connection is None"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        serializer = NodeCreateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # Generate node_id if not provided
        node_id = serializer.validated_data.get('node_id')
        if not node_id:
            # Find next available node ID
            existing_nodes = await db.nodes.find({}, {"node_id": 1}).to_list(100)
            existing_ids = {n.get('node_id', '') for n in existing_nodes}
            node_num = 0
            while f"node_{node_num}" in existing_ids:
                node_num += 1
            node_id = f"node_{node_num}"
        
        # Check if node already exists
        existing = await db.nodes.find_one({"node_id": node_id})
        if existing:
            return Response(
                {"detail": f"Node {node_id} already exists"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create node document
        node_doc = {
            "node_id": node_id,
            "node_type": serializer.validated_data.get('node_type', 'edge_compressor'),
            "status": "pending",
            "config": serializer.validated_data.get('config', {}),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        await db.nodes.insert_one(node_doc)
        
        # Publish node creation via Redis Pub/Sub
        await publish_message('node_create', {
            "node_id": node_id,
            "node_type": node_doc['node_type'],
            "config": node_doc['config']
        })
        
        return Response({
            "status": "success",
            "node_id": node_id,
            "message": f"Node {node_id} created. Starting node..."
        }, status=status.HTTP_201_CREATED)
    
    async def delete(self, request, node_id):
        """Delete node"""
        db = await ensure_db_connection()
        
        # Check if node exists
        node = await db.nodes.find_one({"node_id": node_id})
        if not node:
            return Response(
                {"detail": "Node not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Delete node
        await db.nodes.delete_one({"node_id": node_id})
        
        # Publish node deletion via Redis Pub/Sub
        await publish_message('node_delete', {
            "node_id": node_id
        })
        
        return Response({
            "status": "success",
            "message": f"Node {node_id} deleted"
        })


class NodeConfigurationView(AsyncAPIView):
    """Manage system-wide node configuration"""
    
    async def get(self, request):
        """Get current node configuration"""
        db = await ensure_db_connection()
        
        # Get configuration from system_config collection
        config = await db.system_config.find_one({"type": "node_configuration"})
        if not config:
            # Return defaults
            config = {
                "target_node_count": 2,
                "auto_scale": False,
                "min_nodes": 1,
                "max_nodes": 10
            }
        else:
            config = {k: v for k, v in config.items() if k != '_id' and k != 'type'}
        
        serializer = NodeConfigurationSerializer(config)
        return Response(serializer.data)
    
    async def post(self, request):
        """Update node configuration"""
        db = await ensure_db_connection()
        
        serializer = NodeConfigurationSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        config_data = serializer.validated_data
        config_data['type'] = 'node_configuration'
        config_data['updated_at'] = datetime.utcnow()
        
        # Upsert configuration
        await db.system_config.update_one(
            {"type": "node_configuration"},
            {"$set": config_data},
            upsert=True
        )
        
        # Publish configuration update via Redis Pub/Sub
        await publish_message('node_configuration_update', config_data)
        
        # Auto-scale nodes if enabled
        if config_data.get('auto_scale', False):
            current_count = await db.nodes.count_documents({})
            target_count = config_data.get('target_node_count', 2)
            
            if current_count < target_count:
                # Create missing nodes
                for i in range(current_count, target_count):
                    node_id = f"node_{i}"
                    existing = await db.nodes.find_one({"node_id": node_id})
                    if not existing:
                        await db.nodes.insert_one({
                            "node_id": node_id,
                            "node_type": "edge_compressor",
                            "status": "pending",
                            "created_at": datetime.utcnow()
                        })
                        await publish_message('node_create', {
                            "node_id": node_id,
                            "node_type": "edge_compressor"
                        })
            elif current_count > target_count:
                # Remove excess nodes (remove highest numbered ones first)
                excess = current_count - target_count
                nodes_to_remove = await db.nodes.find().sort("node_id", -1).limit(excess).to_list(excess)
                for node in nodes_to_remove:
                    await db.nodes.delete_one({"node_id": node['node_id']})
                    await publish_message('node_delete', {
                        "node_id": node['node_id']
                    })
        
        return Response({
            "status": "success",
            "configuration": config_data
        })

