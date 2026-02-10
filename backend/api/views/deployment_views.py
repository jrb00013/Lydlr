"""
Deployment-related views
"""
import logging
import shutil
from pathlib import Path
import os
from datetime import datetime
from rest_framework.response import Response
from rest_framework import status

from backend.api.views.base import AsyncAPIView, ensure_db_connection
from backend.api.serializers import DeploymentRequestSerializer
from backend.api.redis_pubsub import publish_message

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv('MODEL_DIR', '/app/models'))


class DeploymentView(AsyncAPIView):
    """Deploy model to nodes"""
    
    async def post(self, request):
        """Deploy"""
        db = await ensure_db_connection()
        serializer = DeploymentRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        model_version = serializer.validated_data['model_version']
        node_ids = serializer.validated_data['node_ids']
        
        # Validate that all nodes exist
        existing_nodes = await db.nodes.find(
            {"node_id": {"$in": node_ids}},
            {"node_id": 1}
        ).to_list(len(node_ids))
        existing_node_ids = {node['node_id'] for node in existing_nodes}
        missing_nodes = set(node_ids) - existing_node_ids
        
        if missing_nodes:
            return Response(
                {
                    "detail": f"Nodes not found: {', '.join(missing_nodes)}",
                    "missing_nodes": list(missing_nodes)
                },
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Validate that model file exists
        # Try different naming conventions
        model_filename = None
        metadata_filename = None
        potential_model = None
        
        # Normalize version (remove 'v' prefix if present)
        normalized_version = model_version.lstrip('v') if model_version.startswith('v') else model_version
        
        # Try different filename patterns
        search_patterns = [
            (f"lydlr_compressor_v{normalized_version}.pth", f"metadata_lydlr_compressor_v{normalized_version}.json"),
            (f"compressor_v{normalized_version}.pth", f"metadata_{normalized_version}.json"),
            (f"{normalized_version}.pth", f"metadata_{normalized_version}.json"),
            # Also try with 'v' prefix in filename
            (f"lydlr_compressor_v{model_version}.pth", f"metadata_lydlr_compressor_v{model_version}.json"),
            (f"compressor_v{model_version}.pth", f"metadata_{model_version}.json"),
            (f"{model_version}.pth", f"metadata_{model_version}.json"),
        ]
        
        for pattern_model, pattern_metadata in search_patterns:
            test_path = MODEL_DIR / pattern_model
            if test_path.exists():
                model_filename = pattern_model
                metadata_filename = pattern_metadata
                potential_model = test_path
                break
        
        # If still not found, search all .pth files for matching version
        if not model_filename and MODEL_DIR.exists():
            for pth_file in MODEL_DIR.glob("*.pth"):
                # Extract version from filename (similar to ModelListView logic)
                if "_v" in pth_file.stem:
                    file_version = pth_file.stem.split("_v")[1]
                    # Compare normalized versions
                    file_version_normalized = file_version.lstrip('v') if file_version.startswith('v') else file_version
                    if file_version_normalized == normalized_version or file_version == model_version:
                        model_filename = pth_file.name
                        # Try to find corresponding metadata
                        for json_file in MODEL_DIR.glob(f"metadata*{file_version}*.json"):
                            metadata_filename = json_file.name
                            break
                        if not metadata_filename:
                            metadata_filename = f"metadata_{file_version}.json"
                        potential_model = pth_file
                        break
        
        if not model_filename or not potential_model or not potential_model.exists():
            searched_paths = [str(MODEL_DIR / pattern[0]) for pattern in search_patterns]
            return Response(
                {
                    "detail": f"Model file not found for version {model_version}",
                    "searched_paths": searched_paths,
                    "available_models": [f.name for f in MODEL_DIR.glob("*.pth")] if MODEL_DIR.exists() else []
                },
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Create deployment record
        deployment = {
            "model_version": model_version,
            "node_ids": node_ids,
            "deployed_at": datetime.utcnow(),
            "status": "deploying"
        }
        
        result = await db.deployments.insert_one(deployment)
        deployment_id = str(result.inserted_id)
        
        # Deploy to each node
        successful_nodes = []
        failed_nodes = []
        
        for node_id in node_ids:
            try:
                # Create node model directory
                node_model_dir = MODEL_DIR / node_id
                node_model_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy model file
                src_model = MODEL_DIR / model_filename
                dst_model = node_model_dir / model_filename
                shutil.copy2(src_model, dst_model)
                
                # Copy metadata file if it exists
                src_metadata = MODEL_DIR / metadata_filename
                if src_metadata.exists():
                    dst_metadata = node_model_dir / metadata_filename
                    shutil.copy2(src_metadata, dst_metadata)
                
                # Update node record with new model version
                await db.nodes.update_one(
                    {"node_id": node_id},
                    {
                        "$set": {
                            "model_version": model_version,
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
                
                successful_nodes.append(node_id)
            except Exception as e:
                failed_nodes.append({"node_id": node_id, "error": str(e)})
        
        # Update deployment status
        if failed_nodes:
            deployment_status = "partial" if successful_nodes else "failed"
            error_details = {node["node_id"]: node["error"] for node in failed_nodes}
        else:
            deployment_status = "success"
            error_details = None
        
        await db.deployments.update_one(
            {"_id": result.inserted_id},
            {
                "$set": {
                    "status": deployment_status,
                    "successful_nodes": successful_nodes,
                    "failed_nodes": [node["node_id"] for node in failed_nodes] if failed_nodes else [],
                    "error_details": error_details,
                    "completed_at": datetime.utcnow()
                }
            }
        )
        
        # Publish deployment command via Redis Pub/Sub
        await publish_message('deployment', {
            "deployment_id": deployment_id,
            "model_version": model_version,
            "node_ids": node_ids,
            "status": deployment_status,
            "successful_nodes": successful_nodes,
            "failed_nodes": [node["node_id"] for node in failed_nodes] if failed_nodes else []
        })
        
        if failed_nodes:
            return Response(
                {
                    "deployment_id": deployment_id,
                    "status": deployment_status,
                    "successful_nodes": successful_nodes,
                    "failed_nodes": failed_nodes,
                    "message": f"Deployment completed with {len(failed_nodes)} failure(s)"
                },
                status=status.HTTP_207_MULTI_STATUS if successful_nodes else status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        return Response({
            "deployment_id": deployment_id,
            "status": deployment_status,
            "successful_nodes": successful_nodes,
            "message": f"Model {model_version} successfully deployed to {len(successful_nodes)} node(s)"
        })
    
    async def get(self, request):
        """Get deployments"""
        db = await ensure_db_connection()
        deployments = await db.deployments.find().sort("deployed_at", -1).limit(50).to_list(50)
        return Response(deployments)

