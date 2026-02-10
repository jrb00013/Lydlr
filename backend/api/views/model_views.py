"""
Model-related views
"""
import json
import os
from pathlib import Path
from datetime import datetime
from rest_framework.views import APIView
from rest_framework.response import Response

from backend.api.serializers import ModelInfoSerializer

MODEL_DIR = Path(os.getenv('MODEL_DIR', '/app/models'))


class ModelListView(APIView):
    """List all available models"""
    
    def get(self, request):
        """Get models - sync wrapper"""
        models = []
        
        if MODEL_DIR.exists():
            for pth_file in MODEL_DIR.glob("*.pth"):
                metadata_file = MODEL_DIR / f"metadata_{pth_file.stem.split('_v')[1]}.json"
                
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                models.append({
                    "version": pth_file.stem.split("_v")[1] if "_v" in pth_file.stem else "unknown",
                    "filename": pth_file.name,
                    "size_mb": pth_file.stat().st_size / (1024 * 1024),
                    "created_at": datetime.fromtimestamp(pth_file.stat().st_mtime).isoformat(),
                    "metadata": metadata
                })
        
        serializer = ModelInfoSerializer(models, many=True)
        return Response(serializer.data)

