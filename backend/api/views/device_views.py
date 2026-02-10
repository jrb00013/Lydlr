"""
Device-related views
"""
import logging
from datetime import datetime
from rest_framework.response import Response
from rest_framework import status

from backend.api.views.base import AsyncAPIView, ensure_db_connection
from backend.api.serializers import (
    DeviceSerializer, DeviceCreateSerializer, SensorSerializer
)
from backend.api.device_topics import generate_device_topics
from backend.api.redis_pubsub import publish_message

logger = logging.getLogger(__name__)


class DeviceListView(AsyncAPIView):
    """List all devices"""
    
    async def get(self, request):
        """Get all devices"""
        db = await ensure_db_connection()
        devices = await db.devices.find().to_list(100)
        # Convert ObjectId to string and enrich with sensors
        for device in devices:
            device['_id'] = str(device['_id'])
            # Get sensors for this device
            sensors = await db.sensors.find({"device_id": device['device_id']}).to_list(50)
            device['sensors'] = [dict(s, _id=str(s['_id'])) for s in sensors]
        serializer = DeviceSerializer(devices, many=True)
        return Response(serializer.data)


class DeviceDetailView(AsyncAPIView):
    """Get specific device"""
    
    async def get(self, request, device_id):
        """Get device"""
        db = await ensure_db_connection()
        device = await db.devices.find_one({"device_id": device_id})
        if not device:
            return Response(
                {"detail": "Device not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        device['_id'] = str(device['_id'])
        # Get sensors for this device
        sensors = await db.sensors.find({"device_id": device_id}).to_list(50)
        device['sensors'] = [dict(s, _id=str(s['_id'])) for s in sensors]
        serializer = DeviceSerializer(device)
        return Response(serializer.data)
    
    async def post(self, request, device_id):
        """Update device"""
        db = await ensure_db_connection()
        
        # Get existing device
        device = await db.devices.find_one({"device_id": device_id})
        if not device:
            return Response(
                {"detail": "Device not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Update device
        update_data = {
            "last_update": datetime.utcnow()
        }
        
        # Check if device type changed (will need to regenerate topics)
        regenerate_topics = False
        if 'device_type' in request.data and request.data['device_type'] != device.get('device_type'):
            update_data['device_type'] = request.data['device_type']
            regenerate_topics = True
        
        if 'device_name' in request.data:
            update_data['device_name'] = request.data['device_name']
        if 'node_id' in request.data:
            update_data['node_id'] = request.data['node_id']
        if 'status' in request.data:
            update_data['status'] = request.data['status']
        if 'ip_address' in request.data:
            update_data['ip_address'] = request.data['ip_address']
        if 'location' in request.data:
            update_data['location'] = request.data['location']
        if 'metadata' in request.data:
            update_data['metadata'] = request.data['metadata']
        
        # Regenerate topics if device type changed or if requested
        if regenerate_topics or request.data.get('regenerate_topics', False):
            # Get current sensors
            sensors = await db.sensors.find({"device_id": device_id}).to_list(50)
            device_data = {
                "device_id": device_id,
                "device_name": update_data.get('device_name', device.get('device_name', device_id)),
                "device_type": update_data.get('device_type', device.get('device_type', 'camera')),
                "sensors": [{"sensor_id": s.get('sensor_id'), "sensor_type": s.get('sensor_type')} for s in sensors]
            }
            topics = generate_device_topics(device_data)
            update_data['pub_topics'] = topics["pub_topics"]
            update_data['sub_topics'] = topics["sub_topics"]
        
        await db.devices.update_one(
            {"device_id": device_id},
            {"$set": update_data}
        )
        
        # Publish update via Redis Pub/Sub
        await publish_message('device_update', {
            "device_id": device_id,
            "update": update_data
        })
        
        return Response({"status": "success"})


class DeviceCreateView(AsyncAPIView):
    """Create a new device"""
    
    async def post(self, request):
        """Create device"""
        db = await ensure_db_connection()
        
        serializer = DeviceCreateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # Generate device_id if not provided
        device_id = serializer.validated_data.get('device_id')
        if not device_id:
            existing_devices = await db.devices.find({}, {"device_id": 1}).to_list(100)
            existing_ids = {d.get('device_id', '') for d in existing_devices}
            device_num = 0
            while f"device_{device_num}" in existing_ids:
                device_num += 1
            device_id = f"device_{device_num}"
        
        # Check if device already exists
        existing = await db.devices.find_one({"device_id": device_id})
        if existing:
            return Response(
                {"detail": f"Device {device_id} already exists"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Generate topics based on device type
        device_data = {
            "device_id": device_id,
            "device_name": serializer.validated_data.get('device_name', device_id),
            "device_type": serializer.validated_data.get('device_type', 'camera'),
            "sensors": []
        }
        topics = generate_device_topics(device_data)
        
        # Create device document
        device_doc = {
            "device_id": device_id,
            "device_name": serializer.validated_data.get('device_name', device_id),
            "device_type": serializer.validated_data.get('device_type', 'camera'),
            "node_id": serializer.validated_data.get('node_id'),
            "status": "active",
            "ip_address": serializer.validated_data.get('ip_address'),
            "location": serializer.validated_data.get('location'),
            "created_at": datetime.utcnow(),
            "last_update": datetime.utcnow(),
            "sensors": [],
            "metadata": serializer.validated_data.get('metadata', {}),
            "pub_topics": topics["pub_topics"],
            "sub_topics": topics["sub_topics"]
        }
        
        await db.devices.insert_one(device_doc)
        
        # Publish device creation via Redis Pub/Sub
        await publish_message('device_create', {
            "device_id": device_id,
            "device_type": device_doc['device_type'],
            "node_id": device_doc.get('node_id')
        })
        
        return Response({
            "status": "success",
            "device_id": device_id,
            "message": f"Device {device_id} created"
        }, status=status.HTTP_201_CREATED)
    
    async def delete(self, request, device_id):
        """Delete device"""
        db = await ensure_db_connection()
        
        # Check if device exists
        device = await db.devices.find_one({"device_id": device_id})
        if not device:
            return Response(
                {"detail": "Device not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Delete device and its sensors
        await db.devices.delete_one({"device_id": device_id})
        await db.sensors.delete_many({"device_id": device_id})
        
        # Publish device deletion via Redis Pub/Sub
        await publish_message('device_delete', {
            "device_id": device_id
        })
        
        return Response({
            "status": "success",
            "message": f"Device {device_id} deleted"
        })


class SensorListView(AsyncAPIView):
    """List sensors for a device"""
    
    async def get(self, request):
        """Get sensors"""
        device_id = request.query_params.get('device_id')
        db = await ensure_db_connection()
        query = {"device_id": device_id} if device_id else {}
        sensors = await db.sensors.find(query).to_list(100)
        for sensor in sensors:
            sensor['_id'] = str(sensor['_id'])
        serializer = SensorSerializer(sensors, many=True)
        return Response(serializer.data)
    
    async def post(self, request):
        """Create sensor"""
        db = await ensure_db_connection()
        
        serializer = SensorSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # Check if device exists
        device = await db.devices.find_one({"device_id": serializer.validated_data['device_id']})
        if not device:
            return Response(
                {"detail": "Device not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Check if sensor already exists
        existing = await db.sensors.find_one({
            "device_id": serializer.validated_data['device_id'],
            "sensor_id": serializer.validated_data['sensor_id']
        })
        if existing:
            return Response(
                {"detail": "Sensor already exists"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        sensor_doc = serializer.validated_data.copy()
        sensor_doc['last_update'] = datetime.utcnow()
        
        await db.sensors.insert_one(sensor_doc)
        
        # Regenerate device topics since sensors changed
        sensors = await db.sensors.find({"device_id": sensor_doc['device_id']}).to_list(50)
        device_data = {
            "device_id": sensor_doc['device_id'],
            "device_name": device.get('device_name', sensor_doc['device_id']),
            "device_type": device.get('device_type', 'camera'),
            "sensors": [{"sensor_id": s.get('sensor_id'), "sensor_type": s.get('sensor_type')} for s in sensors]
        }
        topics = generate_device_topics(device_data)
        
        # Update device with new topics
        await db.devices.update_one(
            {"device_id": sensor_doc['device_id']},
            {"$set": {
                "pub_topics": topics["pub_topics"],
                "sub_topics": topics["sub_topics"],
                "last_update": datetime.utcnow()
            }}
        )
        
        # Publish sensor creation via Redis Pub/Sub
        await publish_message('sensor_create', {
            "device_id": sensor_doc['device_id'],
            "sensor_id": sensor_doc['sensor_id'],
            "sensor_type": sensor_doc['sensor_type']
        })
        
        return Response({
            "status": "success",
            "sensor_id": sensor_doc['sensor_id']
        }, status=status.HTTP_201_CREATED)


class NodeDeviceConnectionView(AsyncAPIView):
    """Manage connections between nodes and devices with pub/sub topics"""
    
    async def get(self, request):
        """Get all node-device connections"""
        db = await ensure_db_connection()
        
        # Get all devices with their connected nodes
        devices = await db.devices.find({}).to_list(100)
        nodes = await db.nodes.find({}).to_list(100)
        
        # Build connection map
        connections = []
        for device in devices:
            if device.get('node_id'):
                node = next((n for n in nodes if n.get('node_id') == device.get('node_id')), None)
                connections.append({
                    "device_id": device.get('device_id'),
                    "device_name": device.get('device_name'),
                    "node_id": device.get('node_id'),
                    "node_status": node.get('status') if node else 'unknown',
                    "topics": device.get('topics', []),
                    "pub_topics": device.get('pub_topics', []),
                    "sub_topics": device.get('sub_topics', [])
                })
        
        return Response(connections)
    
    async def post(self, request):
        """Create or update node-device connection with topics"""
        db = await ensure_db_connection()
        
        device_id = request.data.get('device_id')
        node_id = request.data.get('node_id')
        pub_topics = request.data.get('pub_topics', [])  # Topics device publishes to
        sub_topics = request.data.get('sub_topics', [])  # Topics device subscribes to
        
        if not device_id or not node_id:
            return Response(
                {"detail": "device_id and node_id are required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Verify device exists
        device = await db.devices.find_one({"device_id": device_id})
        if not device:
            return Response(
                {"detail": "Device not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Verify node exists
        node = await db.nodes.find_one({"node_id": node_id})
        if not node:
            return Response(
                {"detail": "Node not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # If topics not provided, generate them automatically based on device type
        if not pub_topics and not sub_topics:
            topics = generate_device_topics(device)
            pub_topics = topics["pub_topics"]
            sub_topics = topics["sub_topics"]
        elif not pub_topics:
            # Generate pub topics if not provided
            topics = generate_device_topics(device)
            pub_topics = topics["pub_topics"]
        elif not sub_topics:
            # Generate sub topics if not provided
            topics = generate_device_topics(device)
            sub_topics = topics["sub_topics"]
        
        # Update device with node connection and topics
        update_data = {
            "node_id": node_id,
            "last_update": datetime.utcnow(),
            "pub_topics": pub_topics,
            "sub_topics": sub_topics
        }
        
        await db.devices.update_one(
            {"device_id": device_id},
            {"$set": update_data}
        )
        
        # Publish connection update via Redis Pub/Sub
        await publish_message('device_node_connection', {
            "device_id": device_id,
            "node_id": node_id,
            "pub_topics": pub_topics,
            "sub_topics": sub_topics,
            "action": "connected"
        })
        
        return Response({
            "status": "success",
            "message": f"Device {device_id} connected to node {node_id}",
            "device_id": device_id,
            "node_id": node_id,
            "pub_topics": pub_topics,
            "sub_topics": sub_topics
        })
    
    async def delete(self, request):
        """Disconnect device from node"""
        db = await ensure_db_connection()
        
        device_id = request.data.get('device_id')
        if not device_id:
            return Response(
                {"detail": "device_id is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        device = await db.devices.find_one({"device_id": device_id})
        if not device:
            return Response(
                {"detail": "Device not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        node_id = device.get('node_id')
        
        # Remove connection
        await db.devices.update_one(
            {"device_id": device_id},
            {"$set": {
                "node_id": None,
                "pub_topics": [],
                "sub_topics": [],
                "last_update": datetime.utcnow()
            }}
        )
        
        # Publish disconnection via Redis Pub/Sub
        if node_id:
            await publish_message('device_node_connection', {
                "device_id": device_id,
                "node_id": node_id,
                "action": "disconnected"
            })
        
        return Response({
            "status": "success",
            "message": f"Device {device_id} disconnected from node"
        })

