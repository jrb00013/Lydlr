"""
WebSocket consumers for Channels
"""
import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from backend.api.redis_pubsub import subscribe_to_channel, CHANNELS


class MetricsConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time metrics"""
    
    async def connect(self):
        await self.accept()
        self.channel_name = self.channel_name
        
        # Subscribe to metrics updates via Redis Pub/Sub
        asyncio.create_task(
            subscribe_to_channel(
                CHANNELS['METRICS_UPDATE'],
                self.on_metrics_update
            )
        )
    
    async def disconnect(self, close_code):
        pass
    
    async def receive(self, text_data):
        """Handle messages from client"""
        try:
            data = json.loads(text_data)
            # Echo or process commands
            await self.send(text_data=json.dumps({"echo": data}))
        except json.JSONDecodeError:
            pass
    
    async def on_metrics_update(self, data: dict):
        """Handle metrics update from Redis Pub/Sub"""
        await self.send(text_data=json.dumps({
            "type": "metrics_update",
            "data": data
        }))

