"""
WebSocket consumers for Channels
"""
import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from backend.api.redis_pubsub import (
    subscribe_to_channel,
    subscribe_to_multiple_channels,
    CHANNELS,
)
from backend.api.pubsub_utils import unwrap_pubsub_payload


class MetricsConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time metrics"""

    async def connect(self):
        await self.accept()
        self._sub_task = asyncio.create_task(
            subscribe_to_channel(
                CHANNELS["METRICS_UPDATE"],
                self.on_metrics_update,
            )
        )

    async def disconnect(self, close_code):
        task = getattr(self, "_sub_task", None)
        if task and not task.done():
            task.cancel()

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            await self.send(text_data=json.dumps({"echo": data}))
        except json.JSONDecodeError:
            pass

    async def on_metrics_update(self, data: dict):
        metric = unwrap_pubsub_payload(data)
        await self.send(
            text_data=json.dumps(
                {
                    "type": "metrics_update",
                    "data": metric,
                }
            )
        )


class FleetEventsConsumer(AsyncWebsocketConsumer):
    """WebSocket for deployment / node command / link-spec events."""

    CHANNEL_LIST = [
        CHANNELS["DEPLOYMENT"],
        CHANNELS["NODE_COMMAND"],
        "node_link_spec_update",
        CHANNELS["NODE_CONFIG_UPDATE"],
    ]

    async def connect(self):
        await self.accept()
        self._sub_task = asyncio.create_task(
            subscribe_to_multiple_channels(self.CHANNEL_LIST, self.on_fleet_event)
        )

    async def disconnect(self, close_code):
        task = getattr(self, "_sub_task", None)
        if task and not task.done():
            task.cancel()

    async def receive(self, text_data):
        pass

    async def on_fleet_event(self, channel: str, data: dict):
        payload = unwrap_pubsub_payload(data)
        await self.send(
            text_data=json.dumps(
                {
                    "type": "fleet_event",
                    "channel": channel,
                    "data": payload,
                }
            )
        )
