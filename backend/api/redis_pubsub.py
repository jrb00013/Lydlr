"""
Redis Pub/Sub implementation for real-time messaging
"""
import json
import asyncio
from typing import Dict, Any, Optional, Callable
from backend.api.connections import redis_client, redis_pubsub


async def publish_message(channel: str, data: Dict[str, Any]) -> bool:
    """
    Publish a message to a Redis channel
    
    Args:
        channel: Channel name to publish to
        data: Data dictionary to publish
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if not redis_client:
            return False
        
        message = json.dumps({
            "type": channel,
            "data": data,
            "timestamp": str(asyncio.get_event_loop().time())
        })
        
        await redis_client.publish(channel, message)
        return True
    except Exception as e:
        print(f"Error publishing to Redis: {e}")
        return False


async def subscribe_to_channel(channel: str, callback: Callable[[Dict[str, Any]], None]):
    """
    Subscribe to a Redis channel and call callback for each message
    
    Args:
        channel: Channel name to subscribe to
        callback: Async function to call with message data
    """
    try:
        if not redis_pubsub:
            return
        
        await redis_pubsub.subscribe(channel)
        
        async for message in redis_pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    await callback(data)
                except json.JSONDecodeError:
                    print(f"Failed to decode message: {message['data']}")
    except Exception as e:
        print(f"Error subscribing to Redis channel {channel}: {e}")


async def subscribe_to_multiple_channels(channels: list, callback: Callable[[str, Dict[str, Any]], None]):
    """
    Subscribe to multiple Redis channels
    
    Args:
        channels: List of channel names
        callback: Async function to call with (channel, data)
    """
    try:
        if not redis_pubsub:
            return
        
        for channel in channels:
            await redis_pubsub.subscribe(channel)
        
        async for message in redis_pubsub.listen():
            if message['type'] == 'message':
                try:
                    channel = message['channel'].decode() if isinstance(message['channel'], bytes) else message['channel']
                    data = json.loads(message['data'])
                    await callback(channel, data)
                except json.JSONDecodeError:
                    print(f"Failed to decode message: {message['data']}")
    except Exception as e:
        print(f"Error subscribing to Redis channels: {e}")


# Common channel names
CHANNELS = {
    'NODE_CONFIG_UPDATE': 'node_config_update',
    'DEPLOYMENT': 'deployment',
    'METRICS_UPDATE': 'metrics_update',
    'NODE_COMMAND': 'node_command',
    'SYSTEM_EVENT': 'system_event',
    'DEVICE_NODE_CONNECTION': 'device_node_connection',
}

