"""
WebSocket routing for Channels
"""
from django.urls import re_path
from backend.api import consumers

websocket_urlpatterns = [
    re_path(r'ws/metrics/$', consumers.MetricsConsumer.as_asgi()),
]

