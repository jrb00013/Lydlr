"""
URL routing for GStreamer/Deepstream endpoints
Comprehensive API endpoints for GStreamer operations
"""
from django.urls import path
from backend.gstreamer import views

urlpatterns = [
    # Status and diagnostics
    path('status/', views.gstreamer_status, name='gstreamer-status'),
    path('health/', views.gstreamer_health, name='gstreamer-health'),
    path('diagnostics/', views.gstreamer_diagnostics, name='gstreamer-diagnostics'),
    path('plugins/', views.gstreamer_plugins, name='gstreamer-plugins'),
    path('plugins/<str:plugin_name>/', views.gstreamer_plugin_detail, name='gstreamer-plugin-detail'),
    path('test-pipeline/', views.test_pipeline, name='test-pipeline'),
    
    # Deepstream
    path('deepstream/status/', views.deepstream_status, name='deepstream-status'),
    path('deepstream/inference/', views.DeepstreamInferenceView.as_view(), name='deepstream-inference'),
    
    # Pipeline creation
    path('pipeline/compression/', views.CompressionPipelineView.as_view(), name='compression-pipeline'),
    path('pipeline/rtsp/', views.RTSPPipelineView.as_view(), name='rtsp-pipeline'),
    path('pipeline/hls/', views.HLSPipelineView.as_view(), name='hls-pipeline'),
    path('pipeline/webrtc/', views.WebRTCPipelineView.as_view(), name='webrtc-pipeline'),
]

