"""
GStreamer integration for Lydlr
Comprehensive video processing, streaming, and AI inference
"""

# Lazy imports to allow backend to start even if GStreamer dependencies are missing
try:
    from backend.gstreamer.gst_pipeline import (
        GStreamerPipeline,
        VideoCompressionPipeline,
        RTSPStreamPipeline,
        HLSPipeline,
        WebRTCPipeline,
        DeepstreamPipeline,
        PipelineState,
        PipelineType
    )
except ImportError as e:
    # GStreamer not available - set to None for graceful degradation
    GStreamerPipeline = None
    VideoCompressionPipeline = None
    RTSPStreamPipeline = None
    HLSPipeline = None
    WebRTCPipeline = None
    DeepstreamPipeline = None
    PipelineState = None
    PipelineType = None
    _gstreamer_error = str(e)

try:
    from backend.gstreamer.deepstream_plugin import (
        DeepstreamManager,
        GStreamerPluginManager
    )
except ImportError:
    DeepstreamManager = None
    GStreamerPluginManager = None

try:
    from backend.gstreamer.utils import (
        GStreamerDiagnostics,
        PipelineBuilder,
        create_compression_pipeline_string
    )
except ImportError:
    GStreamerDiagnostics = None
    PipelineBuilder = None
    create_compression_pipeline_string = None

try:
    from backend.gstreamer.health import GStreamerHealthCheck
except ImportError:
    GStreamerHealthCheck = None

__all__ = [
    'GStreamerPipeline',
    'VideoCompressionPipeline',
    'RTSPStreamPipeline',
    'HLSPipeline',
    'WebRTCPipeline',
    'DeepstreamPipeline',
    'PipelineState',
    'PipelineType',
    'DeepstreamManager',
    'GStreamerPluginManager',
    'GStreamerDiagnostics',
    'PipelineBuilder',
    'create_compression_pipeline_string',
    'GStreamerHealthCheck',
]

