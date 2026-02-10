"""
GStreamer pipeline management for video processing
Comprehensive GStreamer integration with multiple pipeline types
"""
import os
import logging
import threading
import time
from typing import Optional, Dict, Any, Callable, List, TYPE_CHECKING
from enum import Enum

# Try to import GStreamer bindings - allow graceful degradation if not available
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib, GObject
    GSTREAMER_AVAILABLE = True
    # Initialize GStreamer
    Gst.init(None)
except ImportError as e:
    GSTREAMER_AVAILABLE = False
    # Create dummy classes for type hints when GStreamer is not available
    class DummyGst:
        class Pipeline: pass
        class Element: pass
        class State:
            NULL = "NULL"
            READY = "READY"
            PAUSED = "PAUSED"
            PLAYING = "PLAYING"
        class MessageType:
            ERROR = "ERROR"
            WARNING = "WARNING"
            EOS = "EOS"
            STATE_CHANGED = "STATE_CHANGED"
            STREAM_START = "STREAM_START"
            QOS = "QOS"
            STATS = "STATS"
        class StateChangeReturn:
            FAILURE = "FAILURE"
            ASYNC = "ASYNC"
        CLOCK_TIME_NONE = 0
        @staticmethod
        def parse_launch(pipeline_string): return None
        @staticmethod
        def init(args): pass
    class DummyGLib:
        class MainLoop: pass
    class DummyGObject:
        pass
    Gst = DummyGst()
    GLib = DummyGLib()
    GObject = DummyGObject()
    logging.warning(f"GStreamer Python bindings not available: {e}. GStreamer features will be disabled.")

# Set up logging
logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Pipeline state enumeration"""
    NULL = "NULL"
    READY = "READY"
    PAUSED = "PAUSED"
    PLAYING = "PLAYING"


class PipelineType(Enum):
    """Pipeline type enumeration"""
    FILE_TO_FILE = "file_to_file"
    RTSP_STREAM = "rtsp_stream"
    RTSP_SERVER = "rtsp_server"
    WEBRTC = "webrtc"
    HLS = "hls"
    FILE_TO_RTSP = "file_to_rtsp"
    USB_CAMERA = "usb_camera"
    DEEPSTREAM = "deepstream"


class GStreamerPipeline:
    """Comprehensive GStreamer pipeline wrapper with advanced features"""
    
    def __init__(self, pipeline_string: str, name: str = "lydlr-pipeline"):
        self.pipeline_string = pipeline_string
        self.name = name
        self.pipeline: Optional[Any] = None
        self.loop: Optional[Any] = None
        self.thread: Optional[threading.Thread] = None
        self.callbacks: Dict[str, Callable] = {}
        self.state = PipelineState.NULL
        self.start_time: Optional[float] = None
        self.stats = {
            'frames_processed': 0,
            'bytes_processed': 0,
            'errors': 0,
            'warnings': 0
        }
        self._lock = threading.Lock()
    
    def create_pipeline(self) -> bool:
        """Create and configure the pipeline"""
        if not GSTREAMER_AVAILABLE:
            logger.warning("GStreamer not available, cannot create pipeline")
            return False
        try:
            self.pipeline = Gst.parse_launch(self.pipeline_string)
            if not self.pipeline:
                print(f"Failed to create pipeline: {self.pipeline_string}")
                return False
            
            # Set up message handling
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self.on_message)
            
            return True
        except Exception as e:
            print(f"Error creating pipeline: {e}")
            return False
    
    def on_message(self, bus, message):
        """Handle GStreamer messages with comprehensive logging"""
        if not GSTREAMER_AVAILABLE:
            return
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"Pipeline {self.name} error: {err}, {debug}")
            with self._lock:
                self.stats['errors'] += 1
            if 'error' in self.callbacks:
                self.callbacks['error'](err, debug)
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            logger.warning(f"Pipeline {self.name} warning: {warn}, {debug}")
            with self._lock:
                self.stats['warnings'] += 1
            if 'warning' in self.callbacks:
                self.callbacks['warning'](warn, debug)
        elif t == Gst.MessageType.EOS:
            logger.info(f"Pipeline {self.name} reached end of stream")
            if 'eos' in self.callbacks:
                self.callbacks['eos']()
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                state_names = {
                    Gst.State.NULL: "NULL",
                    Gst.State.READY: "READY",
                    Gst.State.PAUSED: "PAUSED",
                    Gst.State.PLAYING: "PLAYING"
                }
                logger.debug(f"Pipeline {self.name} state: {state_names.get(old_state)} -> {state_names.get(new_state)}")
                if 'state_changed' in self.callbacks:
                    self.callbacks['state_changed'](old_state, new_state, pending_state)
        elif t == Gst.MessageType.STREAM_START:
            logger.info(f"Pipeline {self.name} stream started")
            if 'stream_start' in self.callbacks:
                self.callbacks['stream_start']()
        elif t == Gst.MessageType.QOS:
            if 'qos' in self.callbacks:
                self.callbacks['qos'](message)
        elif t == Gst.MessageType.STATS:
            if 'stats' in self.callbacks:
                self.callbacks['stats'](message)
    
    def set_callback(self, event: str, callback: Callable):
        """Set callback for pipeline events"""
        self.callbacks[event] = callback
    
    def start(self) -> bool:
        """Start the pipeline with comprehensive error handling"""
        if not GSTREAMER_AVAILABLE:
            logger.warning("GStreamer not available, cannot start pipeline")
            return False
        if not self.pipeline:
            if not self.create_pipeline():
                logger.error(f"Failed to create pipeline: {self.name}")
                return False
        
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error(f"Failed to start pipeline: {self.name}")
            return False
        elif ret == Gst.StateChangeReturn.ASYNC:
            # Wait for state change
            ret = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
            if ret[0] == Gst.StateChangeReturn.FAILURE:
                logger.error(f"Pipeline state change failed: {self.name}")
                return False
        
        self.state = PipelineState.PLAYING
        self.start_time = time.time()
        logger.info(f"Pipeline {self.name} started successfully")
        return True
    
    def stop(self):
        """Stop the pipeline"""
        if not GSTREAMER_AVAILABLE or not self.pipeline:
            return
        self.pipeline.set_state(Gst.State.NULL)
        self.state = PipelineState.NULL
        logger.info(f"Pipeline {self.name} stopped")
    
    def pause(self):
        """Pause the pipeline"""
        if not GSTREAMER_AVAILABLE or not self.pipeline:
            return
        self.pipeline.set_state(Gst.State.PAUSED)
        self.state = PipelineState.PAUSED
        logger.info(f"Pipeline {self.name} paused")
    
    def resume(self):
        """Resume the pipeline"""
        if not GSTREAMER_AVAILABLE or not self.pipeline:
            return
        self.pipeline.set_state(Gst.State.PLAYING)
        self.state = PipelineState.PLAYING
        logger.info(f"Pipeline {self.name} resumed")
    
    def get_state(self):
        """Get current pipeline state"""
        if not GSTREAMER_AVAILABLE or not self.pipeline:
            return self.state
        state = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)[1]
        state_map = {
            Gst.State.NULL: PipelineState.NULL,
            Gst.State.READY: PipelineState.READY,
            Gst.State.PAUSED: PipelineState.PAUSED,
            Gst.State.PLAYING: PipelineState.PLAYING
        }
        self.state = state_map.get(state, PipelineState.NULL)
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        with self._lock:
            stats = self.stats.copy()
        if self.start_time:
            stats['uptime'] = time.time() - self.start_time
        return stats
    
    def get_element(self, name: str) -> Optional[Any]:
        """Get a pipeline element by name"""
        if not GSTREAMER_AVAILABLE or not self.pipeline:
            return None
        return self.pipeline.get_by_name(name)


class VideoCompressionPipeline(GStreamerPipeline):
    """Video compression pipeline using GStreamer with multiple codec options"""
    
    def __init__(self, input_source: str, output_sink: str, 
                 compression_level: float = 0.8, codec: str = "x264"):
        """
        Create video compression pipeline
        
        Args:
            input_source: Input video file path
            output_sink: Output video file path
            compression_level: Compression level (0.0-1.0)
            codec: Codec to use (x264, x265, vp8, vp9)
        """
        bitrate = int(1000 * compression_level)
        
        codec_pipelines = {
            "x264": f"x264enc bitrate={bitrate} speed-preset=ultrafast tune=zerolatency",
            "x265": f"x265enc bitrate={bitrate} speed-preset=ultrafast",
            "vp8": f"vp8enc target-bitrate={bitrate * 1000} deadline=1",
            "vp9": f"vp9enc target-bitrate={bitrate * 1000} deadline=1"
        }
        
        encoder = codec_pipelines.get(codec, codec_pipelines["x264"])
        muxer = "mp4mux" if codec in ["x264", "x265"] else "webmmux"
        
        pipeline_string = (
            f"filesrc location={input_source} ! "
            "decodebin ! "
            "videoconvert ! "
            "videoscale ! "
            "video/x-raw,format=I420 ! "
            f"{encoder} ! "
            f"{muxer} ! "
            f"filesink location={output_sink}"
        )
        super().__init__(pipeline_string, f"video-compression-{codec}")


class RTSPStreamPipeline(GStreamerPipeline):
    """RTSP streaming pipeline"""
    
    def __init__(self, input_source: str, rtsp_port: int = 8554, 
                 mount_point: str = "/test", codec: str = "h264"):
        """
        Create RTSP streaming pipeline
        
        Args:
            input_source: Input source (file path or device)
            rtsp_port: RTSP server port
            mount_point: RTSP mount point
            codec: Video codec (h264, h265, vp8, vp9)
        """
        if input_source.startswith("/dev/video"):
            source = f"v4l2src device={input_source}"
        elif input_source.startswith("rtsp://") or input_source.startswith("http://"):
            source = f"rtspsrc location={input_source}"
        else:
            source = f"filesrc location={input_source} ! decodebin"
        
        encoder_map = {
            "h264": "x264enc tune=zerolatency speed-preset=ultrafast",
            "h265": "x265enc speed-preset=ultrafast",
            "vp8": "vp8enc deadline=1",
            "vp9": "vp9enc deadline=1"
        }
        encoder = encoder_map.get(codec, encoder_map["h264"])
        
        pipeline_string = (
            f"{source} ! "
            "videoconvert ! "
            "videoscale ! "
            "video/x-raw,width=1280,height=720,framerate=30/1 ! "
            f"{encoder} ! "
            f"rtph264pay name=pay0 pt=96"
        )
        super().__init__(pipeline_string, f"rtsp-stream-{codec}")
        self.rtsp_port = rtsp_port
        self.mount_point = mount_point


class HLSPipeline(GStreamerPipeline):
    """HLS streaming pipeline"""
    
    def __init__(self, input_source: str, output_dir: str, 
                 playlist_name: str = "playlist.m3u8", segment_duration: int = 2):
        """
        Create HLS streaming pipeline
        
        Args:
            input_source: Input source
            output_dir: Output directory for HLS segments
            playlist_name: Playlist filename
            segment_duration: Segment duration in seconds
        """
        os.makedirs(output_dir, exist_ok=True)
        playlist_path = os.path.join(output_dir, playlist_name)
        
        pipeline_string = (
            f"filesrc location={input_source} ! "
            "decodebin ! "
            "videoconvert ! "
            "videoscale ! "
            "video/x-raw,width=1280,height=720,framerate=30/1 ! "
            "x264enc tune=zerolatency speed-preset=ultrafast ! "
            "h264parse ! "
            f"hlssink2 location={output_dir}/segment%05d.ts "
            f"playlist-location={playlist_path} "
            f"target-duration={segment_duration} max-files=5"
        )
        super().__init__(pipeline_string, "hls-stream")


class WebRTCPipeline(GStreamerPipeline):
    """WebRTC streaming pipeline"""
    
    def __init__(self, input_source: str, stun_server: str = "stun://stun.l.google.com:19302"):
        """
        Create WebRTC streaming pipeline
        
        Args:
            input_source: Input source
            stun_server: STUN server URL
        """
        if input_source.startswith("/dev/video"):
            source = f"v4l2src device={input_source}"
        else:
            source = f"filesrc location={input_source} ! decodebin"
        
        pipeline_string = (
            f"{source} ! "
            "videoconvert ! "
            "videoscale ! "
            "video/x-raw,width=1280,height=720,framerate=30/1 ! "
            "x264enc tune=zerolatency speed-preset=ultrafast ! "
            "rtph264pay ! "
            "webrtcbin name=webrtcbin stun-server={stun_server}"
        )
        super().__init__(pipeline_string, "webrtc-stream")
        self.stun_server = stun_server


class DeepstreamPipeline(GStreamerPipeline):
    """NVIDIA Deepstream pipeline for AI inference"""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        pipeline_string = f"deepstream-app -c {config_file}"
        super().__init__(pipeline_string, "deepstream-pipeline")
    
    def create_pipeline(self) -> bool:
        """Create Deepstream pipeline from config"""
        # Deepstream uses its own pipeline creation
        # This is a placeholder - actual implementation would use Deepstream SDK
        logger.info(f"Creating Deepstream pipeline from config: {self.config_file}")
        return True

