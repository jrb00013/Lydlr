"""
Views - import all views from submodules
"""
from backend.api.views.base import (
    AsyncAPIView,
    ensure_db_connection,
    health_check,
    root
)
from backend.api.views.node_views import (
    NodeListView,
    NodeDetailView,
    NodeControlView,
    NodeDeployView,
    NodeCreateView,
    NodeConfigurationView
)
from backend.api.views.device_views import (
    DeviceListView,
    DeviceDetailView,
    DeviceCreateView,
    SensorListView,
    NodeDeviceConnectionView
)
from backend.api.views.workspace_views import WorkspaceView, DiagnosticView, OrchestrationStatusView
from backend.api.views.model_views import ModelListView
from backend.api.views.metrics_views import MetricsView, SystemStatsView
from backend.api.views.deployment_views import DeploymentView

__all__ = [
    # Base
    'AsyncAPIView',
    'ensure_db_connection',
    'health_check',
    'root',
    # Nodes
    'NodeListView',
    'NodeDetailView',
    'NodeControlView',
    'NodeDeployView',
    'NodeCreateView',
    'NodeConfigurationView',
    # Devices
    'DeviceListView',
    'DeviceDetailView',
    'DeviceCreateView',
    'SensorListView',
    'NodeDeviceConnectionView',
    # Workspace
    'WorkspaceView',
    'OrchestrationStatusView',
    'DiagnosticView',
    # Models
    'ModelListView',
    # Metrics
    'MetricsView',
    'SystemStatsView',
    # Deployment
    'DeploymentView',
]

