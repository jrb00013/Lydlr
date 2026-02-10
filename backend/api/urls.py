"""
URL routing for API endpoints
"""
from django.urls import path
from backend.api.views import (
    root, health_check,
    NodeListView, NodeDetailView, NodeControlView, NodeDeployView,
    NodeCreateView, NodeConfigurationView,
    DeviceListView, DeviceDetailView, DeviceCreateView,
    SensorListView, NodeDeviceConnectionView,
    WorkspaceView, ModelListView, MetricsView, SystemStatsView,
    DeploymentView
)

urlpatterns = [
    path('', root, name='root'),
    path('health/', health_check, name='health'),
    path('nodes/', NodeListView.as_view(), name='nodes-list'),
    path('nodes/create/', NodeCreateView.as_view(), name='node-create'),
    path('nodes/config/', NodeConfigurationView.as_view(), name='node-config'),
    path('nodes/<str:node_id>/deploy/', NodeDeployView.as_view(), name='node-deploy'),
    path('nodes/<str:node_id>/delete/', NodeCreateView.as_view(), name='node-delete'),
    path('nodes/<str:node_id>/<str:action>/', NodeControlView.as_view(), name='node-control'),
    path('nodes/<str:node_id>/', NodeDetailView.as_view(), name='node-detail'),
    path('models/', ModelListView.as_view(), name='models-list'),
    path('deploy/', DeploymentView.as_view(), name='deploy'),
    path('deployments/', DeploymentView.as_view(), name='deployments'),
    path('metrics/', MetricsView.as_view(), name='metrics'),
    path('stats/', SystemStatsView.as_view(), name='stats'),
    path('devices/', DeviceListView.as_view(), name='devices-list'),
    path('devices/create/', DeviceCreateView.as_view(), name='device-create'),
    path('devices/<str:device_id>/', DeviceDetailView.as_view(), name='device-detail'),
    path('devices/<str:device_id>/delete/', DeviceCreateView.as_view(), name='device-delete'),
    path('sensors/', SensorListView.as_view(), name='sensors-list'),
    path('connections/', NodeDeviceConnectionView.as_view(), name='node-device-connections'),
    path('workspace/', WorkspaceView.as_view(), name='workspace'),
]

