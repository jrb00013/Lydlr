"""
URL routing for API endpoints
"""
from django.urls import path
from backend.api.views import (
    root, health_check,
    NodeListView, NodeDetailView, NodeControlView, NodeDeployView,
    NodeCreateView, NodeConfigurationView, NodeLinkSpecView,
    DeviceListView, DeviceDetailView, DeviceCreateView,
    SensorListView, NodeDeviceConnectionView,
    WorkspaceView, DiagnosticView, OrchestrationStatusView,
    ModelListView, ModelRegistryView, ModelRegistryTableView,
    ModelArtifactDetailView, ModelSyncView, ModelUploadView,
    MetricsView, SystemStatsView, MetricsRollupsView, MetricsFleetView,
    MetricsExportView,
    DeploymentView, ModelRollbackView,
)
from backend.api.views.fleet_views import FleetLinkPolicyView, FleetLinkHealthView, rl_policy
from backend.api.views.federated_views import FederatedRoundDetailView, FederatedRoundListView

urlpatterns = [
    path('', root, name='root'),
    path('health/', health_check, name='health'),
    path('nodes/', NodeListView.as_view(), name='nodes-list'),
    path('nodes/create/', NodeCreateView.as_view(), name='node-create'),
    path('nodes/config/', NodeConfigurationView.as_view(), name='node-config'),
    path('nodes/<str:node_id>/deploy/', NodeDeployView.as_view(), name='node-deploy'),
    path('nodes/<str:node_id>/delete/', NodeCreateView.as_view(), name='node-delete'),
    path('nodes/<str:node_id>/<str:action>/', NodeControlView.as_view(), name='node-control'),
    path('nodes/<str:node_id>/link-spec/', NodeLinkSpecView.as_view(), name='node-link-spec'),
    path('nodes/<str:node_id>/', NodeDetailView.as_view(), name='node-detail'),
    path('models/', ModelListView.as_view(), name='models-list'),
    path('models/upload/', ModelUploadView.as_view(), name='models-upload'),
    path('models/sync/', ModelSyncView.as_view(), name='models-sync'),
    path('models/registry/', ModelRegistryView.as_view(), name='models-registry'),
    path('models/registry/table/', ModelRegistryTableView.as_view(), name='models-registry-table'),
    path('models/registry/<str:artifact_id>/', ModelArtifactDetailView.as_view(), name='models-artifact-detail'),
    path('deploy/', DeploymentView.as_view(), name='deploy'),
    path('deploy/rollback/', ModelRollbackView.as_view(), name='deploy-rollback'),
    path('deployments/', DeploymentView.as_view(), name='deployments'),
    path('metrics/', MetricsView.as_view(), name='metrics'),
    path('metrics/rollups/', MetricsRollupsView.as_view(), name='metrics-rollups'),
    path('metrics/fleet/', MetricsFleetView.as_view(), name='metrics-fleet'),
    path('metrics/export/', MetricsExportView.as_view(), name='metrics-export'),
    path('fleet/link-policy/', FleetLinkPolicyView.as_view(), name='fleet-link-policy'),
    path('fleet/link-policy/health/', FleetLinkHealthView.as_view(), name='fleet-link-health'),
    path('fleet/rl-policy/', rl_policy, name='fleet-rl-policy'),
    path('stats/', SystemStatsView.as_view(), name='stats'),
    path('devices/', DeviceListView.as_view(), name='devices-list'),
    path('devices/create/', DeviceCreateView.as_view(), name='device-create'),
    path('devices/<str:device_id>/', DeviceDetailView.as_view(), name='device-detail'),
    path('devices/<str:device_id>/delete/', DeviceCreateView.as_view(), name='device-delete'),
    path('sensors/', SensorListView.as_view(), name='sensors-list'),
    path('connections/', NodeDeviceConnectionView.as_view(), name='node-device-connections'),
    path('workspace/', WorkspaceView.as_view(), name='workspace'),
    path('diagnostic/', DiagnosticView.as_view(), name='diagnostic'),
    path('orchestration/status/', OrchestrationStatusView.as_view(), name='orchestration-status'),
    path('federated/rounds/', FederatedRoundListView.as_view(), name='federated-rounds'),
    path('federated/rounds/<str:round_id>/', FederatedRoundDetailView.as_view(), name='federated-round-detail'),
]

