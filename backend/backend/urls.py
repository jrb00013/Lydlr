"""
URL configuration for Lydlr backend.
"""
from django.contrib import admin
from django.urls import path, include, re_path
from django.http import JsonResponse
from backend.api.views import health_check, root, OrchestrationStatusView

def catch_all(request, path=''):
    """
    Catch-all route for frontend routes.
    Returns a JSON response indicating the route should be handled by the frontend.
    In production, nginx should handle routing, but this prevents 404 errors.
    """
    return JsonResponse({
        'message': 'This route is handled by the frontend. Please access through the frontend application.',
        'path': path or '/',
        'note': 'If you are accessing the backend directly, use /api/ endpoints instead.'
    }, status=404)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('backend.api.urls')),
    path('gstreamer/', include('backend.gstreamer.urls')),
    path('orchestration/status/', OrchestrationStatusView.as_view(), name='orchestration-status'),  # Orchestration status endpoint
    path('health/', health_check, name='health'),  # Health check
    path('', root, name='root'),  # Root endpoint
    # Catch-all for any other routes (frontend routes like /nodes, /devices, etc.)
    # This prevents "Not Found" errors when refreshing on frontend routes
    re_path(r'^(?!(api|admin|gstreamer|orchestration|health|$)).+', catch_all, name='catch-all'),
]

