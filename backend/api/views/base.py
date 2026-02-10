"""
Base views and utility functions
"""
import logging
from datetime import datetime
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
import inspect

from backend.api import connections
from backend.api.connections import init_connections

logger = logging.getLogger(__name__)


class AsyncAPIView(APIView):
    """Base APIView that properly handles async methods"""
    
    async def dispatch(self, request, *args, **kwargs):
        """
        Override dispatch to properly handle async methods
        """
        self.args = args
        self.kwargs = kwargs
        request = self.initialize_request(request, *args, **kwargs)
        self.request = request
        self.headers = self.default_response_headers

        try:
            self.initial(request, *args, **kwargs)

            # Get the appropriate handler method
            if request.method.lower() in self.http_method_names:
                handler = getattr(self, request.method.lower(),
                                self.http_method_not_allowed)
            else:
                handler = self.http_method_not_allowed

            # Check if handler is async and await it
            if inspect.iscoroutinefunction(handler):
                response = await handler(request, *args, **kwargs)
            else:
                response = handler(request, *args, **kwargs)

        except Exception as exc:
            response = self.handle_exception(exc)
            # handle_exception might return a coroutine in some cases
            if inspect.iscoroutine(response):
                response = await response

        self.response = self.finalize_response(request, response, *args, **kwargs)
        return self.response


async def ensure_db_connection():
    """Ensure database connection is initialized"""
    if connections.db is None:
        try:
            await init_connections()
        except Exception as e:
            raise Exception(f"Failed to initialize database connection: {str(e)}")
    
    if connections.db is None:
        raise Exception("Database connection is None after initialization")
    
    return connections.db


@api_view(['GET'])
def health_check(request):
    """Health check endpoint"""
    return Response({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    })


@api_view(['GET'])
def root(request):
    """Root endpoint"""
    return Response({
        "message": "Lydlr Revolutionary Compression System API",
        "version": "1.0.0",
        "docs": "/api/docs"
    })

