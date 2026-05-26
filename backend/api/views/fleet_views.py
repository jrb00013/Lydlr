"""Fleet link policy API for ROS coordinator."""
from rest_framework.response import Response

from backend.api.views.base import AsyncAPIView, ensure_db_connection
from backend.api.services.link_policy_service import build_fleet_link_policy


class FleetLinkPolicyView(AsyncAPIView):
    async def get(self, request):
        db = await ensure_db_connection()
        return Response(await build_fleet_link_policy(db))
