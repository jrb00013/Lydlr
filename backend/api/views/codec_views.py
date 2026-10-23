"""Live vision-codec frame. No database. Each call encodes the next clip frame."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

from rest_framework.response import Response
from rest_framework.views import APIView

_PKG = Path(__file__).resolve().parents[3] / "ros2" / "src" / "lydlr_ai"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))


def codec_frame_payload() -> dict:
    from lydlr_ai.model.vision_session import get_session

    tick = get_session().tick()
    metric = tick["metric"]
    previews = {
        side: "data:image/jpeg;base64," + base64.b64encode(blob).decode("ascii")
        for side, blob in tick["previews"].items()
    }
    return {"metric": metric, "previews": previews}


class CodecFrameView(APIView):
    """GET /api/codec/frame/ — one countable temporal-DCT encode."""

    def get(self, request):
        return Response(codec_frame_payload())
