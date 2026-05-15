"""
Canonical ROS2 topic graph for Lydlr.

Namespace layout
----------------
/lydlr/fleet/*           Fleet-wide commands & status
/lydlr/{node_id}/*       Per-edge-node data plane
/lydlr/ground/*          Downlink aggregation (relay → GCS)
/camera|lidar|imu|...    Standard sensor ingress (shared bus)
"""
import os
import re
from typing import List, Optional


FLEET_NS = "/lydlr"
GROUND_NS = f"{FLEET_NS}/ground"


def fleet_node_ids() -> List[str]:
    raw = os.getenv("NODE_IDS", "node_0,node_1,iot_gateway_01")
    return [n.strip() for n in raw.split(",") if n.strip()]


class LydlrTopics:
    """Topic name builder — single source of truth."""

    # Shared sensor bus (synthetic publisher / real sensors)
    CAMERA = "/camera/image_raw"
    LIDAR = "/lidar/data"
    IMU = "/imu/data"
    AUDIO = "/audio/data"
    CMD_VEL = "/cmd_vel"

    # Fleet command plane
    FLEET_DEPLOY = f"{FLEET_NS}/fleet/deploy"
    FLEET_SCRIPT = f"{FLEET_NS}/fleet/script/load"
    FLEET_STATUS = f"{FLEET_NS}/fleet/status"
    FLEET_GRAPH = f"{FLEET_NS}/fleet/graph"

    # Coordinator
    COORDINATOR_PERF = f"{FLEET_NS}/coordinator/performance"

    @staticmethod
    def node_base(node_id: str) -> str:
        return f"{FLEET_NS}/{node_id}"

    @classmethod
    def deploy(cls, node_id: str) -> str:
        return f"{cls.node_base(node_id)}/command/deploy"

    @classmethod
    def script_load(cls, node_id: str) -> str:
        return f"{cls.node_base(node_id)}/command/script_load"

    @classmethod
    def coordination(cls, node_id: str) -> str:
        return f"{cls.node_base(node_id)}/coordination"

    @classmethod
    def compressed_transport(cls, node_id: str) -> str:
        """LYDT wire-format compressed payload (primary transport)."""
        return f"{cls.node_base(node_id)}/transport/compressed"

    @classmethod
    def metrics_transport(cls, node_id: str) -> str:
        return f"{cls.node_base(node_id)}/transport/metrics"

    @classmethod
    def heartbeat(cls, node_id: str) -> str:
        return f"{cls.node_base(node_id)}/heartbeat"

    # Legacy aliases (backward compatible with existing tools)
    @classmethod
    def legacy_compressed(cls, node_id: str) -> str:
        return f"/{node_id}/compressed"

    @classmethod
    def legacy_metrics(cls, node_id: str) -> str:
        return f"/{node_id}/metrics"

    @classmethod
    def legacy_deploy(cls, node_id: str) -> str:
        return f"/{node_id}/model/deploy"

    @classmethod
    def ground_uplink(cls) -> str:
        return f"{GROUND_NS}/uplink/compressed"

    @classmethod
    def ground_metrics(cls) -> str:
        return f"{GROUND_NS}/uplink/metrics"

    @classmethod
    def discover_node_ids_from_topic(cls, topic: str) -> Optional[str]:
        m = re.match(rf"^{re.escape(FLEET_NS)}/([^/]+)/", topic)
        if m:
            return m.group(1)
        m = re.match(r"^/([^/]+)/metrics$", topic)
        if m:
            return m.group(1)
        return None
