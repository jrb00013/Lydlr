# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#federated_sync_node.py
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import threading
import time
import random

class FederatedSyncNode(Node):
    def __init__(self):
        super().__init__('federated_sync_node')

        self.local_model_version = 0
        self.master_model_version = 0

        self.srv = self.create_service(Trigger, 'sync_model', self.handle_sync_model)

        self.lock = threading.Lock()
        threading.Thread(target=self.local_training_loop, daemon=True).start()

    def local_training_loop(self):
        while rclpy.ok():
            time.sleep(random.uniform(1.0, 3.0))
            with self.lock:
                self.local_model_version += 1
                self.get_logger().info(f"Local training done, model version {self.local_model_version}")

    def handle_sync_model(self, request, response):
        with self.lock:
            if self.local_model_version > self.master_model_version:
                self.master_model_version = self.local_model_version
                response.success = True
                response.message = f"Master model updated to version {self.master_model_version}"
            else:
                response.success = False
                response.message = "No update needed"
            self.get_logger().info(response.message)
        return response

def main(args=None):
    rclpy.init(args=args)
    node = FederatedSyncNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
