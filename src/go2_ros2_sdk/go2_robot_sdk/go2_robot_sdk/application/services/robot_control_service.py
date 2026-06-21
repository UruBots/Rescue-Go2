# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

import json
import logging


from ...domain.interfaces import IRobotController
from ..utils.command_generator import gen_mov_command
from ...domain.constants import RTC_TOPIC


logger = logging.getLogger(__name__)


class RobotControlService:
    """Service for robot control"""

    def __init__(self, controller: IRobotController):
        self.controller = controller
        self.last_sent_non_zero = False
        self.last_joy_buttons = []

    def handle_cmd_vel(self, x: float, y: float, z: float, robot_id: str, obstacle_avoidance: bool = False) -> None:
        """Process movement command. Sends non-zero commands and exactly one zero command to stop."""
        try:
            is_zero = (x == 0.0 and y == 0.0 and z == 0.0)
            if not is_zero or self.last_sent_non_zero:
                self.controller.send_movement_command(robot_id, x, y, z)
                self.last_sent_non_zero = not is_zero
        except Exception as e:
            logger.error(f"Error handling cmd_vel: {e}")

    def handle_webrtc_request(self, api_id: int, parameter_str: str, topic: str, msg_id: str, robot_id: str) -> None:
        """Process WebRTC request"""
        try:
            parameter = "" if parameter_str == "" else json.loads(parameter_str)
            self.controller.send_webrtc_request(robot_id, api_id, parameter, topic)
            logger.info(f"WebRTC request sent to robot {robot_id}")
        except ValueError as e:
            logger.error(f"Invalid JSON in WebRTC request: {e}")
        except Exception as e:
            logger.error(f"Error handling WebRTC request: {e}")

    def handle_joy_command(self, joy_buttons: list, robot_id: str) -> None:
        """Process joystick commands using rising edge to prevent queue flooding"""
        try:
            if joy_buttons and len(joy_buttons) > 1:
                # Initialize last_joy_buttons if empty
                if not self.last_joy_buttons or len(self.last_joy_buttons) < len(joy_buttons):
                    self.last_joy_buttons = [0] * len(joy_buttons)
                
                # Button A (index 0) - Stand Up
                if joy_buttons[0] == 1 and self.last_joy_buttons[0] == 0:
                    self.controller.send_stand_up_command(robot_id)
                    logger.info(f"Stand up command sent to robot {robot_id}")
                
                # Button B (index 1) - Stand Down
                elif joy_buttons[1] == 1 and self.last_joy_buttons[1] == 0:
                    self.controller.send_stand_down_command(robot_id)
                    logger.info(f"Stand down command sent to robot {robot_id}")
                
                self.last_joy_buttons = list(joy_buttons)
        except Exception as e:
            logger.error(f"Error handling joy command: {e}")

    def set_obstacle_avoidance(self, enabled: bool, robot_id: str) -> None:
        """Set obstacle avoidance mode"""
        try:
            self.controller.send_webrtc_request(
                robot_id, 
                1004, 
                {"is_remote_commands_from_api": enabled},
                RTC_TOPIC['OBSTACLES_AVOID']
            )
            logger.info(f"Obstacle avoidance set to {enabled} for robot {robot_id}")
        except Exception as e:
            logger.error(f"Error setting obstacle avoidance: {e}") 