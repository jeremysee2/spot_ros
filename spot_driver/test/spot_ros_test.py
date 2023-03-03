#!/usr/bin/env python3
PKG = "spot_ros"
NAME = "spot_ros_test"
SUITE = "spot_ros_test.TestSuiteSpotROS"

import unittest
from unittest.mock import MagicMock, patch
import rospy
from rosservice import get_service_class_by_name
import rosservice
import time
import sys

from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse
from bosdyn.api import image_pb2, robot_state_pb2, lease_pb2, geometry_pb2
from google.protobuf import wrappers_pb2, timestamp_pb2, duration_pb2
from bosdyn.client.frame_helpers import (
    add_edge_to_tree,
    VISION_FRAME_NAME,
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
)

from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistWithCovarianceStamped, Twist, Pose, PoseStamped
from nav_msgs.msg import Odometry
from spot_msgs.msg import Metrics
from spot_msgs.msg import LeaseArray, LeaseResource
from spot_msgs.msg import FootState, FootStateArray
from spot_msgs.msg import EStopState, EStopStateArray
from spot_msgs.msg import WiFiState
from spot_msgs.msg import PowerState
from spot_msgs.msg import BehaviorFault, BehaviorFaultState
from spot_msgs.msg import SystemFault, SystemFaultState
from spot_msgs.msg import BatteryState, BatteryStateArray


class TestRobotStateCB(unittest.TestCase):
    def setUp(self):
        self.data = {}

    def joint_states_cb(self, joint_state: JointState):
        self.data["joint_state"] = joint_state

    def tf_cb(self, tf: TFMessage):
        # Differentiating between foot and body TFs
        if "foot" in tf.transforms[0].child_frame_id:
            self.data["foot_TF"] = tf
        else:
            self.data["TF"] = tf

    def twist_odom_cb(self, twist_odom: TwistWithCovarianceStamped):
        self.data["twist_odom"] = twist_odom

    def odom_cb(self, odom: Odometry):
        self.data["odom"] = odom

    def foot_cb(self, foot: FootStateArray):
        self.data["foot"] = foot

    def estop_cb(self, estop: EStopStateArray):
        self.data["estop"] = estop

    def wifi_cb(self, wifi: WiFiState):
        self.data["wifi"] = wifi

    def battery_cb(self, battery: BatteryStateArray):
        self.data["battery"] = battery

    def power_cb(self, power: PowerState):
        self.data["power"] = power

    def system_fault_cb(self, system_fault: SystemFaultState):
        self.data["system_fault"] = system_fault

    def behaviour_fault_cb(self, behaviour_fault: BehaviorFaultState):
        self.data["behaviour_fault"] = behaviour_fault

    def check_joint_states(self, joint_state: JointState):
        self.assertEqual(joint_state.name[0], "front_left_hip_x")
        self.assertEqual(joint_state.position[0], 1.0)
        self.assertEqual(joint_state.velocity[0], 2.0)
        self.assertEqual(joint_state.effort[0], 4.0)
        self.assertEqual(joint_state.name[1], "front_left_hip_y")
        self.assertEqual(joint_state.position[1], 5.0)
        self.assertEqual(joint_state.velocity[1], 6.0)
        self.assertEqual(joint_state.effort[1], 8.0)
        self.assertEqual(joint_state.header.stamp.secs, 30)
        self.assertEqual(joint_state.header.stamp.nsecs, 100)

    def check_foot_TF_states(self, tf_message: TFMessage):
        self.assertEqual(len(tf_message.transforms), 2)
        self.assertEqual(tf_message.transforms[0].transform.translation.x, 1.0)
        self.assertEqual(tf_message.transforms[0].transform.translation.y, 2.0)
        self.assertEqual(tf_message.transforms[0].transform.translation.z, 3.0)
        self.assertEqual(tf_message.transforms[0].header.frame_id, "body")
        self.assertEqual(tf_message.transforms[0].child_frame_id, "front_left_foot")
        self.assertEqual(tf_message.transforms[1].transform.translation.x, 4.0)
        self.assertEqual(tf_message.transforms[1].transform.translation.y, 5.0)
        self.assertEqual(tf_message.transforms[1].transform.translation.z, 6.0)
        self.assertEqual(tf_message.transforms[1].header.frame_id, "body")
        self.assertEqual(tf_message.transforms[1].child_frame_id, "front_right_foot")

    def check_TF_states(self, tf_message: TFMessage):
        self.assertEqual(len(tf_message.transforms), 2)
        self.assertEqual(tf_message.transforms[0].header.frame_id, "body")
        self.assertEqual(tf_message.transforms[0].child_frame_id, "odom")
        self.assertEqual(tf_message.transforms[0].transform.translation.x, -2.0)
        self.assertEqual(tf_message.transforms[0].transform.translation.y, -3.0)
        self.assertEqual(tf_message.transforms[0].transform.translation.z, -2.0)

        self.assertEqual(tf_message.transforms[1].header.frame_id, "vision")
        self.assertEqual(tf_message.transforms[1].child_frame_id, "body")
        self.assertEqual(tf_message.transforms[1].transform.translation.x, -2.0)
        self.assertEqual(tf_message.transforms[1].transform.translation.y, -3.0)
        self.assertEqual(tf_message.transforms[1].transform.translation.z, -2.0)

    def check_twist_odom_states(self, twist_odom_msg: TwistWithCovarianceStamped):
        self.assertEqual(twist_odom_msg.twist.twist.linear.x, 1.0)
        self.assertEqual(twist_odom_msg.twist.twist.linear.y, 2.0)
        self.assertEqual(twist_odom_msg.twist.twist.linear.z, 3.0)
        self.assertEqual(twist_odom_msg.twist.twist.angular.x, 4.0)
        self.assertEqual(twist_odom_msg.twist.twist.angular.y, 5.0)
        self.assertEqual(twist_odom_msg.twist.twist.angular.z, 6.0)

    def check_odom_states(self, odometry_msg: Odometry):
        # Add mock edges to relate the body, odom and vision frames. Body is the root frame.
        vision_tform_example = geometry_pb2.SE3Pose(
            position=geometry_pb2.Vec3(x=2, y=3, z=2),
            rotation=geometry_pb2.Quaternion(x=0, y=0, z=0, w=1),
        )
        body_tform_example = geometry_pb2.SE3Pose(
            position=geometry_pb2.Vec3(x=-2, y=-3, z=-2),
            rotation=geometry_pb2.Quaternion(x=0, y=0, z=0, w=1),
        )
        none_tform_example = geometry_pb2.SE3Pose(
            position=geometry_pb2.Vec3(x=0, y=0, z=0),
            rotation=geometry_pb2.Quaternion(x=0, y=0, z=0, w=1),
        )

        self.assertEqual(
            odometry_msg.pose.pose.position.x, -vision_tform_example.position.x
        )
        self.assertEqual(
            odometry_msg.pose.pose.position.y, -vision_tform_example.position.y
        )
        self.assertEqual(
            odometry_msg.pose.pose.position.z, -vision_tform_example.position.z
        )
        self.assertEqual(
            odometry_msg.pose.pose.orientation.x, vision_tform_example.rotation.x
        )
        self.assertEqual(
            odometry_msg.pose.pose.orientation.y, vision_tform_example.rotation.y
        )
        self.assertEqual(
            odometry_msg.pose.pose.orientation.z, vision_tform_example.rotation.z
        )
        self.assertEqual(
            odometry_msg.pose.pose.orientation.w, vision_tform_example.rotation.w
        )

    def check_foot_states(self, foot_state_array: FootStateArray):
        self.assertAlmostEqual(
            foot_state_array.states[0].foot_position_rt_body.y, 2.0, places=3
        )
        self.assertAlmostEqual(
            foot_state_array.states[0].foot_position_rt_body.z, 3.0, places=3
        )
        self.assertAlmostEqual(
            foot_state_array.states[0].foot_position_rt_body.x, 1.0, places=3
        )
        self.assertEqual(
            foot_state_array.states[0].contact, robot_state_pb2.FootState.CONTACT_MADE
        )
        self.assertAlmostEqual(
            foot_state_array.states[0].terrain.ground_mu_est, 0.5, places=3
        )
        self.assertEqual(foot_state_array.states[0].terrain.frame_name, "frame1")
        self.assertAlmostEqual(
            foot_state_array.states[0].terrain.foot_slip_distance_rt_frame.x,
            1.0,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[0].terrain.foot_slip_distance_rt_frame.y,
            2.0,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[0].terrain.foot_slip_distance_rt_frame.z,
            3.0,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[0].terrain.foot_slip_velocity_rt_frame.x,
            4.0,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[0].terrain.foot_slip_velocity_rt_frame.y,
            5.0,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[0].terrain.foot_slip_velocity_rt_frame.z,
            6.0,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[0].terrain.ground_contact_normal_rt_frame.x,
            7.0,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[0].terrain.ground_contact_normal_rt_frame.y,
            8.0,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[0].terrain.ground_contact_normal_rt_frame.z,
            9.0,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[0].terrain.visual_surface_ground_penetration_mean,
            0.1,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[0].terrain.visual_surface_ground_penetration_std,
            0.02,
            places=3,
        )

        self.assertAlmostEqual(
            foot_state_array.states[1].foot_position_rt_body.y, 5.0, places=3
        )
        self.assertAlmostEqual(
            foot_state_array.states[1].foot_position_rt_body.z, 6.0, places=3
        )
        self.assertAlmostEqual(
            foot_state_array.states[1].foot_position_rt_body.x, 4.0, places=3
        )
        self.assertAlmostEqual(
            foot_state_array.states[1].contact, robot_state_pb2.FootState.CONTACT_LOST
        )
        self.assertAlmostEqual(
            foot_state_array.states[1].terrain.ground_mu_est, 0.6, places=3
        )
        self.assertEqual(foot_state_array.states[1].terrain.frame_name, "frame2")
        self.assertAlmostEqual(
            foot_state_array.states[1].terrain.foot_slip_distance_rt_frame.x,
            10.0,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[1].terrain.foot_slip_distance_rt_frame.y,
            11.0,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[1].terrain.foot_slip_distance_rt_frame.z,
            12.0,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[1].terrain.foot_slip_velocity_rt_frame.x,
            13.0,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[1].terrain.foot_slip_velocity_rt_frame.y,
            14.0,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[1].terrain.foot_slip_velocity_rt_frame.z,
            15.0,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[1].terrain.ground_contact_normal_rt_frame.x,
            16.0,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[1].terrain.ground_contact_normal_rt_frame.y,
            17.0,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[1].terrain.ground_contact_normal_rt_frame.z,
            18.0,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[1].terrain.visual_surface_ground_penetration_mean,
            0.2,
            places=3,
        )
        self.assertAlmostEqual(
            foot_state_array.states[1].terrain.visual_surface_ground_penetration_std,
            0.03,
            places=3,
        )

    def check_estop_states(self, estop_state_array: EStopStateArray):
        self.assertEqual(estop_state_array.estop_states[0].name, "estop1")
        self.assertEqual(
            estop_state_array.estop_states[0].state,
            robot_state_pb2.EStopState.STATE_ESTOPPED,
        )
        self.assertEqual(
            estop_state_array.estop_states[0].type,
            robot_state_pb2.EStopState.TYPE_HARDWARE,
        )
        self.assertEqual(estop_state_array.estop_states[0].header.stamp.secs, 30)
        self.assertEqual(estop_state_array.estop_states[0].header.stamp.nsecs, 10)

        self.assertEqual(estop_state_array.estop_states[1].name, "estop2")
        self.assertEqual(
            estop_state_array.estop_states[1].state,
            robot_state_pb2.EStopState.STATE_NOT_ESTOPPED,
        )
        self.assertEqual(
            estop_state_array.estop_states[1].type,
            robot_state_pb2.EStopState.TYPE_SOFTWARE,
        )
        self.assertEqual(estop_state_array.estop_states[1].header.stamp.secs, 20)
        self.assertEqual(estop_state_array.estop_states[1].header.stamp.nsecs, 15)

    def check_wifi_states(self, wifi_state: WiFiState):
        self.assertEqual(
            wifi_state.current_mode, robot_state_pb2.WiFiState.MODE_ACCESS_POINT
        )
        self.assertEqual(wifi_state.essid, "test_essid")

    def check_battery_states(self, battery_states: BatteryStateArray):
        self.assertEqual(len(battery_states.battery_states), 1)
        self.assertEqual(battery_states.battery_states[0].header.stamp.secs, 1)
        self.assertEqual(battery_states.battery_states[0].header.stamp.nsecs, 2)
        self.assertEqual(battery_states.battery_states[0].identifier, "battery1")
        self.assertEqual(battery_states.battery_states[0].charge_percentage, 95.0)
        self.assertEqual(battery_states.battery_states[0].estimated_runtime.secs, 100)
        self.assertEqual(battery_states.battery_states[0].current, 10.0)
        self.assertEqual(battery_states.battery_states[0].voltage, 9.0)
        self.assertAlmostEqual(
            battery_states.battery_states[0].temperatures[0], 25.0, places=3
        )
        self.assertAlmostEqual(
            battery_states.battery_states[0].temperatures[1], 26.0, places=3
        )
        self.assertAlmostEqual(
            battery_states.battery_states[0].temperatures[2], 27.0, places=3
        )
        self.assertEqual(
            battery_states.battery_states[0].status,
            robot_state_pb2.BatteryState.STATUS_DISCHARGING,
        )

    def check_power_states(self, power_state_msg: PowerState):
        self.assertEqual(
            power_state_msg.motor_power_state,
            robot_state_pb2.PowerState.MOTOR_POWER_STATE_OFF,
        )
        self.assertEqual(
            power_state_msg.shore_power_state,
            robot_state_pb2.PowerState.SHORE_POWER_STATE_ON,
        )

    def check_system_fault_states(self, system_faults: SystemFaultState):
        self.assertEqual(system_faults.faults[0].name, "fault1")
        self.assertEqual(system_faults.faults[0].header.stamp.secs, 1)
        self.assertEqual(system_faults.faults[0].header.stamp.nsecs, 2)
        self.assertEqual(system_faults.faults[0].duration.secs, 3)
        self.assertEqual(system_faults.faults[0].duration.nsecs, 4)
        self.assertEqual(system_faults.faults[0].code, 42)
        self.assertEqual(system_faults.faults[0].uid, 5)
        self.assertEqual(system_faults.faults[0].error_message, "error message1")
        self.assertEqual(system_faults.faults[0].attributes, ["imu", "power"])
        self.assertEqual(
            system_faults.faults[0].severity, robot_state_pb2.SystemFault.SEVERITY_WARN
        )

        self.assertEqual(system_faults.historical_faults[0].name, "fault2")
        self.assertEqual(system_faults.historical_faults[0].header.stamp.secs, 6)
        self.assertEqual(system_faults.historical_faults[0].header.stamp.nsecs, 7)
        self.assertEqual(system_faults.historical_faults[0].duration.secs, 8)
        self.assertEqual(system_faults.historical_faults[0].duration.nsecs, 9)
        self.assertEqual(system_faults.historical_faults[0].code, 43)
        self.assertEqual(system_faults.historical_faults[0].uid, 10)
        self.assertEqual(
            system_faults.historical_faults[0].error_message, "error message2"
        )
        self.assertEqual(
            system_faults.historical_faults[0].attributes, ["wifi", "vision"]
        )
        self.assertEqual(
            system_faults.historical_faults[0].severity,
            robot_state_pb2.SystemFault.SEVERITY_CRITICAL,
        )

    def check_behaviour_fault_states(self, behavior_faults: BehaviorFaultState):
        self.assertEqual(behavior_faults.faults[0].behavior_fault_id, 1)
        self.assertEqual(behavior_faults.faults[0].header.stamp.secs, 1)
        self.assertEqual(behavior_faults.faults[0].header.stamp.nsecs, 2)
        self.assertEqual(
            behavior_faults.faults[0].cause, robot_state_pb2.BehaviorFault.CAUSE_FALL
        )
        self.assertEqual(
            behavior_faults.faults[0].status,
            robot_state_pb2.BehaviorFault.STATUS_UNCLEARABLE,
        )

        self.assertEqual(behavior_faults.faults[1].behavior_fault_id, 3)
        self.assertEqual(behavior_faults.faults[1].header.stamp.secs, 4)
        self.assertEqual(behavior_faults.faults[1].header.stamp.nsecs, 5)
        self.assertEqual(
            behavior_faults.faults[1].cause,
            robot_state_pb2.BehaviorFault.CAUSE_LEASE_TIMEOUT,
        )
        self.assertEqual(
            behavior_faults.faults[1].status,
            robot_state_pb2.BehaviorFault.STATUS_CLEARABLE,
        )

    def test_robot_state_cb(self):
        # Set up a subscriber to listen to joint_states, TF, twist_odom,
        # odom, foot, estop, wifi, battery, power, system fault, behaviour fault
        self.joint_states = rospy.Subscriber(
            "joint_states", JointState, self.joint_states_cb
        )
        self.tf = rospy.Subscriber("tf", TFMessage, self.tf_cb)
        self.twist_odom = rospy.Subscriber(
            "odometry/twist", TwistWithCovarianceStamped, self.twist_odom_cb
        )
        self.odom = rospy.Subscriber("odometry", Odometry, self.odom_cb)
        self.feet = rospy.Subscriber("status/feet", FootStateArray, self.foot_cb)
        self.estop = rospy.Subscriber("status/estop", EStopStateArray, self.estop_cb)
        self.wifi = rospy.Subscriber("status/wifi", WiFiState, self.wifi_cb)
        self.battery = rospy.Subscriber(
            "status/battery_states", BatteryStateArray, self.battery_cb
        )
        self.power = rospy.Subscriber("status/power_state", PowerState, self.power_cb)
        self.system_fault = rospy.Subscriber(
            "status/system_faults", SystemFaultState, self.system_fault_cb
        )
        self.behaviour_fault = rospy.Subscriber(
            "status/behavior_faults", BehaviorFaultState, self.behaviour_fault_cb
        )

        counter = 0
        while not rospy.is_shutdown() and counter < 10:
            time.sleep(1)
            counter += 1

        # Check if the data is not empty
        self.assertTrue("joint_state" in self.data, "Joint state is empty")
        self.assertTrue("TF" in self.data, "TF is empty")
        self.assertTrue("twist_odom" in self.data, "Twist odom is empty")
        self.assertTrue("odom" in self.data, "Odom is empty")
        self.assertTrue("foot" in self.data, "Foot is empty")
        self.assertTrue("estop" in self.data, "Estop is empty")
        self.assertTrue("wifi" in self.data, "Wifi is empty")
        self.assertTrue("battery" in self.data, "Battery is empty")
        self.assertTrue("power" in self.data, "Power is empty")
        self.assertTrue("system_fault" in self.data, "System fault is empty")
        self.assertTrue("behaviour_fault" in self.data, "Behaviour fault is empty")

        # Check contents of received data
        self.check_joint_states(self.data["joint_state"])
        self.check_foot_TF_states(self.data["foot_TF"])
        self.check_TF_states(self.data["TF"])
        self.check_twist_odom_states(self.data["twist_odom"])
        self.check_odom_states(self.data["odom"])
        self.check_foot_states(self.data["foot"])
        self.check_estop_states(self.data["estop"])
        self.check_wifi_states(self.data["wifi"])
        self.check_battery_states(self.data["battery"])
        self.check_power_states(self.data["power"])
        self.check_system_fault_states(self.data["system_fault"])
        self.check_behaviour_fault_states(self.data["behaviour_fault"])


class TestMetricsCB(unittest.TestCase):
    def setUp(self):
        self.data = {}

    def metrics_cb(self, data):
        self.data["metrics"] = data

    def check_metrics_data(self, metrics: Metrics):
        # Check against the data provided in mock_spot_ros.py
        self.assertEqual(metrics.header.stamp.secs, 1)
        self.assertEqual(metrics.header.stamp.nsecs, 2)
        self.assertAlmostEqual(metrics.distance, 3.0, 2)
        self.assertEqual(metrics.gait_cycles, 4)
        self.assertEqual(metrics.time_moving.secs, 5)
        self.assertEqual(metrics.time_moving.nsecs, 6)
        self.assertEqual(metrics.electric_power.secs, 7)
        self.assertEqual(metrics.electric_power.nsecs, 8)

    def test_metrics_cb(self):
        self.metrics = rospy.Subscriber("status/metrics", Metrics, self.metrics_cb)

        counter = 0
        while not rospy.is_shutdown() and counter < 10:
            time.sleep(1)
            counter += 1

        self.assertTrue("metrics" in self.data, "Metrics is empty")
        self.check_metrics_data(self.data["metrics"])


class TestLeaseCB(unittest.TestCase):
    def setUp(self):
        self.data = {}

    def lease_cb(self, data):
        self.data["lease"] = data

    def check_lease_data(self, lease: LeaseArray):
        # Check against the data provided in mock_spot_ros.py
        self.assertEqual(lease.resources[0].resource, "spot")
        self.assertEqual(lease.resources[0].lease.resource, "lease_id")
        self.assertEqual(lease.resources[0].lease.epoch, "epoch1")
        self.assertEqual(lease.resources[0].lease.sequence[0], 1)
        self.assertEqual(lease.resources[0].lease.sequence[1], 2)
        self.assertEqual(lease.resources[0].lease.sequence[2], 3)
        self.assertEqual(lease.resources[0].lease_owner.client_name, "Adam")
        self.assertEqual(lease.resources[0].lease_owner.user_name, "Dylan")

    def test_lease_cb(self):
        self.lease = rospy.Subscriber("status/leases", LeaseArray, self.lease_cb)

        counter = 0
        while not rospy.is_shutdown() and counter < 10:
            time.sleep(1)
            counter += 1

        self.assertTrue("lease" in self.data, "Lease is empty")
        self.check_lease_data(self.data["lease"])


class TestHandImageCB(unittest.TestCase):
    def setUp(self):
        self.data = {}

    def hand_image_mono_cb(self, data):
        self.data["hand_image_mono"] = data

    def hand_image_color_cb(self, data):
        self.data["hand_color_image"] = data

    def hand_depth_cb(self, data):
        self.data["hand_depth"] = data

    def hand_depth_in_hand_color_cb(self, data):
        self.data["hand_depth_in_hand_color"] = data

    def hand_image_mono_info_cb(self, data):
        self.data["hand_image_mono_info"] = data

    def hand_image_color_info_cb(self, data):
        self.data["hand_image_color_info"] = data

    def hand_depth_info_cb(self, data):
        self.data["hand_depth_info"] = data

    def hand_depth_in_color_info_cb(self, data):
        self.data["hand_depth_in_color_info"] = data

    def check_hand_image_data_mono(self, hand_image: Image):
        # Check against the data provided in mock_spot_ros.py
        self.assertEqual(hand_image.header.frame_id, "hand_image")
        self.assertEqual(hand_image.height, 640)
        self.assertEqual(hand_image.width, 480)
        self.assertEqual(hand_image.encoding, "mono8")
        self.assertEqual(hand_image.is_bigendian, True)
        self.assertEqual(hand_image.step, 480)
        self.assertEqual(hand_image.data, b"hand_image")

    def check_hand_image_data_depth(self, hand_image: Image):
        # Check against the data provided in mock_spot_ros.py
        self.assertEqual(hand_image.header.frame_id, "hand_depth")
        self.assertEqual(hand_image.height, 424)
        self.assertEqual(hand_image.width, 240)
        self.assertEqual(hand_image.encoding, "16UC1")
        self.assertEqual(hand_image.is_bigendian, False)
        self.assertEqual(hand_image.step, 2 * 240)
        self.assertEqual(hand_image.data, b"hand_depth")

    def check_hand_image_data_color(self, hand_image: Image):
        # Check against the data provided in mock_spot_ros.py
        self.assertEqual(hand_image.header.frame_id, "hand_color_image")
        self.assertEqual(hand_image.height, 640)
        self.assertEqual(hand_image.width, 480)
        self.assertEqual(hand_image.encoding, "rgb8")
        self.assertEqual(hand_image.is_bigendian, True)
        self.assertEqual(hand_image.step, 3 * 480)
        self.assertEqual(hand_image.data, b"hand_color_image")

    def check_hand_image_data_depth_in_color(self, hand_image: Image):
        # Check against the data provided in mock_spot_ros.py
        self.assertEqual(hand_image.header.frame_id, "hand_depth_in_color_frame")
        self.assertEqual(hand_image.height, 640)
        self.assertEqual(hand_image.width, 480)
        self.assertEqual(hand_image.encoding, "mono8")
        self.assertEqual(hand_image.is_bigendian, True)
        self.assertEqual(hand_image.step, 480)
        self.assertEqual(hand_image.data, b"hand_depth_in_color_frame")

    def check_hand_image_info_mono(self, hand_image_info: CameraInfo):
        # Check against the data provided in mock_spot_ros.py
        self.assertEqual(hand_image_info.header.frame_id, "hand_image")
        self.assertEqual(hand_image_info.height, 640)
        self.assertEqual(hand_image_info.width, 480)

    def check_hand_image_info_color(self, hand_image_info: CameraInfo):
        # Check against the data provided in mock_spot_ros.py
        self.assertEqual(hand_image_info.header.frame_id, "hand_color_image")
        self.assertEqual(hand_image_info.height, 640)
        self.assertEqual(hand_image_info.width, 480)

    def check_hand_image_info_depth(self, hand_image_info: CameraInfo):
        # Check against the data provided in mock_spot_ros.py
        self.assertEqual(hand_image_info.header.frame_id, "hand_depth")
        self.assertEqual(hand_image_info.height, 424)
        self.assertEqual(hand_image_info.width, 240)

    def check_hand_image_info_depth_in_color(self, hand_image_info: CameraInfo):
        # Check against the data provided in mock_spot_ros.py
        self.assertEqual(hand_image_info.header.frame_id, "hand_depth_in_color_frame")
        self.assertEqual(hand_image_info.height, 640)
        self.assertEqual(hand_image_info.width, 480)

    def test_hand_image_cb(self):
        self.hand_image_mono = rospy.Subscriber(
            "camera/hand_mono/image", Image, self.hand_image_mono_cb
        )
        self.hand_image_color = rospy.Subscriber(
            "camera/hand_color/image", Image, self.hand_image_color_cb
        )
        self.hand_depth = rospy.Subscriber(
            "depth/hand/image", Image, self.hand_depth_cb
        )
        self.hand_depth_in_hand_color = rospy.Subscriber(
            "depth/hand/depth_in_color", Image, self.hand_depth_in_hand_color_cb
        )

        self.hand_image_mono_info = rospy.Subscriber(
            "camera/hand_mono/camera_info", CameraInfo, self.hand_image_mono_info_cb
        )
        self.hand_image_color_info = rospy.Subscriber(
            "camera/hand_color/camera_info", CameraInfo, self.hand_image_color_info_cb
        )

        self.hand_depth_info = rospy.Subscriber(
            "depth/hand/camera_info", CameraInfo, self.hand_depth_info_cb
        )
        self.hand_depth_in_color_info = rospy.Subscriber(
            "camera/hand/depth_in_color/camera_info",
            CameraInfo,
            self.hand_depth_in_color_info_cb,
        )

        counter = 0
        while not rospy.is_shutdown() and counter < 10:
            time.sleep(1)
            counter += 1

        # Check that we got all the data
        self.assertTrue("hand_image_mono" in self.data, "Hand image is empty")
        self.assertTrue("hand_color_image" in self.data, "Hand image color is empty")
        self.assertTrue("hand_depth" in self.data, "Hand depth is empty")
        self.assertTrue(
            "hand_depth_in_hand_color" in self.data, "Hand depth in hand color is empty"
        )

        self.assertTrue(
            "hand_image_mono_info" in self.data, "Hand image mono info is empty"
        )
        self.assertTrue(
            "hand_image_color_info" in self.data, "Hand image color info is empty"
        )
        self.assertTrue("hand_depth_info" in self.data, "Hand depth info is empty")
        self.assertTrue(
            "hand_depth_in_color_info" in self.data, "Hand depth in color info is empty"
        )

        # Check that the data is valid
        self.check_hand_image_data_mono(self.data["hand_image_mono"])
        self.check_hand_image_data_color(self.data["hand_color_image"])
        self.check_hand_image_data_depth(self.data["hand_depth"])
        self.check_hand_image_data_depth_in_color(self.data["hand_depth_in_hand_color"])

        self.check_hand_image_info_mono(self.data["hand_image_mono_info"])
        self.check_hand_image_info_color(self.data["hand_image_color_info"])
        self.check_hand_image_info_depth(self.data["hand_depth_info"])
        self.check_hand_image_info_depth_in_color(self.data["hand_depth_in_color_info"])


# Test suite for SpotROS
class TestSuiteSpotROS(unittest.TestSuite):
    def __init__(self):
        super(TestSuiteSpotROS, self).__init__()
        self.loader = unittest.TestLoader()
        self.addTest(self.loader.loadTestsFromTestCase(TestRobotStateCB))
        self.addTest(self.loader.loadTestsFromTestCase(TestMetricsCB))
        self.addTest(self.loader.loadTestsFromTestCase(TestLeaseCB))
        self.addTest(self.loader.loadTestsFromTestCase(TestHandImageCB))


if __name__ == "__main__":
    print("Starting tests!")
    import rosunit

    rospy.init_node(NAME)

    rosunit.unitrun(PKG, NAME, TestRobotStateCB)
    rosunit.unitrun(PKG, NAME, TestMetricsCB)
    rosunit.unitrun(PKG, NAME, TestLeaseCB)
    rosunit.unitrun(PKG, NAME, TestHandImageCB)

    print("Tests complete!")
