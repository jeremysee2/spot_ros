#!/usr/bin/env python3
import rospy

from bosdyn.api import image_pb2, robot_state_pb2, lease_pb2, geometry_pb2
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from google.protobuf import wrappers_pb2, timestamp_pb2, duration_pb2
from bosdyn.client.frame_helpers import (
    add_edge_to_tree,
    VISION_FRAME_NAME,
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
)

from spot_driver.spot_ros import SpotROS
from spot_driver.spot_wrapper import SpotWrapper


# Stubbed SpotWrapper class for testing
class TestSpotWrapper(SpotWrapper):
    def __init__(self):
        self._robot_state = robot_state_pb2.RobotState()
        self._metrics = robot_state_pb2.RobotMetrics()
        self._lease_list = lease_pb2.ListLeasesResponse()
        self._valid = True
        self._robot_params = {
            "is_standing": False,
            "is_sitting": True,
            "is_moving": False,
            "robot_id": None,
            "estop_timeout": 9.0,
        }
        self._async_tasks = AsyncTasks()
        self._mobility_params = RobotCommandBuilder.mobility_params()

    @property
    def time_skew(self) -> duration_pb2.Duration:
        robot_time_skew = duration_pb2.Duration(seconds=0, nanos=0)
        return robot_time_skew

    @property
    def robot_state(self) -> robot_state_pb2.RobotState:
        """Return latest proto from the _robot_state_task"""
        return self._robot_state

    @robot_state.setter
    def robot_state(self, robot_state: robot_state_pb2.RobotState):
        """Set the robot_state_task proto"""
        self._robot_state = robot_state

    @property
    def metrics(self) -> robot_state_pb2.RobotMetrics:
        """Return latest proto from the _robot_metrics_task"""
        return self._metrics

    @metrics.setter
    def metrics(self, robot_metrics: robot_state_pb2.RobotMetrics):
        """Set the robot_metrics_task proto"""
        self._metrics = robot_metrics

    @property
    def lease(self) -> lease_pb2.ListLeasesResponse:
        """Return latest proto from the _lease_task"""
        return self._lease_list

    @lease.setter
    def lease(self, lease_list: lease_pb2.ListLeasesResponse):
        """Set the lease_task proto"""
        self._lease_list = lease_list

    def disconnect(self):
        pass


# Run the mock SpotROS class as a node
class MockSpotROS:
    def __init__(self):
        self.spot_ros = SpotROS()
        self.spot_ros.node_name = "mock_spot_ros"
        self.spot_ros.spot_wrapper = TestSpotWrapper()

    def set_joint_states(self, state: robot_state_pb2.RobotState):
        # Test getJointStatesFromState with two joints and acquisition_timestamp
        state.kinematic_state.joint_states.add(
            name="fl.hx",
            position=wrappers_pb2.DoubleValue(value=1.0),
            velocity=wrappers_pb2.DoubleValue(value=2.0),
            acceleration=wrappers_pb2.DoubleValue(value=3.0),
            load=wrappers_pb2.DoubleValue(value=4.0),
        )
        state.kinematic_state.joint_states.add(
            name="fl.hy",
            position=wrappers_pb2.DoubleValue(value=5.0),
            velocity=wrappers_pb2.DoubleValue(value=6.0),
            acceleration=wrappers_pb2.DoubleValue(value=7.0),
            load=wrappers_pb2.DoubleValue(value=8.0),
        )
        state.kinematic_state.acquisition_timestamp.seconds = 30
        state.kinematic_state.acquisition_timestamp.nanos = 100
        return state

    def set_TF_states(self, state: robot_state_pb2.RobotState):
        # Test with vision frame transformation
        # Add mock edges to relate the body, odom and vision frames. Body is the root frame.
        body_tform_example = geometry_pb2.SE3Pose(
            position=geometry_pb2.Vec3(x=-2, y=-3, z=-2),
            rotation=geometry_pb2.Quaternion(x=0, y=0, z=0, w=1),
        )
        vision_tform_example = geometry_pb2.SE3Pose(
            position=geometry_pb2.Vec3(x=2, y=3, z=2),
            rotation=geometry_pb2.Quaternion(x=0, y=0, z=0, w=1),
        )
        none_tform_example = geometry_pb2.SE3Pose(
            position=geometry_pb2.Vec3(x=0, y=0, z=0),
            rotation=geometry_pb2.Quaternion(x=0, y=0, z=0, w=1),
        )
        edges = {}
        edges = add_edge_to_tree(
            edges, body_tform_example, BODY_FRAME_NAME, ODOM_FRAME_NAME
        )
        edges = add_edge_to_tree(
            edges, vision_tform_example, BODY_FRAME_NAME, VISION_FRAME_NAME
        )
        edges = add_edge_to_tree(edges, none_tform_example, "", BODY_FRAME_NAME)

        snapshot = geometry_pb2.FrameTreeSnapshot(child_to_parent_edge_map=edges)
        state.kinematic_state.transforms_snapshot.CopyFrom(snapshot)
        return state

    def set_twist_odom_states(self, state: robot_state_pb2.RobotState):
        state.kinematic_state.velocity_of_body_in_odom.linear.x = 1.0
        state.kinematic_state.velocity_of_body_in_odom.linear.y = 2.0
        state.kinematic_state.velocity_of_body_in_odom.linear.z = 3.0
        state.kinematic_state.velocity_of_body_in_odom.angular.x = 4.0
        state.kinematic_state.velocity_of_body_in_odom.angular.y = 5.0
        state.kinematic_state.velocity_of_body_in_odom.angular.z = 6.0
        return state

    def set_odom_states(self, state: robot_state_pb2.RobotState):
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
        edges = {}
        edges = add_edge_to_tree(
            edges, body_tform_example, BODY_FRAME_NAME, ODOM_FRAME_NAME
        )
        edges = add_edge_to_tree(
            edges, vision_tform_example, BODY_FRAME_NAME, VISION_FRAME_NAME
        )
        edges = add_edge_to_tree(edges, none_tform_example, "", BODY_FRAME_NAME)

        snapshot = geometry_pb2.FrameTreeSnapshot(child_to_parent_edge_map=edges)
        state.kinematic_state.transforms_snapshot.CopyFrom(snapshot)
        return state

    def set_foot_states(self, state: robot_state_pb2.RobotState):
        # Test getFeetFromState with two feet
        state.foot_state.add(
            foot_position_rt_body=geometry_pb2.Vec3(x=1.0, y=2.0, z=3.0),
            contact=robot_state_pb2.FootState.CONTACT_MADE,
            terrain=robot_state_pb2.FootState.TerrainState(
                ground_mu_est=0.5,
                frame_name="frame1",
                foot_slip_distance_rt_frame=geometry_pb2.Vec3(x=1.0, y=2.0, z=3.0),
                foot_slip_velocity_rt_frame=geometry_pb2.Vec3(x=4.0, y=5.0, z=6.0),
                ground_contact_normal_rt_frame=geometry_pb2.Vec3(x=7.0, y=8.0, z=9.0),
                visual_surface_ground_penetration_mean=0.1,
                visual_surface_ground_penetration_std=0.02,
            ),
        )
        state.foot_state.add(
            foot_position_rt_body=geometry_pb2.Vec3(x=4.0, y=5.0, z=6.0),
            contact=robot_state_pb2.FootState.CONTACT_LOST,
            terrain=robot_state_pb2.FootState.TerrainState(
                ground_mu_est=0.6,
                frame_name="frame2",
                foot_slip_distance_rt_frame=geometry_pb2.Vec3(x=10.0, y=11.0, z=12.0),
                foot_slip_velocity_rt_frame=geometry_pb2.Vec3(x=13.0, y=14.0, z=15.0),
                ground_contact_normal_rt_frame=geometry_pb2.Vec3(
                    x=16.0, y=17.0, z=18.0
                ),
                visual_surface_ground_penetration_mean=0.2,
                visual_surface_ground_penetration_std=0.03,
            ),
        )

        return state

    def set_estop_states(self, state: robot_state_pb2.RobotState):
        # Test getEStopStateFromState, hardware estopped
        state.estop_states.add(
            timestamp=timestamp_pb2.Timestamp(seconds=30, nanos=10),
            name="estop1",
            state=robot_state_pb2.EStopState.STATE_ESTOPPED,
            type=robot_state_pb2.EStopState.TYPE_HARDWARE,
        )
        # Add software type estop state, not estopped
        state.estop_states.add(
            timestamp=timestamp_pb2.Timestamp(seconds=20, nanos=15),
            name="estop2",
            state=robot_state_pb2.EStopState.STATE_NOT_ESTOPPED,
            type=robot_state_pb2.EStopState.TYPE_SOFTWARE,
        )

        return state

    def set_wifi_states(self, state: robot_state_pb2.RobotState):
        initial_wifi_state = robot_state_pb2.WiFiState(
            current_mode=robot_state_pb2.WiFiState.MODE_ACCESS_POINT, essid="test_essid"
        )
        state.comms_states.add(wifi_state=initial_wifi_state)

        return state

    def set_battery_states(self, state: robot_state_pb2.RobotState):
        # Test with one battery state
        state.battery_states.add(
            timestamp=timestamp_pb2.Timestamp(seconds=1, nanos=2),
            identifier="battery1",
            charge_percentage=wrappers_pb2.DoubleValue(value=95.0),
            estimated_runtime=duration_pb2.Duration(seconds=100),
            current=wrappers_pb2.DoubleValue(value=10.0),
            voltage=wrappers_pb2.DoubleValue(value=9.0),
            temperatures=[25.0, 26.0, 27.0],
            status=robot_state_pb2.BatteryState.STATUS_DISCHARGING,
        )
        return state

    def set_power_states(self, state: robot_state_pb2.RobotState):
        state.power_state.motor_power_state = (
            robot_state_pb2.PowerState.MOTOR_POWER_STATE_OFF
        )
        state.power_state.shore_power_state = (
            robot_state_pb2.PowerState.SHORE_POWER_STATE_ON
        )
        return state

    def set_system_fault_states(self, state: robot_state_pb2.RobotState):
        state.system_fault_state.faults.add(
            name="fault1",
            onset_timestamp=timestamp_pb2.Timestamp(seconds=1, nanos=2),
            duration=duration_pb2.Duration(seconds=3, nanos=4),
            code=42,
            uid=5,
            error_message="error message1",
            attributes=["imu", "power"],
            severity=robot_state_pb2.SystemFault.SEVERITY_WARN,
        )

        state.system_fault_state.historical_faults.add(
            name="fault2",
            onset_timestamp=timestamp_pb2.Timestamp(seconds=6, nanos=7),
            duration=duration_pb2.Duration(seconds=8, nanos=9),
            code=43,
            uid=10,
            error_message="error message2",
            attributes=["wifi", "vision"],
            severity=robot_state_pb2.SystemFault.SEVERITY_CRITICAL,
        )

        return state

    def set_behaviour_fault_states(self, state: robot_state_pb2.RobotState):
        state.behavior_fault_state.faults.add(
            behavior_fault_id=1,
            onset_timestamp=timestamp_pb2.Timestamp(seconds=1, nanos=2),
            cause=robot_state_pb2.BehaviorFault.CAUSE_FALL,
            status=robot_state_pb2.BehaviorFault.STATUS_UNCLEARABLE,
        )

        state.behavior_fault_state.faults.add(
            behavior_fault_id=3,
            onset_timestamp=timestamp_pb2.Timestamp(seconds=4, nanos=5),
            cause=robot_state_pb2.BehaviorFault.CAUSE_LEASE_TIMEOUT,
            status=robot_state_pb2.BehaviorFault.STATUS_CLEARABLE,
        )

        return state

    def set_robot_state(self):
        # Create a robot state message inside the spot_wrapper object
        self.spot_ros.spot_wrapper.robot_state = robot_state_pb2.RobotState()  # type: ignore

        # Set the robot state message's joint state field
        self.set_joint_states(self.spot_ros.spot_wrapper.robot_state)
        self.set_TF_states(self.spot_ros.spot_wrapper.robot_state)
        self.set_twist_odom_states(self.spot_ros.spot_wrapper.robot_state)
        self.set_odom_states(self.spot_ros.spot_wrapper.robot_state)
        self.set_foot_states(self.spot_ros.spot_wrapper.robot_state)
        self.set_estop_states(self.spot_ros.spot_wrapper.robot_state)
        self.set_wifi_states(self.spot_ros.spot_wrapper.robot_state)
        self.set_battery_states(self.spot_ros.spot_wrapper.robot_state)
        self.set_power_states(self.spot_ros.spot_wrapper.robot_state)
        self.set_system_fault_states(self.spot_ros.spot_wrapper.robot_state)
        self.set_behaviour_fault_states(self.spot_ros.spot_wrapper.robot_state)

    def set_robot_metrics(self):
        # Create a robot metrics message inside the spot_wrapper object
        self.spot_ros.spot_wrapper.metrics = robot_state_pb2.RobotMetrics()  # type: ignore

        # Populate the Metrics message field with a timestamp and metrics
        self.spot_ros.spot_wrapper.metrics.timestamp.seconds = 1
        self.spot_ros.spot_wrapper.metrics.timestamp.nanos = 2
        self.spot_ros.spot_wrapper.metrics.metrics.add(
            label="distance", float_value=3.0
        )
        self.spot_ros.spot_wrapper.metrics.metrics.add(label="gait cycles", int_value=4)
        self.spot_ros.spot_wrapper.metrics.metrics.add(
            label="time moving", duration=duration_pb2.Duration(seconds=5, nanos=6)
        )
        self.spot_ros.spot_wrapper.metrics.metrics.add(
            label="electric power", duration=duration_pb2.Duration(seconds=7, nanos=8)
        )

    def set_robot_lease(self):
        # Create a robot lease message inside the spot_wrapper object
        list_lease_resp = lease_pb2.ListLeasesResponse()  # type: ignore

        # Populate the lease message field with a timestamp and lease
        list_lease_resp.resources.add(
            resource="spot",
            lease=lease_pb2.Lease(
                resource="lease_id",
                epoch="epoch1",
                sequence=[1, 2, 3],
                client_names=["Adam", "Bob", "Charlie"],
            ),
            lease_owner=lease_pb2.LeaseOwner(client_name="Adam", user_name="Dylan"),
            stale_time=timestamp_pb2.Timestamp(seconds=4, nanos=5),
        )

        self.spot_ros.spot_wrapper.lease = list_lease_resp.resources  # type: ignore

    def main(self):
        rospy.init_node(self.spot_ros.node_name, anonymous=True)
        # Initialize variables for transforms
        self.spot_ros.mode_parent_odom_tf = "vision"

        # Set up the publishers
        self.spot_ros.initialize_publishers()

        # Manually set robot_state, metrics
        self.set_robot_state()
        self.set_robot_metrics()
        self.set_robot_lease()

        while not rospy.is_shutdown():
            # Call publish callbacks
            self.spot_ros.RobotStateCB("robot_state_test")
            self.spot_ros.MetricsCB("metrics_test")
            self.spot_ros.LeaseCB("lease_test")


if __name__ == "__main__":
    run_spot_ros = MockSpotROS()
    run_spot_ros.main()
