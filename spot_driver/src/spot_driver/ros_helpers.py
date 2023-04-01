import typing
import numpy as np

import rospy

from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import JointState
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseWithCovariance
from geometry_msgs.msg import TwistWithCovarianceStamped
from geometry_msgs.msg import TransformStamped, Transform
from geometry_msgs.msg import Pose, Point, Quaternion, Polygon, Vector3, Point32
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage

from spot_msgs.msg import FootState, FootStateArray
from spot_msgs.msg import EStopState, EStopStateArray
from spot_msgs.msg import WiFiState
from spot_msgs.msg import PowerState
from spot_msgs.msg import BehaviorFault, BehaviorFaultState
from spot_msgs.msg import SystemFault, SystemFaultState
from spot_msgs.msg import BatteryState, BatteryStateArray
from spot_msgs.msg import DockState
from spot_msgs.msg import (
    WorldObject,
    WorldObjectArray,
    AprilTagProperties,
    ImageProperties,
)
from spot_msgs.msg import FrameTreeSnapshot, ParentEdge
from spot_msgs.msg import GraphNavGraph, GraphNavWaypoint
from spot_msgs.srv import SpotCheckResponse
from spot_msgs.srv import UploadGraphRequest, UploadGraphResponse

from bosdyn.api import image_pb2, robot_state_pb2, point_cloud_pb2
from bosdyn.api import world_object_pb2, geometry_pb2
from bosdyn.api.docking import docking_pb2
from bosdyn.api.graph_nav import map_pb2
from bosdyn.client.spot_check import spot_check_pb2
from bosdyn.client.math_helpers import SE3Pose
from bosdyn.client.frame_helpers import get_odom_tform_body, get_vision_tform_body

from google.protobuf.timestamp_pb2 import Timestamp

from .spot_wrapper import SpotWrapper


friendly_joint_names = {}
"""Dictionary for mapping BD joint names to more friendly names"""
friendly_joint_names["fl.hx"] = "front_left_hip_x"
friendly_joint_names["fl.hy"] = "front_left_hip_y"
friendly_joint_names["fl.kn"] = "front_left_knee"
friendly_joint_names["fr.hx"] = "front_right_hip_x"
friendly_joint_names["fr.hy"] = "front_right_hip_y"
friendly_joint_names["fr.kn"] = "front_right_knee"
friendly_joint_names["hl.hx"] = "rear_left_hip_x"
friendly_joint_names["hl.hy"] = "rear_left_hip_y"
friendly_joint_names["hl.kn"] = "rear_left_knee"
friendly_joint_names["hr.hx"] = "rear_right_hip_x"
friendly_joint_names["hr.hy"] = "rear_right_hip_y"
friendly_joint_names["hr.kn"] = "rear_right_knee"

# arm joints
friendly_joint_names["arm0.sh0"] = "arm_joint1"
friendly_joint_names["arm0.sh1"] = "arm_joint2"
friendly_joint_names["arm0.el0"] = "arm_joint3"
friendly_joint_names["arm0.el1"] = "arm_joint4"
friendly_joint_names["arm0.wr0"] = "arm_joint5"
friendly_joint_names["arm0.wr1"] = "arm_joint6"
friendly_joint_names["arm0.f1x"] = "arm_gripper"


class DefaultCameraInfo(CameraInfo):
    """Blank class extending CameraInfo ROS topic that defaults most parameters"""

    def __init__(self):
        super().__init__()
        self.distortion_model = "plumb_bob"

        self.D.append(0)
        self.D.append(0)
        self.D.append(0)
        self.D.append(0)
        self.D.append(0)

        self.K[1] = 0
        self.K[3] = 0
        self.K[6] = 0
        self.K[7] = 0
        self.K[8] = 1

        self.R[0] = 1
        self.R[1] = 0
        self.R[2] = 0
        self.R[3] = 0
        self.R[4] = 1
        self.R[5] = 0
        self.R[6] = 0
        self.R[7] = 0
        self.R[8] = 1

        self.P[1] = 0
        self.P[3] = 0
        self.P[4] = 0
        self.P[7] = 0
        self.P[8] = 0
        self.P[9] = 0
        self.P[10] = 1
        self.P[11] = 0


def populateTransformStamped(
    time: rospy.Time, parent_frame: str, child_frame: str, transform: Transform
) -> TransformStamped:
    """Populates a TransformStamped message

    Args:
        time: The time of the transform
        parent_frame: The parent frame of the transform
        child_frame: The child_frame_id of the transform
        transform: A transform to copy into a StampedTransform object. Should have position (x,y,z) and rotation (x,
        y,z,w) members
    Returns:
        TransformStamped message. Empty if transform does not have position or translation attribute
    """
    if hasattr(transform, "position"):
        position = transform.position
    elif hasattr(transform, "translation"):
        position = transform.translation
    else:
        rospy.logerr(
            "Trying to generate StampedTransform but input transform has neither position nor translation "
            "attributes"
        )
        return TransformStamped()

    new_tf = TransformStamped()
    new_tf.header.stamp = time
    new_tf.header.frame_id = parent_frame
    new_tf.child_frame_id = child_frame
    new_tf.transform.translation.x = position.x
    new_tf.transform.translation.y = position.y
    new_tf.transform.translation.z = position.z
    new_tf.transform.rotation.x = transform.rotation.x
    new_tf.transform.rotation.y = transform.rotation.y
    new_tf.transform.rotation.z = transform.rotation.z
    new_tf.transform.rotation.w = transform.rotation.w

    return new_tf


def GetImageMsg(
    data: image_pb2.ImageResponse, spot_wrapper: SpotWrapper
) -> typing.Tuple[Image, CameraInfo]:
    """Takes the image and camera data and populates the necessary ROS messages

    Args:
        data: Image proto
        spot_wrapper: A SpotWrapper object
    Returns:
        (tuple):
            * Image: message of the image captured
            * CameraInfo: message to define the state and config of the camera that took the image
    """
    image_msg = Image()
    local_time = spot_wrapper.robotToLocalTime(data.shot.acquisition_time)
    image_msg.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
    image_msg.header.frame_id = data.shot.frame_name_image_sensor
    image_msg.height = data.shot.image.rows
    image_msg.width = data.shot.image.cols

    # Color/greyscale formats.
    # JPEG format
    if data.shot.image.format == image_pb2.Image.FORMAT_JPEG:
        image_msg.encoding = "rgb8"
        image_msg.is_bigendian = True
        image_msg.step = 3 * data.shot.image.cols
        image_msg.data = data.shot.image.data

    # Uncompressed.  Requires pixel_format.
    if data.shot.image.format == image_pb2.Image.FORMAT_RAW:
        # One byte per pixel.
        if data.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            image_msg.encoding = "mono8"
            image_msg.is_bigendian = True
            image_msg.step = data.shot.image.cols
            image_msg.data = data.shot.image.data

        # Three bytes per pixel.
        if data.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            image_msg.encoding = "rgb8"
            image_msg.is_bigendian = True
            image_msg.step = 3 * data.shot.image.cols
            image_msg.data = data.shot.image.data

        # Four bytes per pixel.
        if data.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            image_msg.encoding = "rgba8"
            image_msg.is_bigendian = True
            image_msg.step = 4 * data.shot.image.cols
            image_msg.data = data.shot.image.data

        # Little-endian uint16 z-distance from camera (mm).
        if data.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            image_msg.encoding = "16UC1"
            image_msg.is_bigendian = False
            image_msg.step = 2 * data.shot.image.cols
            image_msg.data = data.shot.image.data

    camera_info_msg = DefaultCameraInfo()
    local_time = spot_wrapper.robotToLocalTime(data.shot.acquisition_time)
    camera_info_msg.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
    camera_info_msg.header.frame_id = data.shot.frame_name_image_sensor
    camera_info_msg.height = data.shot.image.rows
    camera_info_msg.width = data.shot.image.cols

    camera_info_msg.K[0] = data.source.pinhole.intrinsics.focal_length.x
    camera_info_msg.K[2] = data.source.pinhole.intrinsics.principal_point.x
    camera_info_msg.K[4] = data.source.pinhole.intrinsics.focal_length.y
    camera_info_msg.K[5] = data.source.pinhole.intrinsics.principal_point.y

    camera_info_msg.P[0] = data.source.pinhole.intrinsics.focal_length.x
    camera_info_msg.P[2] = data.source.pinhole.intrinsics.principal_point.x
    camera_info_msg.P[5] = data.source.pinhole.intrinsics.focal_length.y
    camera_info_msg.P[6] = data.source.pinhole.intrinsics.principal_point.y

    return image_msg, camera_info_msg


def GetPointCloudMsg(
    data: point_cloud_pb2.PointCloudResponse, spot_wrapper: SpotWrapper
) -> PointCloud2:
    """Takes the imag and  camera data and populates the necessary ROS messages

    Args:
        data: PointCloud proto (PointCloudResponse)
        spot_wrapper: A SpotWrapper object
    Returns:
           PointCloud: message of the point cloud (PointCloud2)
    """
    point_cloud_msg = PointCloud2()
    local_time = spot_wrapper.robotToLocalTime(data.point_cloud.source.acquisition_time)
    point_cloud_msg.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
    point_cloud_msg.header.frame_id = data.point_cloud.source.frame_name_sensor
    if data.point_cloud.encoding == point_cloud_pb2.PointCloud.ENCODING_XYZ_32F:
        point_cloud_msg.height = 1
        point_cloud_msg.width = data.point_cloud.num_points
        point_cloud_msg.fields = []
        for i, ax in enumerate(("x", "y", "z")):
            field = PointField()
            field.name = ax
            field.offset = i * 4
            field.datatype = PointField.FLOAT32
            field.count = 1
            point_cloud_msg.fields.append(field)
        point_cloud_msg.is_bigendian = False
        point_cloud_np = np.frombuffer(data.point_cloud.data, dtype=np.uint8)
        point_cloud_msg.point_step = 12  # float32 XYZ
        point_cloud_msg.row_step = point_cloud_msg.width * point_cloud_msg.point_step
        point_cloud_msg.data = point_cloud_np.tobytes()
        point_cloud_msg.is_dense = True
    else:
        rospy.logwarn("Not supported point cloud data type.")
    return point_cloud_msg


def GetJointStatesFromState(
    state: robot_state_pb2.RobotState, spot_wrapper: SpotWrapper
):
    """Maps joint state data from robot state proto to ROS JointState message

    Args:
        data: Robot State proto
        spot_wrapper: A SpotWrapper object
    Returns:
        JointState message
    """
    joint_state = JointState()
    local_time = spot_wrapper.robotToLocalTime(
        state.kinematic_state.acquisition_timestamp
    )
    joint_state.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
    for joint in state.kinematic_state.joint_states:
        # there is a joint with name arm0.hr0 in the robot state, however this
        # joint has no data and should not be there, this is why we ignore it
        if joint.name == "arm0.hr0":
            continue
        joint_state.name.append(friendly_joint_names.get(joint.name, "ERROR"))
        joint_state.position.append(joint.position.value)
        joint_state.velocity.append(joint.velocity.value)
        joint_state.effort.append(joint.load.value)

    return joint_state


def GetEStopStateFromState(
    state: robot_state_pb2.RobotState, spot_wrapper: SpotWrapper
) -> EStopStateArray:
    """Maps eStop state data from robot state proto to ROS EStopArray message

    Args:
        data: Robot State proto
        spot_wrapper: A SpotWrapper object
    Returns:
        EStopArray message
    """
    estop_array_msg = EStopStateArray()
    for estop in state.estop_states:
        estop_msg = EStopState()
        local_time = spot_wrapper.robotToLocalTime(estop.timestamp)
        estop_msg.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
        estop_msg.name = estop.name
        estop_msg.type = estop.type
        estop_msg.state = estop.state
        estop_msg.state_description = estop.state_description
        estop_array_msg.estop_states.append(estop_msg)

    return estop_array_msg


def GetFeetFromState(
    state: robot_state_pb2.RobotState, spot_wrapper: SpotWrapper
) -> FootStateArray:
    """Maps foot position state data from robot state proto to ROS FootStateArray message

    Args:
        data: Robot State proto
        spot_wrapper: A SpotWrapper object
    Returns:
        FootStateArray message
    """
    foot_array_msg = FootStateArray()
    for foot in state.foot_state:
        foot_msg = FootState()
        foot_msg.foot_position_rt_body.x = foot.foot_position_rt_body.x
        foot_msg.foot_position_rt_body.y = foot.foot_position_rt_body.y
        foot_msg.foot_position_rt_body.z = foot.foot_position_rt_body.z
        foot_msg.contact = foot.contact

        if foot.HasField("terrain"):
            terrain = foot.terrain
            foot_msg.terrain.ground_mu_est = terrain.ground_mu_est
            foot_msg.terrain.frame_name = terrain.frame_name
            foot_msg.terrain.foot_slip_distance_rt_frame = (
                terrain.foot_slip_distance_rt_frame
            )
            foot_msg.terrain.foot_slip_velocity_rt_frame = (
                terrain.foot_slip_velocity_rt_frame
            )
            foot_msg.terrain.ground_contact_normal_rt_frame = (
                terrain.ground_contact_normal_rt_frame
            )
            foot_msg.terrain.visual_surface_ground_penetration_mean = (
                terrain.visual_surface_ground_penetration_mean
            )
            foot_msg.terrain.visual_surface_ground_penetration_std = (
                terrain.visual_surface_ground_penetration_std
            )

        foot_array_msg.states.append(foot_msg)

    return foot_array_msg


def GetOdomTwistFromState(
    state: robot_state_pb2.RobotState, spot_wrapper: SpotWrapper
) -> TwistWithCovarianceStamped:
    """Maps odometry data from robot state proto to ROS TwistWithCovarianceStamped message

    Args:
        data: Robot State proto
        spot_wrapper: A SpotWrapper object
    Returns:
        TwistWithCovarianceStamped message
    """
    twist_odom_msg = TwistWithCovarianceStamped()
    local_time = spot_wrapper.robotToLocalTime(
        state.kinematic_state.acquisition_timestamp
    )
    twist_odom_msg.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
    twist_odom_msg.twist.twist.linear.x = (
        state.kinematic_state.velocity_of_body_in_odom.linear.x
    )
    twist_odom_msg.twist.twist.linear.y = (
        state.kinematic_state.velocity_of_body_in_odom.linear.y
    )
    twist_odom_msg.twist.twist.linear.z = (
        state.kinematic_state.velocity_of_body_in_odom.linear.z
    )
    twist_odom_msg.twist.twist.angular.x = (
        state.kinematic_state.velocity_of_body_in_odom.angular.x
    )
    twist_odom_msg.twist.twist.angular.y = (
        state.kinematic_state.velocity_of_body_in_odom.angular.y
    )
    twist_odom_msg.twist.twist.angular.z = (
        state.kinematic_state.velocity_of_body_in_odom.angular.z
    )
    return twist_odom_msg


def GetOdomFromState(
    state: robot_state_pb2.RobotState, spot_wrapper: SpotWrapper, use_vision=True
):
    """Maps odometry data from robot state proto to ROS Odometry message

    Args:
        data: Robot State proto
        spot_wrapper: A SpotWrapper object
    Returns:
        Odometry message
    """
    odom_msg = Odometry()
    local_time = spot_wrapper.robotToLocalTime(
        state.kinematic_state.acquisition_timestamp
    )
    odom_msg.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
    if use_vision == True:
        odom_msg.header.frame_id = "vision"
        tform_body = get_vision_tform_body(state.kinematic_state.transforms_snapshot)
    else:
        odom_msg.header.frame_id = "odom"
        tform_body = get_odom_tform_body(state.kinematic_state.transforms_snapshot)
    odom_msg.child_frame_id = "body"
    pose_odom_msg = PoseWithCovariance()
    pose_odom_msg.pose.position.x = tform_body.position.x
    pose_odom_msg.pose.position.y = tform_body.position.y
    pose_odom_msg.pose.position.z = tform_body.position.z
    pose_odom_msg.pose.orientation.x = tform_body.rotation.x
    pose_odom_msg.pose.orientation.y = tform_body.rotation.y
    pose_odom_msg.pose.orientation.z = tform_body.rotation.z
    pose_odom_msg.pose.orientation.w = tform_body.rotation.w

    odom_msg.pose = pose_odom_msg
    twist_odom_msg = GetOdomTwistFromState(state, spot_wrapper).twist
    odom_msg.twist = twist_odom_msg
    return odom_msg


def GetWifiFromState(
    state: robot_state_pb2.RobotState, spot_wrapper: SpotWrapper
) -> WiFiState:
    """Maps wireless state data from robot state proto to ROS WiFiState message

    Args:
        data: Robot State proto
        spot_wrapper: A SpotWrapper object
    Returns:
        WiFiState message
    """
    wifi_msg = WiFiState()
    for comm_state in state.comms_states:
        if comm_state.HasField("wifi_state"):
            wifi_msg.current_mode = comm_state.wifi_state.current_mode
            wifi_msg.essid = comm_state.wifi_state.essid

    return wifi_msg


def GenerateFeetTF(
    foot_states_msg: FootStateArray, time_now: typing.Optional[rospy.Time] = None
) -> TFMessage:
    """
    Generate a tf message containing information about foot states

    Args:
        foot_states_msg: FootStateArray message containing the foot states from the robot state

    Returns: tf message with foot states

    """
    foot_ordering = ["front_left", "front_right", "rear_left", "rear_right"]
    foot_tfs = TFMessage()
    if not time_now:
        time_now = rospy.Time.now()
    for idx, foot_state in enumerate(foot_states_msg.states):
        foot_transform = Transform()
        # Rotation of the foot is not given
        foot_transform.rotation.w = 1
        foot_transform.translation.x = foot_state.foot_position_rt_body.x
        foot_transform.translation.y = foot_state.foot_position_rt_body.y
        foot_transform.translation.z = foot_state.foot_position_rt_body.z

        foot_tfs.transforms.append(
            populateTransformStamped(
                time_now, "body", foot_ordering[idx] + "_foot", foot_transform
            )
        )

    return foot_tfs


def GetTFFromState(
    state: robot_state_pb2.RobotState,
    spot_wrapper: SpotWrapper,
    inverse_target_frame: str,
) -> TFMessage:
    """Maps robot link state data from robot state proto to ROS TFMessage message

    Args:
        data: Robot State proto
        spot_wrapper: A SpotWrapper object
        inverse_target_frame: A frame name to be inversed to a parent frame.
    Returns:
        TFMessage message
    """
    tf_msg = TFMessage()

    for (
        frame_name
    ) in state.kinematic_state.transforms_snapshot.child_to_parent_edge_map:
        if state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get(
            frame_name
        ).parent_frame_name:
            try:
                transform = state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get(
                    frame_name
                )
                local_time = spot_wrapper.robotToLocalTime(
                    state.kinematic_state.acquisition_timestamp
                )
                tf_time = rospy.Time(local_time.seconds, local_time.nanos)
                if inverse_target_frame == frame_name:
                    geo_tform_inversed = SE3Pose.from_obj(
                        transform.parent_tform_child
                    ).inverse()
                    new_tf = populateTransformStamped(
                        tf_time,
                        frame_name,
                        transform.parent_frame_name,
                        geo_tform_inversed,
                    )
                else:
                    new_tf = populateTransformStamped(
                        tf_time,
                        transform.parent_frame_name,
                        frame_name,
                        transform.parent_tform_child,
                    )
                tf_msg.transforms.append(new_tf)
            except Exception as e:
                spot_wrapper.logger.error("Error: {}".format(e))

    return tf_msg


def GetBatteryStatesFromState(
    state: robot_state_pb2.RobotState, spot_wrapper: SpotWrapper
) -> BatteryStateArray:
    """Maps battery state data from robot state proto to ROS BatteryStateArray message

    Args:
        data: Robot State proto
        spot_wrapper: A SpotWrapper object
    Returns:
        BatteryStateArray message
    """
    battery_states_array_msg = BatteryStateArray()
    for battery in state.battery_states:
        battery_msg = BatteryState()
        local_time = spot_wrapper.robotToLocalTime(battery.timestamp)
        battery_msg.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)

        battery_msg.identifier = battery.identifier
        battery_msg.charge_percentage = battery.charge_percentage.value
        battery_msg.estimated_runtime = rospy.Time(
            battery.estimated_runtime.seconds, battery.estimated_runtime.nanos
        )
        battery_msg.current = battery.current.value
        battery_msg.voltage = battery.voltage.value
        for temp in battery.temperatures:
            battery_msg.temperatures.append(temp)
        battery_msg.status = battery.status
        battery_states_array_msg.battery_states.append(battery_msg)

    return battery_states_array_msg


def GetPowerStatesFromState(
    state: robot_state_pb2.RobotState, spot_wrapper: SpotWrapper
) -> PowerState:
    """Maps power state data from robot state proto to ROS PowerState message

    Args:
        data: Robot State proto
        spot_wrapper: A SpotWrapper object
    Returns:
        PowerState message
    """
    power_state_msg = PowerState()
    local_time = spot_wrapper.robotToLocalTime(state.power_state.timestamp)
    power_state_msg.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
    power_state_msg.motor_power_state = state.power_state.motor_power_state
    power_state_msg.shore_power_state = state.power_state.shore_power_state
    power_state_msg.locomotion_charge_percentage = (
        state.power_state.locomotion_charge_percentage.value
    )
    power_state_msg.locomotion_estimated_runtime = rospy.Time(
        state.power_state.locomotion_estimated_runtime.seconds,
        state.power_state.locomotion_estimated_runtime.nanos,
    )
    return power_state_msg


def GetDockStatesFromState(state: docking_pb2.DockState) -> DockState:
    """Maps dock state data from robot state proto to ROS DockState message

    Args:
        state: Robot State proto
    Returns:
        DockState message
    """
    dock_state_msg = DockState()
    dock_state_msg.status = state.status
    dock_state_msg.dock_type = state.dock_type
    dock_state_msg.dock_id = state.dock_id
    dock_state_msg.power_status = state.power_status
    return dock_state_msg


def GetBehaviorFaults(
    behavior_faults: typing.List[robot_state_pb2.BehaviorFault],
    spot_wrapper: SpotWrapper,
) -> typing.List[BehaviorFault]:
    """Helper function to strip out behavior faults into a list

    Args:
        behavior_faults: List of BehaviorFaults
        spot_wrapper: A SpotWrapper object
    Returns:
        List of BehaviorFault messages
    """
    faults = []

    for fault in behavior_faults:
        new_fault = BehaviorFault()
        new_fault.behavior_fault_id = fault.behavior_fault_id
        local_time = spot_wrapper.robotToLocalTime(fault.onset_timestamp)
        new_fault.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
        new_fault.cause = fault.cause
        new_fault.status = fault.status
        faults.append(new_fault)

    return faults


def GetBehaviorFaultsFromState(
    state: robot_state_pb2.RobotState, spot_wrapper: SpotWrapper
) -> BehaviorFaultState:
    """Maps behavior fault data from robot state proto to ROS BehaviorFaultState message

    Args:
        data: Robot State proto
        spot_wrapper: A SpotWrapper object
    Returns:
        BehaviorFaultState message
    """
    behavior_fault_state_msg = BehaviorFaultState()
    behavior_fault_state_msg.faults = GetBehaviorFaults(
        state.behavior_fault_state.faults, spot_wrapper
    )
    return behavior_fault_state_msg


def GetSystemFaults(
    system_faults: typing.List[robot_state_pb2.SystemFault], spot_wrapper: SpotWrapper
) -> typing.List[SystemFault]:
    """Helper function to strip out system faults into a list

    Args:
        systen_faults: List of SystemFaults
        spot_wrapper: A SpotWrapper object
    Returns:
        List of SystemFault messages
    """
    faults = []

    for fault in system_faults:
        new_fault = SystemFault()
        new_fault.name = fault.name
        local_time = spot_wrapper.robotToLocalTime(fault.onset_timestamp)
        new_fault.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
        new_fault.duration = rospy.Time(fault.duration.seconds, fault.duration.nanos)
        new_fault.code = fault.code
        new_fault.uid = fault.uid
        new_fault.error_message = fault.error_message

        for att in fault.attributes:
            new_fault.attributes.append(att)

        new_fault.severity = fault.severity
        faults.append(new_fault)

    return faults


def GetSystemFaultsFromState(
    state: robot_state_pb2.RobotState, spot_wrapper: SpotWrapper
) -> SystemFaultState:
    """Maps system fault data from robot state proto to ROS SystemFaultState message

    Args:
        data: Robot State proto
        spot_wrapper: A SpotWrapper object
    Returns:
        SystemFaultState message
    """
    system_fault_state_msg = SystemFaultState()
    system_fault_state_msg.faults = GetSystemFaults(
        state.system_fault_state.faults, spot_wrapper
    )
    system_fault_state_msg.historical_faults = GetSystemFaults(
        state.system_fault_state.historical_faults, spot_wrapper
    )
    return system_fault_state_msg


def GetSpotCheckResultsMsg(
    data: spot_check_pb2.SpotCheckFeedbackResponse, resp: typing.Tuple[bool, str]
) -> SpotCheckResponse:
    """Build the SpotCheckReponse message from the SpotCheckFeedbackResponse"""
    ros_resp = SpotCheckResponse()
    ros_resp.success = resp[0]
    ros_resp.message = resp[1]

    ros_resp.camera_names = list(data.camera_results.keys())
    ros_resp.camera_results = list(data.camera_results.values())
    ros_resp.load_cell_names = list(data.load_cell_results.keys())
    ros_resp.load_cell_results = list(data.load_cell_results.values())
    ros_resp.kinematic_joint_names = list(data.kinematic_cal_results.keys())
    ros_resp.kinematic_cal_results = list(data.kinematic_cal_results.values())
    ros_resp.payload_result = data.payload_result
    ros_resp.leg_names = list(data.hip_range_of_motion_results.keys())
    ros_resp.hip_range_of_motion_results = list(
        data.hip_range_of_motion_results.values()
    )
    ros_resp.progress = data.progress
    ros_resp.last_cal_timestamp.secs = data.last_cal_timestamp.seconds
    ros_resp.last_cal_timestamp.nsecs = data.last_cal_timestamp.nanos

    return ros_resp


def GetFrameTreeSnapshotMsg(data: geometry_pb2.FrameTreeSnapshot) -> FrameTreeSnapshot:
    """Build the FrameTreeSnapshot message from the FrameTreeSnapshot proto"""
    frame_tree_snapshot_msg = FrameTreeSnapshot()
    child_parent_map: typing.Dict[
        str, geometry_pb2.SE3Pose
    ] = data.child_to_parent_edge_map

    children, parents = [], []
    for child, parent_edge in child_parent_map.items():
        children.append(child)
        parent = ParentEdge()
        parent.parent_frame_name = parent_edge.parent_frame_name
        parent.parent_tform_child = Pose(
            Point(
                parent_edge.parent_tform_child.position.x,
                parent_edge.parent_tform_child.position.y,
                parent_edge.parent_tform_child.position.z,
            ),
            Quaternion(
                parent_edge.parent_tform_child.rotation.x,
                parent_edge.parent_tform_child.rotation.y,
                parent_edge.parent_tform_child.rotation.z,
                parent_edge.parent_tform_child.rotation.w,
            ),
        )
        parents.append(parent)

    frame_tree_snapshot_msg.child_edges = children
    frame_tree_snapshot_msg.parent_edges = parents

    return frame_tree_snapshot_msg


def GetAprilTagPropertiesMsg(
    data: world_object_pb2.AprilTagProperties,
) -> AprilTagProperties:
    """Build the AprilTagProperties message from the AprilTagProperties proto"""
    april_tag_properties_msg = AprilTagProperties()
    april_tag_properties_msg.tag_id = data.tag_id
    april_tag_properties_msg.x = data.dimensions.x
    april_tag_properties_msg.y = data.dimensions.y
    april_tag_properties_msg.frame_name_fiducial = data.frame_name_fiducial
    april_tag_properties_msg.fiducial_pose_status = data.fiducial_pose_status
    april_tag_properties_msg.frame_name_fiducial_filtered = (
        data.frame_name_fiducial_filtered
    )
    april_tag_properties_msg.fiducial_filtered_pose_status = (
        data.fiducial_filtered_pose_status
    )
    april_tag_properties_msg.frame_name_camera = data.frame_name_camera
    april_tag_properties_msg.detection_covariance = PoseWithCovariance()
    april_tag_properties_msg.detection_covariance.pose = Pose()
    april_tag_properties_msg.detection_covariance.covariance = list(
        data.detection_covariance.matrix.values
    )
    april_tag_properties_msg.detection_covariance_reference_frame = (
        data.detection_covariance_reference_frame
    )

    return april_tag_properties_msg


def GetImagePropertiesMsg(
    data: world_object_pb2.ImageProperties, spot_wrapper: "SpotWrapper"
) -> ImageProperties:
    """Build the ImageProperties message from the ImageProperties proto"""
    image_properties_msg = ImageProperties()
    image_properties_msg.camera_source = data.camera_source
    if data.coordinates:
        data_polygon_coordinates: typing.List[
            geometry_pb2.Vec2
        ] = data.coordinates.vertexes
        image_properties_msg.image_data_coordinates = Polygon(
            [Point32(vec.x, vec.y, 0) for vec in data_polygon_coordinates]
        )
    elif data.keypoints:
        image_properties_msg.image_data_keypoint_type = data.keypoints.type
        image_properties_msg.keypoint_coordinate_x = [
            i.coordinates.x for i in data.keypoints.keypoints
        ]
        image_properties_msg.keypoint_coordinate_y = [
            i.coordinates.y for i in data.keypoints.keypoints
        ]
        image_properties_msg.binary_descriptor = [
            i.binary_descriptor for i in data.keypoints.keypoints
        ]
        image_properties_msg.keypoint_score = [
            i.score for i in data.keypoints.keypoints
        ]
        image_properties_msg.keypoint_size = [i.size for i in data.keypoints.keypoints]
        image_properties_msg.keypoint_angle = [
            i.angle for i in data.keypoints.keypoints
        ]

    image_properties_msg.image_source.name = data.image_source.name
    image_properties_msg.image_source.cols = data.image_source.cols
    image_properties_msg.image_source.rows = data.image_source.rows
    image_properties_msg.image_source.depth_scale = data.image_source.depth_scale
    image_properties_msg.image_source.focal_length_x = (
        data.image_source.pinhole.intrinsics.focal_length.x
    )
    image_properties_msg.image_source.focal_length_y = (
        data.image_source.pinhole.intrinsics.focal_length.y
    )
    image_properties_msg.image_source.principal_point_x = (
        data.image_source.pinhole.intrinsics.principal_point.x
    )
    image_properties_msg.image_source.principal_point_y = (
        data.image_source.pinhole.intrinsics.principal_point.y
    )
    image_properties_msg.image_source.skew_x = (
        data.image_source.pinhole.intrinsics.skew.x
    )
    image_properties_msg.image_source.skew_y = (
        data.image_source.pinhole.intrinsics.skew.y
    )

    image_properties_msg.image_source.image_type = data.image_source.image_type
    image_properties_msg.image_source.pixel_formats = list(
        data.image_source.pixel_formats
    )
    image_properties_msg.image_source.image_formats = list(
        data.image_source.image_formats
    )

    local_time = spot_wrapper.robotToLocalTime(data.image_capture.acquisition_time)

    image_properties_msg.image_capture.acquisition_time.secs = local_time.seconds
    image_properties_msg.image_capture.acquisition_time.nsecs = local_time.nanos

    image_properties_msg.image_capture.transforms_snapshot = GetFrameTreeSnapshotMsg(
        data.image_capture.transforms_snapshot
    )
    image_properties_msg.image_capture.frame_name_image_sensor = (
        data.image_capture.frame_name_image_sensor
    )
    image_properties_msg.image_capture.image, _ = GetImageMsg(
        image_pb2.ImageResponse(shot=data.image_capture, source=data.image_source),
        spot_wrapper,
    )
    image_properties_msg.image_capture.capture_exposure_duration.secs = (
        data.image_capture.capture_params.exposure_duration.seconds
    )
    image_properties_msg.image_capture.capture_exposure_duration.nsecs = (
        data.image_capture.capture_params.exposure_duration.nanos
    )
    image_properties_msg.image_capture.capture_sensor_gain = (
        data.image_capture.capture_params.gain
    )

    image_properties_msg.frame_name_image_coordinates = (
        data.frame_name_image_coordinates
    )

    return image_properties_msg


def GetWorldObjectsMsg(
    data: world_object_pb2.ListWorldObjectResponse,
    spot_wrapper: SpotWrapper,
) -> WorldObjectArray:
    """Build the WorldObjectsResponse message from the WorldObjectsResponse"""
    world_object_msg = WorldObjectArray()

    world_objects: typing.List[world_object_pb2.WorldObject] = data.world_objects
    for world_object in world_objects:
        id: int = world_object.id
        name: str = world_object.name
        acquisition_time: Timestamp = world_object.acquisition_time
        frame_tree_snapshot: geometry_pb2.FrameTreeSnapshot = (
            world_object.transforms_snapshot
        )
        apriltag_properties: world_object_pb2.AprilTagProperties = (
            world_object.apriltag_properties
        )
        image_properties: world_object_pb2.ImageProperties = (
            world_object.image_properties
        )
        dock_properties: world_object_pb2.DockProperties = world_object.dock_properties
        ray_properties: world_object_pb2.RayProperties = world_object.ray_properties
        bounding_box_properties: world_object_pb2.BoundingBoxProperties = (
            world_object.bounding_box_properties
        )
        additional_properties = world_object.additional_properties

        # Put properties into ROS message
        new_world_object = WorldObject()
        new_world_object.id = id
        new_world_object.name = name
        new_world_object.acquisition_time.secs = acquisition_time.seconds
        new_world_object.acquisition_time.nsecs = acquisition_time.nanos
        new_world_object.frame_tree_snapshot = GetFrameTreeSnapshotMsg(
            frame_tree_snapshot
        )
        new_world_object.apriltag_properties = GetAprilTagPropertiesMsg(
            apriltag_properties
        )
        new_world_object.image_properties = GetImagePropertiesMsg(
            image_properties, spot_wrapper
        )

        # Dock properties
        new_world_object.dock_id = world_object.dock_properties.dock_id
        new_world_object.dock_type = world_object.dock_properties.type
        new_world_object.frame_name_dock = world_object.dock_properties.frame_name_dock
        new_world_object.dock_unavailable = world_object.dock_properties.unavailable
        new_world_object.from_prior_detection = world_object.dock_properties.from_prior

        # Ray properties
        new_world_object.ray_frame = world_object.ray_properties.frame
        new_world_object.ray_origin = Vector3(
            x=world_object.ray_properties.ray.origin.x,
            y=world_object.ray_properties.ray.origin.y,
            z=world_object.ray_properties.ray.origin.z,
        )
        new_world_object.ray_direction = Vector3(
            x=world_object.ray_properties.ray.direction.x,
            y=world_object.ray_properties.ray.direction.y,
            z=world_object.ray_properties.ray.direction.z,
        )

        # Bounding box properties
        new_world_object.bounding_box_frame = world_object.bounding_box_properties.frame
        new_world_object.bounding_box_size_ewrt_frame = Vector3(
            x=world_object.bounding_box_properties.size_ewrt_frame.x,
            y=world_object.bounding_box_properties.size_ewrt_frame.y,
            z=world_object.bounding_box_properties.size_ewrt_frame.z,
        )

        world_object_msg.world_objects.append(new_world_object)

    return world_object_msg


def GetGraphNavGraphMsg(data: UploadGraphRequest) -> map_pb2.Graph:
    """Build the Graph message for the upload_graph RPC call, from the ROS service message"""

    """ protobuf definition for map_pb2.Graph
    message Graph {
        // The waypoints for the graph (containing frames, annotations, and sensor data).
        repeated Waypoint waypoints = 1;
        // The edges connecting the graph's waypoints.
        repeated Edge edges = 2;

        // The anchoring (mapping from waypoints to their pose in a shared reference frame).
        Anchoring anchoring = 3;
    }



message Waypoint {
    // Identifier of the waypoint. Unique across all maps.
    // This identifier does not have to be updated when its fields change.
    string id = 1;

    // Identifier of this waypoint's Snapshot data.
    string snapshot_id = 2;

    // Transform from the KO frame (at time of recording) to the waypoint.
    SE3Pose waypoint_tform_ko = 3;

    enum WaypointSource {
        WAYPOINT_SOURCE_UNKNOWN = 0;
        // Waypoints from the robot's location during recording.
        WAYPOINT_SOURCE_ROBOT_PATH = 1;
        // Waypoints with user-requested placement.
        WAYPOINT_SOURCE_USER_REQUEST = 2;
        // Waypoints that may help find alternate routes.
        WAYPOINT_SOURCE_ALTERNATE_ROUTE_FINDING = 3;
    };

    // Annotations understood by BostonDynamics systems.
    message Annotations {
        // Human-friendly name of the waypoint. For example, "Kitchen Fridge"
        string name = 1;

        // The time that the waypoint was created while recording a map.
        google.protobuf.Timestamp creation_time = 4;

        // Estimate of the variance of ICP when performed at this waypoint, collected at record time.
        bosdyn.api.SE3Covariance icp_variance = 2;

        message LocalizeRegion {
            // Check this before reading other fields.
            AnnotationState state = 1;

            // Use the default region to localize in.
            message Default {
            }
            // Do not localize to this waypoint.
            message Empty {
            }
            // Indicates the number of meters away we can be from this waypoint we can be before scan
            // matching.
            // - If zero, the default value is used.
            // - If less than zero, no scan matching will be performed at this waypoint.
            // - If greater than zero, scan matching will only be performed if the robot is at most this
            //   far away from the waypoint.
            // Distance calculation is done in the 2d plane with respect to the waypoint.
            message Circle2D {
                double dist_2d = 1; // meters.
            }

            oneof region {
                // Oneof field that describes the waypoint's location as a default region (no special features/traits).
                Default default_region = 2;
                // Oneof field that describes the waypoint's location as a empty/featureless region.
                Empty empty = 3;
                // Oneof field that describes the waypoint's location as a circular region.
                Circle2D circle = 4;
            }
        }
        // Options for how to localize to a waypoint (if at all).
        LocalizeRegion scan_match_region = 3;

        // How this waypoint was made.
        WaypointSource waypoint_source = 5;

        // Information about the state of the client when this waypoint was created.
        ClientMetadata client_metadata = 6;
    }
    // Annotations specific to the current waypoint.
    Annotations annotations = 4;
}
    """

    graph: GraphNavGraph = data.graph
    graph_msg = map_pb2.Graph()

    # Waypoints
    for waypoint in graph.waypoints:
        waypoint: GraphNavWaypoint = waypoint  # temp for linting
        new_waypoint = map_pb2.Waypoint()
        new_waypoint.id = waypoint.id
        new_waypoint.snapshot_id = waypoint.snapshot_id
        new_waypoint.waypoint_tform_ko = geometry_pb2.SE3Pose(
            position=geometry_pb2.Vec3(
                x=waypoint.waypoint_tform_ko.position.x,
                y=waypoint.waypoint_tform_ko.position.y,
                z=waypoint.waypoint_tform_ko.position.z,
            ),
            rotation=geometry_pb2.Quaternion(
                x=waypoint.waypoint_tform_ko.orientation.x,
                y=waypoint.waypoint_tform_ko.orientation.y,
                z=waypoint.waypoint_tform_ko.orientation.z,
                w=waypoint.waypoint_tform_ko.orientation.w,
            ),
        )
        new_waypoint.annotations.name = waypoint.name
        new_waypoint.annotations.creation_time = waypoint.creation_time
        # Row-major covariance matrix
        new_waypoint.annotations.icp_variance = geometry_pb2.SE3Covariance(
            geometry_pb2.Matrix(
                rows=6,
                cols=6,
                values=[
                    waypoint.icp_variance[0][0],
                    waypoint.icp_variance[0][1],
                    waypoint.icp_variance[0][2],
                    waypoint.icp_variance[0][3],
                    waypoint.icp_variance[0][4],
                    waypoint.icp_variance[0][5],
                    waypoint.icp_variance[1][0],
                    waypoint.icp_variance[1][1],
                    waypoint.icp_variance[1][2],
                    waypoint.icp_variance[1][3],
                    waypoint.icp_variance[1][4],
                    waypoint.icp_variance[1][5],
                    waypoint.icp_variance[2][0],
                    waypoint.icp_variance[2][1],
                    waypoint.icp_variance[2][2],
                    waypoint.icp_variance[2][3],
                    waypoint.icp_variance[2][4],
                    waypoint.icp_variance[2][5],
                    waypoint.icp_variance[3][0],
                    waypoint.icp_variance[3][1],
                    waypoint.icp_variance[3][2],
                    waypoint.icp_variance[3][3],
                    waypoint.icp_variance[3][4],
                    waypoint.icp_variance[3][5],
                    waypoint.icp_variance[4][0],
                    waypoint.icp_variance[4][1],
                    waypoint.icp_variance[4][2],
                    waypoint.icp_variance[4][3],
                    waypoint.icp_variance[4][4],
                    waypoint.icp_variance[4][5],
                    waypoint.icp_variance[5][0],
                    waypoint.icp_variance[5][1],
                    waypoint.icp_variance[5][2],
                    waypoint.icp_variance[5][3],
                    waypoint.icp_variance[5][4],
                    waypoint.icp_variance[5][5],
                ],
            )
        )
        new_waypoint.annotations.scan_match_region.state = (
            waypoint.scan_match_region_state
        )
        if waypoint.default_region:
            new_waypoint.annotations.scan_match_region.region = (
                map_pb2.Waypoint.Annotations.LocalizeRegion.Default()
            )
        elif waypoint.empty:
            new_waypoint.annotations.scan_match_region.region = (
                map_pb2.Waypoint.Annotations.LocalizeRegion.Empty()
            )
        elif waypoint.circle_2d:
            new_waypoint.annotations.scan_match_region.region = (
                map_pb2.Waypoint.Annotations.LocalizeRegion.Circle2D(
                    dist_2d=waypoint.circle_region_dist_2d
                )
            )

        new_waypoint.annotations.waypoint_source = waypoint.waypoint_source
        new_waypoint.annotations.client_metadata = map_pb2.ClientMetadata(
            session_name=waypoint.session_name,
            client_username=waypoint.client_username,
            client_software_version=waypoint.client_software_version,
            client_id=waypoint.client_id,
            client_type=waypoint.client_type,
        )

        graph_msg.waypoints.append(new_waypoint)

    return graph_msg
