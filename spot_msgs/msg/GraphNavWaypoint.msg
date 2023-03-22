string id
string snapshot_id

# Transform from the KO frame (at time of recording) to the waypoint.
geometry_msgs/Pose waypoint_tform_ko

#################### Annotations ####################
# Annotation state enums
uint8 ANNOTATION_STATE_UNKNOWN = 0
uint8 ANNOTATION_STATE_SET = 1
uint8 ANNOTATION_STATE_NONE = 2

string name
time creation_time
float64[36] icp_variance

uint8 scan_match_region_state
bool default_region
bool empty
bool circle_2d

# Indicates the number of meters away we can be from this waypoint before we can begin scan
# matching.
# - If zero, the default value is used.
# - If less than zero, no scan matching will be performed at this waypoint.
# - If greater than zero, scan matching will only be performed if the robot is at most this
#   far away from the waypoint.
# Distance calculation is done in the 2d plane with respect to the waypoint.
float64 circle_region_dist_2d

# Waypoint source enums
uint8 WAYPOINT_SOURCE_UNKNOWN = 0
uint8 WAYPOINT_SOURCE_ROBOT_PATH = 1
uint8 WAYPOINT_SOURCE_USER_REQUEST = 2
uint8 WAYPOINT_SOURCE_ALTERNATE_ROUTE_FINDING = 3

# How this waypoint was made
uint8 waypoint_source

# Client metadata
string session_name
string client_username
string client_software_version
string client_id
string client_type