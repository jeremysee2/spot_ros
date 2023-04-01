string from_waypoint
string to_waypoint

string snapshot_id
# Describes the transform between the "from" waypoint and the "to" waypoint.
geometry_msgs/Pose from_tform_to

#################### Annotations ####################
uint8 ANNOTATION_STATE_UNKNOWN = 0
uint8 ANNOTATION_STATE_SET = 1
uint8 ANNOTATION_STATE_NONE = 2

uint8 stairs_annotation_state
StraightStaircase straight_staircase

# Direction constraint enums
uint8 DIRECTION_CONSTRAINT_UNKNOWN = 0
uint8 DIRECTION_CONSTRAINT_NO_TURN = 1
uint8 DIRECTION_CONSTRAINT_FORWARD = 2
uint8 DIRECTION_CONSTRAINT_REVERSE = 3
uint8 DIRECTION_CONSTRAINT_NONE = 4

uint8 direction_constraint

bool require_alignment
bool flat_ground


# Overrides the following fields of the mobility parameters to whatever is
# stored in the map. For example, if this FieldMask contains "stairs_mode" and
# "terrain_params.enable_grated_floor", then the map will be
# annotated with "stairs_mode" and "enable_grated_floor" settings. An empty FieldMask means all fields are active
# annotations. Note that the more conservative of the velocity limit stored in the mobility parameters and the
# TravelParams of the entire route will be used for this edge (regardless of what override_mobility_params says).
string[] override_mobility_params

MobilityParams mobility_params

# Assign edges a cost, used when finding the "shortest" (lowest cost) path.
float64 cost

# Edge source enums
uint8 EDGE_SOURCE_UNKNOWN = 0
uint8 EDGE_SOURCE_ODOMETRY = 1
uint8 EDGE_SOURCE_SMALL_LOOP_CLOSURE = 2
uint8 EDGE_SOURCE_FIDUCIAL_LOOP_CLOSURE = 3
uint8 EDGE_SOURCE_ALTERNATE_ROUTE_FINDING = 4
uint8 EDGE_SOURCE_USER_REQUEST = 5

uint8 edge_source
bool disable_alternate_route_finding

# Path following enums
uint8 PATH_MODE_UNKNOWN = 0
uint8 PATH_MODE_DEFAULT = 1
uint8 PATH_MODE_STRICT = 2

uint8 path_following_mode
bool disable_directed_exploration

string[] area_callbacks_keys
string[] area_callbacks_region_service_names
string[] area_callbacks_region_descriptions

# Ground clutter enums
uint8 GROUND_CLUTTER_UNKNOWN = 0
uint8 GROUND_CLUTTER_OFF = 1
uint8 GROUND_CLUTTER_FROM_FOOTFALLS = 2

uint8 ground_clutter_mode