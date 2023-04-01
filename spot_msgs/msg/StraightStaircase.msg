# It is expressed in ko frame of the from_waypoint, only GraphNav.
geometry_msgs/Pose from_ko_tform_stairs

# Outside GraphNav, from bottom center step
geometry_msgs/PoseStamped stair_tform

float64[] stairs_rise
float64[] stairs_run

# Pose of the bottom landing's center relative to the stairs frame.
geometry_msgs/Pose stairs_tform_bottom_landing_center
float64 bottom_landing_extent_x
float64 bottom_landing_extent_y

# Pose of the top landing's center relative to the stairs frame.
geometry_msgs/Pose stairs_tform_top_landing_center
float64 top_landing_extent_x
float64 top_landing_extent_y