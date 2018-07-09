#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree # suggested in walkthrough video

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        # Initially from 'partial walkthrough' video:
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None

        self.loop() # instead of rospy.spin() so we can control rate
        # rospy.spin()

    def loop(self)
        # Initially from 'partial walkthrough' video
        rate = rospy.Rate(50) # May be excessive
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
            # Get closest waypoint
            closest_waypoint_idx = self.get_closest_waypoint_idx()
            self.publish_waypoints(closest_waypoint_idx)
        rate.sleep()

    def get_closest_waypoint_idx(self):
        # Initially from 'partial walkthrough' video
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        # Only want 1 point returned, and want index (2nd item returned) hence 1, [1]
        closest_idx = self.waypoint_tree.query([x,y], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1] # CW may index off array?!

        # Equation for hyperplane through closest coords
        cl_vect = np.array(closest_coord) # closest waypoint
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])
        # Take dot product (get distance?)
        # Plane is perpendicular to vector from prev_vect to cl_vect.
        # Want to check if pos_vect is ahead 
        val = np.dot(cl_vect, prev_vect, pos_vect, cl_vect)
        # If dot product is negative, vector from prev_vect to cl_vect and
        # vector from cl_vect to pos_vect are pointing in opposite directions
        # relative to that plane, implying pos_vect is on the same side of
        # the plane as prev_vect, i.e. we are still behind the closest
        # waypoint, which is in front of us. Otherwise our position is beyond
        # the plane so the closest waypoint is already behind us.
        # (See 'partial walkthrough' video at ~11 min.) Could do other
        # things, e.g. considering car orientation.
        if val > 0:
            # waypoint was behind us so take next one
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
            # CW: no check that next-nearest is also in front of us?!

        return closest_idx

    def publish_waypoints(self, closest_idx)
        # Initially from 'partial walkthrough' video
        lane = lane()
        lane.header = self.base_waypoints.header
        lane_waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx + LOOKAHEAD_WPS] # Python will just go to end of list, but won't wrap around
        self.final_waypoints_pub.publish(lane)

    def pose_cb(self, msg):
        # TODO: Implement
        # From 'partial walkthrough' video, just grab latest pose, will be called often:
        self.pose = msg 
        pass

    def waypoints_cb(self, waypoints):
        # TODO: Implement
	# Starting with suggested code from 'partial walkthrough' video

        # Video explained this is a latched subscriber, so we should get
        # this message only once, rather than saving the whole set of base
        # waypoints periodically, which would be very wasteful
        self.base_waypoings = waypoints
        if not self.waypoints_2d:
            # Collapse waypoints to just 2D coordinates
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoint]
            # KDTree will allow very efficient search for nearest points
            self_waypoint_tree = KDTree(self.waypoints_2d)
        pass

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
