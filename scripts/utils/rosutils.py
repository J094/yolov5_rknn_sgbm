#!/usr/bin/env python
# J094
"""
Utils for ROS
"""

import rospy
from std_msgs.msg import Float64
from detector.srv import ObstacleDist, ObstacleDistResponse

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class rosImgPublisher(object):
    def __init__(self, fps) -> None:
        #rospy.init_node('ImgPub', anonymous=True)
        self._img_pub = rospy.Publisher('img', Image, queue_size=10)
        self.rate = rospy.Rate(fps) # publish frequency
        self.bridge = CvBridge()

    def pub_img(self, img):
        rospy.loginfo(f"Image published!")
        self._img_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        self.rate.sleep()


class rosPublisher(object):
    def __init__(self, fps) -> None:
        #rospy.init_node('Detector', anonymous=True)
        self._dist_pub = rospy.Publisher('Distance', Float64, queue_size=10)
        self.rate = rospy.Rate(fps) # publish frequency

    def pub_dist(self, dist):
        rospy.loginfo(f"Dist: {dist}m")
        self._dist_pub.publish(dist)
        self.rate.sleep()


class rosListener(object):
    def __init__(self) -> None:
        #rospy.init_node('Listener', anonymous=True)
        self._dist_sub = None

    def _callback(self, data):
        rospy.loginfo(rospy.get_caller_id() + "I get %s", data.data)

    def sub_dist(self):
        self._dist_sub = rospy.Subscriber('Distance', Float64, self._callback)
        rospy.spin()


class rosService(object):
    def __init__(self) -> None:
        #rospy.init_node('Service')
        self.s = None
    
    def handle_req(self, req):
        rospy.loginfo(f"Obtain {req.distance}")
        if req.distance == -1:
            rospy.loginfo(f"Entering safe state...")
        else:
            rospy.loginfo(f"Danger state: there is a obstacle in {req.distance}cm")
        return ObstacleDistResponse(True, f"Dist: {req.distance}m")
    
    def run(self):
        self.s = rospy.Service('distance_server', ObstacleDist, self.handle_req)
        rospy.loginfo("Ready to obtain req...")
        rospy.spin()


class rosClient(object):
    def __init__(self) -> None:
        #rospy.init_node('Client')
        pass

    def run(self, dist):
        rospy.wait_for_service('distance_server')
        try:
           resp_process = rospy.ServiceProxy('distance_server', ObstacleDist) 
           resp = resp_process(dist)
           if resp.success:
               rospy.loginfo(resp.message)
        except rospy.ServiceException as e:
            rospy.loginfo(f'Service call failed: {e}')
