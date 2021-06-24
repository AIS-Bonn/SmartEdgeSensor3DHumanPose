#!/usr/bin/env python
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image as ROS_Image
from person_msgs.msg import Person2DList

CocoColors = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
              (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (50, 0, 255), (100, 0, 255),
              (170, 0, 255), (255, 0, 255), (255, 150, 0), (85, 170, 0), (42, 128, 85), (0, 85, 170),
              (255, 0, 170), (255, 0, 85), (242, 165, 65)]
CocoColors_inv = [(255 - color[0], 255 - color[1] , 255 - color[2]) for color in CocoColors]
CocoPairs = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10),
               (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)]

def draw_humans(img, humans):
        _CONF_THRESHOLD_DRAW = 0.25
        
        image_w = img.shape[1]
        image_h = img.shape[0]
        
        num_joints = 17
        colors = CocoColors
        pairs = CocoPairs
            
        for human in humans:
            centers = {}
            body_parts = {}
            # draw point
            for i in range(num_joints):
                if isinstance(human['keypoints'], dict):
                    if str(i) not in human['keypoints'].keys() or human['keypoints'][str(i)][2] < _CONF_THRESHOLD_DRAW:
                        continue
                    body_part = human['keypoints'][str(i)]
                    
                elif isinstance(human['keypoints'], list):
                    if human['keypoints'][i] is None or human['keypoints'][i][2] < _CONF_THRESHOLD_DRAW:
                        continue
                    body_part = human['keypoints'][i]
                    
                center = (int(body_part[0] + 0.5), int(body_part[1] + 0.5))

                centers[i] = center
                body_parts[i] = body_part
                img = cv2.circle(img, center, max(1, int(img.shape[1] / 360)) * 5, colors[i], thickness=-1, lineType=8, shift=0)

            # draw line
            for pair_order, pair in enumerate(pairs):
                if pair[0] not in centers.keys() or pair[1] not in centers.keys():
                    continue

                img = cv2.line(img, centers[pair[0]], centers[pair[1]], colors[pair[1]], max(1, int(img.shape[1] / 360)) * 4)
                
            # draw bounding box
            x1, y1, x2, y2 = human['bbox']
            x1 = int(x1 + 0.5)-6
            y1 = int(y1 + 0.5)-6
            x2 = int(x2 + 0.5)+6
            y2 = int(y2 + 0.5)+6
                

            cv2.rectangle(img, (x1, y1), (x2, y2), color = colors[human['id'] % len(colors)], thickness = max(1, int(img.shape[1] / 360)) * 2)
            
            #cv2.putText(img, 'ID: {} - {:.1f}%'.format(max(0,human['id']) , human['score'] * 100), (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, colors[human['id'] % len(colors)], 2)
            #cv2.putText(img, '{:.1f}%'.format(human['score'] * 100), (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, colors[human['id'] % len(colors)], 2)

        return img

class pose_analyzer:
    
    def __init__(self):
        
        self.bridge = CvBridge()
        
        self.num_joints = 17
                
        self.json_pose_sub = rospy.Subscriber('/human_joints', Person2DList, self.callback_pose,  queue_size = 3)
                
        self.publisher_image_overlay = rospy.Publisher('image_overlay_from_json', ROS_Image, queue_size=1)
        
    def callback_pose(self, msg):
        
        humans = [{'id': 0, 'score': p.score, 'bbox': p.bbox, 'keypoints': [[kp.x, kp.y, kp.score] for kp in p.keypoints]} for p in msg.persons]
        bg_image = 255 * np.ones((480,640,3), dtype=np.uint8)
        
        img = draw_humans(bg_image, humans)
        
        img_msg = self.bridge.cv2_to_imgmsg(img, "rgb8")
        img_msg.header = msg.header
            
        self.publisher_image_overlay.publish(img_msg)
 
def main():
    
    rospy.init_node('pose2D_plot_node')
    
    panalyzer = pose_analyzer()
    
    rospy.spin()

if __name__ == '__main__':
    main()
