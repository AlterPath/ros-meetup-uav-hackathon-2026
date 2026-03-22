import os
import sys
import time

user_site = os.path.expanduser('~/.local/lib/python3.10/site-packages')
if user_site not in sys.path:
    sys.path.insert(0, user_site)

sys.path.append('/opt/ros/humble/lib/python3.10/site-packages')
sys.path.append('/opt/ros/humble/local/lib/python3.10/dist-packages')

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('yolo_vision_node')

        self.inference_interval = 0.5  
        self.last_inference_time = 0.0
        
        model_path = '/home/sverk/Hack/classifier/weights/best.pt'
        self.model = YOLO(model_path)
        self.model.to('cpu') 
        self.get_logger().info('YOLO Node запущен. Режим: CPU')

        self.bridge = CvBridge()

        self.img_publisher = self.create_publisher(Image, 'camera_1/yolo_visual', 10)
        self.result_publisher = self.create_publisher(String, 'yolo_result', 10)

        self.label_publisher = self.create_publisher(String, 'yolo_label', 10)

        self.subscription = self.create_subscription(
            Image, 'camera_1/image_raw', self.listener_callback, 10)

    def listener_callback(self, msg):
        current_time = time.time()
        
        if (current_time - self.last_inference_time) < self.inference_interval:
            return

        try:
            self.last_inference_time = current_time
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            results = self.model(cv_image, verbose=False, device='cpu')
            res = results[0]
            
            conf = res.probs.top1conf.item()
            cls_name = res.names[res.probs.top1]
            label_msg = String()

            if conf > 0.90:
                res_msg = String()
                res_msg.data = f"{cls_name}"
                self.result_publisher.publish(res_msg)
                
                label_msg.data = f"{cls_name}"
                
                label = f"DETECT: {cls_name} ({conf:.2%})"
                color = (0, 255, 0)
            else:
                label_msg.data = "Nothing found"
                label = f"Searching... ({conf:.2%})"
                color = (0, 0, 255)

            self.label_publisher.publish(label_msg)

            cv2.putText(cv_image, label, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            out_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.img_publisher.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'Ошибка: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = YoloVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()