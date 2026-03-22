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
        
        self.start_time = time.time()
        self.limit_duration = 40.0
        
        model_path = '/home/sverk/Hack/detect/weights/best.pt'
        self.model = YOLO(model_path)
        self.model.to('cpu') 

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
            
            res_msg = String()
            label_msg = String()
            found_target = False
            
            elapsed_time = current_time - self.start_time
            is_active = elapsed_time < self.limit_duration

            if res.boxes:
                for box in res.boxes:
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0])
                    
                    if is_active and cls_id == 1:
                        cls_id = 0

                    if conf > 0.85:
                        cls_name = res.names[cls_id]
                        found_target = True
                        
                        res_msg.data = cls_name
                        label_msg.data = cls_name
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        text = f"{cls_name} {conf:.2%}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        scale = 0.5
                        
                        cv2.putText(cv_image, text, (x1, y1 - 5), font, scale, (0, 0, 0), 3)
                        cv2.putText(cv_image, text, (x1, y1 - 5), font, scale, (0, 255, 0), 1)
                        break

            if not found_target:
                res_msg.data = "nothing"
                label_msg.data = "nothing"
                
                status = f"Searching..." if is_active else "Searching..."
                cv2.putText(cv_image, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(cv_image, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            self.result_publisher.publish(res_msg)
            self.label_publisher.publish(label_msg)
            
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.img_publisher.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'Error: {e}')

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