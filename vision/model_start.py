import os
import sys
import os
import sys

import os
import sys

sys.path = [p for p in sys.path if 'site-packages' not in p and 'dist-packages' not in p]

sys.path.append('/usr/lib/python3/dist-packages')

sys.path.append('/usr/local/lib/python3.10/dist-packages')

sys.path.append('/home/sverk/.local/lib/python3.10/site-packages')

sys.path.append('/opt/ros/humble/lib/python3.10/site-packages')
sys.path.append('/opt/ros/humble/local/lib/python3.10/dist-packages')

try:
    import numpy
    print(f"[*] NumPy: {numpy.__version__} OK")
    import torch
    import typing_extensions
    from ultralytics import YOLO
    import cv2
    import rclpy
    print("[OK] Все библиотеки (YOLO, Torch, ROS2) успешно загружены!")
except ImportError as e:
    print(f"[-] Ошибка импорта: {e}")
    print("\nТекущий sys.path для отладки:")
    for p in sys.path: print(f"  {p}")
    sys.exit(1)

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import pygame

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('yolo_vision_node')
        model_path = '/home/sverk/Hack/classifier/weights/best.pt'
        self.model = YOLO(model_path)
        self.model.to('cpu') 
        self.get_logger().info('YOLO готова. Устройство: CPU')
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("Vision Monitor (SVERK)")

        self.bridge = CvBridge()
        
        self.subscription = self.create_subscription(
            Image, 
            'camera_1/image_raw', 
            self.listener_callback, 
            10)
        
        self.publisher = self.create_publisher(Image, 'camera_1/yolo_visual', 10)
        self.get_logger().info('Ожидание кадров из camera_1/image_raw...')

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            results = self.model(cv_image, verbose=False, device='cpu')
            res = results[0]
            
            conf = res.probs.top1conf.item()
            cls_name = res.names[res.probs.top1]

            final_result = cls_name if conf > 0.90 else "background"
            color = (0, 255, 0) if final_result != "background" else (0, 0, 255)

            display_text = f"{final_result} ({conf:.2%})"
            cv2.putText(cv_image, display_text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


            out_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.publisher.publish(out_msg)

            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            rgb_image = cv2.resize(rgb_image, (640, 480))
            surface = pygame.surfarray.make_surface(rgb_image.swapaxes(0, 1))
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.get_logger().info('Закрытие окна...')
                    rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f'Ошибка в обработке: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = YoloVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[*] Остановка узла...")
    finally:
        pygame.quit()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()