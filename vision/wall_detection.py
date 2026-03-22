import os
import sys
import time
import math
import numpy as np
import cv2

# --- Настройка путей ---
sys.path = [p for p in sys.path if 'site-packages' not in p and 'dist-packages' not in p]
sys.path.append('/usr/lib/python3/dist-packages')
sys.path.append('/usr/local/lib/python3.10/dist-packages')
sys.path.append('/home/sverk/.local/lib/python3.10/site-packages')
sys.path.append('/opt/ros/humble/lib/python3.10/site-packages')
sys.path.append('/opt/ros/humble/local/lib/python3.10/dist-packages')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

# Метки секторов
SECTOR_LABELS = ['FRONT', 'RIGHT', 'BACK', 'LEFT'] # По часовой стрелке

class FisheyeMapBuilderNode(Node):
    def __init__(self):
        super().__init__('fisheye_map_builder')
        
        # --- НАСТРОЙКИ (подбери под свою камеру) ---
        self.declare_parameter('canny_low', 50)
        self.declare_parameter('canny_high', 150)
        self.declare_parameter('hough_threshold', 40)
        self.declare_parameter('min_line_len', 60)
        
        # Виртуальный лидар: параметры
        self.num_readings = 4 # Количество секторов/лучей (4, 8, 16...)
        self.scan_range_min = 0.2  # Метры (ближе - стена)
        self.scan_range_max = 2.0  # Метры (дальше - свободно)
        self.frame_id = 'camera_link' # Фрейм камеры для Rviz

        # --- ИНИЦИАЛИЗАЦИЯ ---
        self.bridge = CvBridge()
        self.canny_low = self.get_parameter('canny_low').value
        self.canny_high = self.get_parameter('canny_high').value
        
        # Подписка на видео
        self.sub_img = self.create_subscription(
            Image, 'camera_1/image_raw', self.image_callback, 10)
        
        # --- ПАБЛИШЕРЫ ---
        # 1. Отладочное видео с секторами
        self.pub_debug_img = self.create_publisher(Image, 'camera_1/wall_debug', 10)
        # 2. Виртуальный лазерный скан для карты
        self.pub_scan = self.create_publisher(LaserScan, 'virtual_scan', 10)

        self.get_logger().info('>>> УЗЕЛ КАРТОГРАФИРОВАНИЯ СТЕН ЗАПУЩЕН.')

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w = cv_img.shape[:2]
            center = (w // 2, h // 2)

            # --- 1. ВЫДЕЛЕНИЕ КРАЕВ (OpenCV) ---
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

            # --- 2. МАСКИРОВАНИЕ ЗОНЫ (Ободок) ---
            mask = np.zeros_like(edges)
            # Внешний радиус (почти весь кадр)
            cv2.circle(mask, center, int(min(w, h) * 0.48), 255, -1) 
            # Внутренний радиус (зона безопасности - игнорируем)
            cv2.circle(mask, center, int(min(w, h) * 0.35), 0, -1)
            masked_edges = cv2.bitwise_and(edges, mask)

            # --- 3. ПОИСК ЛИНИЙ (HoughLinesP) ---
            lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 
                                    threshold=self.get_parameter('hough_threshold').value, 
                                    minLineLength=self.get_parameter('min_line_len').value, 
                                    maxLineGap=10)

            # --- 4. ЗОНИРОВАНИЕ И ОПРЕДЕЛЕНИЕ СТОРОН ---
            # Создаем пустой массив для подсчета линий в 4 секторах (F, R, B, L)
            sector_counts = np.zeros(self.num_readings, dtype=int)
            debug_img = cv_img.copy()

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Вычисляем угол центра линии относительно центра кадра (atan2)
                    line_cx = (x1 + x2) / 2
                    line_cy = (y1 + y2) / 2
                    # Угол в радианах (-pi ... pi)
                    angle_rad = math.atan2(line_cy - center[1], line_cx - center[0])
                    # Угол в градусах (0 ... 360)
                    angle_deg = (math.degrees(angle_rad) + 360) % 360
                    
                    # Маппинг угла на индекс сектора (0=FRONT, 1=RIGHT, 2=BACK, 3=LEFT)
                    # Корректируем на 45 градусов, чтобы FRONT был с 315 по 45
                    sector_idx = int(((angle_deg + 45) % 360) / (360 / self.num_readings))
                    sector_counts[sector_idx] += 1
                    
                    # Отрисовка линий (красным)
                    cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # --- 5. ФОРМИРОВАНИЕ И ПУБЛИКАЦИЯ LASERSCAN ---
            self.publish_virtual_scan(sector_counts)

            # --- 6. ВИЗУАЛИЗАЦИЯ ДЛЯ ОТЛАДКИ ---
            self.draw_debug_overlay(debug_img, center, sector_counts)
            self.pub_debug_img.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))

        except Exception as e:
            self.get_logger().error(f'Ошибка кадра: {e}')

    def publish_virtual_scan(self, sector_counts):
        """Превращает данные секторов в стандартное сообщение LaserScan"""
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = self.frame_id
        
        # Лидар смотрит вниз, поэтому лучи идут по кругу (2pi)
        scan.angle_min = 0.0
        scan.angle_max = 2.0 * math.pi
        scan.angle_increment = (2.0 * math.pi) / self.num_readings
        
        scan.time_increment = 0.0
        scan.range_min = self.scan_range_min
        scan.range_max = self.scan_range_max
        
        # Формируем массив дальностей (ranges)
        scan.ranges = []
        for count in sector_counts:
            # Порог: если в секторе > 5 линий - считаем, что там стена (близко)
            if count > 5:
                scan.ranges.append(self.scan_range_min + 0.1) # Стена на 0.3м
            else:
                scan.ranges.append(self.scan_range_max) # Свободно
                
        self.pub_scan.publish(scan)

    def draw_debug_overlay(self, img, center, sector_counts):
        """Рисует сектора и статистику поверх изображения"""
        h, w = img.shape[:2]
        cv2.circle(img, center, int(min(w, h) * 0.35), (255, 0, 0), 2) # Зона безопасности (синий)
        
        # Рисуем разделители секторов (45, 135, 225, 315 градусов)
        for angle in [45, 135, 225, 315]:
            rad = math.radians(angle)
            end_x = int(center[0] + w * math.cos(rad))
            end_y = int(center[1] + h * math.sin(rad))
            cv2.line(img, center, (end_x, end_y), (0, 255, 0), 1)

        # Выводим статистику по секторам
        text_y_offset = 30
        max_count = np.max(sector_counts) if np.max(sector_counts) > 0 else 1
        
        for i, (label, count) in enumerate(zip(SECTOR_LABELS, sector_counts)):
            # Чем больше линий, тем краснее текст
            intensity = int((count / max_count) * 255)
            color = (0, 255 - intensity, intensity) # BGR
            
            status_text = f"{label}: {count} lines"
            cv2.putText(img, status_text, (10, text_y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            text_y_offset += 25

def main():
    rclpy.init()
    node = FisheyeMapBuilderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()