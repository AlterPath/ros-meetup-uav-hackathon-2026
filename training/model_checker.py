import pygame
import cv2
import numpy as np
from ultralytics import YOLO
import sys
import time

# === НАСТРОЙКИ ===
MODEL_PATH = '/home/alterpath/drones/catkin_ws/src/hack_drones_20-22/runs/classify/drone_results/my_classifier9/weights/best.pt'
CONF_THRESHOLD = 0.90
WIN_W, WIN_H = 800, 600
# =================

class ModelChecker:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("YOLO Checker - Wait for Camera...")
        self.font = pygame.font.SysFont("Arial", 22)
        
        print(f"[*] Загрузка модели: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        
        # Пробуем разные индексы, если 0 не работает (бывает 0, 1, 2)
        self.cap = cv2.VideoCapture(0)
        
        # Даем камере время "проснуться" (важно для драйверов в симуляторе)
        print("[*] Ожидание инициализации камеры (3 сек)...")
        time.sleep(3) 
        
        self.clock = pygame.time.Clock()

    def cv_to_pygame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        return pygame.transform.flip(frame, True, False)

    def run(self):
        # Проверка: открылась ли камера вообще?
        if not self.cap.isOpened():
            print("[-] ОШИБКА: Камера не найдена. Проверь индекс VideoCapture(0/1/2)")
            self.show_exit_screen("ERROR: CAMERA NOT FOUND")
            return

        running = True
        print("[*] Поток запущен. Нажми ESC или закрыть окно для выхода.")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            success, frame = self.cap.read()
            
            if not success:
                # Если кадр не считан, не выходим сразу, а пробуем еще пару раз
                print("[-] Кадр не получен, пробую еще раз...", end='\r')
                continue 

            # КЛАССИФИКАЦИЯ
            results = self.model(frame, verbose=False)
            res = results[0]
            conf = res.probs.top1conf.item()
            cls_name = res.names[res.probs.top1]
            final_result = cls_name if conf >= CONF_THRESHOLD else "nothing"

            # ОТРИСОВКА
            self.screen.fill((30, 30, 30))
            
            img_surface = self.cv_to_pygame(frame)
            img_surface = pygame.transform.scale(img_surface, (640, 480))
            self.screen.blit(img_surface, (80, 20))

            color = (0, 255, 0) if final_result != "nothing" else (200, 200, 200)
            msg = f"OBJECT: {final_result.upper()} | CONF: {conf:.2%}"
            txt = self.font.render(msg, True, color)
            self.screen.blit(txt, (80, 530))

            pygame.display.flip()
            self.clock.tick(30)

        self.cap.release()
        self.show_exit_screen("TEST FINISHED. PRESS ANY KEY.")

    def show_exit_screen(self, message):
        waiting = True
        while waiting:
            self.screen.fill((0, 0, 0))
            msg_surf = self.font.render(message, True, (255, 255, 0))
            rect = msg_surf.get_rect(center=(WIN_W//2, WIN_H//2))
            self.screen.blit(msg_surf, rect)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type in [pygame.QUIT, pygame.KEYDOWN]:
                    waiting = False
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    checker = ModelChecker()
    checker.run()
