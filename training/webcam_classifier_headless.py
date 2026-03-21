
from ultralytics import YOLO
import cv2

model = YOLO('/Hack/classifier/weights/best.pt')
cap = cv2.VideoCapture(0)

print("Нажми Ctrl+C для выхода. Смотри результат в файле live_view.jpg")

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        results = model(frame, verbose=False)
        res = results[0]
        
        # Рисуем текст прямо на кадре (для истории)
        cls_name = res.names[res.probs.top1]
        conf = res.probs.top1conf.item()
        cv2.putText(frame, f"{cls_name} {conf:.2%}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ВМЕСТО cv2.imshow — СОХРАНЯЕМ
        cv2.imwrite("live_view.jpg", frame)
        
        print(f"Вижу: {cls_name} ({conf:.2%})", end='\r')

        # Запускаем классификацию
        results = model(frame, verbose=False)
        res = results[0]
        
        # Получаем вероятность лучшего класса
        conf = res.probs.top1conf.item()
        cls_idx = res.probs.top1
        cls_name = res.names[cls_idx]
        
        # ЖЕСТКИЙ ПОРОГ УВЕРЕННОСТИ
        if conf < 0.90:  # Если уверенность меньше 90%
            final_result = "nothing"
        else:
            final_result = cls_name
                
        print(f"Результат: {final_result} (Уверенность: {conf:.2%})", end='\r')
        # print(f"Результат: {final_result} (Уверенность: {conf:.2%})", end='\r')
except KeyboardInterrupt:
    print("\nСтоп.")
finally:
    cap.release()

