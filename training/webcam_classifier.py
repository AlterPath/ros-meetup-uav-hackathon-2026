from ultralytics import YOLO
import cv2

# Загружаем модель
model = YOLO('/home/alterpath/drones/catkin_ws/src/hack_drones_20-22/runs/classify/drone_results/my_classifier2/weights/best.pt')

# Открываем камеру (0 - обычно встроенная или первая USB-камера)
cap = cv2.VideoCapture(0)

print("Нажми 'q' для выхода")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Запускаем классификацию кадра
    results = model(frame, verbose=False) # verbose=False чтобы не спамить в консоль
    
    res = results[0]
    cls_idx = res.probs.top1
    cls_name = res.names[cls_idx]
    conf = res.probs.top1conf.item()

    # Рисуем текст на кадре
    text = f"{cls_name} ({conf:.2%})"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Показываем окно
    cv2.imshow("YOLOv8 Classification Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

