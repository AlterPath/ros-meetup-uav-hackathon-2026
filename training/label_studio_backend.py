from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO

class YOLOv8nBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Загружаем pretrained модель YOLOv8n
        self.model = YOLO('yolov8n.pt')  # можно заменить на кастомный путь

    def predict(self, tasks, **kwargs):
        results = []
        for task in tasks:
            image_url = task['data']['image']  # путь к картинке в Label Studio
            predictions = self.model.predict(image_url)
            # формируем результат в формате Label Studio
            annotations = []
            for pred in predictions[0].boxes.data.tolist():  # [[x1,y1,x2,y2,score,class]]
                x1, y1, x2, y2, score, class_id = pred
                annotations.append({
                    'from_name': 'bbox',
                    'to_name': 'image',
                    'type': 'rectanglelabels',
                    'value': {
                        'x': x1,
                        'y': y1,
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'rotation': 0,
                        'rectanglelabels': [str(int(class_id))]
                    }
                })
            results.append({'result': annotations})
        return results
