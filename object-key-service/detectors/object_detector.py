from ultralytics import YOLO


class ObjectDetector:

    def __init__(
        self,
        model_path="yolo11n.pt",
        confidence_threshold=0.55
    ):
        self.model = YOLO(model_path)

        self.confidence_threshold = (
            confidence_threshold
        )


    def detect(self, frame):

        results = self.model(
            frame,
            verbose=False
        )

        result = results[0]

        detections = []


        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )


            if (
                confidence
                <
                self.confidence_threshold
            ):
                continue


            class_id = int(
                box.cls[0]
            )


            object_name = (
                self.model.names[
                    class_id
                ]
            )


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            detections.append({
                "name": object_name,
                "confidence": confidence,
                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                )
            })


        return detections