import cv2
import argparse


class AgeGenderDetector:
    def __init__(self, face_model, face_proto, age_model, age_proto, gender_model, gender_proto):
        self.face_net = cv2.dnn.readNet(face_model, face_proto)
        self.age_net = cv2.dnn.readNet(age_model, age_proto)
        self.gender_net = cv2.dnn.readNet(gender_model, gender_proto)
        self.model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_list = ['(25-32)', '(25-32)', '(25-32)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['LALAKI', 'BABAE']
        self.padding = 20
        self.conf_threshold = 0.5

    def highlight_faces(self, frame):
        frame_copy = frame.copy()
        frame_height, frame_width = frame_copy.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], True, False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        face_boxes = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                face_boxes.append([x1, y1, x2, y2])

                # Set border color based on confidence
                border_color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), border_color, int(round(frame_height / 150)), 8)

        return frame_copy, face_boxes

    def detect_age_gender(self, face):
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.model_mean_values, swapRB=False)

        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.gender_list[gender_preds[0].argmax()]

        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = self.age_list[age_preds[0].argmax()]

        return gender, age[1:-1]

    def process_frame(self, frame):
        result_img, face_boxes = self.highlight_faces(frame)
        if not face_boxes:
            print("No face detected")

        for face_box in face_boxes:
            face = frame[max(0, face_box[1] - self.padding):min(face_box[3] + self.padding, frame.shape[0] - 1), max(0, face_box[0] - self.padding):min(face_box[2] + self.padding, frame.shape[1] - 1)]

            gender, age = self.detect_age_gender(face)
            print(f'Gender: {gender}, Age: {age} years')

            # Create the text to be displayed
            text = f'Gender: {gender}, Age: {age}'

            # Calculate text size and position to center it
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            x_centered = face_box[0] + (face_box[2] - face_box[0] - text_width) // 2
            y_position = face_box[1] - 10

            cv2.putText(result_img, text, (x_centered, y_position), font, font_scale, (0, 255, 0), thickness,
                        cv2.LINE_AA)

        return result_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='Videos/nas.mp4')  # Default to the video file inside the Videos folder
    args = parser.parse_args()

    face_proto = "opencv_face_detector.pbtxt"
    face_model = "opencv_face_detector_uint8.pb"
    age_proto = "age_deploy.prototxt"
    age_model = "age_net.caffemodel"
    gender_proto = "gender_deploy.prototxt"
    gender_model = "gender_net.caffemodel"

    detector = AgeGenderDetector(face_model, face_proto, age_model, age_proto, gender_model, gender_proto)
    video = cv2.VideoCapture(args.video)

    try:
        while True:
            has_frame, frame = video.read()
            if not has_frame:
                break

            result_img = detector.process_frame(frame)
            cv2.imshow("Detecting age and gender", result_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()