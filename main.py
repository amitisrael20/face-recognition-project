import os
import cv2


DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "models/face_recognition_sface_2021dec.onnx"

KNOWN_FACES_DIR = "known_faces"

THRESHOLD = 1.128


detector = cv2.FaceDetectorYN_create(
    DETECTOR_MODEL,
    "",
    (320, 320)
)

recognizer = cv2.FaceRecognizerSF_create(
    RECOGNIZER_MODEL,
    ""
)


def get_face_embedding(image):
    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None:
        return None

    face = faces[0]

    aligned_face = recognizer.alignCrop(image, face)

    embedding = recognizer.feature(aligned_face)

    return embedding


def load_known_faces():
    known_faces = []

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_path = os.path.join(KNOWN_FACES_DIR, person_name)

        if not os.path.isdir(person_path):
            continue

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)

            image = cv2.imread(image_path)

            if image is None:
                continue

            embedding = get_face_embedding(image)

            if embedding is None:
                continue

            known_faces.append(
                {
                    "name": person_name,
                    "embedding": embedding
                }
            )

    return known_faces


known_faces = load_known_faces()

print(f"Loaded {len(known_faces)} known face embeddings")


camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        break

    height, width = frame.shape[:2]
    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            aligned_face = recognizer.alignCrop(frame, face)

            current_embedding = recognizer.feature(aligned_face)

            best_name = "Unknown"
            best_distance = float("inf")

            for known_face in known_faces:
                distance = recognizer.match(
                    current_embedding,
                    known_face["embedding"],
                    cv2.FaceRecognizerSF_FR_NORM_L2
                )

                if distance < best_distance:
                    best_distance = distance
                    best_name = known_face["name"]

            if best_distance > THRESHOLD:
                best_name = "Unknown"

            x, y, face_width, face_height = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + face_width, y + face_height),
                (0, 200, 0),
                2
            )

            cv2.putText(
                frame,
                f"{best_name} | Distance: {best_distance:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("Face Identification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()