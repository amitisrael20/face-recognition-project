import cv2


DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "models/face_recognition_sface_2021dec.onnx"

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

    aligned_face = recognizer.alignCrop(
        image,
        face
    )

    embedding = recognizer.feature(
        aligned_face
    )

    return embedding

def identify_face(current_embedding, known_faces):
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

    return best_name, best_distance