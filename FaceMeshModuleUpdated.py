import cv2
import mediapipe as mp
import math

class FaceMeshDetector:

    """
    Enhanced Face Mesh Detector to find 468 face landmarks + 10 iris landmarks (5 per eye).
    Helps acquire the landmark points in pixel format.
    """

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):

        """
        :param staticMode: In static mode, detection is done on each image: slower
        :param maxFaces: Maximum number of faces to detect
        :param minDetectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            refine_landmarks=True,  # Enable iris detection
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

        # Iris landmark indices (MediaPipe)
        self.LEFT_IRIS = [468, 469, 470, 471, 472]  # Left eye pupil landmarks
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]  # Right eye pupil landmarks


    def findFaceMesh(self, img, draw=True):

        """
        Finds face and iris landmarks in BGR Image.
        :param img: Image to find the face landmarks in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings, list of face landmarks, list of iris landmarks
        """

        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        irises = []  # Stores left and right iris landmarks for each face

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                        self.drawSpec, self.drawSpec
                    )

                # Extract face landmarks
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])

                # Extract iris landmarks (pupils)
                iris = {}
                iris['left'] = [face[idx] for idx in self.LEFT_IRIS]  # Left pupil points
                iris['right'] = [face[idx] for idx in self.RIGHT_IRIS]  # Right pupil points

                faces.append(face)
                irises.append(iris)

                """ Draw pupils (optional) """
                # if draw:
                #     for point in iris['left']:
                #         cv2.circle(img, point, 2, (0, 0, 255), -1)  # Red for left pupil
                #     for point in iris['right']:
                #         cv2.circle(img, point, 2, (255, 0, 0), -1)  # Blue for right pupil

        return img, faces, irises


    def findDistance(self, p1, p2, img=None):

        """
        Find the distance between two landmarks.
        :param p1: Point1 (x, y)
        :param p2: Point2 (x, y)
        :param img: Image to draw on.
        :return: Distance, info (coordinates), and optionally the image with drawings
        """
        
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)

        if img is not None:
            cv2.circle(img, (x1, y1), 3, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 3, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
            cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5)

    while True:
        success, img = cap.read()
        if not success:
            break

        # Get face mesh and iris landmarks
        img, faces, irises = detector.findFaceMesh(img, draw=True)

        if faces and irises:
            for i, (face, iris) in enumerate(zip(faces, irises)):
                # Example: Get the center of the left pupil (first landmark in LEFT_IRIS)
                left_pupil_center = iris['left'][0]  # Index 0 is the center of the iris
                right_pupil_center = iris['right'][0]

                # Print pupil positions
                print(f"Face {i+1} - Left Pupil: {left_pupil_center}, Right Pupil: {right_pupil_center}")

                # Draw a larger circle at pupil centers
                cv2.circle(img, left_pupil_center, 5, (0, 255, 0), -1)  # Green for left pupil
                cv2.circle(img, right_pupil_center, 5, (0, 255, 255), -1)  # Yellow for right pupil

        cv2.imshow("Face and Pupil Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
