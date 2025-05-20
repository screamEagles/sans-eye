import cv2
import numpy as np
from FaceMeshModuleUpdated import FaceMeshDetector
import pygame
import time


# Initialisation
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("./sans.wav") # You have to upload your own
is_playing = False
last_play_time = 0

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = FaceMeshDetector(staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5)

# Eye landmarks
LEFT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 263, 249, 390, 373, 374, 380, 381, 382]

# E.A.R (Eye Aspect Ratio) calculation points
LEFT_EAR_POINTS = [159, 23, 130, 243]  # For eye openness detection
RIGHT_EAR_POINTS = [386, 374, 362, 263]

# Thresholds (with trackbar support)
LEFT_EAR_THRESHOLD, RIGHT_EAR_THRESHOLD = 0.26, 0.23


def get_ear(eye_points, face):
    """ Calculate Eye Aspect Ratio (E.A.R) """
    A = face[eye_points[0]]  # Top
    B = face[eye_points[1]]  # Bottom
    C = face[eye_points[2]]  # Left
    D = face[eye_points[3]]  # Right
    
    vertical = np.linalg.norm(np.array(A) - np.array(B))
    horizontal = np.linalg.norm(np.array(C) - np.array(D))
    
    return vertical / horizontal if horizontal != 0 else 0.0


def overlay_transparent(background, overlay, x, y):
    """Overlay a transparent image onto a background"""
    bg_height, bg_width = background.shape[:2]
    
    # Convert overlay to BGRA if it isn't already
    if overlay.shape[2] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
    
    # Get overlay dimensions
    h, w = overlay.shape[:2]
    
    # Limit overlay to background dimensions
    if x + w > bg_width or y + h > bg_height:
        h, w = min(h, bg_height - y), min(w, bg_width - x)
        overlay = overlay[:h, :w]
    
    # Extract the alpha channel
    alpha = overlay[:, :, 3] / 255.0
    inv_alpha = 1.0 - alpha
    
    # Blend each color channel
    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = (
            alpha * overlay[:, :, c] + 
            inv_alpha * background[y:y+h, x:x+w, c]
        )
    return background


# Create window with trackbars
cv2.namedWindow("Advanced Eye Tracking")
cv2.createTrackbar("Left", "Advanced Eye Tracking", int(LEFT_EAR_THRESHOLD * 100), 50, lambda x: None)
cv2.createTrackbar("Right", "Advanced Eye Tracking", int(RIGHT_EAR_THRESHOLD * 100), 50, lambda x: None)


while True:
    success, img = cap.read()
    if not success:
        break

    sans_img = cv2.imread("./sans.png", cv2.IMREAD_UNCHANGED)
    sans_img = cv2.resize(sans_img, (600, 400))

    # Update thresholds from trackbars
    LEFT_EAR_THRESHOLD = cv2.getTrackbarPos("Left", "Advanced Eye Tracking") / 100
    RIGHT_EAR_THRESHOLD = cv2.getTrackbarPos("Right", "Advanced Eye Tracking") / 100

    img, faces, irises = detector.findFaceMesh(img, draw=False)
    img = overlay_transparent(img, sans_img, 350, 150) # tune the numbers to set it on your face

    if faces and irises:
        for face, iris in zip(faces, irises):
            # Calculate E.A.R for both eyes
            left_ear = get_ear(LEFT_EAR_POINTS, face)
            right_ear = get_ear(RIGHT_EAR_POINTS, face)

            # Draw eye contours (always visible)
            left_eye_pts = np.array([face[idx] for idx in LEFT_EYE_CONTOUR], np.int32).reshape((-1, 1, 2))
            right_eye_pts = np.array([face[idx] for idx in RIGHT_EYE_CONTOUR], np.int32).reshape((-1, 1, 2))

            # Draw pupils only if eyes are open
            if left_ear > LEFT_EAR_THRESHOLD and right_ear > RIGHT_EAR_THRESHOLD:
                # Eyes open - stop sound if playing
                if is_playing:
                    alert_sound.stop()
                    is_playing = False
                continue
            else:
                # Eyes closed - play sound (with cooldown)
                current_time = time.time()
                if not is_playing and current_time - last_play_time > 1.0:  # 1 second cooldown
                    alert_sound.play()
                    is_playing = True
                    last_play_time = current_time

                if left_ear > LEFT_EAR_THRESHOLD:
                    cv2.fillPoly(img, [left_eye_pts], (255, 0, 0))
                    cv2.circle(img, iris['left'][0], 3, (0, 0, 0), -1)
                if right_ear > RIGHT_EAR_THRESHOLD:
                    cv2.fillPoly(img, [right_eye_pts], (255, 0, 0))
                    cv2.circle(img, iris['right'][0], 3, (0, 0, 0), -1)

            """ Display metrics (for debugging purposes) """
            # cv2.putText(img, f"L-EAR: {left_ear:.2f} (>{LEFT_EAR_THRESHOLD:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # cv2.putText(img, f"R-EAR: {right_ear:.2f} (>{RIGHT_EAR_THRESHOLD:.2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (252, 182, 3), 2)

    cv2.imshow("Advanced Eye Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


pygame.mixer.quit()
cap.release()
cv2.destroyAllWindows()
