
import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

canvas = None
prev_x, prev_y = None, None
smooth_x, smooth_y = None, None
drawing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = hands.process(rgb)
    rgb.flags.writeable = True

    drawing = False

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        thumb = hand.landmark[4]
        index = hand.landmark[8]

        tx, ty = int(thumb.x * w), int(thumb.y * h)
        ix, iy = int(index.x * w), int(index.y * h)

        distance = np.hypot(tx - ix, ty - iy)

        # Barmaq hərəkətini yumşalt
        if smooth_x is None:
            smooth_x, smooth_y = ix, iy
        else:
            smooth_x = int(smooth_x * 0.65 + ix * 0.35)
            smooth_y = int(smooth_y * 0.65 + iy * 0.35)

        x, y = smooth_x, smooth_y

        # Pinch - barmaqları birləşdir
        if distance < 40:
            drawing = True

            cv2.circle(frame, (tx, ty), 7, (0, 255, 0), -1)
            cv2.circle(frame, (ix, iy), 7, (0, 255, 0), -1)

            if prev_x is not None:
                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (x, y),
                    (0, 255, 0),
                    6
                )

            prev_x, prev_y = x, y
        else:
            prev_x, prev_y = None, None

    else:
        prev_x, prev_y = None, None
        smooth_x, smooth_y = None, None

    output = cv2.add(frame, canvas)

    status = "DRAWING" if drawing else "READY"
    cv2.putText(output, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)

    cv2.putText(output, "Pinch: Draw | C: Clear | Q: Quit",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    cv2.imshow("Pinch Drawing", output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        canvas[:] = 0

    if key == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
