import json
import os
import time

import cv2

from detectors.object_detector import ObjectDetector
from detectors.gesture_detector import GestureDetector

from mapping import (
    OBJECT_ACTION_MAP,
    OBJECT_LABEL_MAP,
    GESTURE_VALUE_MAP,
)


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

OBJECT_DETECTION_INTERVAL = 3
OBJECT_REARM_SECONDS = 1.0

GESTURE_HOLD_SECONDS = 0.7
GESTURE_RELEASE_SECONDS = 0.5


# ============================================================
# PATH DATA.JSON
# ============================================================
# Struktur project:
#
# object-detection/
# ├── keylistener/
# │   ├── index.html
# │   └── data.json
# │
# └── object-key-service/
#     ├── main.py
#     └── mapping.py
#
# Jadi dari main.py:
# ../keylistener/data.json
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_JSON_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "keylistener",
        "data.json"
    )
)


# ============================================================
# JSON FUNCTION
# ============================================================

def save_number_to_json(number=""):
    """
    Menyimpan angka gesture ke data.json.

    Format JSON sengaja tetap:

    {
        "number": ""
    }

    supaya index.html lama tidak perlu diubah.
    """

    data = {
        "number": str(number)
    }

    # File temporary dipakai supaya browser
    # tidak membaca JSON ketika file sedang ditulis.
    temp_path = DATA_JSON_PATH + ".tmp"

    try:

        with open(
            temp_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=4
            )

        os.replace(
            temp_path,
            DATA_JSON_PATH
        )

    except OSError as error:

        print(
            f"[JSON ERROR] {error}"
        )


# ============================================================
# RESET JSON SAAT PROGRAM DIMULAI
# ============================================================

save_number_to_json("")


print(
    f"Data JSON: {DATA_JSON_PATH}"
)


# ============================================================
# INITIALIZE OBJECT DETECTOR
# ============================================================

print("Memuat YOLO...")

object_detector = ObjectDetector(
    model_path="yolo11n.pt",
    confidence_threshold=0.55,
)


# ============================================================
# INITIALIZE GESTURE DETECTOR
# ============================================================

print("Memuat Gesture Detector...")

gesture_detector = GestureDetector(
    confidence_threshold=0.75
)


# ============================================================
# INITIALIZE CAMERA
# ============================================================

camera = cv2.VideoCapture(
    CAMERA_INDEX
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

camera.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not camera.isOpened():

    raise RuntimeError(
        "Webcam tidak dapat dibuka."
    )


# ============================================================
# FRAME COUNTER
# ============================================================

frame_counter = 0


# ============================================================
# OBJECT STATE
# ============================================================

active_objects = set()

last_object_seen = {}

cached_object_detections = []


# ============================================================
# GESTURE STATE
# ============================================================

gesture_candidate = None

gesture_candidate_since = None

last_saved_gesture = None

last_gesture_seen_time = time.time()


# ============================================================
# OBJECT ACTION
# ============================================================

def trigger_object_action(object_name):
    """
    Menjalankan action object.

    Saat ini hanya:

    cell phone / HP
        ->
    Enter
        ->
    reset data.json
        ->
    browser menjadi kosong.
    """

    action = OBJECT_ACTION_MAP.get(
        object_name
    )

    if action is None:
        return


    label = OBJECT_LABEL_MAP.get(
        object_name,
        object_name
    )


    # ========================================================
    # HP = ENTER = RESET
    # ========================================================

    if action == "enter":

        save_number_to_json("")

        print(
            f"[OBJECT] "
            f"{label} -> ENTER / RESET"
        )


# ============================================================
# GESTURE -> JSON
# ============================================================

def trigger_gesture_value(label):
    """
    Mengubah hasil gesture 0-9
    menjadi nilai yang disimpan ke data.json.
    """

    value = GESTURE_VALUE_MAP.get(
        label
    )

    if value is None:
        return


    save_number_to_json(
        value
    )


    print(
        f"[GESTURE JSON] "
        f"{label} -> {value}"
    )


# ============================================================
# START INFORMATION
# ============================================================

print()

print(
    "============================================"
)

print(
    " OBJECT + GESTURE JSON SERVICE"
)

print(
    "============================================"
)

print()

print("OBJECT:")

print(
    "HP -> ENTER / RESET"
)

print()

print("GESTURE:")

print(
    "0-9 -> data.json -> browser"
)

print()

print(
    "Jumlah jari dari kedua tangan "
    "akan dijumlahkan."
)

print()

print(
    "Contoh:"
)

print(
    "1 jari        -> 1"
)

print(
    "2 jari        -> 2"
)

print(
    "5 + 1 jari    -> 6"
)

print(
    "5 + 2 jari    -> 7"
)

print(
    "5 + 3 jari    -> 8"
)

print(
    "5 + 4 jari    -> 9"
)

print()

print(
    f"Resolusi kamera : "
    f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"
)

print(
    f"YOLO berjalan setiap "
    f"{OBJECT_DETECTION_INTERVAL} frame"
)

print()

print(
    "Tekan Q pada jendela kamera "
    "untuk keluar."
)

print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()


    if not success:

        print(
            "Gagal membaca frame webcam."
        )

        break


    # ========================================================
    # CAMERA MIRROR
    # ========================================================
    # Tetap mirror seperti project sebelumnya.
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )


    current_time = time.time()

    frame_counter += 1


    # ========================================================
    # OBJECT DETECTION
    # ========================================================

    run_object_detection = (
        frame_counter == 1
        or
        frame_counter
        % OBJECT_DETECTION_INTERVAL
        == 0
    )


    if run_object_detection:

        cached_object_detections = (
            object_detector.detect(
                frame
            )
        )


    object_detections = (
        cached_object_detections
    )


    objects_seen_now = set()

    visible_objects = []


    # ========================================================
    # PROCESS OBJECT DETECTIONS
    # ========================================================

    for detection in object_detections:

        object_name = detection[
            "name"
        ]


        # ====================================================
        # HANYA OBJECT YANG ADA DI MAPPING
        # ====================================================
        # Saat ini hanya:
        #
        # cell phone -> HP
        # ====================================================

        if (
            object_name
            not in
            OBJECT_LABEL_MAP
        ):

            continue


        confidence = detection[
            "confidence"
        ]


        x1, y1, x2, y2 = detection[
            "box"
        ]


        display_name = (
            OBJECT_LABEL_MAP[
                object_name
            ]
        )


        objects_seen_now.add(
            object_name
        )


        if (
            display_name
            not in
            visible_objects
        ):

            visible_objects.append(
                display_name
            )


        if run_object_detection:

            last_object_seen[
                object_name
            ] = current_time


        # ====================================================
        # DRAW OBJECT BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 190, 255),
            2
        )


        object_label_text = (
            f"{display_name} "
            f"{confidence:.0%}"
        )


        cv2.putText(
            frame,
            object_label_text,
            (
                x1,
                max(
                    y1 - 10,
                    25
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 190, 255),
            2
        )


        # ====================================================
        # TRIGGER OBJECT ACTION
        # ====================================================
        # HP tidak akan melakukan reset terus-menerus
        # selama masih berada di depan kamera.
        #
        # Harus hilang terlebih dahulu,
        # baru bisa aktif lagi.
        # ====================================================

        if run_object_detection:

            if (
                object_name
                in
                OBJECT_ACTION_MAP
            ):

                if (
                    object_name
                    not in
                    active_objects
                ):

                    trigger_object_action(
                        object_name
                    )


                    active_objects.add(
                        object_name
                    )


    # ========================================================
    # RESET OBJECT STATE
    # ========================================================

    if run_object_detection:

        for object_name in list(
            active_objects
        ):

            if (
                object_name
                in
                objects_seen_now
            ):

                continue


            last_seen_time = (
                last_object_seen.get(
                    object_name,
                    0
                )
            )


            if (
                current_time
                -
                last_seen_time
                >=
                OBJECT_REARM_SECONDS
            ):

                active_objects.remove(
                    object_name
                )


    # ========================================================
    # GESTURE DETECTION
    # ========================================================

    gesture_results = (
        gesture_detector.detect(
            frame
        )
    )


    current_gesture = None

    gesture_detail_text = ""

    gesture_confidence = 0.0


    # ========================================================
    # HITUNG JUMLAH JARI SEMUA TANGAN
    # ========================================================

    if gesture_results:

        total_fingers = 0

        gesture_details = []

        confidences = []


        for hand_number, gesture in enumerate(
            gesture_results,
            start=1
        ):

            finger_count = int(
                gesture.get(
                    "finger_count",
                    0
                )
            )


            confidence = float(
                gesture.get(
                    "confidence",
                    0.0
                )
            )


            # =================================================
            # JUMLAHKAN JARI
            # =================================================

            total_fingers += (
                finger_count
            )


            confidences.append(
                confidence
            )


            gesture_details.append(
                f"Tangan {hand_number}: "
                f"{finger_count}"
            )


        # ====================================================
        # BATASI HASIL 0-9
        # ====================================================
        # Dua tangan terbuka penuh = 10.
        #
        # Karena yang dibutuhkan hanya 0-9,
        # nilai 10 tidak disimpan.
        # ====================================================

        if (
            0
            <=
            total_fingers
            <=
            9
        ):

            current_gesture = str(
                total_fingers
            )


        else:

            current_gesture = None


        # ====================================================
        # CONFIDENCE
        # ====================================================

        if confidences:

            gesture_confidence = min(
                confidences
            )


        gesture_detail_text = (
            " | ".join(
                gesture_details
            )
        )


        last_gesture_seen_time = (
            current_time
        )


    # ========================================================
    # GESTURE HOLD
    # ========================================================
    # Gesture harus stabil selama 0.7 detik
    # sebelum disimpan ke JSON.
    # ========================================================

    if current_gesture is not None:

        # ====================================================
        # GESTURE BERUBAH
        # ====================================================

        if (
            current_gesture
            !=
            gesture_candidate
        ):

            gesture_candidate = (
                current_gesture
            )


            gesture_candidate_since = (
                current_time
            )


        else:

            # =================================================
            # GESTURE HARUS STABIL
            # =================================================

            if (
                gesture_candidate_since
                is not None
                and
                current_time
                -
                gesture_candidate_since
                >=
                GESTURE_HOLD_SECONDS
            ):

                # =============================================
                # JANGAN SIMPAN BERULANG-ULANG
                # =============================================

                if (
                    current_gesture
                    !=
                    last_saved_gesture
                ):

                    if (
                        current_gesture
                        in
                        GESTURE_VALUE_MAP
                    ):

                        trigger_gesture_value(
                            current_gesture
                        )


                        print(
                            f"[GESTURE] "
                            f"{gesture_detail_text} "
                            f"= "
                            f"{current_gesture}"
                        )


                        last_saved_gesture = (
                            current_gesture
                        )


    # ========================================================
    # RESET GESTURE
    # ========================================================
    # Setelah tangan hilang selama 0.5 detik,
    # gesture yang sama boleh digunakan lagi.
    # ========================================================

    else:

        if (
            current_time
            -
            last_gesture_seen_time
            >=
            GESTURE_RELEASE_SECONDS
        ):

            gesture_candidate = None

            gesture_candidate_since = None

            last_saved_gesture = None


    # ========================================================
    # HEADER BACKGROUND
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (
            frame.shape[1],
            125
        ),
        (25, 25, 25),
        -1
    )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "OBJECT + GESTURE JSON SERVICE",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2
    )


    # ========================================================
    # OBJECT INFORMATION
    # ========================================================

    if visible_objects:

        object_text = (
            "Objek: "
            +
            ", ".join(
                visible_objects[:5]
            )
        )

    else:

        object_text = (
            "Objek: -"
        )


    cv2.putText(
        frame,
        object_text,
        (18, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 210, 255),
        2
    )


    # ========================================================
    # GESTURE INFORMATION
    # ========================================================

    if current_gesture is not None:

        gesture_text = (
            f"Gesture: "
            f"{current_gesture}"
        )


        if gesture_detail_text:

            gesture_text += (
                f" "
                f"({gesture_detail_text})"
            )


    elif gesture_results:

        gesture_text = (
            f"Gesture: - "
            f"({gesture_detail_text})"
        )


    else:

        gesture_text = (
            "Gesture: -"
        )


    cv2.putText(
        frame,
        gesture_text,
        (18, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (100, 255, 130),
        2
    )


    # ========================================================
    # PROCESS INFORMATION
    # ========================================================

    if run_object_detection:

        process_text = (
            "YOLO: SCAN"
        )

    else:

        process_text = (
            "YOLO: CACHE"
        )


    cv2.putText(
        frame,
        process_text,
        (18, 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (180, 180, 180),
        1
    )


    # ========================================================
    # FOOTER
    # ========================================================

    cv2.putText(
        frame,
        "Q = Keluar",
        (
            18,
            frame.shape[0] - 18
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


    # ========================================================
    # SHOW WINDOW
    # ========================================================

    cv2.imshow(
        "Object + Gesture JSON Service",
        frame
    )


    pressed_key = (
        cv2.waitKey(1)
        &
        0xFF
    )


    if (
        pressed_key
        ==
        ord("q")
    ):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()


# ============================================================
# RESET JSON SAAT SERVICE DIHENTIKAN
# ============================================================

save_number_to_json("")


print()

print(
    "Service dihentikan."
)