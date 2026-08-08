import math

import cv2
import mediapipe as mp


class GestureDetector:

    def __init__(
        self,
        model_path=None,
        classes_path=None,
        confidence_threshold=0.75
    ):
        """
        Gesture detector menggunakan MediaPipe Hands.

        model_path dan classes_path tetap disediakan
        agar kompatibel jika main.py lama masih
        mengirim parameter tersebut.

        Tetapi pada versi ini ONNX sudah tidak digunakan.
        """

        self.confidence_threshold = confidence_threshold

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_draw = mp.solutions.drawing_utils

    # HITUNG JARAK 2 TITIK
    def distance(self, point_a, point_b):

        return math.sqrt(
            (point_a.x - point_b.x) ** 2
            +
            (point_a.y - point_b.y) ** 2
            +
            (point_a.z - point_b.z) ** 2
        )

    # HITUNG SUDUT 3 TITIK
    def calculate_angle(
        self,
        point_a,
        point_b,
        point_c
    ):
        """
        Menghitung sudut A-B-C.
        Titik B adalah titik tengah/sendi.
        """

        vector_ba = (
            point_a.x - point_b.x,
            point_a.y - point_b.y,
            point_a.z - point_b.z,
        )

        vector_bc = (
            point_c.x - point_b.x,
            point_c.y - point_b.y,
            point_c.z - point_b.z,
        )


        dot_product = (
            vector_ba[0] * vector_bc[0]
            +
            vector_ba[1] * vector_bc[1]
            +
            vector_ba[2] * vector_bc[2]
        )


        length_ba = math.sqrt(
            vector_ba[0] ** 2
            +
            vector_ba[1] ** 2
            +
            vector_ba[2] ** 2
        )


        length_bc = math.sqrt(
            vector_bc[0] ** 2
            +
            vector_bc[1] ** 2
            +
            vector_bc[2] ** 2
        )


        if (
            length_ba == 0
            or
            length_bc == 0
        ):
            return 0.0


        cosine_angle = (
            dot_product
            /
            (
                length_ba
                *
                length_bc
            )
        )


        # Hindari error floating point
        cosine_angle = max(
            -1.0,
            min(
                1.0,
                cosine_angle
            )
        )


        angle = math.degrees(
            math.acos(
                cosine_angle
            )
        )


        return angle

    # CEK JARI BIASA TERBUKA
    def is_finger_open(
        self,
        landmarks,
        mcp_index,
        pip_index,
        dip_index,
        tip_index
    ):
        """
        Digunakan untuk:
        - telunjuk
        - tengah
        - manis
        - kelingking

        Tidak bergantung pada tangan kiri/kanan.
        """

        mcp = landmarks[mcp_index]
        pip = landmarks[pip_index]
        dip = landmarks[dip_index]
        tip = landmarks[tip_index]


        # Sudut PIP
        pip_angle = self.calculate_angle(
            mcp,
            pip,
            dip
        )


        # Sudut DIP
        dip_angle = self.calculate_angle(
            pip,
            dip,
            tip
        )


        # Jarak ujung jari ke pergelangan
        wrist = landmarks[0]

        tip_to_wrist = self.distance(
            tip,
            wrist
        )

        pip_to_wrist = self.distance(
            pip,
            wrist
        )


        # Jari dianggap terbuka jika:
        # 1. sendinya relatif lurus
        # 2. ujung jari lebih jauh dari wrist
        #    dibanding sendi PIP
        is_straight = (
            pip_angle > 150
            and
            dip_angle > 150
        )


        is_extended = (
            tip_to_wrist
            >
            pip_to_wrist * 1.08
        )


        return (
            is_straight
            and
            is_extended
        )

    # CEK IBU JARI TERBUKA
    def is_thumb_open(
        self,
        landmarks
    ):
        """
        Deteksi ibu jari tanpa bergantung
        pada handedness Left / Right.
        """

        wrist = landmarks[0]

        thumb_cmc = landmarks[1]
        thumb_mcp = landmarks[2]
        thumb_ip = landmarks[3]
        thumb_tip = landmarks[4]

        index_mcp = landmarks[5]


        # Sudut ibu jari
        thumb_mcp_angle = self.calculate_angle(
            thumb_cmc,
            thumb_mcp,
            thumb_ip
        )


        thumb_ip_angle = self.calculate_angle(
            thumb_mcp,
            thumb_ip,
            thumb_tip
        )


        # Jarak ibu jari ke pangkal telunjuk
        tip_to_index = self.distance(
            thumb_tip,
            index_mcp
        )


        ip_to_index = self.distance(
            thumb_ip,
            index_mcp
        )


        # Jarak ke wrist
        tip_to_wrist = self.distance(
            thumb_tip,
            wrist
        )


        mcp_to_wrist = self.distance(
            thumb_mcp,
            wrist
        )


        # Ibu jari harus cukup lurus
        is_straight = (
            thumb_mcp_angle > 130
            and
            thumb_ip_angle > 140
        )


        # Ujung ibu jari menjauh dari telapak
        is_away_from_index = (
            tip_to_index
            >
            ip_to_index * 1.10
        )


        # Ujung ibu jari juga lebih jauh dari wrist
        is_extended = (
            tip_to_wrist
            >
            mcp_to_wrist * 1.20
        )


        return (
            is_straight
            and
            is_away_from_index
            and
            is_extended
        )

    # HITUNG JUMLAH JARI
    def count_fingers(
        self,
        hand_landmarks
    ):

        landmarks = (
            hand_landmarks.landmark
        )


        fingers = 0

        # IBU JARI
        thumb_open = (
            self.is_thumb_open(
                landmarks
            )
        )


        if thumb_open:
            fingers += 1

        # TELUNJUK
        # landmark:
        # MCP = 5
        # PIP = 6
        # DIP = 7
        # TIP = 8
        index_open = (
            self.is_finger_open(
                landmarks,
                5,
                6,
                7,
                8
            )
        )


        if index_open:
            fingers += 1


        # =====================================================
        # JARI TENGAH
        # MCP = 9
        # PIP = 10
        # DIP = 11
        # TIP = 12
        # =====================================================

        middle_open = (
            self.is_finger_open(
                landmarks,
                9,
                10,
                11,
                12
            )
        )


        if middle_open:
            fingers += 1


        # =====================================================
        # JARI MANIS
        # MCP = 13
        # PIP = 14
        # DIP = 15
        # TIP = 16
        # =====================================================

        ring_open = (
            self.is_finger_open(
                landmarks,
                13,
                14,
                15,
                16
            )
        )


        if ring_open:
            fingers += 1


        # =====================================================
        # KELINGKING
        # MCP = 17
        # PIP = 18
        # DIP = 19
        # TIP = 20
        # =====================================================

        pinky_open = (
            self.is_finger_open(
                landmarks,
                17,
                18,
                19,
                20
            )
        )


        if pinky_open:
            fingers += 1


        return fingers


    # =========================================================
    # DETECT
    # =========================================================

    def detect(
        self,
        frame
    ):

        # OpenCV = BGR
        # MediaPipe = RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        result = (
            self.hands.process(
                rgb_frame
            )
        )


        # Tidak ada tangan
        if not result.multi_hand_landmarks:
            return []


        detected_hands = []


        # =====================================================
        # LOOP SEMUA TANGAN
        # =====================================================

        for hand_index, hand_landmarks in enumerate(
            result.multi_hand_landmarks
        ):

            # =================================================
            # GAMBAR LANDMARK
            # =================================================

            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )


            # =================================================
            # HANDEDNESS
            # =================================================
            # Hanya sebagai informasi.
            # TIDAK digunakan untuk menghitung jumlah jari.
            # =================================================

            hand_name = (
                f"Hand {hand_index + 1}"
            )

            confidence = 1.0


            if (
                result.multi_handedness
                and
                hand_index
                <
                len(
                    result.multi_handedness
                )
            ):

                classification = (
                    result
                    .multi_handedness[
                        hand_index
                    ]
                    .classification[0]
                )


                hand_name = (
                    classification.label
                )


                confidence = float(
                    classification.score
                )


            # =================================================
            # HITUNG JARI
            # =================================================

            finger_count = (
                self.count_fingers(
                    hand_landmarks
                )
            )


            # =================================================
            # HASIL
            # =================================================

            detected_hands.append({

                "label": str(
                    finger_count
                ),

                "finger_count":
                    finger_count,

                "confidence":
                    confidence,

                "hand":
                    hand_name,

            })


        return detected_hands