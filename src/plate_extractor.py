import cv2
import numpy as np
from matplotlib import pyplot as plt

# 이미지 경로 리스트
image_paths = [
    '../img/car_01.jpg',
    '../img/car_02.jpg',
    '../img/car_03.jpg',
    '../img/car_04.jpg',
    '../img/car_05.jpg'
]

# 전역 변수 선언
pts = np.zeros((4, 2), dtype=np.float32)
pts_cnt = 0
draw = None
img = None

# 번호판 전용 적응형 임계처리 함수
def adaptive_threshold_plate(enhanced_plate):
    """번호판 전용 적응형 임계처리"""

    # 1단계: 가벼운 블러링 (노이즈 제거, 글자는 보존)
    blurred = cv2.GaussianBlur(enhanced_plate, (3, 3), 0)

    # 2단계: 번호판 최적화 적응형 임계처리
    thresh_adaptive = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,  # 일반 BINARY 사용
        blockSize=11,
        C=2
    )

    # 3단계: Otsu 임계처리와 비교
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4단계: 결과 비교 시각화
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(enhanced_plate, cmap='gray')
    plt.title('Enhanced Plate')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(blurred, cmap='gray')
    plt.title('Blurred')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(thresh_adaptive, cmap='gray')
    plt.title('Adaptive Threshold')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(thresh_otsu, cmap='gray')
    plt.title('Otsu Threshold')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return thresh_adaptive, thresh_otsu

# 마우스 콜백 함수
def onMouse(event, x, y, flags, param):
    global pts_cnt, pts, draw, img

    if event == cv2.EVENT_LBUTTONDOWN and pts_cnt < 4:
        pts[pts_cnt] = [x, y]
        pts_cnt += 1

        cv2.circle(draw, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(draw, str(pts_cnt), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("License Plate Extractor - Original", draw)

        if pts_cnt == 4:
            width, height = 300, 100
            dst_pts = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(pts, dst_pts)
            result = cv2.warpPerspective(img, M, (width, height))

            # 1. 그레이스케일 변환
            gray_plate = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            # 2. 대비 극대화를 위한 전처리 (OCR 최적화)
            enhanced_plate = cv2.adaptiveThreshold(
                gray_plate,
                maxValue=255,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY,
                blockSize=11,
                C=2
            )

            # 3. 번호판 전용 적응형 임계처리 함수 호출
            thresh_adaptive, thresh_otsu = adaptive_threshold_plate(enhanced_plate)

            # 결과 출력
            cv2.imshow("License Plate Extractor - Warped (Color)", result)
            cv2.imshow("License Plate Extractor - Grayscale", gray_plate)
            cv2.imshow("License Plate Extractor - Contrast Enhanced", enhanced_plate)
            cv2.imshow("Adaptive Threshold Result", thresh_adaptive)
            cv2.imshow("Otsu Threshold Result", thresh_otsu)

# 이미지 하나씩 반복
for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"이미지 로드 실패: {path}")
        continue

    pts = np.zeros((4, 2), dtype=np.float32)
    pts_cnt = 0
    draw = img.copy()

    cv2.imshow("License Plate Extractor - Original", draw)
    cv2.setMouseCallback("License Plate Extractor - Original", onMouse)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
