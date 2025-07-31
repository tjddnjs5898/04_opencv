import cv2
import numpy as np

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
            # - 한국 번호판은 흰 바탕 + 검정 글자이므로 선명한 이진 이미지가 효과적
            enhanced_plate = cv2.adaptiveThreshold(
                gray_plate,
                maxValue=255,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY,
                blockSize=11,
                C=2
            )

            # 결과 출력
            cv2.imshow("License Plate Extractor - Warped (Color)", result)
            cv2.imshow("License Plate Extractor - Grayscale", gray_plate)
            cv2.imshow("License Plate Extractor - Contrast Enhanced", enhanced_plate)

# 이미지 하나씩 반복
for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"이미지 로드 실패: {path}")
        continue

    pts = np.zeros((4, 2), dtype=np.float32)
    pts_cnt = 0
    draw = img.copy()

    print(f"\n{path} 파일: 번호판 영역을 시계 방향으로 4점 클릭하세요.")
    cv2.imshow("License Plate Extractor - Original", draw)
    cv2.setMouseCallback("License Plate Extractor - Original", onMouse)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
