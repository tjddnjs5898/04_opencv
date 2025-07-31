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
    blurred = cv2.GaussianBlur(enhanced_plate, (3, 3), 0)
    thresh_adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

# 윤곽선 검출 함수
def find_contours_in_plate(thresh_plate):
    contours, hierarchy = cv2.findContours(thresh_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = thresh_plate.shape
    contour_image = cv2.cvtColor(thresh_plate, cv2.COLOR_GRAY2BGR)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 0, 128), (255, 165, 0)]

    for i, contour in enumerate(contours):
        color = colors[i % len(colors)]
        cv2.drawContours(contour_image, [contour], -1, color, 2)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(contour_image, str(i+1), (cx-5, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 바운딩 박스 이미지
    contour_info = np.zeros((height, width, 3), dtype=np.uint8)
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(contour_info, (x, y), (x+w, y+h), colors[i % len(colors)], 1)
        cv2.putText(contour_info, f'A:{int(area)}', (x, y-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

    # 시각화
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(thresh_plate, cmap='gray')
    plt.title('Binary Plate')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(contour_image)
    plt.title(f'Contours Detected: {len(contours)}')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(contour_info)
    plt.title('Bounding Rectangles')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 정보 출력
    print("=== 윤곽선 검출 결과 ===")
    print(f"총 윤곽선 개수: {len(contours)}")
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        print(f"윤곽선 {i+1}: 면적={area:.0f}, 크기=({w}×{h}), 비율={aspect_ratio:.2f}")

    return contours, contour_image

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

            gray_plate = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            enhanced_plate = cv2.adaptiveThreshold(
                gray_plate, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            thresh_adaptive, thresh_otsu = adaptive_threshold_plate(enhanced_plate)

            # 윤곽선 검출 및 시각화
            contours, contour_result = find_contours_in_plate(thresh_adaptive)

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

    print(f"\n{path} 파일: 번호판 영역을 시계 방향으로 4점 클릭하세요.")
    cv2.imshow("License Plate Extractor - Original", draw)
    cv2.setMouseCallback("License Plate Extractor - Original", onMouse)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
