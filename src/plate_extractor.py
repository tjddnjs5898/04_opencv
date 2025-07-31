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

# 이미지 하나씩 반복
for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"이미지 로드 실패: {path}")
        continue

    # 전역 변수 초기화
    pts = np.zeros((4, 2), dtype=np.float32)
    pts_cnt = 0
    draw = img.copy()

    # 마우스 콜백 함수 정의
    def onMouse(event, x, y, flags, param):
        global pts_cnt, pts, draw

        if event == cv2.EVENT_LBUTTONDOWN and pts_cnt < 4:
            # 좌표 저장 및 표시
            pts[pts_cnt] = [x, y]
            pts_cnt += 1

            cv2.circle(draw, (x, y), 5, (0, 255, 255), -1)  # 노란색 점
            cv2.putText(draw, str(pts_cnt), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("License Plate Extractor - Original", draw)

            # 4점 클릭 완료 시 원근 변환
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
                cv2.imshow("License Plate Extractor - Warped", result)

    print(f"{path} 파일: 번호판 영역을 시계 방향으로 4점 클릭하세요.")
    cv2.imshow("License Plate Extractor - Original", draw)
    cv2.setMouseCallback("License Plate Extractor - Original", onMouse)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
