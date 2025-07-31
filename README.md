# 실습
- 어핀 변환
<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/396d9c61-3347-45c7-b1b3-5196bca74ce3" />

~~~
# 어핀 변환 (getAffine.py)

import cv2
import numpy as np
from matplotlib import pyplot as plt

file_name = '../img/fish.jpg'
img = cv2.imread(file_name)
rows, cols = img.shape[:2]

# ---① 변환 전, 후 각 3개의 좌표 생성
pts1 = np.float32([[100, 50], [200, 50], [100, 200]])
pts2 = np.float32([[80, 70], [210, 60], [250, 120]])

# ---② 변환 전 좌표를 이미지에 표시
cv2.circle(img, (100,50), 5, (255,0), -1)
cv2.circle(img, (200,50), 5, (0,255,0), -1)
cv2.circle(img, (100,200), 5, (0,0,255), -1)

#---③ 짝지은 3개의 좌표로 변환 행렬 계산
mtrx = cv2.getAffineTransform(pts1, pts2)
#---④ 어핀 변환 적용
dst = cv2.warpAffine(img, mtrx, (int(cols*1.5), rows))

#---⑤ 결과 출력
cv2.imshow('origin',img)
cv2.imshow('affin', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~
- 원근 변환
~~~
# 원근 변환 (perspective.py)

import cv2
import numpy as np

file_name = "../img/fish.jpg"
img = cv2.imread(file_name)
rows, cols = img.shape[:2]

#---① 원근 변환 전 후 4개 좌표
pts1 = np.float32([[0,0], [0,rows], [cols, 0], [cols,rows]])
pts2 = np.float32([[100,50], [10,rows-50], [cols-100, 50], [cols-10,rows-50]])

#---② 변환 전 좌표를 원본 이미지에 표시
cv2.circle(img, (0,0), 10, (255,0,0), -1)
cv2.circle(img, (0,rows), 10, (0,255,0), -1)
cv2.circle(img, (cols,0), 10, (0,0,255), -1)
cv2.circle(img, (cols,rows), 10, (0,255,255), -1)

#---③ 원근 변환 행렬 계산
mtrx = cv2.getPerspectiveTransform(pts1, pts2)
#---④ 원근 변환 적용
dst = cv2.warpPerspective(img, mtrx, (cols, rows))

cv2.imshow("origin", img)
cv2.imshow('perspective', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~
- 문서 스캔 효과 내기
~~~
#마우스와 원근 변환으로 문서 스캔 효과 내기 (perspective_scan.py)

import cv2
import numpy as np

win_name = "scanning"
img = cv2.imread("../img/paper.jpg")
rows, cols = img.shape[:2]
draw = img.copy()
pts_cnt = 0
pts = np.zeros((4,2), dtype=np.float32)

def onMouse(event, x, y, flags, param):  #마우스 이벤트 콜백 함수 구현 ---① 
    global  pts_cnt                     # 마우스로 찍은 좌표의 갯수 저장
    if event == cv2.EVENT_LBUTTONDOWN:  
        cv2.circle(draw, (x,y), 10, (0,255,0), -1) # 좌표에 초록색 동그라미 표시
        cv2.imshow(win_name, draw)

        pts[pts_cnt] = [x,y]            # 마우스 좌표 저장
        pts_cnt+=1
        if pts_cnt == 4:                       # 좌표가 4개 수집됨 
            # 좌표 4개 중 상하좌우 찾기 ---② 
            sm = pts.sum(axis=1)                 # 4쌍의 좌표 각각 x+y 계산
            diff = np.diff(pts, axis = 1)       # 4쌍의 좌표 각각 x-y 계산

            topLeft = pts[np.argmin(sm)]         # x+y가 가장 값이 좌상단 좌표
            bottomRight = pts[np.argmax(sm)]     # x+y가 가장 큰 값이 우하단 좌표
            topRight = pts[np.argmin(diff)]     # x-y가 가장 작은 것이 우상단 좌표
            bottomLeft = pts[np.argmax(diff)]   # x-y가 가장 큰 값이 좌하단 좌표

            # 변환 전 4개 좌표 
            pts1 = np.float32([topLeft, topRight, bottomRight , bottomLeft])

            # 변환 후 영상에 사용할 서류의 폭과 높이 계산 ---③ 
            w1 = abs(bottomRight[0] - bottomLeft[0])    # 상단 좌우 좌표간의 거리
            w2 = abs(topRight[0] - topLeft[0])          # 하당 좌우 좌표간의 거리
            h1 = abs(topRight[1] - bottomRight[1])      # 우측 상하 좌표간의 거리
            h2 = abs(topLeft[1] - bottomLeft[1])        # 좌측 상하 좌표간의 거리
            width = int(max([w1, w2]))                       # 두 좌우 거리간의 최대값이 서류의 폭
            height = int(max([h1, h2]))                      # 두 상하 거리간의 최대값이 서류의 높이
            
            # 변환 후 4개 좌표
            pts2 = np.float32([[0,0], [width-1,0], 
                                [width-1,height-1], [0,height-1]])

            # 변환 행렬 계산 
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            # 원근 변환 적용
            result = cv2.warpPerspective(img, mtrx, (width, height))
            cv2.imshow('scanned', result)
cv2.imshow(win_name, img)
cv2.setMouseCallback(win_name, onMouse)    # 마우스 콜백 함수를 GUI 윈도우에 등록 ---④
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~
- 변환행렬 구하기
~~~
# OpenCv로 회전 변환행렬 구하기 (rotate_getmatrix.py)

import cv2

img = cv2.imread('../img/fish.jpg')
rows,cols = img.shape[0:2]

#---① 회전을 위한 변환 행렬 구하기
# 회전축:중앙, 각도:45, 배율:0.5
m45 = cv2.getRotationMatrix2D((cols/2,rows/2),45,0.5) 
# 회전축:중앙, 각도:90, 배율:1.5
m90 = cv2.getRotationMatrix2D((cols/2,rows/2),90,1.5) 

#---② 변환 행렬 적용
img45 = cv2.warpAffine(img, m45,(cols, rows))
img90 = cv2.warpAffine(img, m90,(cols, rows))

#---③ 결과 출력
cv2.imshow('origin',img)
cv2.imshow("45", img45)
cv2.imshow("90", img90)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~
- 이미지 회전
~~~
# 변환행렬을 이용한 이미지 회전 (rotate_martix.py)

import cv2
import numpy as np

img = cv2.imread('../img/fish.jpg')
rows,cols = img.shape[0:2]

# ---① 라디안 각도 계산(60진법을 호도법으로 변경)
d45 = 45.0 * np.pi / 180    # 45도
d90 = 90.0 * np.pi / 180    # 90도

# ---② 회전을 위한 변환 행렬 생성
m45 = np.float32( [[ np.cos(d45), -1* np.sin(d45), rows//2],
[np.sin(d45), np.cos(d45), -1*cols//4]])
m90 = np.float32( [[ np.cos(d90), -1* np.sin(d90), rows],
[np.sin(d90), np.cos(d90), 0]])

# ---③ 회전 변환 행렬 적용
r45 = cv2.warpAffine(img,m45,(cols,rows))
r90 = cv2.warpAffine(img,m90,(rows,cols))

# ---④ 결과 출력
cv2.imshow("origin", img)
cv2.imshow("45", r45)
cv2.imshow("90", r90)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~
- 이미지 확대 및 축소
~~~
# 행렬을 이용한 이미지 확대 및 축소 (scale_matrix.py)

import cv2
import numpy as np

img = cv2.imread('../img/fish.jpg')
height, width = img.shape[:2]

# --① 0.5배 축소 변환 행렬
m_small = np.float32([[0.5, 0, 0],
[0, 0.5,0]])  
# --② 2배 확대 변환 행렬
m_big = np.float32([[2, 0, 0],
[0, 2, 0]])  

# --③ 보간법 적용 없이 확대 축소
dst1 = cv2.warpAffine(img, m_small, (int(height*0.5), int(width*0.5)))
dst2 = cv2.warpAffine(img, m_big, (int(height*2), int(width*2)))

# --④ 보간법 적용한 확대 축소
dst3 = cv2.warpAffine(img, m_small, (int(height*0.5), int(width*0.5)), \
None, cv2.INTER_AREA)
dst4 = cv2.warpAffine(img, m_big, (int(height*2), int(width*2)), \
None, cv2.INTER_CUBIC)

# 결과 출력
cv2.imshow("original", img)
cv2.imshow("small", dst1)
cv2.imshow("big", dst2)
cv2.imshow("small INTER_AREA", dst3)
cv2.imshow("big INTER_CUBIC", dst4)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~
- 평행 이동
~~~
# 평행 이동 (translate.py)

import cv2
import numpy as np

img = cv2.imread('../img/fish.jpg')
rows,cols = img.shape[0:2]  # 영상의 크기

dx, dy = 100, 50            # 이동할 픽셀 거리

# ---① 변환 행렬 생성 
mtrx = np.float32([[1, 0, dx],
[0, 1, dy]])  
# ---② 단순 이동
dst = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy))   

# ---③ 탈락된 외곽 픽셀을 파랑색으로 보정
dst2 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None, \
cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (255,0,0) )

# ---④ 탈락된 외곽 픽셀을 원본을 반사 시켜서 보정
dst3 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None, \
cv2.INTER_LINEAR, cv2.BORDER_REFLECT)

cv2.imshow('original', img)
cv2.imshow('trans',dst)
cv2.imshow('BORDER_CONSTATNT', dst2)
cv2.imshow('BORDER_FEFLECT', dst3)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~
- 자동차 번호판 추출 실습 가이드
  1.그레이스케일 변환
  2.대비 최대화
  3.적응형 임계처리
  4.윤곽선 검출

~~~
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

#이미지 경로 리스트
image_paths = [
    '../img/car_01.jpg',
    '../img/car_02.jpg',
    '../img/car_03.jpg',
    '../img/car_04.jpg',
    '../img/car_05.jpg'
]

#전역 변수 선언
pts = np.zeros((4, 2), dtype=np.float32)
pts_cnt = 0
draw = None
img = None

#번호판 전용 적응형 임계처리 함수
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

#윤곽선 검출 함수
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

    contour_info = np.zeros((height, width, 3), dtype=np.uint8)
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(contour_info, (x, y), (x+w, y+h), colors[i % len(colors)], 1)
        cv2.putText(contour_info, f'A:{int(area)}', (x, y-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

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

    print("=== 윤곽선 검출 결과 ===")
    print(f"총 윤곽선 개수: {len(contours)}")
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        print(f"윤곽선 {i+1}: 면적={area:.0f}, 크기=({w}×{h}), 비율={aspect_ratio:.2f}")

    return contours, contour_image

#마우스 콜백 함수
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

            contours, contour_result = find_contours_in_plate(thresh_adaptive)

            cv2.imshow("License Plate Extractor - Warped (Color)", result)
            cv2.imshow("License Plate Extractor - Grayscale", gray_plate)
            cv2.imshow("License Plate Extractor - Contrast Enhanced", enhanced_plate)
            cv2.imshow("Adaptive Threshold Result", thresh_adaptive)
            cv2.imshow("Otsu Threshold Result", thresh_otsu)

#-- 여기부터 새로 추가한 후처리, 전체 파이프라인, 배치 처리 함수 --

def save_processed_results(plate_name, gray_plate, enhanced_plate, thresh_plate, contour_result):
    """처리된 번호판 이미지들을 체계적으로 저장"""
    save_dir = '../processed_plates'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cv2.imwrite(f'{save_dir}/{plate_name}_1_gray.png', gray_plate)
    cv2.imwrite(f'{save_dir}/{plate_name}_2_enhanced.png', enhanced_plate)  
    cv2.imwrite(f'{save_dir}/{plate_name}_3_threshold.png', thresh_plate)
    cv2.imwrite(f'{save_dir}/{plate_name}_4_contours.png', contour_result)
    
    print(f"처리 결과 저장 완료: {save_dir}/{plate_name}_*.png")

def process_extracted_plate(plate_name):
    """추출된 번호판의 완전한 처리 파이프라인"""
    print(f"=== {plate_name} 처리 시작 ===")
    
    plate_img = load_extracted_plate(plate_name)
    if plate_img is None:
        print(f"이미지 로드 실패: {plate_name}")
        return None
    
    gray_plate = convert_to_grayscale(plate_img)
    enhanced_plate = maximize_contrast(gray_plate)
    thresh_plate, _ = adaptive_threshold_plate(enhanced_plate)
    contours, contour_result = find_contours_in_plate(thresh_plate)
    save_processed_results(plate_name, gray_plate, enhanced_plate, thresh_plate, contour_result)
    potential_chars = prepare_for_next_step(contours, thresh_plate)
    
    print(f"처리 완료 - 검출된 윤곽선: {len(contours)}개, 잠재적 글자: {potential_chars}개")
    
    return {
        'original': plate_img,
        'gray': gray_plate, 
        'enhanced': enhanced_plate,
        'threshold': thresh_plate,
        'contours': len(contours),
        'potential_chars': potential_chars,
        'contour_result': contour_result
    }

def batch_process_plates():
    """extracted_plates 폴더의 모든 번호판 처리"""
    plate_dir = '../extracted_plates'
    if not os.path.exists(plate_dir):
        print(f"폴더를 찾을 수 없습니다: {plate_dir}")
        return {}
        
    plate_files = [f for f in os.listdir(plate_dir) if f.endswith('.png')]
    
    if len(plate_files) == 0:
        print("처리할 번호판 이미지가 없습니다.")
        return {}
    
    results = {}
    for plate_file in plate_files:
        plate_name = plate_file.replace('.png', '')
        result = process_extracted_plate(plate_name)
        if result:
            results[plate_name] = result
    
    print(f"\n=== 전체 처리 완료: {len(results)}개 번호판 ===")
    return results

#이미지 하나씩 반복
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

~~~
<img width="168" height="127" alt="image" src="https://github.com/user-attachments/assets/a571a3a7-33ff-4e31-941e-0560c388e460" />
<img width="170" height="120" alt="image" src="https://github.com/user-attachments/assets/e7aec2ce-a684-4e0e-8bb7-89a681c2c920" />
<img width="1595" height="459" alt="image" src="https://github.com/user-attachments/assets/6317d6b6-1bf3-4619-ba94-589b60a992f3" />
<img width="1489" height="559" alt="image" src="https://github.com/user-attachments/assets/90e25267-fc76-4ed2-8dde-db94adfa8a66" />


