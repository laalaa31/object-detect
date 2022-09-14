# 정면 얼굴 인식, 인식된 얼굴의 실세계 좌표 계산(cm단위)

import cv2

"""
원본 동영상 크기 정보 640 픽셀*480픽셀 4:3
카메라에서 52cm 떨어진 상태 가정
"""
# 화면 상 비디오 넓이 및 높이(픽셀단위)
VIDEO_WIDTH_PIXEL=640
VIDEO_HIGHT_PIXEL=480

# 화면 상 비디오 넓이 및 높이(cm단위)
VIDEO_WIDTH=14.5
VIDEO_HIGHT=10.5

# 52cm 떨어진 곳의 실세계 넓이 및 높이
REAL_WIDTH=52
REAL_HIGHT=43

#px to cm
PX_TO_CM=0.02645

# 색상 및 폰트
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX

# 비디오 캡쳐
cap = cv2.VideoCapture(0)

# 얼굴 탐지
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

# 화면의 중점을 (0,0)으로 만드는 함수
def change_midpoint(y,z):
    y=y-VIDEO_WIDTH_PIXEL/2
    z=VIDEO_HIGHT_PIXEL/2-z
    
    return (y,z)

# 탐지된 얼굴 표시
def face_data(image):
    is_exist=0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), WHITE, 1)
        is_exist=h
    
    return is_exist

# 탐지된 얼굴의 중점 좌표 실세계 cm로 반환
def coordinate_finder(image):
    yc=0
    zc=0
    y_result=0
    z_result=0
    """
    중점= (x+w/2, y+h/2)
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

    for (x, y, h, w) in faces:
        #화면상의 좌표 그대로 출력
        yc=x+w/2
        zc=y+h/2

        #기준점(원점)을 화면의 중점으로 변환
        yc,zc=change_midpoint(yc,zc)

        #실세계 좌표로 변환
        yc=yc*(REAL_WIDTH/VIDEO_WIDTH)
        zc=zc*(REAL_HIGHT/VIDEO_HIGHT)

        #픽셀->cm 변환
        y_result=yc*PX_TO_CM
        z_result=zc*PX_TO_CM


    return (round(y_result,2),round(z_result,2))

while True:
    _, frame = cap.read()

    # 좌표 탐색 함수 호출
    is_exist=face_data(frame)
    coordinate=coordinate_finder(frame)
    
    # 화면에 좌표 및 바운딩 박스 표시
    if is_exist!=0 and coordinate != 0:
        cv2.putText(
            frame, f"Coord=({coordinate[0]}, {coordinate[1]})cm", (50,50), fonts, 1, (GREEN), 2
        )
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()