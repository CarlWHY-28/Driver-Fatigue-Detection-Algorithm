import dlib
import cv2
import numpy as np
import os


# 加载预训练的人脸检测器和特征点预测器
def LoadModel():
    
    detector = dlib.get_frontal_face_detector()
    file_path = "models/shape_predictor_68_face_landmarks.dat"
    #看看找不找得到文件
    
    
    if  not os.path.exists(file_path):
        print("找不到文件")
    #打印当前运行路径
    print(os.getcwd())
    predictor = dlib.shape_predictor(file_path)
    return detector, predictor


def detect_eye_landmarks(frame, detector, predictor):
    # 转换为灰度图像，用于人脸检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = detector(gray, 1)

    # 初始化关键点列表
    landmarks_list = []

    # 对于检测到的每个人脸
    for i, face_rect in enumerate(faces):
        # 获取特征点
        landmarks = predictor(gray, face_rect)

        # 提取37-48号关键点（眼睛周围的点）
        eye_landmarks = [landmarks.part(n) for n in range(36, 48)]
        #提取61-68号关键点（嘴巴周围的点）
        mouth_landmarks = [landmarks.part(n) for n in range(60, 68)]
        

        # 将关键点坐标添加到列表
        landmarks_list.append([landmark.x, landmark.y] for landmark in eye_landmarks)
        landmarks_list.append([landmark.x, landmark.y] for landmark in mouth_landmarks)

    return landmarks_list


def EAR(eye_landmarks):
    # calculate eye width and height
    #print(eye_landmarks)
    eye_width = np.linalg.norm(abs(eye_landmarks[0][0] - eye_landmarks[3][0]))
    eye_height = np.linalg.norm(abs(
        eye_landmarks[1][1] - eye_landmarks[5][1])+abs(eye_landmarks[2][1] - eye_landmarks[4][1]))

    # 计算眼睛的纵横比
    ear = eye_height / (eye_width)
    return ear
# (|2-8|+|3-7|+|4-6|)/3*|1-5|


def MAR(mouth_landmarks):
    # 计算嘴巴的高度
    # (|2-8|+|3-7|+|4-6|)/3*|1-5|
    mouth_height = np.linalg.norm(abs(mouth_landmarks[1][1] - mouth_landmarks[7][1])+abs(
        mouth_landmarks[2][1] - mouth_landmarks[6][1])+abs(mouth_landmarks[3][1] - mouth_landmarks[5][1]))
    # 计算嘴巴的宽度
    mouth_width = np.linalg.norm(
        abs(mouth_landmarks[0][0] - mouth_landmarks[4][0]))
    mar = mouth_height / (2 * mouth_width)
    return mar

# 示例：使用接口
def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    c=0
    detector, predictor = LoadModel()

    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头")
            break

        # 使用接口检测关键点
        eye_landmarks = detect_eye_landmarks(frame, detector, predictor)

        # 在图像上绘制特征点（可选）
        for landmarks in eye_landmarks:
            for x, y in landmarks:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

        # 显示结果
        cv2.imshow("Eye Landmarks", frame)
        #print(eye_landmarks)
        print(c)
        c+=1    
        # 按'q'退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()
from ultralytics import YOLO
if __name__ == "__main__":
    #main()
    face_model = YOLO("my_eye_best.pt")
    cap=cv2.VideoCapture(0)
    while True:
        frame= cap.read()[1]
        frame = cv2.resize(frame, (640, 480))
        results = face_model.predict(source=frame, save=False)
        if results[0].keypoints==None:
            continue
        print(results[0].keypoints)


    
    
