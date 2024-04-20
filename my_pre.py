from ultralytics import YOLO
from PIL import Image
import cv2
import os
import random
import time
import numpy as np

from face_point_detect import LoadModel, detect_eye_landmarks, EAR, MAR
# Load model
model = YOLO("best.pt")
pose_model = YOLO("new_best.pt")
f_detector, f_predictor = LoadModel()

# Define the order of keypoints detected by YOLOv8：“nose”,“left_eye”, “right_eye”,“left_ear”, “right_ear”,
# “left_shoulder”, “right_shoulder”,“left_elbow”, “right_elbow”,“left_wrist”, “right_wrist”,“left_hip”, “right_hip”
connections = [
    (0, 1), (0, 2), (1, 3), (2, 4), (1, 2),  # from nose to eyes, between eyes
    (3, 5), (4, 6), (5, 6), (7, 5),  # from ears to shoulders, from shoulder to shoulder, from nose to shoulders
    (7, 9), (6, 8), (10, 8), (6, 12), (5, 11),  # 左肩到左肘，左肘到左腕，右肩到右肘，右肘到右腕,
    (11, 12), (13, 11), (13, 15), (14, 12), (16, 14),  # 左腕到左髋，左髋到左膝，左膝到左踝，右腕到右髋，右髋到右膝，右膝到右踝
]
# Initialize stacks to store the state of eyes and mouth, keeping the earliest 50 frames
con_eye_stack = []
con_mouth_stack = []
con_flag = False   # Flag to initialize the state

# Initialize stacks for a moving window to save the most recent 10 frames of eye and mouth states
alarm_eye_stack = []
alarm_mouth_stack = []

# Define thresholds for different activities
thresholds = {
    'head_stability_threshold': 10,  # If the nose moves less than 10 pixels, it is considered stable
    'arm_activity_threshold': 50,  # If the arm moves less than 50 pixels, it is considered stable
    'mar_threshold': 0.6,  # If the mouth aspect ratio is greater than 0.6, it is considered yawning
}

#Initialize stacks for a moving window to save the most recent 20 frames of head and arm poses
alarm_head_stack = []
alarm_l_shoulder_stack = []
alarm_r_shoulder_stack = []

# Initialize variables to store the mean and standard deviation of eye aspect ratio (EAR)
eye_mean = -1
eye_std = -1






# Define a function to evaluate the driver's status based on an array of status
def evaluate_driver(status_arry):
    # Check if more than half of the statuses indicate abnormal behavior
    return sum(status_arry) > len(status_arry) / 2


# 钓鱼检测
# Define a function to detect 'fishing' behavior (leaning forward) using facial landmarks
def fishing_detect(landmarks):
    x0, y0 = landmarks[0][0].item(), landmarks[0][1].item()  # 鼻子
    x1, y1 = landmarks[2][0].item(), landmarks[2][1].item()  # 左眼
    x2, y2 = landmarks[1][0].item(), landmarks[1][1].item()  # 右眼
    x5, y5 = landmarks[5][0].item(), landmarks[5][1].item()  # 左肩
    x6, y6 = landmarks[6][0].item(), landmarks[6][1].item()  # 右肩
    if abs(y1 - y0) + abs(y2 - y0) > (abs(y1 - y5) + abs(y2 - y6)) / 2:
        print('钓鱼!!!')#fishing!!!
        return True
    return False

# Define a function to calculate head activity score based on the movement of the head landmarks
def calculate_head_activity_score(landmarks):
    global alarm_head_stack
    if len(landmarks) == 0:
        return False

    landmarks = landmarks[0]
    if len(alarm_head_stack) == 0:
        alarm_head_stack.append(landmarks)
    if len(alarm_head_stack) < 20:

        x1, y1 = landmarks.numpy()
        x2, y2 = alarm_head_stack[-1].numpy()
        # 计算移动距离
        dis = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # print(dis)
        if dis < thresholds['head_stability_threshold']:#Check if the head movement exceeds the threshold for stability
            alarm_head_stack.append(landmarks)
            if len(alarm_head_stack) == 20:
                alarm_head_stack.pop()  # 使得alarm_head_stack动态窗口只有5个
                print('鼻子不动')
                return True
        else:
            alarm_head_stack = []
    return False


#  Define a function to calculate arm activity score based on the movement of the arm landmarks
def calculate_arm_activity_score(landmarks):

    global alarm_r_shoulder_stack
    global alarm_l_shoulder_stack

    if len(landmarks) == 0:
        return False
    landmarks = landmarks[5:9]
    if len(alarm_l_shoulder_stack) == 0:
        alarm_l_shoulder_stack.append([landmarks[0][0].item(), landmarks[0][1].item()])
        alarm_r_shoulder_stack.append([landmarks[1][0].item(), landmarks[1][1].item()])
        return False

    if len(alarm_l_shoulder_stack) < 20:
        x1, y1 = landmarks[0][0].item(), landmarks[0][1].item()
        x2, y2 = alarm_l_shoulder_stack[-1]
        x3, y3 = landmarks[1][0].item(), landmarks[1][1].item()
        x4, y4 = alarm_r_shoulder_stack[-1]
        left_dis = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        right_dis = np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)
        if left_dis < thresholds['arm_activity_threshold'] and right_dis < thresholds['arm_activity_threshold']:
            alarm_l_shoulder_stack.append([landmarks[0][0].item(), landmarks[0][1].item()])
            alarm_r_shoulder_stack.append([landmarks[1][0].item(), landmarks[1][1].item()])
            if len(alarm_l_shoulder_stack) == 20:
                print('手臂不动')
                alarm_l_shoulder_stack.pop()
                alarm_r_shoulder_stack.pop()
                return True


        else:
            alarm_l_shoulder_stack = []
            alarm_r_shoulder_stack = []
    return False



# Define a function to analyze the state of the driver's eyes and mouth
def eye_and_mouth_analyse(ear, mar):
    global eye_std
    global eye_mean
    global con_flag
    global con_eye_stack
    global con_mouth_stack
    global alarm_eye_stack
    global alarm_mouth_stack
    global alarm_pose_stack

    result = [True, -1, -1, -1, '']  # 司机正常驾驶、司机窗口ear、司机阈值ear、司机窗口mar、司机状态
    if len(con_eye_stack) < 50:  # 初始化
        con_eye_stack.append(ear)
        con_mouth_stack.append(mar)
    else:
        if not con_flag:
            # 计算标准差
            eye_std = np.std(con_eye_stack)
            eye_mean = np.mean(con_eye_stack)

            con_flag = True
            print('初始化完成\n\n')
            print('eye_mean:', eye_mean)
            print('eye_std:', eye_std)

    if len(alarm_eye_stack) < 10:
        alarm_eye_stack.append(ear)
        alarm_mouth_stack.append(mar)
    else:
        alarm_eye_stack.pop(0)
        alarm_eye_stack.append(ear)
        alarm_mouth_stack.pop(0)
        alarm_mouth_stack.append(mar)

        if con_flag:

            if ear > eye_mean + 2 * eye_std:  # over 2 times of standard deviation, abnormal
                print("眼睛异常\n\n")
                result[0] = False
                result[1] = np.mean(alarm_eye_stack)
                result[2] = eye_mean
                result[3] = np.mean(alarm_mouth_stack)
                result[4] = 'eye acting abnormal'
            else:
                print("数据正常\n")
                print("当前窗口ear:", np.mean(alarm_eye_stack))
                if np.mean(alarm_eye_stack) < np.mean(con_eye_stack) - 0.1:
                    print("司机正常ear:", np.mean(con_eye_stack))# driver normal ear
                    print("司机眯眼\n\n")# eye closed
                    result[0] = False
                    result[1] = np.mean(alarm_eye_stack)
                    result[2] = eye_mean
                    result[3] = np.mean(alarm_mouth_stack)
                    result[4] = 'eye closed '
                else:
                    print("司机正常ear:", np.mean(con_eye_stack))
                    print("司机正常\n\n")
                    result[0] = True
                    result[1] = np.mean(alarm_eye_stack)
                    result[2] = eye_mean
                    result[3] = np.mean(alarm_mouth_stack)
                    result[4] = ''
            if np.mean(alarm_mouth_stack) > thresholds['mar_threshold']:
                print("当前mar:", np.mean(alarm_mouth_stack))
                print("司机打哈欠\n\n")
                result[0] = False
                result[3] = np.mean(alarm_mouth_stack)
                result[4] += 'yawning'
            return result

# Define a function to detect facial landmarks and return their coordinates
def face_point_detect(f_detector, f_predictor, org_frame):
    gray = cv2.cvtColor(org_frame, cv2.COLOR_BGR2GRAY)
    faces = f_detector(gray, 1)
    landmarks_list = []
    new_list = []

    for i, face_rect in enumerate(faces):
        # predict facial landmarks
        landmarks = f_predictor(gray, face_rect)

        # extract the eye and mouth landmarks
        eye_landmarks = [landmarks.part(n) for n in range(36, 48)]
        mouth_landmarks = [landmarks.part(n) for n in range(60, 68)]

        landmarks_list += (eye_landmarks)
        landmarks_list += (mouth_landmarks)

        new_list = []
        for landmark in landmarks_list:
            new_list.append([int(landmark.x), int(landmark.y)])

    return new_list

# Define a function to draw the pose keypoints on an image
def draw_pose(landmarks, image):
    # draw points
    for i, landmark in enumerate(landmarks):
        if landmark[0] != 0 and landmark[1] != 0:  # filter out invalid points
            cv2.circle(image, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 0), -1)

    if len(landmarks) == 0:
        return
    # draw connections
    for (a, b) in connections:
        # print(landmarks[a],'   ', landmarks[b])
        if landmarks[a][0] != 0 and landmarks[a][1] != 0 and landmarks[b][0] != 0 and landmarks[b][
            1] != 0:  # 确保两个端点都不是无效点
            cv2.line(image, (int(landmarks[a][0]), int(landmarks[a][1])), (int(landmarks[b][0]), int(landmarks[b][1])),
                     (0, 255, 0), 2)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    frame_count = 0
    start_time = time.time()
    frame_start_time = start_time

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))

        # show the original frame
        cv2.imshow('org_frame', frame)
        if not ret:
            break

        # detect face points
        face_point_list = face_point_detect(f_detector, f_predictor, frame)
        if not len(face_point_list) == 0:
            # print(face_point_list)
            mar = MAR(face_point_list[12:20])
            ear = (EAR(face_point_list[0:6]) + EAR(face_point_list[6:12])) / 2
            drow_result = eye_and_mouth_analyse(ear, mar)
            for x, y in face_point_list:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
            if con_flag:
                text = 'ear:' + str(drow_result[1].__format__('.3f')) + ' mar:' + str(
                    drow_result[3].__format__('.3f')) + ' ' + drow_result[4]
                cv2.putText(frame, text, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        pose_results = pose_model.predict(source=frame, save=False)

        # evaluate the driver's status
        if not len(pose_results[0].keypoints.cpu().xy[0]) == 0:
            xy_list = pose_results[0].keypoints.cpu().xy[0]

            draw_pose(xy_list, frame)
            # evaluate the driver's status based on the pose keypoints
            eva_result = evaluate_driver(
                [calculate_head_activity_score(xy_list), calculate_arm_activity_score(xy_list),
                 fishing_detect(xy_list)])

            if eva_result:
                cv2.putText(frame, 'Driver abnormal', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame_count += 1
        current_time = time.time()
        if current_time - frame_start_time >= 1:
            fps = frame_count / (current_time - frame_start_time)
        else:
            fps = 0
        fps = fps.__format__('.2f')

        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == 27:
            break
