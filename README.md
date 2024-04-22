**目录**

- `Research thoughts`
- `Methods`



# Driver Fatigue Detection Algorithm Intro

`Driver Fatigue Detection Algorithm` (Graduate Project), 'DFDA' for short, is an algorithm to evaluate driver status, in which `deep learning` algorithm, as a means of feature point detection, is the cornerstone of subsequent analysis, YOLOv8-pose is adopted as a pose estimation method, and facial keypoint detection is realized by using functions in Dlib.
A prototype system based on the algorithm is still under development, but the basic functions of the algorithm have been implemented and can be demonstrated. At present, the efficiency of the algorithm is 7 fps on m3 chip and 6 fps on Nvidia 2070s.

Please start with the `my_pre.py`.

## Research thoughts
**Preliminary work**
+ Read paper
    + In the past two years, few studies have realized fatigue state analysis based on facial key points and pose estimation key points. Therefore, I mainly refer to the feasibility of the algorithm implementation.
+ Compare different algorithms
    + I mainly compared AlphaPose, OpenPose and YOLOv8-pose algorithms. The main reason for the victory of YOLOv8-pose is that it is too convenient on the basis of high efficiency and accuracy.

**Feature engineering**
+ Pose keypoint
    + The order of key points in yolov8 starting from 0 is (the following points are not written because they are not used) : “nose”,“left_eye”, “right_eye”,“left_ear”, “right_ear”,“left_shoulder”, “right_shoulder”,“left_elbow”,  “right_elbow”, etc. However, only the first 7 points can be always detected in this case.
+ Face keypoint
    + Extracting 6 points of each eye and 8 point of mouth, 20 in total
 
**Algorithm development**
+ Basic idea
    + The algorithm needs to take into account some basic physiological activities of people (blinking, speaking, etc.), and at the same time, everyone has different blinking habits, so each frame cannot be directly used as the basis for analysis, and the algorithm also needs to adapt to the particularity of individuals. 
    + The algorithm used the `sliding window` to record the normal condition of the driver's eyes and the situation of nearly 10 frames, and distinguished the driver's state based on the mean `value and standard deviation`.The opening of the mouth is also analyzed.
    + For the analysis of posture, I calculated the standard deviation of the first 7 points of the human body (mentioned above) in the process of algorithm design (before establishing rules). Under the resolution of 640*480, the `standard deviation` of the coordinate of the human nose image is usually more than 30 (the last 20 frames), and the `standard deviation` of the shoulder coordinate is more than 20.


## Methods

**Preprocessing**

+ Model Loading
    + Face predictor
        + Using the Dlib open source model
    + Pose predictor
        + Using model trained on the coco dataset
    
+ Feature extraction
    + Face point detection
    + Pose estimation

**Initialization**
+ When initialization is not complete, fill the const stack with data. After that, the detection would start.
``` python
# examples
    # Initialize stacks to store the state of eyes and mouth, keeping the earliest 50 frames
    con_eye_stack = []
    con_mouth_stack = []

    # Initialize stacks for a moving window to save the most recent 10 frames of eye and mouth states
    alarm_eye_stack = []
    alarm_mouth_stack = []
```

**Evaluation algorithm**

+ Face analysis
    + The analysis is based on EAR(Eye aspect ratio) and MAR(Mouth aspect ratio). The algorithm analyzes the state of the eyes and mouth, including the degree to which the eyes are open (EAR) and the mouth is open (MAR), to detect if the driver is fatigued or distracted.
``` python
# Examples
# Ear
    def EAR(eye_landmarks):
    #Compute ear
    eye_width = np.linalg.norm(abs(eye_landmarks[0][0] - eye_landmarks[3][0]))
    eye_height = np.linalg.norm(abs(
        eye_landmarks[1][1] - eye_landmarks[5][1])+abs(eye_landmarks[2][1] - eye_landmarks[4][1]))
    ear = eye_height / (eye_width)
    return ear
    
# Analysis
    if ear > eye_mean + 2 * eye_std:  # Abnormal motion state
    # The result is what the function returns
    result[0] = False
    result[1] = np.mean(alarm_eye_stack)
    result[2] = eye_mean
    result[3] = np.mean(alarm_mouth_stack)
    result[4] = 'eye acting abnormal'
else:

    if np.mean(alarm_eye_stack) < np.mean(con_eye_stack) - 0.1:# When the mean is too low
        print("normal ear:", np.mean(con_eye_stack))# driver normal ear
        print("eye closed\n\n")
        result[0] = False
        result[1] = np.mean(alarm_eye_stack)
        result[2] = eye_mean
        result[3] = np.mean(alarm_mouth_stack)
        result[4] = 'eye closed '
```


+ Pose evaluation
    + Rule setting
        + Based on the previous analysis, three rules are laid down.
            + The nose can't stay still for long. In the code, the standard deviation of the nose cannot be too small.
            + The shoulders can't both stay still for long(considering someone is driving with one hand). 
            + The nose should not be too low. The vertical distance from nose to shoulders should not be less than nose to eyes.
    + Driver evaluation
        + When two of the three rules are broken, the driver is judged to be tired.
    + Threshold Settings 
    ``` python
    #examples
    # Define thresholds for different activities
        thresholds = {
            'head_stability_threshold': 10,  # If the nose moves less than 10 pixels, it is considered stable
            'arm_activity_threshold': 50,  # If the arm moves less than 50 pixels, it is considered stable
            'mar_threshold': 0.6,  # If the mouth aspect ratio is greater than 0.6, it is considered yawning
        } ```

## Result Analysis
+ Algorithm accuracy
    + The algorithm is close to 90% accurate in the analysis function (based on my own labeled videos).
    + In the detection function, the accuracy of facial key point detection is not high, around 70%.
+ Algorithm efficiency
    + The current efficiency is 7 fps, considering that my computer has an efficiency of 17 fps using YOLOv8-pose alone, I don't think the time complexity of the algorithm is very high
