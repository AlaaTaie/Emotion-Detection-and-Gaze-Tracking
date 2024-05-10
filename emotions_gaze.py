import cv2
import mediapipe as mp
import numpy as np
from keras.models import model_from_json
import csv

emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# Load emotion detection model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.h5")

# Initialize MediaPipe Face Detection
mp_face_mesh=mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

maxindex = 0
cap = cv2.VideoCapture(0)
filename = 'subject_1.csv'
first_row = True


with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection, \
        mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ct = 0
        expression_list = []
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image=cv2.cvtColor(cv2.flip(frame,1),cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        results=face_mesh.process(image)
        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        h,w,c=image.shape
        face_3d=[]
        face_2d=[]
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx,lm in enumerate(face_landmarks.landmark):
                    if idx==33 or idx ==263 or idx==1 or idx== 61 or idx== 291 or idx== 199:
                        if idx==1:
                            nose_2d=(lm.x*w,lm.y*h)
                            nose_3d=(lm.x*w,lm.y*h,lm.z*3000)
                        x,y=int(lm.x*w),int(lm.y*h)
                        face_2d.append([x,y])
                        face_3d.append([x,y,lm.z])
                face_2d=np.array(face_2d,dtype=np.float64)
                face_3d=np.array(face_3d,dtype=np.float64)
                focal_length=1*w
                cam_matrix=np.array([[focal_length,0,h/2],
                                    [0,focal_length,w/2],
                                    [0,0,1]])
                dist_matrix=np.zeros((4, 1), dtype=np.float64)
                success,rot_vec,trans_vec=cv2.solvePnP(face_3d,face_2d,cam_matrix,dist_matrix)
                rmat,jac=cv2.Rodrigues(rot_vec)
                angels, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                x=angels[0]*360
                y=angels[1]*360
                z=angels[2]*360

                if y<-10:
                    text="Looking Left"
                elif y>10:
                    text="Looking Right"
                elif x<-10:
                    text="Looking Down"
                elif x>10:
                    text="Looking Up"
                else:
                    text="Forward"
                
                nose_3d_projection,jacobian=cv2.projectPoints(nose_3d,rot_vec,trans_vec,cam_matrix,dist_matrix)

                p1=(int(nose_2d[0]),int(nose_2d[1]))
                p2=(int(nose_2d[0]+y*10),int(nose_2d[1]-x*10))
                cv2.line(image,p1,p2,(255,0,0),3)

                cv2.putText(image,text,(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
                cv2.putText(image,"x"+str(np.round(x,2)),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                cv2.putText(image,"y"+str(np.round(y,2)),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                cv2.putText(image,"z"+str(np.round(z,2)),(500,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.7).process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections is not None:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, c = image.shape
                x, y, width, height = int(
                    bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                face_image = image[y:y + height, x:x + width]
                if not face_image.size == 0:
                    # Perform emotion detection on the face image
                    gray_frame = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                    face_roi_gray_frame = cv2.resize(gray_frame, (48, 48))
                    cropped_img = np.expand_dims(
                        np.expand_dims(face_roi_gray_frame, -1), 0)

                    # Predict emotions
                    emotion_prediction = emotion_model.predict(cropped_img)
                    maxindex = int(np.argmax(emotion_prediction))
                    expression_list.append(maxindex)
                    if maxindex == 3 or maxindex == 4 or maxindex == 6:
                        ct += 1
                    cv2.putText(image, "face expression: " + str(emotion_dict[maxindex]), (5, 400),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Display the frame in full screen
        cv2.namedWindow('Emotion, Face, and Pose Detection', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Emotion, Face, and Pose Detection',
                              cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Emotion, Face, and Pose Detection', image_bgr)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()