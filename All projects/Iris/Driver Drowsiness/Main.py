from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2

# The algorithm
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	val = (A + B) / (2.0 * C)
	return val

# Threshold value below which we alert
threshold = 0.25
# Number of Frames to observe
frame_check = 50
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


cap=cv2.VideoCapture(0)
flag=0
for each in subjects:
	shape = predict(gray, subject)
	shape = face_utils.shape_to_np(shape)#converting to NumPy Array
    leftEye = shape[lStart:lEnd]
	rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
	rightEAR = eye_aspect_ratio(rightEye)
	val = (leftEAR + rightEAR) / 2.0
    leftEyeHull = cv2.convexHull(leftEye)
	rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
	cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
	if val < thresh:
		flag += 1
		#print (flag)
		if flag >= frame_check:
			cv2.putText(frame, "Closed Eye", (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "Closed Eye", (10,325),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	else:
		flag = 0
cv2.imshow("Frame", frame)
key = cv2.waitKey(1) & 0xFF





































