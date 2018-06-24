from keras.models import load_model
import pickle
import cv2 
import numpy as np
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
from sklearn.linear_model import LogisticRegression
model = load_model("modelxc.hdf5")
model.load_weights("weights_mini_xception.62-0.65.hdf5")
model2 = load_model("model2.hdf5")
with open("regr", "rb") as f:
    regr = pickle.loads(f.read())


cap = cv2.VideoCapture(0)
i = 0
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

while True:
    
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
       
        # prediction1 = model.predict(cropped_img/255)
        prediction2 = model.predict(cropped_img/255)
        # both = np.hstack([prediction2, prediction1])
        # final_prediction = regr.predict_proba(both)
        cv2.putText(frame, emotion_dict[int(np.argmax(prediction2))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('window', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()