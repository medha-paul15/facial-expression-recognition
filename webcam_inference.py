import cv2 as cv
import numpy as np
import tensorflow as tf
import mediapipe as mp

model=tf.keras.models.load_model('best_phase3.keras')
class_names=['angry','disgust','fear','happy','neutral','sad','surprise']
face_cascade=cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap=cv.VideoCapture(0)

while True:
    ret, frame=cap.read()
    if not ret:
        break

    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(48,48))

    for(x,y,w,h) in faces:
        face=frame[y:y+h,x:x+w]
        face_resized=cv.resize(face,(96,96)) #tensorflow expects images of shape (96,96)
        face_array=tf.keras.utils.img_to_array(face_resized) #converts image to a format compatible w tf pipeline
        face_array = tf.expand_dims(face_array, 0) #expands or adds batch dimension to array, shape: [batch_size, height, width, channels]), so now (96,96,3)->(1,96,96,3)

        predictions=model.predict(face_array,verbose=0)
        #emotion=class_names[np.argmax(predictions)]
        confidence=np.max(predictions)

        cv.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2) #to put a rectangle around the face detected

        if confidence>0.4:
            emotion=class_names[np.argmax(predictions)]
            cv.putText(frame,f"{emotion} ({confidence:.0%})", (x, y-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv.putText(frame,"uncertain",(x, y-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)   
 
    cv.imshow('FER', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break    

cap.release()
cv.destroyAllWindows()
