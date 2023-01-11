import csv
import cv2
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
from tensorflow import keras
from SignLanguageModel import SignLanguage


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

model = keras.models.load_model('Model/model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

sl = SignLanguage()

# Load CSV File
sign = []
with open('Data\\SignList.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        sign.append(row[1])
        
actions = np.array(sign)


plt.figure(figsize=(18,18))
# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.8

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        
        # Make detections
        image, results = sl.mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        sl.draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = sl.extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-15:]
        
        if len(sequence) == 15:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
        # #3. Viz logic      
            if predictions[-10:].count(predictions[-1]) >= 10:
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # # Viz probabilities
            # image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()