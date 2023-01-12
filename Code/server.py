import asyncio 
import websockets
from matplotlib import pyplot as plt 
from SignLanguageModel import SignLanguage
from NaverDictionaryParsing import NaverDictionaryParsing
import warnings
import numpy as np
import base64, cv2

warnings.simplefilter("ignore", DeprecationWarning)

IP = "localhost"
PORT = 3000

# call back for websockets.serve(accept,
class Server():
    def __init__(self):
        plt.figure(figsize=(18,18))
        self.sl = SignLanguage()
        self.nd = NaverDictionaryParsing()
        self.Kor_actions = np.array([i[0] for i in self.sl.actions])
        self.Eng_actions = np.array([i[1] for i in self.sl.actions])
        self.url = np.array([i[2] for i in self.sl.actions])
        self.sequence = []
        self.sentence = []
        self.predictions = []
        self.threshold = 0.8
        
    async def sendMessage(self, websocket, data):
        x = data.split('#')
        
        if x[0] != 'x':
            print("송신 : ", data)
            await websocket.send(data); # send received data
        
    async def accept(self, websocket): 
        print("client connected")
        
        async def sendWordThumbnail():
            self.sentence.append(self.Eng_actions[np.argmax(res)])
            thumnail = self.nd.getThumnail(self.Kor_actions[np.argmax(res)], self.url[np.argmax(res)])
            await self.sendMessage(websocket, f"{self.Kor_actions[np.argmax(res)]}#{thumnail}")
            
        with self.sl.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while True:
                msg = await websocket.recv()
                img = cv2.imdecode(np.fromstring(base64.b64decode(msg.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
                
                # x flip
                img = cv2.flip(img, 1)
            
                # Make detections
                image, results = self.sl.mediapipe_detection(img, holistic)
                
                # Draw landmarks
                self.sl.draw_styled_landmarks(image, results)
                
                # Prediction logic
                keypoints = self.sl.extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-15:]
                
                if len(self.sequence) == 15:
                    # predict
                    res = self.sl.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                    
                    # print predictEng_actions
                    print(self.Eng_actions[np.argmax(res)])
                    
                    self.predictions.append(np.argmax(res))
                                        
                    #3. add to word list
                    if self.predictions[-7:].count(self.predictions[-1]) >= 7:
                        if res[np.argmax(res)] > self.threshold:
                            if len(self.sentence) > 0: 
                                if self.Eng_actions[np.argmax(res)] != self.sentence[-1]:
                                    await sendWordThumbnail()

                            else:
                                await sendWordThumbnail()
                                
                    
                    # print total five words
                    if len(self.sentence) > 5:
                        self.sentence = self.sentence[-5:]

                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(self.sentence), (3,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                                
                # Show to screen
                cv2.imshow('OpenCV Feed', image)
            cv2.destroyAllWindows()
                
            

sv = Server()
# websocket server creation
websoc_svr = websockets.serve(sv.accept ,IP ,PORT)
print("Ready")

# waiting 
asyncio.get_event_loop().run_until_complete(websoc_svr)
asyncio.get_event_loop().run_forever()