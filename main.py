import numpy as np
import cv2
from keras.preprocessing import image

from threading import Timer
from functools import partial

from mysql.connector import MySQLConnection, Error
import mysql.connector
from mysql.connector import Error

conn = mysql.connector.connect(host='localhost',database='test',user='root',password='password')
if conn.is_connected():
    print('Connected to MySQL database')

face_cascade = cv2.CascadeClassifier('/home/ubantu/Desktop/FER/dataset/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
#-----------------------------
#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open("/home/ubantu/Desktop/FER/model/facial_expression_model_structure.json", "r").read())
model.load_weights('/home/ubantu/Desktop/FER/model/facial_expression_model_weights.h5') #load weights

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

def insert(emotion,time): 
    query = "INSERT INTO eg(emotion,time) " \
            "VALUES(%s,%s)"
    args = (emotion, time)
   
    try:
        cursor = conn.cursor()
        cursor.execute(query, args)
	print("Inserted in db")
 	conn.commit()
    except Error as e:
        print(e)
	
class Interval(object):

    def __init__(self, interval, function, args=[], kwargs={}):
        """
        Runs the function at a specified interval with given arguments.
        """
        self.interval = interval
        self.function = partial(function, *args, **kwargs)
        self.running  = False 
        self._timer   = None 

    def __call__(self):
        """
        Handler function for calling the partial and continuting. 
        """
        self.running = False  # mark not running
        self.start()          # reset the timer for the next go 
        self.function()       # call the partial function 

    def start(self):
        """
        Starts the interval and lets it run. 
        """
        if self.running:
            # Don't start if we're running! 
            return 
            
        # Create the timer object, start and set state. 
        self._timer = Timer(self.interval, self)
        self._timer.start() 
        self.running = True
        

    def stop(self):
        """
        Cancel the interval (no more function calls).
        """
        if self._timer:
            self._timer.cancel() 
        self.running = False 
        self._timer  = None


if __name__ == "__main__":
    import time 
    import random
    
    def clock(start):
        diff = time.time() - start
        print("Inside clock: "+str(emotion))
        insert(emotion, diff)
   	print(diff)
   	
   	#Create an interval. 
    interval = Interval(5, clock, args=[time.time(),])
    print "Starting Interval, press CTRL+C to stop."
    interval.start() 
    
    while(True):
		time.sleep(0.1)
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		#print(faces)
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
				
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
			detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
			
			img_pixels = image.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
	
			img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
	
			predictions = model.predict(img_pixels) #store probabilities of 7 expressions
	
			#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
			max_index = np.argmax(predictions[0])
		
			emotion = emotions[max_index]
			#print(emotion)
			#write emotion text above rectangle
			cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
			#print(emotion)
		cv2.imshow('img',img)
			
		if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
			interval.stop()
			break            

#kill open cv things		
cap.release()
cv2.destroyAllWindows()
