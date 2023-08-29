import cv2 as cv
import numpy as np 
import datetime
import math
import time
from ultralytics import YOLO
import os



file_dir = os.path.dirname(os.path.abspath(__file__))


os.chdir(file_dir)


### EVENTS
def Start():
   print('start')
    
def Stop():
    print('stop')

def counter_rest():
    print('reset')
    
    
def button_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        reset = [1287, 209,1490,309]
        start =[900,700,50]
        end = [1200, 700,50]
        start_dist = math.sqrt((x -start[0])**2 + (y - start[1])**2)
        stop_dist = math.sqrt((x -end[0])**2 + (y - end[1])**2)
       
        if reset[0] < x < reset[2] and reset[1] < y < reset[3]:
            counter_rest()
        if start_dist <= start[2]:
            Start()
        if stop_dist <= end[2]:
            Stop()

        







###background
background = img_3 = np.zeros([800,1500,3],dtype=np.uint8)
cv.rectangle(background, (0, 0), (1500, 800),(180,187,133),thickness=-1)
font = cv.FONT_HERSHEY_COMPLEX
### element
cv.putText(background, 'Alkaline=',(800,200),font,0.8, (0,0,0))
cv.putText(background, 'Lithium=',(800,300),font,0.8, (0,0,0))
cv.putText(background, 'Ni-cd=',(800,400),font,0.8, (0,0,0))
cv.putText(background, 'Ni-Mh=',(800,500),font,0.8, (0,0,0))
###numbers
cv.rectangle(background, (950, 170), (1100, 220),(255,255,255),thickness=-1)
cv.rectangle(background, (950, 270), (1100, 320),(255,255,255),thickness=-1)
cv.rectangle(background, (950, 370), (1100, 420),(255,255,255),thickness=-1)
cv.rectangle(background, (950, 470), (1100, 520),(255,255,255),thickness=-1)


cv.putText(background,str(0),(980, 200),font,0.8, (0,0,0))
cv.putText(background, str(0),(980, 300),font,0.8, (0,0,0))
cv.putText(background, str(0),(980, 400),font,0.8, (0,0,0))
cv.putText(background, str(0),(980, 500),font,0.8, (0,0,0))



####counter
cv.rectangle(background, (1287, 209), (1490, 309),(255,0,0),thickness=-1)
cv.putText(background, 'counter reset',(1290, 180),font,0.8, (0,0,0))



###start & stop
cv.circle(background, (900, 700),50, (0,255,0),-1)
cv.circle(background, (1200, 700),50, (0,0,255),-1)
cv.putText(background, 'Start',(850, 630),font,1, (0,0,0))
cv.putText(background, 'Stop',(1150, 630),font,1, (0,0,0))

### Date & time
cv.putText(background, 'Date/time',(1300, 700),font,1, (0,0,0))






### Detection 

model = YOLO('best.pt')

file = open('classname.txt','r')
data =file.read()
class_list = data.split("\n") 

colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(11)]


###frame
cap = cv.VideoCapture(0)
while True :
        ret,frame = cap.read()

        time = datetime.datetime.now()
       
        cv.putText(background,str(0),(980, 200),font,0.8, (0,0,0))
        cv.putText(background, str(0),(980, 300),font,0.8, (0,0,0))
        cv.putText(background, str(0),(980, 400),font,0.8, (0,0,0))
        cv.putText(background, str(0),(980, 500),font,0.8, (0,0,0))
     

        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1 =int(result[0])
            y1 =int(result[1])
            x2 =int(result[2])
            y2 =int(result[3])
            score = result[4]
            class_id = int(result[5])
       
            if score > 0.4:
                cv.rectangle(frame,(x1, y1), (x2, y2), colors[class_id], 4)
                cv.putText(frame,class_list[class_id], (x1, y1-10),
                        cv.FONT_HERSHEY_SIMPLEX, 1.3, colors[class_id], 3, cv.LINE_AA)
    
        
        image = cv.resize(frame,(710,727))
        background[50:777,20:730]= image
        copy = background.copy()
        
        copy = cv.resize(background, (1200 ,700))
        cv.putText(copy,str(time),( 1000,650),font,0.5, (0,0,0))
       
        cv.imshow('Battery Sorting Technology',copy)
        
        cv.setMouseCallback("Battery Sorting Technology", button_click)
        key = cv.waitKey(1) & 0xFF

        
        if key == ord("q"):
            break




cap.release()

cv.destroyAllWindows()
