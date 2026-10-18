from ultralytics import YOLO
import cv2
from sort import Sort
import numpy as np


model = YOLO('yolov8n.pt')
#live web cam
#cap = cv2.VideoCapture(0)
#video
cap = cv2.VideoCapture("cars.mp4")


#img = cv2.imread('car_image.jpg')
#model(img,show=True)
#cv2.waitKey(0)

#mask = cv2.imread('mask.png')
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
count_line = [0,297,1280,297]
totalCount = []
while True:
    sucess, img = cap.read()
    #imgRegion = cv2.bitwise_and(img,mask)
    results = model(img,stream=True)
    detections = np.empty((0,5))

    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in ['car','truck','bus','mtorbike']:
              #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255), 2)
              #cv2.putText(img,f"{label}{conf:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
              current = np.array([x1,y1,x2,y2,conf])
              detections = np.vstack((detections,current))

    resultTracker = tracker.update(detections)
    cv2.line(img,(count_line[0],count_line[1]),(count_line[2],count_line[3]),(0,0,255),5)
    for x1, y1, x2, y2 ,_id in resultTracker:
        x1, y1, x2, y2 ,_id = map(int,(x1,y1,x2,y2,_id))
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(img, f"ID:{_id}", (x1, max(y1-10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
        
        if count_line[0] < cx < count_line[2] and count_line[1]-15 <cy <count_line[3]+15:
            if _id not in totalCount:
                totalCount.append(_id)
                cv2.line(img,(count_line[0],count_line[1]),(count_line[2],count_line[3]),(0,255,0),5)


    cv2.putText(img,f"Count:{len(totalCount)}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,255),4)           
    cv2.imshow("webcam",img)
    cv2.waitKey(1)

