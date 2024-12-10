from ultralytics import YOLO
import cv2
from time import sleep

source = input('Enter video source (default webcam): ')
cnf = input('Enter confidence (default 0.3): ')
crowd_density = input('Enter crowd density (Default 100): ')


if crowd_density.isnumeric():
    crowd_density = int(crowd_density)
else:
    crowd_density = 100

if source.isnumeric():
    source = int(source)

if source == '':
    source = 0

try:
    cnf = float(cnf)
    if cnf<0.1 or cnf>0.9:
        cnf = 0.3
except:
    cnf = 0.3

cap = cv2.VideoCapture(source)
model = YOLO('traininResult/weights/best.pt')

print(f'STARTING WITH\n\nCrowd density: {crowd_density}\n\nVideo source: {source}\n\nConfidence:{cnf}\n\n')
sleep(3)

while True:
    ret, frame = cap.read()

    if not ret:
        print('Video source expired...')
        cap.release()
        cap = cv2.VideoCapture(source)
        continue

    results = model(frame,conf=cnf)

    people_found = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            conf = round(box.conf[0].item(), 2)
            name = r.names[box.cls[0].item()]

            color = (0,255,0)
            x1, y1, x2, y2 = int(b[0].item()), int(b[1].item()), int(b[2].item()), int(b[3].item())
            
            cv2.circle(frame, (x1, y1),5,color,-1)
            # cv2.putText(frame, f'{str(name)} {str(conf)}', (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,1.5, color, 1)
            people_found += 1

    if people_found >= crowd_density:
        text_p = 'Crowded'
    else:
        text_p = 'Not crowded'

    cv2.rectangle(frame, (0, 0), (250, 100),(0,0,0), -1)
    cv2.putText(frame, text_p, (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL,1.5, (255,255,255), 1)
    cv2.imshow('Output',frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
