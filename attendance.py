import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'opencvproject\\photos'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)

for pic in mylist:
    images.append(cv2.imread(f'{path}\\{pic}'))
    classNames.append(os.path.splitext(pic)[0])
# print(images)
print(classNames)

print(len(images))
def encodeImages(images: list) -> list:
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodedKnown = encodeImages(images)
# print(len(encodedKnown))

def markAttendance(name: str):
    """
    Mark the person in the attendance sheet on arrival time 
    """
    with open('opencvproject\\attendance.csv', 'r+') as a:
        data = a.readlines()
        # print(data)
        namelist = []
        for line in data:
            entry = line.split(',')
            namelist.append(entry[0])

        if name not in namelist:
            now = datetime.now()
            date_time_string = now.strftime('%H : %M : %S')
            a.writelines(f'\n{name},{date_time_string}')


# webcam configuration

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgCompressed = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgCompressed = cv2.cvtColor(imgCompressed, cv2.COLOR_BGR2RGB)

    faceLoc_currFrame = face_recognition.face_locations(imgCompressed)
    encode_curr_frame = face_recognition.face_encodings(imgCompressed, faceLoc_currFrame)

    for face_loc, encode in zip(faceLoc_currFrame, encode_curr_frame):
        matches = face_recognition.compare_faces(encodedKnown, encode)
        distance = face_recognition.face_distance(encodedKnown, encode)

        # print(distance)
    
        matchIndex = np.argmin(distance)
        if matches[matchIndex]:
            # print(classNames[matchIndex].upper())
            img_name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 # scale back to normal cuz we minimized it for faster performing on imagecompressed to 0.25
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED) # to display box for the name
            cv2.putText(img, img_name,(x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)
            markAttendance(img_name)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)

