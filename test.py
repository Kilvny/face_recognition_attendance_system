import cv2
import numpy as np
import face_recognition


viniJR = face_recognition.load_image_file('D:\\python\\opencvproject\\photos\\vini jr1.jpg')
viniJR = cv2.cvtColor(viniJR,cv2.COLOR_BGR2RGB)

viniJRtwo = face_recognition.load_image_file('D:\\python\\opencvproject\\photos\\63d30c7697b4a.webp')
viniJRtwo = cv2.cvtColor(viniJRtwo,cv2.COLOR_BGR2RGB)

faceCord = face_recognition.face_locations(viniJR)[0] # result of this is (top,right,bottom,left) 
encodeVini = face_recognition.face_encodings(viniJR)[0]
cv2.rectangle(viniJR, (faceCord[3], faceCord[0]), (faceCord[1], faceCord[2]), (255,0,200), 2)

faceCordtwo = face_recognition.face_locations(viniJRtwo)[0] # result of this is (top,right,bottom,left) 
encodeVinitwo = face_recognition.face_encodings(viniJRtwo)[0]
# print(faceCordtwo)
cv2.rectangle(viniJRtwo, (faceCordtwo[3], faceCordtwo[0]), (faceCordtwo[1], faceCordtwo[2]), (255,0,200), 2)

# basically we will compare the encodings of the two images to check the distance between them and print either false or true based on the if the distance is close or not 
res = face_recognition.compare_faces([encodeVini], encodeVinitwo)
print(res) # this just check if theres similarity
# this will check how close the distance are
dist = face_recognition.face_distance([encodeVini], encodeVinitwo)
print(dist)

cv2.putText(viniJRtwo,f'{res} {round(dist[0],2)}', (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
cv2.imshow('Vini Jr', viniJR)
cv2.imshow('Vini JR', viniJRtwo)
cv2.waitKey(0)