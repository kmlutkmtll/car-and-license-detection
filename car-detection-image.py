import cv2
import imutils
import numpy as np

lic_data = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
cascade_src = 'car_cascade_own.xml'

img = cv2.imread('img.jpeg', cv2.IMREAD_COLOR)
img = np.array(img)

aspectRatio = (img.shape[0] / img.shape[1])
print(aspectRatio)

if (aspectRatio >= 0.5 and aspectRatio <= 0.7):
    weight = 600
    height = 370
elif (aspectRatio <= 1.5 and aspectRatio >= 1.1):
    weight = 550
    height = 750
elif (aspectRatio >= 0.7 and aspectRatio <= 0.85):
    weight = 600
    height = 400

print(img.shape)
img = cv2.resize(img,(weight,height))
print(img.shape)

car_cascade = cv2.CascadeClassifier(cascade_src)
number = lic_data.detectMultiScale(img,1.22)
print("Number plate detected : " + str(len(number)))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cars = car_cascade.detectMultiScale(gray, 1.2, 8)

for (x,y,w,h) in cars:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

for numbers in number:
    (x,y,w,h) = numbers
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+h]
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

cv2.imshow('img', img)
cv2.waitKey(0)

cv2.destroyAllWindows()