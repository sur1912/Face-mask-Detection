import numpy as np
import cv2
import label_image
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
img_name = 't11.jpg'
im = cv2.imread('./testing/'+ img_name)

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.3,3)
o = 0
p = 0
font = cv2.FONT_HERSHEY_TRIPLEX
for (x,y,w,h) in faces:
    # im = cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
    sub_face = im[y:y + h, x:x + w]
    FaceFileName = "test.jpg"  # Saving the current image from the webcam for testing.
    cv2.imwrite(FaceFileName, sub_face)
    text = label_image.main(FaceFileName)  # Getting the Result from the label_image file, i.e., Classification Result.
    text = text.title()  # Title Case looks Stunning.
    print(text)
    if text == 'Mask Found':
        text = 'With Mask'
        o+=1
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 5)
        cv2.putText(im, text, (x + h, y), font, 1, (0, 260, 0), 2)

    if text == 'Mask Not Found':
        text = 'No mask'
        p+=1
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 25, 255), 5)
        cv2.putText(im, text, (x + h, y), font, 1, (0, 25, 255), 2)
cm = str(o)+" people with mask"
cwm = str(p)+" people without mask"
print("%d people wear the mask" % o)
print("%d people doesn't wear the mask"%p)

cv2.putText(im,cm,(40,350),font,1,(0,260,0),2)
cv2.putText(im,cwm,(40,380),font,1,(0,25,250),2)
#resized_image = cv2.resize(im, (900, 500))
#cv2.imshow('img',resized_image)
cv2.imshow('Face mask Detection',im)
#cv2.imwrite('./testing/tested_'+img_name,im)
cv2.waitKey(0)
cv2.destroyAllWindows()