import cv2
import label_image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

size = 4
# We load the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
global text
webcam = cv2.VideoCapture(0)  # Using default WebCam connected to the PC.
#for video writing
# frame_width = int(webcam.get(3))
# frame_height = int(webcam.get(4))
# out = cv2.VideoWriter('outpy2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

font = cv2.FONT_HERSHEY_TRIPLEX
while True:
    (rval, im) = webcam.read()
    im = cv2.flip(im, 1, 0)  # Flip to act as a mirror
    # Resize the image to speed up detection
    mini = cv2.resize(im, (int(im.shape[1] / size), int(im.shape[0] / size)))
    # detect MultiScale / faces
    faces = classifier.detectMultiScale(mini)
    o = 0
    p = 0
    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f]  # Scale the shapesize backup
        sub_face = im[y:y + h, x:x + w]
        FaceFileName = "test.jpg"  # Saving the current image from the webcam for testing.
        cv2.imwrite(FaceFileName, sub_face)
        text = label_image.main(FaceFileName)  # Getting the Result from the label_image file, i.e., Classification Result.
        text = text.title()  # Title Case looks Stunning.
        font = cv2.FONT_HERSHEY_TRIPLEX
        print(text)

        if text == 'Mask Found':
            text = 'With Mask'
            o += 1
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 5)
            cv2.putText(im, text, (x + h, y), font, 1, (0, 260, 0), 2)

        if text == 'Mask Not Found':
            text = 'No mask'
            p += 1
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 25, 255), 5)
            cv2.putText(im, text, (x + h, y), font, 1, (0, 25, 255), 2)
    cm = str(o) + " people with mask"
    cwm = str(p) + " people without mask"
    # Show the image/
    cv2.putText(im, cm, (40, 370), font, 1, (0, 260, 0), 2)
    cv2.putText(im, cwm, (40, 400), font, 1, (0, 25, 250), 2)
 #   out.write(im) #For video writing
    cv2.imshow('Face Mask Detection', im)
    key = cv2.waitKey(30)& 0xff
    if key == 27:  # The Esc key
        break

cv2.destroyAllWindows()