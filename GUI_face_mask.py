from PIL import ImageTk
import PIL.Image
import cv2
from tkinter import *
import tkinter as tk
from tkinter import filedialog
import numpy as np
import label_image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

windo = Tk()
windo.configure(background='white')
windo.title("Face Mask Detection App")
width  = windo.winfo_screenwidth()
height = windo.winfo_screenheight()
windo.geometry(f'{width}x{height}')

windo.iconbitmap('./images/mask.ico')
windo.resizable(0,0)

#Size for displaying Image
w = 400;h = 280
size = (w, h)

def upload_im():
    global scale1,scale
    def print_value(val):
        global volume
        volume = int(val) / 100
        return volume

    scale = tk.Scale(orient='horizontal', from_=110, to=190, resolution=10, command=print_value,width =20,
                     borderwidth=0,background = 'white',   activebackground='blue')
    scale.set(130)
    scale.place(x=410, y=490)

    s = tk.Label(windo, text='Set lower if people is more:', width=29, height=1, fg="black", bg="gold",
                    font=('times', 15, ' bold '))
    s.place(x=20, y=500)

    def print_value1(val1):
        pass

    scale1 = tk.Scale(orient='horizontal', from_=1, to=9, resolution=1, command=print_value1,width = 20,
                      borderwidth=0,background = 'white',activebackground='blue')
    scale1.set(3)
    scale1.place(x=410, y=555)

    s1 = tk.Label(windo, text='Set Higher for good detection accuracy', width=29, height=1, fg="white", bg="black",
                    font=('times', 15, ' bold '))
    s1.place(x=20, y=562)

    try:
        global im,resized,cp,path
        imageFrame = tk.Frame(windo)
        imageFrame.place(x=415, y=60)
        path = filedialog.askopenfilename()
        im = PIL.Image.open(path)
        resized = im.resize(size, PIL.Image.ANTIALIAS)
        tkimage = ImageTk.PhotoImage(resized)
        display = tk.Label(imageFrame)
        display.imgtk = tkimage
        display.configure(image=tkimage)
        display.grid()
        dn1 = tk.Label(windo, text='Original\ud83d\ude80 Image ', width=20, height=1, fg="white", bg="deep pink",
                       font=('times', 22, ' bold '))
        dn1.place(x=444, y=20)
        cp = tk.Button(windo, text='Detect Face Mask',command = prediction, bg="blue", fg="white", width=20,
                       height=1, font=('times', 22, 'italic bold '),activebackground = 'yellow')
        cp.place(x=440, y=370)
    except:
        noti = tk.Label(windo, text = 'Please upload an Image\ud83d\ude80 File', width=29, height=1, fg="white", bg="blue",
                            font=('times', 15, ' bold '))
        noti.place(x=20, y=450)
        windo.after(5000, destroy_widget, noti)
        scale1.destroy()
        scale.destroy()

def destroy_widget(widget):
    widget.destroy()

def prediction():
    try:
        global op,tkimage4,img
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        # img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        o = scale1.get()
        print(volume,o)
        faces = face_cascade.detectMultiScale(gray,volume,o)
        o = 0
        p = 0
        font = cv2.FONT_HERSHEY_TRIPLEX
        for (x, y, w, h) in faces:
            sub_face = img[y:y + h, x:x + w]
            FaceFileName = "test.jpg"  # Saving the current image from the webcam for testing.
            cv2.imwrite(FaceFileName, sub_face)
            text = label_image.main(FaceFileName)  # Getting the Result from the label_image file, i.e., Classification Result.
            text = text.title()  # Title Case looks Stunning.
            print(text)
            if text == 'Mask Found':
                text = 'With Mask'
                o += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 260, 0), 5)
                cv2.putText(img, text, (x + h, y), font, 1, (0, 260, 0), 2)
            if text == 'Mask Not Found':
                text = 'No mask'
                p += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 25, 255), 5)
                cv2.putText(img, text, (x + h, y), font, 1, (0, 25, 255), 2)
        cm = str(o) + " people with mask"
        cwm = str(p) + " people without mask"
        no = tk.Label(windo, text= cm, width=33, height=1,
                          fg="white", bg="midnightblue",
                          font=('times', 15, ' bold '))
        no.place(x=844, y=370)

        no1 = tk.Label(windo, text= cwm, width=33, height=1,
                          fg="white", bg="red",
                          font=('times', 15, ' bold '))
        no1.place(x=844, y=405)
        sv = tk.Button(windo, text='Save\ud83d\ude80 Image', bg="medium spring green", fg="black", width=20,
                       height=1, font=('times', 22, 'italic bold '), command=save_img, activebackground='yellow')
        sv.place(x=870, y=450)
        # windo.after(8000, destroy_widget, no)
        # windo.after(8000, destroy_widget, no1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        op = PIL.Image.fromarray(img)
        resi = op.resize(size, PIL.Image.ANTIALIAS)
        tkimage4 = ImageTk.PhotoImage(resi)
        imageFrame4 = tk.Frame(windo)
        imageFrame4.place(x=845, y=60)
        dn4 = tk.Label(windo, text='Face Mask Detection', width=20, height=1, fg="white", bg="navy",
                           font=('times', 22, ' bold '))
        dn4.place(x=874, y=20)
        display4 = tk.Label(imageFrame4)
        display4.imgtk = tkimage4
        display4.configure(image=tkimage4)
        display4.grid()
    except Exception as e:
        notip = tk.Label(windo, text = 'Faces not found in Image\ud83d\ude80!!', width=29, height=1, fg="white", bg="midnightblue",
                            font=('times', 15, ' bold '))
        notip.place(x=20, y=450)
        windo.after(7000, destroy_widget, notip)
        print(e)

def save_img():
    name = "FMD_"+os.path.basename(path)
    op.save(name)
    not3 = tk.Label(windo, text= name+' Saved', width=29, height=1, fg="white",
                     bg="midnightblue",
                     font=('times', 15, ' bold '))
    not3.place(x=20, y=450)
    windo.after(5000, destroy_widget, not3)

my_name = tk.Label(windo, text="©Developed by Surbhi Verma", bg="blue", fg="white", width=58,
                   height=1, font=('times', 30, 'italic bold '))
my_name.place(x=00, y=640)

dn = tk.Label(windo, text='Face Mask Detection', width=20, height=1, fg="white", bg="blue2",
              font=('times', 22, ' bold '))
dn.place(x=24, y=20)

ri = PIL.Image.open('./images/man.png')
ri =ri.resize((351,303), PIL.Image.ANTIALIAS)
sad_img = ImageTk.PhotoImage(ri)
panel4 = Label(windo, image=sad_img,bg = 'white')
panel4.pack()
panel4.place(x=20, y=60)

up = tk.Button(windo,text = 'Upload\ud83d\ude80 Image',bg="medium spring green", fg="black", width=20,
                   height=1, font=('times', 22, 'italic bold '),command = upload_im, activebackground = 'yellow')
up.place(x=20, y=370)


windo.mainloop()
