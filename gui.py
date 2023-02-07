import tkinter as tk
import os
from tkinter import *

with open('data/results.txt','r') as f:
    text_list=f.readlines()
for i in range(len(text_list)):
    text_list[i]='Chances de casses: '+str(round(100-100*float(text_list[i].strip('\n')),2))+'%.'

N=len(text_list)

def update_image(image_index):
    if image_index == len(curves_list) - 1:
        # End of animation, stop updating
        return
    curve_photo = curves_list[image_index]
    curve_label.config(image=curve_photo)
    curve_label.image = curve_photo
    
    fft_photo = fft_list[image_index]
    fft_label.config(image=fft_photo)
    fft_label.image = fft_photo
    
    text.config(text=text_list[image_index])
    root.after(100, update_image, image_index + 1)

def gen_curve():
    update_image(0)

root = tk.Tk()
root.iconbitmap('icon.ico')  
root.title('Ventilation Default Detector')
C = Canvas(root, bg="blue", height=350, width=1000)    

#Then change the frame's background
background_img = PhotoImage(file = os.getcwd()+'\\background.png')  
background_label = Label(root, image=background_img)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

#We place the frame on the window
C.pack(fill=BOTH,expand=YES)

curves_list = [tk.PhotoImage(file=os.getcwd()+'\\figures\\courbe_'+str(i)+'.png') for i in range(N)]
curve_label = tk.Label(root)
curve_label.place(x=20,y=10)

fft_list = [tk.PhotoImage(file=os.getcwd()+'\\figures\\fft_'+str(i)+'.png') for i in range(N)]
fft_label = tk.Label(root)
fft_label.place(x=500,y=10)

text = tk.Label(root, text='Analysez la qualité de vos ventilateurs',fg='white',bg='black',font=('Times New Roman', 14))
text.place(x=50,y=320)

tk.Button(root, text='Generate Curve',fg='white',bg='black', command=gen_curve).pack()

root.mainloop()