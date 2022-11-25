# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 17:09:52 2021

@author: CEOSpaceTech
"""
from os import makedirs
from tkinter import * 
import pandas as pd
from PIL import ImageTk, Image
import numpy as np
import shutil
import glob
# from src.ranking import ranking
apply_model_data_path='D:/Omid/UPB/SVM/Galaxy-classification-master/data/california/apply_model.csv'
path='D:/Omid/UPB/SVM/Galaxy-classification-master/data/LA'
images = glob.glob(path + '/unlabeled/*.png')
root = Tk()
root.grid_rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

makedirs(path + "/labeled/positive", exist_ok=True)
makedirs(path + "/labeled/negative", exist_ok=True)
positive_folder = path + "/labeled/positive"
negative_folder = path + "/labeled/negative"
# images_sorted = np.array(pd.read_csv(apply_model_data_path).iloc[:, 0])

# images_sorted = np.take_along_axis(images,np.array(ind),axis=0)
# relevant=images_sorted[:32]
# irrelevant=images_sorted[-32:]
# images=np.concatenate((relevant,irrelevant),axis=0)
img = []
# images= images_sorted
for i in range(len(images)):
    
    img.append(ImageTk.PhotoImage(Image.open(images[i]).resize((64, 64), Image.ANTIALIAS)))
    
frame_main =Frame(root)    
frame_main.grid(sticky='news')
# Create a frame for the canvas with non-zero row&column weights
frame_canvas = Frame(frame_main)
frame_canvas.grid(row=2, column=0, pady=(5, 0), sticky='nw')
frame_canvas.grid_rowconfigure(0, weight=1)
frame_canvas.grid_columnconfigure(0, weight=1)
# Set grid_propagate to False to allow 5-by-5 buttons resizing later
frame_canvas.grid_propagate(False)   

# Add a canvas in that frame
canvas = Canvas(frame_canvas, bg="yellow")
canvas.grid(row=0, column=0, sticky="news")

# Link a scrollbar to the canvas
vsb = Scrollbar(frame_canvas, orient="vertical", command=canvas.yview)
vsb.grid(row=0, column=1, sticky='ns')
canvas.configure(yscrollcommand=vsb.set)
canvas.config(width=400, height=200)

# Create a frame to contain the buttons
frame_buttons = Frame(canvas, bg="blue")
canvas.create_window((0, 0), window=frame_buttons, anchor='nw')

def positive(x):   
    shutil.move(images[x], positive_folder)
def negative(x):
    shutil.move(images[x], negative_folder)
def train():
    root.destroy()
    # ranking.main()
# Add buttons to the frame
btn_nr = -1
columns = 7
rows = int(len(images)/columns)
buttons = [[Button() for j in range(columns)] for i in range(rows)]
# buttons = []
for i in range(0, rows):
    for j in range(0, columns):
        btn_nr += 1
        buttons[i][j] = Button(frame_buttons, image=img[btn_nr]) #, command=lambda x=btn_nr: action(x))
        buttons[i][j].grid(row=i, column=j, sticky='news')
        buttons[i][j].bind('<Button-1>', lambda event, x=btn_nr: positive(x))#lambda x=btn_nr:positive(x))
        buttons[i][j].bind('<Button-3>', lambda event, x=btn_nr: negative(x))
        # buttons.append(Button(frame_buttons,image=img[btn_nr], command=lambda x=btn_nr: action(x)))
        # buttons[btn_nr].grid(row=i, column=j, sticky='news')
        
exit_button = Button(canvas,text='Train', command=train)
exit_button.grid(row=1, column=1, columnspan=15)        
# Update buttons frames idle tasks to let tkinter calculate buttons sizes
frame_buttons.update_idletasks()

# Resize the canvas frame to show exactly 5-by-5 buttons and the scrollbar
first5columns_width = 1834#sum([buttons[0][j].winfo_width() for j in range(0, columns)])
first5rows_height = 1834#sum([buttons[i][0].winfo_height() for i in range(0, columns)])
frame_canvas.config(width=first5columns_width + vsb.winfo_width(),
                    height=first5rows_height)

canvas.config(scrollregion=canvas.bbox("all"))


root.mainloop()