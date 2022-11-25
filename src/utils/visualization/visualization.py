# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 11:57:49 2021

@author: CEOSpaceTech
"""



from os import makedirs
from tkinter import * 
import pandas as pd
from PIL import ImageTk, Image
import numpy as np
import shutil
def visualization(path, apply_model_data_path, ind_0,ind_1):
    root = Tk()
    root.grid_rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    makedirs(path + "/Train/positive", exist_ok=True)
    makedirs(path + "/Train/negative", exist_ok=True)
    positive_folder = path + "/Train/positive"
    negative_folder = path + "/Train/negative"
    images = np.array(pd.read_csv(apply_model_data_path).iloc[:, 0])
    images_0 = np.take_along_axis(images,np.array(ind_0),axis=0)
    images_1 = np.take_along_axis(images,np.array(ind_1),axis=0)
    relevant=images_0[:14]
    irrelevant=images_1[:14]
    images_sorted=np.concatenate((relevant,irrelevant),axis=0)    
    img = []
    for i in range(len(images_sorted)):
        img.append(ImageTk.PhotoImage(file=images_sorted[i]))
        
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
    
    # Create a frame to contain the buttons
    frame_buttons = Frame(canvas, bg="blue")
    canvas.create_window((0, 0), window=frame_buttons, anchor='nw')
    
    def positive(x):   
        shutil.move(images_sorted[x], positive_folder)
    def negative(x):
        shutil.move(images_sorted[x], negative_folder)
    # def train():
    #     root.destroy()
    #     ranking.main()
    # Add buttons to the frame
    btn_nr = -1
    columns = 7
    rows = int(len(images_sorted)/columns)
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
    # exit_button = Button(canvas,text='Train Again', command=train)
    # exit_button.grid(row=1, column=1, columnspan=15)        
    # Update buttons frames idle tasks to let tkinter calculate buttons sizes
    frame_buttons.update_idletasks()
    
    # Resize the canvas frame to show exactly 5-by-5 buttons and the scrollbar
    first5columns_width =262*columns# sum([buttons[0][j].winfo_width() for j in range(0, columns)])
    first5rows_height = 262*columns#sum([buttons[i][0].winfo_height() for i in range(0, columns)])
    frame_canvas.config(width=first5columns_width + vsb.winfo_width(),
                        height=first5rows_height)
    
    canvas.config(scrollregion=canvas.bbox("all"))
    
    # exit_button = Button(canv,text='Exit Game', command=frame.destroy)
    # exit_button.grid(row=4, column=1, columnspan=15)
    root.mainloop()
    
    
def visualization1(path, apply_model_data_path, ind):
    root = Tk()
    root.grid_rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    makedirs(path + "/Train/positive", exist_ok=True)
    makedirs(path + "/Train/negative", exist_ok=True)
    positive_folder = path + "/Train/positive"
    negative_folder = path + "/Train/negative"
    images = np.array(pd.read_csv(apply_model_data_path).iloc[:, 0])
    images = np.take_along_axis(images,np.array(ind),axis=0)
    relevant=images[:56]
    irrelevant=images[-56:]
    images_sorted=np.concatenate((relevant,irrelevant),axis=0)    
    img = []
    for i in range(len(images_sorted)):
        img.append(ImageTk.PhotoImage(Image.open(images_sorted[i]).resize((64, 64), Image.ANTIALIAS)))
        
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
    
    # Create a frame to contain the buttons
    frame_buttons = Frame(canvas, bg="blue")
    canvas.create_window((0, 0), window=frame_buttons, anchor='nw')
    
    def positive(x):   
        shutil.move(images_sorted[x], positive_folder)
    def negative(x):
        shutil.move(images_sorted[x], negative_folder)
    # def train():
    #     root.destroy()
    #     ranking.main()
    # Add buttons to the frame
    btn_nr = -1
    columns = 14
    rows = int(len(images_sorted)/columns)
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
    # exit_button = Button(canvas,text='Train Again', command=train)
    # exit_button.grid(row=1, column=1, columnspan=15)        
    # Update buttons frames idle tasks to let tkinter calculate buttons sizes
    frame_buttons.update_idletasks()
    
    # Resize the canvas frame to show exactly 5-by-5 buttons and the scrollbar
    first5columns_width =262*columns# sum([buttons[0][j].winfo_width() for j in range(0, columns)])
    first5rows_height = 262*columns#sum([buttons[i][0].winfo_height() for i in range(0, columns)])
    frame_canvas.config(width=first5columns_width + vsb.winfo_width(),
                        height=first5rows_height)
    
    canvas.config(scrollregion=canvas.bbox("all"))
    
    # exit_button = Button(canv,text='Exit Game', command=frame.destroy)
    # exit_button.grid(row=4, column=1, columnspan=15)
    root.mainloop()