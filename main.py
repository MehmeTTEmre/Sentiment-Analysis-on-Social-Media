from sqlite3 import Row
from tkinter import *
import tkinter as tk
from tkinter.font import BOLD
from turtle import bgcolor, color, width
from PIL import Image, ImageTk

root = tk.Tk()
root.title('Twitter Sentiment Analysis')
root.iconbitmap(r"image/twitter.ico")
canvas = tk.Canvas(root, width=1000, height=600, bg="#87F0FC")
canvas.grid(columnspan=3, rowspan=4)

def nextPage():
    root.destroy()
    import page2

# Logo 
logo = Image.open("image/Ankara_Üniversitesi_logosu.png")
logo = ImageTk.PhotoImage(logo)
logo_label = tk.Label(root,
                      image=logo,
                      background="#87F0FC"
)

# Title
title = Label(root, 
              text="Twitter Sentiment Analysis",
              font=("Helvatica", 20, BOLD),
              justify="center",
              background="#87F0FC"
)

word = Label(root, 
              text="Mehmet Emre Şahin\n Akın Deniz",
              font=("Helvatica", 20),
              justify="center",
              background="#87F0FC"
)

# Button
login = Button(root,
               text="Login", 
               font= ("Times bold", 14),
               command=nextPage,
               width=91,
               height=1
)



logo_label.grid(column=1, row=1)
title.grid(column=1, row=0)
word.place(x=730, y=485)
login.place(x=0, y=565)

root.mainloop()