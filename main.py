from tkinter import *
import tkinter as tk
from tkinter.font import BOLD
from PIL import Image, ImageTk

#
root = tk.Tk()
root.title('Twitter Sentiment Analysis')
root.iconbitmap(r"image/twitter.ico")
root.geometry("1000x600")
root.minsize(1000,600)
root.maxsize(1000,600)
root.configure(background="#1DA1F2")

def nextPage():
    root.destroy()
    import page2

# Logo 
logo = Image.open("image/Ankara_Üniversitesi_logosu.png")
logo = ImageTk.PhotoImage(logo)
logo_label = tk.Label(root,
                      image=logo,
                      background="#1DA1F2"
)

# Title
title = Label(root, 
              text="Twitter Sentiment Analysis",
              font=("Helvatica", 20, BOLD),
              justify="center",
              background="#1DA1F2"
)

word = Label(root, 
              #text="Mehmet Emre Şahin\n Akın Deniz",
              font=("Helvatica", 20),
              justify="center",
              background="#1DA1F2"
)

# Button
login = Button(root,
               text="Login", 
               font= ("Times bold", 14),
               command=nextPage,
               width=91,
               height=1
)



logo_label.place(x=350, y=120)
title.place(x=350, y=30)
word.place(x=730, y=485)
login.place(x=0, y=561)

root.mainloop()