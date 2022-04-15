from tkinter import *
import tkinter as tk
from tkinter.font import BOLD
from PIL import Image, ImageTk
import webbrowser


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

def github():
   webbrowser.open("https://github.com/MehmeTTEmre")

def linkedin():
   webbrowser.open("https://www.linkedin.com/in/mehmet-emre-şahin-805107199/")


# Github logo
github_btn = Image.open("image/github.ico")
github_btn = ImageTk.PhotoImage(github_btn)
img_label = Label(image=github_btn, 
                  background="#1DA1F2"
)

# Linkedin logo
linkedin_btn = Image.open("image/linkedin.ico")
linkedin_btn = ImageTk.PhotoImage(linkedin_btn)
img_label2 = Label(image=linkedin_btn, 
                   background="#1DA1F2"
)

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


# Button
login = Button(root,
               text="Login", 
               font= ("Times bold", 14),
               command=nextPage,
               width=91,
               height=1
)

button = Button(root, 
                image=github_btn,
                command=github,
                borderwidth=0,
                background="#1DA1F2",
                activebackground="#1DA1F2"
)

button2 = Button(root, 
                 image=linkedin_btn,
                 command= linkedin,
                 borderwidth=0,
                 background="#1DA1F2",
                 activebackground="#1DA1F2"
)


logo_label.place(x=350, y=120)
title.place(x=350, y=30)
login.place(x=0, y=561)
button.place(x=850, y=485)
button2.place(x=910, y=492)
root.mainloop()