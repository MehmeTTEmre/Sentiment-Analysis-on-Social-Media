from tkinter import *
import tkinter as tk
from tkinter.font import BOLD
from PIL import Image, ImageTk
import webbrowser
import customtkinter

customtkinter.set_appearance_mode("System")
root=customtkinter.CTk(fg_color="#f0f0f0")
root.title('Twitter Sentiment Analysis')
root.iconbitmap(r"image/ico/twitter.ico")
root.geometry("1000x600")
root.minsize(1000,600)
root.maxsize(1000,600)
root.configure()

def nextPage():
    root.destroy()
    import page2

def github():
   webbrowser.open("https://github.com/MehmeTTEmre")

def linkedin():
   webbrowser.open("https://www.linkedin.com/in/mehmet-emre-şahin-805107199/")

# Github logo
github_btn = Image.open("image/ico/github.ico")
github_btn = ImageTk.PhotoImage(github_btn)
img_label = Label(image=github_btn)

# Linkedin logo
linkedin_btn = Image.open("image/ico/linkedin.ico")
linkedin_btn = ImageTk.PhotoImage(linkedin_btn)
img_label2 = Label(image=linkedin_btn)

# Logo 
logo = Image.open("image/ico/AnkaraUni_logo.png")
logo = ImageTk.PhotoImage(logo)
logo_label = tk.Label(root, image=logo, bg="#f0f0f0")

img = ImageTk.PhotoImage(Image.open("image/walpaper/AU_PC.jpeg"))
# Create a Label Widget to display the text or Image
label = Label(root, image = img, bg="#f0f0f0")
label.place(x=0, y=70)

# Title
title = Label(root, 
              text="Twitter Sentiment Analysis",
              font=("Helvatica", 30, BOLD),
              justify="center",
              bg="#f0f0f0"
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
                relief="flat"
)

button2 = Button(root, 
                 image=linkedin_btn,
                 command= linkedin,
                 borderwidth=0,
                 relief="flat"
)

title.place(x=250, y=10)
logo_label.place(x=50, y=2)
login.place(x=0, y=561)
button.place(x=850, y=0)
button2.place(x=910, y=8)
root.mainloop()