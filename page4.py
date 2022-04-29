import tkinter as tk
from tkinter import *
import customtkinter

customtkinter.set_appearance_mode("System")
root=customtkinter.CTk()
root.geometry("1000x600")
root.minsize(1000,600)
root.maxsize(1000,600)
root.configure()
root.title('Twitter Sentiment Analysis')
root.iconbitmap(r"image/ico/twitter.ico")

frame_1 = customtkinter.CTkFrame(master=root, width=940, height=150)
frame_1.place(x=30, y=25)

frame_2 = customtkinter.CTkFrame(master=root, width=940, height=360)
frame_2.place(x=30, y=200)

def prevPage():
    root.destroy()
    import page3

def homePage():
    root.destroy()
    import main

def start_training():
    import train2
    accuracy_label["text"] = "Accuracy: {:0.2f}".format(train2.accr1)
    tweetBtn.state = "normal"

lst = []
def submit():
    import train2
    tweet=tweet_var.get()
    tweet_var.set("")
    tweet_sentiment["text"] = train2.predict(tweet)
    class Table:
        def __init__(self, frame_2):
            # code for creating table
            for i in range(total_rows):
                for j in range(total_columns):
                    self.e = customtkinter.CTkEntry(frame_2, width=470)
                    self.e.grid(row=i, column=j)
                    self.e.insert(END, lst[i][j])
    # take the data
    lst.append((tweet, tweet_sentiment["text"]))
    total_rows = len(lst)
    total_columns = 2
    table = customtkinter.CTkLabel(frame_2, fg_color="#2e2e2e")
    table.place(x=0, y=10)
    Table(table)

tweet_var=tk.StringVar()
textEntry = customtkinter.CTkEntry(frame_1, width=840, placeholder_text="Text", placeholder_text_color="white", textvariable=tweet_var)
textEntry.place(x=50, y=30)
textEntry.configure()

tweetBtn=customtkinter.CTkButton(frame_1,text = 'Submit', command = submit, width=30, height=25, fg_color="orange", text_color_disabled="black", text_color="black", state="disabled", hover_color="green")
tweetBtn.place(relx=0.45, rely=0.5)

machine_btn=customtkinter.CTkButton(frame_1,text = 'Start Training', command = start_training, width=50, height=25, fg_color="orange", text_color="black", hover_color="green")
machine_btn.place(relx=0.43, rely=0.75)

accuracy_label = tk.Label(frame_1, text="", font=('calibre',20, 'bold'), bg="#2e2e2e", fg="orange")
accuracy_label.place(x=600, y=85)

tweet_sentiment = tk.Label(frame_2, text="", font=('calibre',20, 'bold'))

homepage = Button(root,
               text="Homepage", 
               font= ("Times bold", 14),
               command=homePage,
               width=46,
               height=1
)
homepage.place(x=0, y=561)

page2 = Button(root, 
               text="Prev Page", 
               font = ("Times bold", 14),           
               command=prevPage,
               width=45,
               height=1
)
page2.place(x=500,y=561)

root.mainloop()