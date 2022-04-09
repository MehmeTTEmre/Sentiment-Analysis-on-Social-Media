import tkinter as tk
from tkinter import *
from tkinter import ttk, filedialog
import pandas as pd

root=tk.Tk()
root.geometry("1000x600")
root.minsize(1000,600)
root.maxsize(1000,600)
root.configure()
root.title('Twitter Sentiment Analysis')
root.iconbitmap(r"image/twitter.ico")


def prevPage():
    root.destroy()
    import page2

def homePage():
    root.destroy()
    import main
    
# Create an object of Style widget
style = ttk.Style()
style.theme_use('clam')

# Create a Frame
frame = Frame(root, background="#1DA1F2")
frame.pack(expand=True, fill=BOTH)
# Define a function for opening the file
def open_file():
   filename = filedialog.askopenfilename(title="Open a File",filetype=(("xlxs files", ".*xlsx"), ("All Files", "*.")))

   if filename:
      try:
         filename = r"{}".format(filename)
         df = pd.read_excel(filename)
      except ValueError:
         label.config(text="File could not be opened")
      except FileNotFoundError:
         label.config(text="File Not Found")

   # Clear all the previous data in tree
   clear_treeview()

   # Add new data in Treeview widget
   tree["column"] = list(df.columns)
   tree["show"] = "headings"

   # For Headings iterate over the columns
   for col in tree["column"]:
      tree.heading(col, text=col)

   # Put Data in Rows
   df_rows = df.to_numpy().tolist()
   for row in df_rows:
      tree.insert("", "end", values=row)

   tree.pack(expand=True, fill=BOTH)

# Clear the Treeview Widget
def clear_treeview():
   tree.delete(*tree.get_children())

# Create a Treeview widget
tree = ttk.Treeview(frame)

# Add a Menu
m = Menu(root)
root.config(menu=m)

# Add Menu Dropdown
file_menu = Menu(m, tearoff=False)
m.add_cascade(label="Open", menu=file_menu)
file_menu.add_command(label="Open Spreadsheet", command=open_file)
#file_menu.add_command(label="Open Image", command=open_file)

# Add a Label widget to display the file content
label = Label(root)
label.pack()


homepage = Button(root,
               text="Homepage", 
               font= ("Times bold", 14),
               command=homePage,
               width=46,
               height=1
)
homepage.place(x=0, y=561)

page2 = Button(root, 
               text="Next Page", 
               font = ("Times bold", 14),
               command=prevPage,
               width=45,
               height=1
)
page2.place(x=495,y=561)

root.mainloop()