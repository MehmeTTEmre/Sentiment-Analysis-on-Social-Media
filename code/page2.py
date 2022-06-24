import tkinter as tk
from tkinter import *
import xlsxwriter
import pandas as pd
import tweepy
import emoji
import customtkinter
import re
from PIL import ImageTk, Image

# function to display data of each tweet
def printtweetdata(n, ith_tweet):
    print()
    print(f"Tweet {n}:")
    print(f"Username:{ith_tweet[0]}")
    print(f"Description:{ith_tweet[1]}")
    print(f"Location:{ith_tweet[2]}")
    print(f"Following Count:{ith_tweet[3]}")
    print(f"Follower Count:{ith_tweet[4]}")
    print(f"Total Tweets:{ith_tweet[5]}")
    print(f"Retweet Count:{ith_tweet[6]}")
    print(f"Tweet Text:{ith_tweet[7]}")
    print(f"Hashtags Used:{ith_tweet[8]}")

def cleantText(text):
    text = re.sub(r"@[A-Za-z0-9]+", "", text) # Removed @mentions
    text = re.sub(r"#", "", text) # Removed the "# symbol"
    text = re.sub(r"RT[\s]+", "", text) # Removed RT
    text = re.sub(r"https?:\/\/\S+", "", text) # Removed the hyper link     
    text = text.lstrip() # Removing the leading whitespace from the string.
    text = text.rstrip() # Removing the trailing whitespace from the string.
    text = text.lstrip("_") # Removing the leading underscore from the string.
    return text

# Removing all the emojis from the text.
def remove_emoji(text):
    return emoji.get_emoji_regexp().sub(u'', text)

# Enter your own credentials obtained
# from your developer account
consumer_key = "your key"
consumer_secret = "your key"
access_key = "your key"
access_secret = "your key"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

def scrape(words, date_since, numtweet):

    # Creating DataFrame using pandas
    db = pd.DataFrame(columns=['username', 'description', 'location', 'following', 'followers', 'totaltweets', 'retweetcount', 'text', 'hashtags', 'tweet_id'])
    new_words = words + " -filter:retweets"
    # We are using .Cursor() to search through twitter for the required tweets.
    # The number of tweets can be restricted using .items(number of tweets)
    # .Cursor() returns an iterable object. Each item in the iterator has various attributes that you can access to get information about each tweet
    tweets = tweepy.Cursor(api.search_tweets, q=new_words, since=date_since, tweet_mode="extended").items(numtweet)
    list_tweets = [tweet for tweet in tweets]
    i = 1
    array_tweets=[]
    array_tweets_id=[]
    array_tweets_username = []
    # we will iterate over each tweet in the list for extracting information about each tweet
    for tweet in list_tweets:
        username = tweet.user.screen_name
        description = tweet.user.description
        location = tweet.user.location
        following = tweet.user.friends_count
        followers = tweet.user.followers_count
        totaltweets = tweet.user.statuses_count
        retweetcount = tweet.retweet_count
        hashtags = tweet.entities['hashtags']
        tweet_id = tweet.id

        # Retweets can be distinguished by a retweeted_status attribute, in case it is an invalid reference, except block will be executed
        try:
            text = tweet.retweeted_status.text
        except AttributeError:
            text = tweet.full_text
        hashtext = list()
        for j in range(0, len(hashtags)):
            hashtext.append(hashtags[j]['text'])
        
        # Here we are appending all the extracted information in the DataFrame
        ith_tweet = [username, description, location, following, followers, totaltweets, retweetcount, text, hashtext, tweet_id]
        db.loc[len(db)] = ith_tweet

        if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
            array_tweets.append(remove_emoji(cleantText(ith_tweet[7])))
            array_tweets_id.append(ith_tweet[9])
            array_tweets_username.append(ith_tweet[0])
        
	# Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook('data/TwitterSentimentAnalysis.xlsx')
    worksheet = workbook.add_worksheet()

	# Add a bold format to use to highlight cells.
    bold = workbook.add_format({'bold': 1})

	# Adjust the column width.
    worksheet.set_column(0, 1, 20) 
    worksheet.set_column(2, 2, 280) 
    worksheet.set_column(3, 3, 10) 
	
	# Write some data headers.
    worksheet.write('A1', 'ID', bold)
    worksheet.write('B1', 'Username', bold)
    worksheet.write('C1', 'Tweet', bold)
    worksheet.write('D1', 'Sentiment', bold)

	# Start from the first cell below the headers.
    row = 1
    col = 0

    for i in array_tweets_id:
        worksheet.write_string(row, col,str(i))
        row += 1
    row = 1
    for i in array_tweets_username:
        worksheet.write_string(row, col+1,str(i))
        row += 1
    row = 1
    for i in array_tweets:
        worksheet.write_string(row, col+2,str(i))
        row += 1
    workbook.close()

customtkinter.set_appearance_mode("System")
root=customtkinter.CTk()
root.geometry("1000x600")
root.minsize(1000,600)
root.maxsize(1000,600)
root.configure()
root.title('Twitter Sentiment Analysis')
root.iconbitmap(r"image/ico/twitter.ico")

# Creating a variable that can be used to store the value of the entry box.
name_var=tk.StringVar()
date_since_var=tk.StringVar()
numtweet_var=tk.StringVar()

frame_1 = customtkinter.CTkFrame(master=root, width=1000, height=90)
frame_1.place(x=0, y=0)

frame_2 = customtkinter.CTkFrame(master=root, width=1000, height=471)
frame_2.place(x=0, y=90)

img = ImageTk.PhotoImage(Image.open("image/walpaper/analysis.jpg"))
# Create a Label Widget to display the text or Image
label = Label(frame_2, image = img, bg="black")
label.place(x=0, y=0)

def submit():
    name=name_var.get()
    date_since=date_since_var.get()
    numtweet=numtweet_var.get()
    name_var.set("")
    date_since_var.set("")
    numtweet_var.set("")
    scrape(name, date_since, int(numtweet))
    text_label["text"] = 'Scraping has completed!'
    machine_btn.state = "normal"

def homePage():
    root.destroy()
    import main

def nextPage():
    root.destroy()
    import page3

def start_training():
    import train
    text_label["text"] = ""
    accuracy_label["text"] = "Accuracy: {:0.2f}".format(train.accr1)
    graph_img = ImageTk.PhotoImage(Image.open("image/analysis/ConfusionMatrix5.jpg"))
    graph_label = Label(frame_2, image=graph_img)
    graph_label.place(x=0, y=0)

    graph2_img = ImageTk.PhotoImage(Image.open("image/analysis/ROC5.jpg"))
    graph2_label = Label(frame_2, image=graph2_img)
    graph2_label.place(x=500, y=0)
    label.destroy()
      
accuracy_label = tk.Label(frame_1, text="", font=('calibre',25, 'bold'), bg="#2e2e2e", fg="white")
text_label = tk.Label(frame_1, text="", font=('calibre',15, 'bold'), bg="#2e2e2e", fg="white")
  
# Creating a entry for input name using widget Entry.
name_entry = customtkinter.CTkEntry(frame_1,textvariable = name_var, width=350, placeholder_text="Twitter Account", placeholder_text_color="white")
date_since_entry = customtkinter.CTkEntry(frame_1,textvariable = date_since_var, width=350, placeholder_text="Since Date  (YYYY-MM-DD)", placeholder_text_color="white")
numtweet_entry = customtkinter.CTkEntry(frame_1,textvariable = numtweet_var, width=350, placeholder_text="Number of Tweets", placeholder_text_color="white")

# creating a button using the widget Button that will call the submit function
submit_btn=customtkinter.CTkButton(frame_1 ,text = 'Submit', command = submit, width=10, height=90, border_color="gray", fg_color="orange", text_color="black", hover_color="green")   
machine_btn=customtkinter.CTkButton(frame_1 ,text = 'Start Training', command = start_training, width=10, height=90, state="disabled", fg_color="orange", text_color="black", text_color_disabled="black", hover_color="green")
  
name_entry.place(x=0,y=0)
date_since_entry.place(x=0,y=30)
numtweet_entry.place(x=0,y=60)
accuracy_label.place(x=540, y=20)
text_label.place(x=535, y=30)
submit_btn.place(relx=0.35, rely=0.0)
machine_btn.place(relx=0.9, rely=0.0)

# Button
homepage = Button(root,
               text="Homepage", 
               font= ("Times bold", 14),
               command=homePage,
               width=46,
               height=1
)
homepage.place(x=0, y=561)

page3 = Button(root, 
               text="Next Page", 
               font = ("Times bold", 14),
               command=nextPage,
               width=45,
               height=1
)
page3.place(x=500,y=561)

root.mainloop()