import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfile
from tkinter import ttk, filedialog
from turtle import home
import pandas as pd
import tweepy
import re
  

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
    text = text.lstrip()
    text = text.rstrip()
    text = text.lstrip("_")
    return text

#def deEmojify(inputString):
#   return inputString.encode('ascii', 'ignore').decode('ascii')


# Enter your own credentials obtained
# from your developer account
consumer_key = "**************************"
consumer_secret = "***********************************"
access_key = "**********************************"
access_secret = "*************************************"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)


def scrape(words, date_since, numtweet):
    
    # Creating DataFrame using pandas
    db = pd.DataFrame(columns=['username', 'description', 'location', 'following', 'followers', 'totaltweets', 'retweetcount', 'text', 'hashtags', 'tweet_id'])
    
    # We are using .Cursor() to search through twitter for the required tweets.
    # The number of tweets can be restricted using .items(number of tweets)
    tweets = tweepy.Cursor(api.search_tweets, q=words, since=date_since, tweet_mode="extended").items(numtweet)
    # .Cursor() returns an iterable object. Each item in
    # the iterator has various attributes that you can access to
    # get information about each tweet
    list_tweets = [tweet for tweet in tweets]
    
    
    # Counter to maintain Tweet Count
    i = 1
    
    array_tweets=[]
    array_tweets_id=[]
    array_tweets_username = []
    array_tum_tweets = []
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

        
        # Retweets can be distinguished by a retweeted_status attribute,
        # in case it is an invalid reference, except block will be executed
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

        array_tum_tweets.append(ith_tweet[7]) #includes uncleaned and retweeted tweets

        if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
            array_tweets.append(cleantText(ith_tweet[7]))
            array_tweets_id.append(ith_tweet[9])
            array_tweets_username.append(ith_tweet[0])
        
        # Function call to print tweet data on screen
        #printtweetdata(i, ith_tweet)
        #i = i+1

    count = 0
    with open("data/uncleaned_tweets.txt", "w", encoding="utf-8", errors="ignore") as txt_file:
        for tivit in array_tum_tweets:
            count += 1
            txt_file.write(str(count) + "-) " + tivit + "\n")



    count = 0
    with open("data/tweets.txt", "w", encoding="utf-8", errors="ignore") as txt_file:
        for tivit in array_tweets:
            count += 1
            txt_file.write(str(count) + "-) " + tivit + "\n")

    count = 0
    with open("data/tweets_id.txt", "w", encoding="utf-8", errors="ignore") as txt_file:
        for tivit in array_tweets_id:
            count += 1
            txt_file.write(str(count) + "-) " + str(tivit) + "\n")

    count = 0
    with open("data/tweets_username.txt", "w", encoding="utf-8", errors="ignore") as txt_file:
        for tivit in array_tweets_username:
            count += 1
            txt_file.write(str(count) + "-) " + str(tivit) + "\n")

    
    # we will save our database as a CSV file.
    #db.to_csv(filename)


root=tk.Tk()
root.geometry("1000x600")
root.configure(background="#87F0FC")
root.title('Twitter Sentiment Analysis')
root.iconbitmap(r"image/twitter.ico")
  
# declaring string variable
# for storing name and password
name_var=tk.StringVar()
date_since_var=tk.StringVar()
numtweet_var=tk.StringVar()

# defining a function that will
# get the name and password and
# print them on the screen
def submit():
 
    name=name_var.get()
    date_since=date_since_var.get()
    numtweet=numtweet_var.get()
    name_var.set("")
    date_since_var.set("")
    numtweet_var.set("")
    scrape(name, date_since, int(numtweet))
    print('Scraping has completed!')

def homePage():
    root.destroy()
    import main

def nextPage():
    root.destroy()
    import page3
      
# creating a label for
# name using widget Label
name_label = tk.Label(root, text = 'Twitter Account: ', font=('calibre',10, 'bold'), bg="#87F0FC")
date_since_label = tk.Label(root, text = 'Since Date: ', font=('calibre',10, 'bold'), bg="#87F0FC")
numtweet_label = tk.Label(root, text = 'Number of Tweets: ', font=('calibre',10, 'bold'), bg="#87F0FC")
  
# creating a entry for input
# name using widget Entry
name_entry = tk.Entry(root,textvariable = name_var, font=('calibre',10,'normal'), width=35)
date_since_entry = tk.Entry(root,textvariable = date_since_var, font=('calibre',10,'normal'), width=35)
numtweet_entry = tk.Entry(root,textvariable = numtweet_var, font=('calibre',10,'normal'), width=35)

# creating a button using the widget
# Button that will call the submit function
submit_btn=tk.Button(root,text = 'Submit', command = submit, width=6, height=4, bd=1)
  
# placing the label and entry in
# the required position using grid
# method
name_label.grid(row=0,column=0)
name_entry.grid(row=0,column=1)

date_since_label.grid(row=1,column=0)
date_since_entry.grid(row=1,column=1)

numtweet_label.grid(row=2,column=0)
numtweet_entry.grid(row=2,column=1)

submit_btn.place(x=380, y=0)

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
page3.place(x=495,y=561)

  
# performing an infinite loop
# for the window to display
root.mainloop()