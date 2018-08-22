#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 20:07:35 2018

@author: dmitriy
"""
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
import random

df0 = pd.read_csv("books_uniq_weeks.csv", encoding="ISO-8859-1", 
                 usecols=["publisher","author","date","title","weeks_on_list"])
df = df0
df["date"] = pd.to_datetime(df["date"])
df["weeks_on_list"] = df["weeks_on_list"].astype(int)
df["wol"] = pd.to_numeric(df["weeks_on_list"])
print(df.head())
print(df.dtypes)
df["weeks_on_list"] = df.iloc[0:,-1].astype(str).replace("0","NaN")
df = df[~df["weeks_on_list"].astype(str).str.contains("NaN")]
df = df.reset_index(drop=True)
print(df.head())
dff = df.iloc[:100,:]
df2 = df.groupby(["author","weeks_on_list"]).count().reset_index()
df3 = df.groupby(["author","weeks_on_list"]).sum().reset_index()
print(df2.head())
print(df3.head())
df4 = df.groupby(["author","title"]).count().reset_index()
df5 = df.groupby(["author","title"]).sum().reset_index()
df4 = df4.drop_duplicates()
df5 = df5.drop_duplicates()
stew = df[df["author"] == "David Baldacci"].reset_index(drop=True)
print(df4.head())
print(df5.head())
print(stew)
data = [go.Bar(x=dff["author"],
               y=dff["weeks_on_list"],
               text=dff["weeks_on_list"],
               textposition = 'auto',
               hoverinfo = "text",
               marker=dict(
                   color='rgb(158,202,225)',
                   line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                    ),
                opacity=0.6          
)]
title = 'Author and weeks_on_list'
layout = go.Layout(
    title=title,
    xaxis=dict(
        title='Name of the book',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='How long book be in top',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    legend=dict(
        x=1.0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    bargap=0.15,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename=title+'.html', auto_open = False)

def pieplot(labels,values,title):
    #labels = k["authors"]    
    trace = go.Pie(labels=labels, values=values)    
    plotly.offline.init_notebook_mode(connected=True)    
    plotly.offline.plot({"data": [trace],
        "layout": go.Layout(title=title)
    }, auto_open=False, filename= str(title) + '.html')

pieplot(df["author"],df["weeks_on_list"],"Authors")
pieplot(df["publisher"],df["weeks_on_list"],"Publishers")
list = np.unique(df["author"], return_index=False, return_inverse=False, return_counts=False, axis=None)
list = list.tolist()
df.sort_values(["author"], ascending = True)
print(df.head())
#print(list)
print(len(list))
df6 = pd.DataFrame({"author":df2["author"],"counts":df3["weeks_on_list"]})
df6 = df6.sort_values(by="counts")
df6 = df6.reset_index(drop=True)
one = df.groupby("weeks_on_list").sum().reset_index()
two = df.groupby("author").count().reset_index()
three = df.groupby("author").sum().reset_index()
three = three.sort_values(by="wol",ascending=False).reset_index(drop=True)
print(one.head())
print(two.head())
print(three.head())
wow = three[three["author"] == "Stephen King"].reset_index(drop=True)
#avg = two["counts"]/three["counts"]
#print(avg)
df7 = pd.DataFrame({"author":two["author"],"counts":two["weeks_on_list"],"sum":three["wol"]})
df7 = df7.sort_values(by="counts")
wiw = df.groupby("author")["weeks_on_list"].agg("sum")
print(df6.head())
print(df7.head())
print(wow)
print(wiw.head())
print(df.columns)
print(df.info())
print(three)
threesmall = three.iloc[:49,:]
data = [go.Bar(x=threesmall["author"],
               y=threesmall["wol"],
               text=threesmall["wol"],
               textposition = 'auto',
               hoverinfo = "text",
               marker=dict(
                   color='rgb(158,202,225)',
                   line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                    ),
                opacity=0.6          
)]
title = 'Author and weeks_on_list'
layout = go.Layout(
    title=title,
    xaxis=dict(
        title='Name of the book',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='How long book from author be in top',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    bargap=0.15,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename=title+'.html', auto_open = False)
nopi = df0.groupby("author").sum().reset_index()
nopi = nopi.sort_values(by="wol",ascending=False).reset_index(drop=True)
wiwi = nopi.groupby(nopi["wol"]).count().reset_index()
counterone = np.sum(wiwi["author"])
print(counterone)
firstname =  []
lastname = []
authors = df["author"].tolist()
for i in range(len(authors)):
    f = authors[i].split(" ")[0]
    firstname.append(f)
for i in range(len(authors)):
    l = authors[i].split(" ")[-1]
    lastname.append(l)
print(firstname)
print(lastname)
wiwi["vol"] = round((wiwi["author"]/counterone)*100,2)
summa = np.sum(wiwi["vol"])
wiwi["cumvol"] = np.cumsum(wiwi["vol"])
wiwi["reverse"] = 100 - wiwi["cumvol"]
print(nopi)
print(summa)
print(wiwi)
nopi = df0.groupby("title").sum().reset_index()
nopi = nopi.sort_values(by="wol",ascending=False).reset_index(drop=True)
wiwi = nopi.groupby(nopi["wol"]).count().reset_index()
counterone = np.sum(wiwi["title"])
print(counterone)
wiwi["vol"] = round((wiwi["title"]/counterone)*100,2)
summa = np.sum(wiwi["vol"])
wiwi["cumvol"] = np.cumsum(wiwi["vol"])
wiwi["reverse"] = 100 - wiwi["cumvol"]
print(nopi)
print(summa)
print(wiwi)
nopi = df0.groupby("publisher").sum().reset_index()
nopi = nopi.sort_values(by="wol",ascending=False).reset_index(drop=True)
wiwi = nopi.groupby(nopi["wol"]).count().reset_index()
counterone = np.sum(wiwi["publisher"])
print(counterone)
wiwi["vol"] = round((wiwi["publisher"]/counterone)*100,2)
summa = np.sum(wiwi["vol"])
wiwi["cumvol"] = np.cumsum(wiwi["vol"])
wiwi["reverse"] = 100 - wiwi["cumvol"]
print(nopi)
print(summa)
print(wiwi)
domi = df0.groupby(df0["publisher"]).sum().reset_index()
domi = domi.sort_values(by="wol",ascending=False).reset_index(drop=True)
domi = domi[domi["wol"] >= 100]
print(domi)
def name(N):
    for i in range(N):
        first = random.sample(set(firstname), 1)
        last = random.sample(set(lastname), 1)
        print("Your name should be " + str(first) + " " + str(last))
name(10)

df1 = df0.loc[~df0["wol"].isin(["0","1","2"])]
print(df1)
list0 = df0["title"].tolist()
list1 = df["title"].tolist()
list10 = df1["title"].tolist()
print(len(list0))
print(len(list1))
print(len(list10))
bw0 = []
list0 = [bw0.extend(i.split(' ')[:]) for i in list0] 
print(bw0)
bw1 = []
list1 = [bw1.extend(i.split(' ')[:]) for i in list1] 
print(bw1)
bw10 = []
list10 = [bw10.extend(i.split(' ')[:]) for i in list10]
print(bw10)
import nltk
nltk.download("stopwords")
sw = nltk.corpus.stopwords.words('english')


# Printing out the first eight stop words
print(sw[0:8])
print(len(sw))
# A new list to hold Moby Dick with No Stop words
words_ns0 = []

# Appending to words_ns all words that are in words but not in sw
for word in bw0:
    if word not in sw:
        words_ns0.append(word)

# Printing the first 5 words_ns to check that stop words are gone
print(words_ns0[0:5])

#%matplotlib inline

# Creating the word frequency distribution
freqdist0 = nltk.FreqDist(words_ns0)

thelist0 = words_ns0

thefile0 = open('test0.txt', 'w')

for item in thelist0:
  thefile0.write("%s\n" % item)
# Plotting the word frequency distribution
freqdist0.plot(25, cumulative=False)
#freqdist.hapaxes()

thefile0.close()

#days_file = open('test0.txt','r+')

#print(days_file.read())
#print(days_file.readline())
#print(days_file)
#days_file = str(days_file)
# A new list to hold Moby Dick with No Stop words
words_ns1 = []

# Appending to words_ns all words that are in words but not in sw
for word in bw1:
    if word not in sw:
        words_ns1.append(word)

# Printing the first 5 words_ns to check that stop words are gone
print(words_ns1[0:5])

#%matplotlib inline

# Creating the word frequency distribution
freqdist1 = nltk.FreqDist(words_ns1)

thelist1 = words_ns1

thefile1 = open('test1.txt', 'w')

for item in thelist1:
  thefile1.write("%s\n" % item)
# Plotting the word frequency distribution
freqdist1.plot(25, cumulative=False)
#freqdist.hapaxes()

thefile1.close()

#days_file1 = open('test1.txt','r+')

#print(days_file.read())
#print(days_file.readline())
#print(days_file)
#days_file = str(days_file)
# A new list to hold Moby Dick with No Stop words
words_ns10 = []

# Appending to words_ns all words that are in words but not in sw
for word in bw10:
    if word not in sw:
        words_ns10.append(word)

# Printing the first 5 words_ns to check that stop words are gone
print(words_ns10[0:5])

#%matplotlib inline

# Creating the word frequency distribution
freqdist10 = nltk.FreqDist(words_ns10)

thelist10 = words_ns10

thefile10 = open('test10.txt', 'w')

for item in thelist10:
  thefile10.write("%s\n" % item)
# Plotting the word frequency distribution
freqdist10.plot(25, cumulative=False)
#freqdist.hapaxes()

thefile10.close()

#days_file = open('test10.txt','r+')
#
#print(days_file.read())
#print(days_file.readline())
#print(days_file)
#days_file = str(days_file)
from textgenrnn import textgenrnn
t = textgenrnn()
t.train_from_file("test0.txt", num_epochs = 1, gen_epochs = 0) 
#                  batchsize = 256, train_size = 0.8, dropout = 0.2, word_level=True, set_validation=False)
t.generate_samples(temperatures=[0.2, 0.5, 0.8, 1.2, 1.5])
t.generate_to_file('textgenrnn_texts.txt', n=5)
t.save('name.hdf5')
lenpub = df["publisher"].tolist()
lenpub1 = []
for i in range(len(lenpub)):
    k = lenpub[i].split(" ")
    lenpub1.append(k)
lenpub1 = [y for x in lenpub for y in x.split(' ')]
print(lenpub[:20])
print(len(lenpub1))
lenbook = df["title"].tolist()
lenbook1 = []
for i in range(len(lenbook)):
    k = lenbook[i].split(" ")
    lenbook1.append(k)
lenbook1 = [y for x in lenbook for y in x.split(' ')]
print(lenbook[:20])
print(len(lenbook1))
a = len(lenbook1)
print(a)
b = len(df["title"])
print(b)
c = a/b
print(c)
length = int(round(len(lenbook1)/len(df["title"]),0))
length2 = int(round(len(lenpub1)/len(df["title"]),0))
print(length)
print(length2)
def name(N):
    list = []
    for i in range(N):
        k = i + 1
        first = [*random.sample(set(firstname), 1)]
        last = [*random.sample(set(lastname), 1)]
        book = [*random.sample(set(lenbook1), length)]
        publisher = [*random.sample(set(lenpub), 1)]
        tup = [str(k)," Your name should be " , str(first[0]) , " " , str(last[0]) , " " , 
              "and you are author of " , str(book[0]) , " " , str(book[1]) , " "
              , str(book[2]) , ". " , 
              "This book was published by " , str(publisher[0]) , "."," ","\n"]
        print(tup)
        list.extend(tup)
    print(list)
#    file = open('dreams.txt', 'w')

#    for item in list:
#        file.write("%s\n" % item)
    b = ''.join(list)
    print(b)
#    file.write(b)
#    file.close()
    np.savetxt('dreams.txt', [b], fmt='%s')
name(10)
check = open('dreams.txt', 'r', encoding='UTF-8')
dreams = check.read()
print(dreams)
check.close()