from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
from tkinter import *
from tkinter import messagebox
import random

data = loadarff("./hepatitis.arff")
df = pd.DataFrame(data[0])

### DECODE BYTES TO STRING
for i in range(1,13):
    df.iloc[:, i] = df.iloc[:, i].apply(lambda x : x.decode('utf-8'), 1)

df.iloc[:, 18] = df.iloc[:, 18].apply(lambda x : x.decode('utf-8'), 1)
df.iloc[:, 19] = df.iloc[:, 19].apply(lambda x : x.decode('utf-8'), 1)

### REPLACE ? WITH OTHER STRING
def str_repl(str, weights = None):
    if str == '?':
        return random.choices(["yes", "no"], weights = weights)[0]
    else:
        return str

for i in range(2,13):
        row = df.iloc[:, i]
        y = sum(row.str.count("yes"))
        n = sum(row.str.count("no"))
        weights = [y/(y+n), n/(y+n)]
        df.iloc[:, i] = df.iloc[:, i].apply(lambda x : str_repl(x, weights = weights), 1)

row = df.iloc[:, 18]
y = sum(row.str.count("yes"))
n = sum(row.str.count("no"))
weights = [y/(y+n), n/(y+n)]
df.iloc[:, 18] = df.iloc[:, 18].apply(lambda x : str_repl(x, weights = weights), 1)

### CLEAR NAN
df.iloc[:, 0] = df.iloc[:,0].fillna(0)
for i in range(13,18):
    df.iloc[:, i] = df.iloc[:,i].fillna(0)

### GUI STARTS
window = Tk()

window.title("Welcome to this DSS")

window.geometry('300x800')

lbl = Label(window, text="Age")
lbl.grid(column=0, row=0)
age = Entry(window,width=10)
age.grid(column=1, row=0)

lbl = Label(window, text="Sex")
lbl.grid(column=0, row=1)
sex = Entry(window,width=10)
sex.grid(column=1, row=1)

lbl = Label(window, text="Steroid")
lbl.grid(column=0, row=2)
ster = Entry(window,width=10)
ster.grid(column=1, row=2)

lbl = Label(window, text="Antivirals")
lbl.grid(column=0, row=3)
anti = Entry(window,width=10)
anti.grid(column=1, row=3)

lbl = Label(window, text="Fatigue")
lbl.grid(column=0, row=4)
fat = Entry(window,width=10)
fat.grid(column=1, row=4)

lbl = Label(window, text="Malaise")
lbl.grid(column=0, row=5)
mal = Entry(window,width=10)
mal.grid(column=1, row=5)

lbl = Label(window, text="Anorexia")
lbl.grid(column=0, row=6)
ano = Entry(window,width=10)
ano.grid(column=1, row=6)

lbl = Label(window, text="Liver Big")
lbl.grid(column=0, row=7)
livb = Entry(window,width=10)
livb.grid(column=1, row=7)

lbl = Label(window, text="Liver Firm")
lbl.grid(column=0, row=8)
livf = Entry(window,width=10)
livf.grid(column=1, row=8)

lbl = Label(window, text="Spleen Palpable")
lbl.grid(column=0, row=9)
spl = Entry(window,width=10)
spl.grid(column=1, row=9)

lbl = Label(window, text="Spiders")
lbl.grid(column=0, row=10)
spi = Entry(window,width=10)
spi.grid(column=1, row=10)

lbl = Label(window, text="Ascites")
lbl.grid(column=0, row=11)
asc = Entry(window,width=10)
asc.grid(column=1, row=11)

lbl = Label(window, text="Varices")
lbl.grid(column=0, row=12)
var = Entry(window,width=10)
var.grid(column=1, row=12)

lbl = Label(window, text="Bilirubin")
lbl.grid(column=0, row=13)
bil = Entry(window,width=10)
bil.grid(column=1, row=13)

lbl = Label(window, text="Alk Phosphate")
lbl.grid(column=0, row=14)
alk = Entry(window,width=10)
alk.grid(column=1, row=14)

lbl = Label(window, text="SGOT")
lbl.grid(column=0, row=15)
sgo = Entry(window,width=10)
sgo.grid(column=1, row=15)

lbl = Label(window, text="Albumin")
lbl.grid(column=0, row=16)
alb = Entry(window,width=10)
alb.grid(column=1, row=16)

lbl = Label(window, text="Protime")
lbl.grid(column=0, row=17)
pro = Entry(window,width=10)
pro.grid(column=1, row=17)

lbl = Label(window, text="Histology")
lbl.grid(column=0, row=18)
hist = Entry(window,width=10)
hist.grid(column=1, row=18)

def clicked():
    vals = [age,sex,ster,anti,fat,mal,ano,livb,livf,spl,spi,asc,var,bil,alk,sgo,alb,pro,hist]
    vals = list(map(lambda x : x.get(), vals))
    vals.append('LIVE')
 #    vals = ['30','male','no','no','no', 'no','no','no', 'no', 'no','no', 'no',
 # 'no',
 # '1',
 # '85',
 # '18',
 # '4',
 # '54',
 # 'no', 'LIVE']

    df2 = df.append(pd.DataFrame(np.array(vals).reshape(1,-1), columns = df.columns)).reset_index(drop=True)

    ### TRANSFORM DATA
    lb = preprocessing.LabelBinarizer()
    for i in range(1,13):
            df2.iloc[:, i] = lb.fit_transform(df2.iloc[:, i])

    df2.iloc[:, 18] = lb.fit_transform(df2.iloc[:, 18])
    df2.iloc[:, 19] = lb.fit_transform(df2.iloc[:, 19])

    df2, ex = df2.iloc[0:-1, :], df2.iloc[-1,0:-1]


    ### NAIVE BAYES
    X = df2.loc[:, "AGE":"HISTOLOGY"]
    y = df2.loc[:, "Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    results = clf.predict(np.array(list(map(float, ex.to_list()))).reshape(1,-1))
    if results == 1:
        result = "Live"
    else:
        result = "Die"

    messagebox.showinfo('Hepatitis Prediction:', f"{result}")


btn = Button(window, text="Submit", command=clicked)
btn.grid(column=1, row=20)

window.mainloop()
