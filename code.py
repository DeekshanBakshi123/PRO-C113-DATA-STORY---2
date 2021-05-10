import pandas as pd
import statistics
import plotly.express as px


from google.colab import files
dataLoad = files.upload()

df = pd.read_csv('data.csv')
fig = px.scatter(df, y = 'quant_saved', color = 'rem_any')
fig.show()

import csv
import plotly.graph_objects as pg

with open('data.csv', newline = '') as f:
    reader  = csv.reader(f)
    data = list(reader)

data.pop(0)


totalEntries = len(data)

reminder = 0

for i in data:
    if int(i[3]) == 1:
        reminder += 1

fig = pg.Figure(pg.Bar(x = ['reminded', 'not reminded'], y = [reminder, totalEntries - reminder]))

fig.show()


allSavings = []

for a in data:
    allSavings.append(float(a[0]))

mean = statistics.mean(allSavings)
print(mean)

median = statistics.median(allSavings)
print(median)

mode = statistics.mode(allSavings)
print(mode)


notReminded = []
reminded = []

for b in data:
    if int(b[3]) == 1:
        reminded.append(float(b[0]))
    else:
        notReminded.append(float(b[0]))

print("Reminded Peoples Data")
print("Mean", statistics.mean(reminded))
print("Median", statistics.median(reminded))
print("Mode", statistics.mode(reminded))

print("Not Reminded Peoples Data")
print("Mean", statistics.mean(notReminded))
print("Median", statistics.median(notReminded))
print("Mode", statistics.mode(notReminded))



stdAll = statistics.stdev(allSavings)
print("Std Of All", stdAll)

stdRem = statistics.stdev(reminded)
print("Std Of Rem", stdRem)

stdNot = statistics.stdev(notReminded)
print("Std Of Not Rem", stdNot)


import numpy as np

age = []
savings = []

for c in data:
    if float(c[5]) != 0:
        age.append(float(c[5]))
        savings.append(float(c[0]))
cor = np.corrcoef(age, savings)
print("Correlation between age and saving is ", {cor[0,1]})


import plotly.figure_factory as pf

fig = pf.create_distplot([df['quant_saved'].tolist()], ['savings'], show_hist = False)
fig.show()

import seaborn as sb
sb.boxplot(data = df, x = df['quant_saved'])

q1 = df['quant_saved'].quantile(.25)
q3 = df['quant_saved'].quantile(.75)
iqr = q3 - q1
print("Q1",q1)
print("Q3", q3)
print("Iqr", iqr)
lowerRange = q1 - 1.5 * iqr
upperRange = q3 + 1.5 * iqr
print("Lower Range", lowerRange)
print("Upper Range", upperRange)

newdf = df[df['quant_saved'] < upperRange]

all_savings = newdf['quant_saved'].tolist()

print("Mean of New Df", statistics.mean(all_savings))
print("Median of New Df", statistics.median(all_savings))
print("Mode of New Df", statistics.mode(all_savings))
print("Std of New Df", statistics.stdev(all_savings))

fig1 = pf.create_distplot([newdf['quant_saved'].tolist()],  ['Savings'], show_hist = False)
fig1.show()

import random 

samplingMeanList = []

for i in range(1000):
    temp_list = []
    for a in range(100):
        temp_list.append(random.choice(all_savings))
    samplingMeanList.append(statistics.mean(temp_list))

meanSampling = statistics.mean(samplingMeanList)
print("Std Of Sampling Data", statistics.stdev(samplingMeanList))
print("Mean of Sampling Data", meanSampling)

fig2 = pf.create_distplot([samplingMeanList], ["Savings For Sampling"], show_hist = False)
fig2.add_trace(pg.Scatter(x = [meanSampling, meanSampling], y = [0,0.1], mode = 'lines', name = 'Sampling Mean'))
fig2.show()

tempDf = newdf[newdf.age != 0]
age1 = tempDf['age'].tolist()
savings1 = tempDf['quant_saved'].tolist()

corr = np.corrcoef(age1, savings1)
print("Corelation of Age And Savings of Sampling Data is", {corr[0,1]})

reminded_df = newdf.loc[newdf['rem_any'] == 1]
notreminded_df = newdf.loc[newdf['rem_any'] == 0]
print(reminded_df.head())
print(notreminded_df.head())

fig3 = pf.create_distplot([notreminded_df['quant_saved'].tolist()],['Not Reminded Savings'], show_hist = False)
fig3.show()

notRemindedSavings = notreminded_df['quant_saved'].tolist()
notRemindedMeanList = []

for x in range(1000):
    newList = []
    for c in range(100):
        newList.append(random.choice(notRemindedSavings))
    notRemindedMeanList.append(statistics.mean(newList))

notRemindedSamplingMean = statistics.mean(notRemindedMeanList)
notRemindedSamplingStd = statistics.stdev(notRemindedMeanList)

print("Sampling Mean of Not Reminded", notRemindedSamplingMean)
print("Sampling Std of Not Reminded", notRemindedSamplingStd)

fig4 = pf.create_distplot([notRemindedMeanList], ["Savings For The Sampling Of Not Reminded"], show_hist = False)
fig4.show()

firstStdStart, firstStdEnd = notRemindedSamplingMean - notRemindedSamplingStd, notRemindedSamplingMean + notRemindedSamplingStd 
secondStdStart, secondStdEnd = notRemindedSamplingMean - 2* notRemindedSamplingStd, notRemindedSamplingMean + 2 * notRemindedSamplingStd
thirdStdStart, thirdStdEnd = notRemindedSamplingMean - 3 * notRemindedSamplingStd, notRemindedSamplingMean + 3 * notRemindedSamplingStd

print(firstStdStart, firstStdEnd)
print(secondStdStart, secondStdEnd)
print(thirdStdStart, thirdStdEnd)

RemindedSavings = reminded_df['quant_saved'].tolist()
RemindedMeanList = []

for x in range(1000):
    newList = []
    for c in range(100):
        newList.append(random.choice(RemindedSavings))
    RemindedMeanList.append(statistics.mean(newList))

RemindedSamplingMean = statistics.mean(RemindedMeanList)
RemindedSamplingStd = statistics.stdev(RemindedMeanList)

print("Sampling Mean of Reminded", RemindedSamplingMean)
print("Sampling Std of Reminded", RemindedSamplingStd)

fig5 = pf.create_distplot([RemindedMeanList], ["Savings For The Sampling Of Reminded"], show_hist = False)
fig5.show()

Zscore = (RemindedSamplingMean - notRemindedSamplingMean) / notRemindedSamplingStd
print("ZScore is", Zscore)
