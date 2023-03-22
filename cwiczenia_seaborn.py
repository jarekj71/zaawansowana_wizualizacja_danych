#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:36:10 2022

@author: jarekj
"""

#%%
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
import numpy as np

#%% Ćwiczenie 1 Seaborn vs matplotlib

flights = sns.load_dataset("flights")
flights.head()
sns.set_style("white")
sns.lineplot(data=flights, x="year", y="passengers", hue="month")


#%% 

fw = flights.pivot(index="year", columns="month", values="passengers") # pivot to metoda obiektu pandas

#%%
plt.style.use("seaborn-white")
colors = sns.color_palette("husl", 12)
fig,ax = plt.subplots()
for column,color in zip(fw.columns,colors):
    ax.plot(fw.index,fw.loc[:,column],label=column,c=color)
ax.set_xlabel("year")
ax.set_ylabel("passengers")
ax.legend()

#%%
plt.style.use("default")
sns.set_style("white")
plt.style.use("seaborn-white")


fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(12,6))
sns.lineplot(data=flights, x="year", y="passengers", hue="month",ax=ax1)
for column,color in zip(fw.columns,colors):
    ax2.plot(fw.index,fw.loc[:,column],label=column,c=color)
ax2.set_xlabel("year")
ax2.set_ylabel("passengers")
ax2.legend(title="month")

#%%
fig.savefig("cw1.pdf")
#%% Ćwiczenie 2  Boxplots
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(12,6))
sns.boxplot(data=fw,ax=ax1)
sns.boxplot(data=fw,ax=ax2,palette="hsv")

#%%
colors = cm.get_cmap("hsv") 
colorlist= colors(np.linspace(0,1,14))[1:13] # geoernowanie kolorów oparte o linspace trochę zawiłe
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(12,6))
sns.boxplot(data=fw,ax=ax1,palette="hsv",saturation=1) # aby uzyskać identyczne kolory trzeba zmienić saturation, w przeciwną stronę jest to dużo trudniejsze

b = ax2.boxplot(fw,labels=fw.columns,patch_artist=True,widths=0.8) # zapis do obiektu dict
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
     plt.setp(b[element], color="#444444",lw=1.5)
for patch,color in zip(b['boxes'],colorlist):
    patch.set(facecolor=color)
ax2.set_xlabel("month")

fig.savefig("cw2.pdf")
#%% Ćwiczenie 3 Wykresy warstwowe

dm = sns.load_dataset("diamonds")
p = sns.histplot(dm, x="cut",hue="color",multiple="fill",alpha=1) # tu dla odmiany wyrównanie kolorów przy pomocy parametru alpha

#%%


total = dm.groupby("cut").count().carat
height = dm[dm.color=="E"].groupby("cut").count().carat/total # potrzebujemy tylko jedną kolumnę

#%%
plt.style.use("default")
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(12,6))

height = np.ones(len(dm.cut.unique()))
for color in dm.color.unique().sort_values():
    ax2.bar(x=total.index,height=height,width=1,edgecolor="black",label=color,align="edge")
    height = height - dm[dm.color==color].groupby("cut").count().carat/total 
ax2.set_ylim(0,1)
ax2.set_xlim(0,5)
ax2.legend(title="color",loc="center left")
ax2.set_xlabel("cut")
ax2.set_ylabel("Count")

fig.savefig("cw3.pdf")
#%%  opcja alternatywna bez nakładania prostokątów

bottom = np.zeros(len(dm.cut.unique()))
for color in dm.color.unique().sort_values():
    height = dm[dm.color==color].groupby("cut").count().carat/total 
    ax2.bar(x=total.index,bottom=bottom,height=height,width=1,edgecolor="black",label=color,align="edge")
    bottom = bottom + height


#%% ćwiczenie 4 żłożony wykres seaborn


planets = sns.load_dataset("planets")
cmap = cm.get_cmap('gist_earth')
from sklearn.preprocessing import minmax_scale

#%% FIGURE LEVEL
g = sns.relplot(
    data=planets,
    x="distance", y="orbital_period",
    hue="year", size="mass",
    palette=cmap, sizes=(10, 200),
)
g.set(xscale="log", yscale="log")
g.ax.xaxis.grid(True, "minor", linewidth=.25)
g.ax.yaxis.grid(True, "minor", linewidth=.25)
g.ax.xaxis.grid(True, "major", linewidth=1)
g.ax.yaxis.grid(True, "major", linewidth=1)
g.despine(left=True, bottom=True)
g.savefig("cw4.pdf")
#%% mapowanie i skalowanie danych
selector = ~planets.isnull().any(axis=1)
p = planets[selector]
x = p.distance
y = p.orbital_period
color = p.year
size = minmax_scale(p.mass,feature_range=(10,200))
#%%
fig,ax1 = plt.subplots(figsize=(8,8))
ax1.scatter(x,y,s=size,cmap=cmap,c=color)
ax1.loglog() # alternatywa
#ax1.set_xscale("log")
#ax1.set_yscale("log")
ax1.grid(visible=True,which='major',lw=1)
ax1.grid(visible=True,which='minor',lw=.25)

#despine
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.set_xlabel("distance")
ax1.set_ylabel("orbital_period")

#legenda
ax1.scatter(x=[],y=[],s=0,label="$\\bf{year}$")
plt.legend()
for cl,label in zip([0.1,0.3,0.5,0.7,0.9],[1990,1995,2000,2005,2010]):
    ax1.scatter(x=[],y=[],color=cmap(cl),label="{}".format(label))
plt.legend()


values = [0,5,10,15,20,25]
scaled = minmax_scale(values,feature_range=(10,200))

ax1.scatter(x=[],y=[],s=0,label="$\\bf{mass}$")
plt.legend()
for v,s in zip(values[1:],scaled[1:]):
    ax1.scatter(x=[],y=[],s=s,label="{}".format(v),c="#000000")
plt.legend()

h,l = ax1.get_legend_handles_labels()
#ph = plt.plot([],marker="", ls="")[0]
#h.insert(0,ph)
#h.insert(6,ph)
#l.insert(0,"$\\bf{year}$")
#l.insert(6,"$\\bf{mass}$")

ax1.legend(h,l,bbox_to_anchor=(1.2, 0.5),loc='center right',frameon=False)

#%%
