import pandas as pd
import numpy as np
from plotly.plotly import iplot, sign_in
import plotly.graph_objs as go
from load_data import artist_mean_rate

with open("../data/key.txt", "r") as file:
    key = file.readline()[:-1]

sign_in("ger94", key)
plot_mean = artist_mean_rate(20e3)

fig = {"data":[{
       "x": plot_mean.count_ratings, 
       "y": plot_mean.mean_rating,
       "text": plot_mean.artist,
       "mode": "markers"
    }],
       "layout": go.Layout(xaxis=go.XAxis(title="Number Ratings"),
                           yaxis=go.YAxis(title="Mean Rating"),
                           title="Per Band Rating")
       }

iplot(fig, filename = "music_ratings")
