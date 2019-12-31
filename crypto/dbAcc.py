#!/usr/bin/env python3

from numpy import genfromtxt
import tensorflow as tf
import numpy as np
import pymongo
import matplotlib.pyplot as plt
import getpass
from pprint import pprint


def isIndice(ticker: dict):
    if ticker.get("tag"):
        return False
    return True


pwd = "891563"  # getpass.getpass("Pass:")
isHome = "y"  # input("Doma? (y/n):")
uname = "stanke"
ipLocal = "192.168.1.100"
ipHome = "188.121.161.140"
regenData = input("Regenerate new dataset? (y/n): ").upper() == "Y"

print("Connecting to database...")

mongo = pymongo.MongoClient(ipLocal if isHome == "y" else ipHome)

db = mongo.crypto
col = db.ticks
db.authenticate(
        name=uname,
        password=pwd
    )

smpl = col.find_one()
futuresOnly = [future if not isIndice(future) else None for future in smpl["tickers"]]
while None in futuresOnly:
    futuresOnly.remove(None)

symbols = [future["symbol"] for future in futuresOnly]

print("Connected. Downloading data from DB...")

lTickers = len(smpl["tickers"])
lFutures = len(futuresOnly)

lEntries = col.estimated_document_count()
attrs = [
        "open24h",
        "markPrice",
        "last",
        "vol24h",
        "ask",
        "lastSize",
        "askSize",
        "bidSize",
        "openInterest",
        "bid",
    ]
prices = [
        "last",
        "bid",
        "ask",
    ]

lAttrs = len(attrs)

# create 3-D array
#   1st.D are time steps
#   2nd.D are futures
#   3rd.D are values for attributes
history = np.ndarray((lEntries, lFutures, lAttrs))
timeline = np.ndarray((lEntries))
it = col.find()

print(f"Got {lEntries} time steps for {lFutures} futures with {lAttrs} attributes. Parsing data...")

if regenData:
    for i in range(lEntries):
        tick = it.next()
        for j in range(lFutures):
            future = tick["tickers"][j]
            for k in range(lAttrs):
                history[i][j][k] = future[attrs[k]]

        timeline[i] = tick["_id"]

    np.save("data/history", history)
    np.save("data/timeline", timeline)
else:
    history = np.load("data/history.npy")
    timeline = np.load("data/timeline.npy")

print(f"RAM footprint: {memoryview(history).nbytes/2**20}MiB")

fig, axs = plt.subplots(int(np.ceil(lAttrs/2)), 2, sharex=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0)

fig.suptitle(symbols[0])

for i in range(lAttrs):
    # axs[int(i/2)][int(i%2)].set_title(attrs[i])
    axs[int(i / 2)][int(i % 2)].set_ylabel(attrs[i])
    axs[int(i / 2)][int(i % 2)].plot(timeline, history[:, 0, i])

# plt.legend()
plt.show()
