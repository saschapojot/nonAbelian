import matplotlib.pyplot as plt
import pandas as pd

import numpy as np



name="example1"
T=1e3



inDir=name+"Evo/"

inEvolution=inDir+name+"drift.csv"
inIbc=inDir+"ibc.csv"

inEvoData=pd.read_csv(inEvolution)

xEvo=np.array(inEvoData.loc[:,"drift"])

QQ=len(xEvo)-1

tEvo=[T/QQ*qq for qq in range(0,QQ+1)]

inIbcData=pd.read_csv(inIbc)

xtopo=np.array(inIbcData.loc[:,"topological"])
xibc=np.array(inIbcData.loc[:,"ibc"])

xTopoIbc=xtopo+xibc
Q=len(xTopoIbc)-1

tIbc=[T/Q*q for q in range(0,Q+1)]

plt.figure()
plt.plot(tEvo,xEvo,color="black",label="numerical evolution")
plt.xlabel("$t$")
plt.ylabel("drift")
plt.title("$T=$"+str(T))
plt.scatter(tIbc,xTopoIbc,color="red",label="theory")
plt.legend(loc="best")
plt.savefig(inDir+"compare.png")

