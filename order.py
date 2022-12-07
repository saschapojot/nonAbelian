import pandas as pd
import mpmath
from mpmath import mp
import numpy as np
mp.dps=20




def intVal(A):

    f=lambda x,y: x**2*mpmath.sin(y)*y*mpmath.sin(A*(mpmath.exp(x+y*mpmath.cos(y))*y**2))

    return mpmath.quad(f,[0,1],[0,1],maxdegree=10)

AValsAll=[1,10,50,100,500,1000]

intValsAll=[intVal(A) for A in AValsAll]

outData=np.array([AValsAll,intValsAll]).T

pdOut=pd.DataFrame(data=outData,columns=["A","int"])
pdOut.to_csv("order.csv")