import numpy as np
import pandas as pd
from scipy.linalg import expm
from multiprocessing import Pool
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import sympy as sp
#this script computes evolution using Schrodinger equation

r=1
alpha1=np.pi/4
alpha2=np.pi*np.sqrt(2)

T=1e3*2

QQ=int(1e5)

dt=T/QQ

N=500#momentum values
kValsAll=np.array([2*np.pi/N*n for n in range(0,N)])

name="example12"
#also see example 10


####################################symbolic computation
###########functions and time derivatives
from sympy import pi as PI
tS=sp.symbols("t",cls=sp.Symbol)
jcS=r*sp.cos(2*PI*tS/T)
jdS=r*sp.sin(2*PI*tS/T)*sp.cos(alpha1+2*PI*tS/T)
jb1S=r*sp.sin(2*PI*tS/T)*sp.sin(alpha1+2*PI*tS/T)*sp.cos(alpha2+2*PI*tS/T)
jb2S=r*sp.sin(2*PI*tS/T)*sp.sin(alpha1+2*PI*tS/T)*sp.sin(alpha2+2*PI*tS/T)
# dtJcSymbol=sp.diff(jcS,tS)
# dtJdSymbol=sp.diff(jdS,tS)
# dtJb1Symbol=sp.diff(jb1S,tS)
# dtJb2Symbol=sp.diff(jb2S,tS)
#########lambdify
###
jcLambdified=sp.lambdify(tS,jcS,"numpy")
jdLambdified=sp.lambdify(tS,jdS,"numpy")
jb1Lambdified=sp.lambdify(tS,jb1S,"numpy")
jb2Lambdified=sp.lambdify(tS,jb2S,"numpy")
###
# dtJcLambdified=sp.lambdify(tS,dtJcSymbol,"numpy")
# dtJdLambdified=sp.lambdify(tS,dtJdSymbol,"numpy")
# dtJb1Lambdified=sp.lambdify(tS,dtJb1Symbol,"numpy")
# dtJb2Lambdified=sp.lambdify(tS,dtJb2Symbol,"numpy")
##

def jcExpression(t):
    return jcLambdified(t)

def jdExpression(t):
    return jdLambdified(t)


def jb1Expression(t):
    return jb1Lambdified(t)

def jb2Expression(t):
    return jb2Lambdified(t)

def jbExpression(k,t):
    return jb1Expression(t)+np.exp(1j*k)*jb2Expression(t)

def Delta(k,t):
    return np.sqrt(np.abs(jbExpression(k,t))**2+jcExpression(t)**2+jdExpression(t)**2)


def delta(t):
    return np.sqrt(jcExpression(t)**2+jdExpression(t)**2)


#initial coefficients
#populate subspace 1 and 2
cTensor=np.ones((N,4),dtype=complex)
cTensor[:,0]*=0
cTensor[:,1]*=0

R0=N/2
for n in range(0,N):
    cTensor[n,:]*=np.exp(-1j*kValsAll[n]*R0)

#norm
# nm=np.sqrt(np.real(np.sum(np.conj(cTensor)*cTensor)))
nm=np.linalg.norm(cTensor,"fro")
cTensor/=nm

def psik0(n):
    """

    :param n:
    :return: init vector for momentum value kn
    """

    kTmp=kValsAll[n]
    tTmp=0
    JbVal=jbExpression(kTmp,tTmp)
    JcVal=jcExpression(tTmp)
    JdVal=jdExpression(tTmp)
    DeltaVal=Delta(kTmp,tTmp)
    deltaVal=delta(tTmp)

    psi00=np.array([-1/np.sqrt(2),np.conj(JbVal)/(np.sqrt(2)*DeltaVal),
                    JcVal/(np.sqrt(2)*DeltaVal),JdVal/(np.sqrt(2)*DeltaVal)])
    psi10=np.array([0,0,-JdVal/deltaVal,JcVal/deltaVal])

    psi11=np.array([0,deltaVal/DeltaVal,-JbVal*JcVal/(deltaVal*DeltaVal),-JbVal*JdVal/(deltaVal*DeltaVal)])

    psi20=np.array([1/np.sqrt(2),np.conj(JbVal)/(np.sqrt(2)*DeltaVal),
                    JcVal/(np.sqrt(2)*DeltaVal),JdVal/(np.sqrt(2)*DeltaVal)])

    ck=cTensor[n,:]

    initVec=ck[0]*psi00+ck[1]*psi10+ck[2]*psi11+ck[3]*psi20

    return initVec


def HkMat(k,t):
    JbVal=jbExpression(k,t)
    JcVal=jcExpression(t)
    JdVal=jdExpression(t)

    retMat=np.zeros((4,4),dtype=complex)

    retMat[0,1]=JbVal
    retMat[0,2]=JcVal
    retMat[0,3]=JdVal

    retMat[1,0]=np.conj(JbVal)
    retMat[2,0]=JcVal
    retMat[3,0]=JdVal

    return retMat



def expH(nqq):
    n,qq=nqq
    k=kValsAll[n]
    t=dt*qq
    return [n,qq,expm(-1j*dt*HkMat(k,t))]


expHTensor=np.zeros((N,QQ,4,4),dtype=complex)

inExpHInds=[[n,qq] for n in range(0,N) for qq in range(0,QQ)]

procNum=48

pool0=Pool(procNum)

tExpHStart=datetime.now()

retExpH=pool0.map(expH,inExpHInds)

tExpHEnd=datetime.now()

print("exp H time: ",tExpHEnd-tExpHStart)

for item in retExpH:
    n,qq,tmp=item
    expHTensor[n,qq,:,:]=np.copy(tmp)

tInitPsiStart=datetime.now()
psi0Tensor=np.zeros((N,4),dtype=complex)
for n in range(0,N):
    vec=psik0(n)
    psi0Tensor[n,:]=np.copy(vec)

tInitPsiEnd=datetime.now()

print("init psi time: ",tInitPsiEnd-tInitPsiStart)


sol00=np.zeros((N,QQ+1),dtype=complex)
sol10=np.zeros((N,QQ+1),dtype=complex)
sol11=np.zeros((N,QQ+1),dtype=complex)
sol20=np.zeros((N,QQ+1),dtype=complex)


def evolution(n):
    """

    :param n:
    :return: evolution vectors corresponding to kn
    """
    vecsAll=[psi0Tensor[n,:]]
    for qq in range(0,QQ):
        psiCurr=vecsAll[qq]
        expHTmp=expHTensor[n,qq,:,:]
        psiNext=expHTmp.dot(psiCurr)
        vecsAll.append(psiNext)

    return [n,vecsAll]

#####################evolution using multiprocessing
# procNum=48
# pool1=Pool(procNum)
#
# tEvoStart=datetime.now()
# retVecsAll=pool1.map(evolution,range(0,N))
# tEvoEnd=datetime.now()
# print("evolution time: ",tEvoEnd-tEvoStart)
# for item in retVecsAll:
#     n,vecsAll=item
#     #vecsAll corresponds to 0,1,...,QQ
#     for q in range(0,len(vecsAll)):
#         oneVec=vecsAll[q]
#         sol00[n,q]=oneVec[0]
#         sol10[n,q]=oneVec[1]
#         sol11[n,q]=oneVec[2]
#         sol20[n,q]=oneVec[3]
####################################
#################################evolution serially

tEvoStart=datetime.now()

for n in range(0,N):
    n,vecsAll=evolution(n)
    for q in range(0,len(vecsAll)):
        oneVec = vecsAll[q]
        sol00[n,q]=oneVec[0]
        sol10[n,q]=oneVec[1]
        sol11[n,q]=oneVec[2]
        sol20[n,q]=oneVec[3]




tEvoEnd=datetime.now()
print("evolution time: ",tEvoEnd-tEvoStart)
####################################
z00=np.zeros((N,QQ+1),dtype=complex)
z10=np.zeros((N,QQ+1),dtype=complex)
z11=np.zeros((N,QQ+1),dtype=complex)
z20=np.zeros((N,QQ+1),dtype=complex)

tfftStart=datetime.now()

for qq in range(0,QQ+1):
    z00[:,qq]=np.fft.fft(sol00[:,qq],norm="ortho")
    z10[:, qq] = np.fft.fft(sol10[:, qq], norm="ortho")
    z11[:, qq] = np.fft.fft(sol11[:, qq], norm="ortho")
    z20[:, qq] = np.fft.fft(sol20[:, qq], norm="ortho")

tfftEnd=datetime.now()

print("fft time: ",tfftEnd-tfftStart)



squareMat=np.conj(z00)*z00+np.conj(z10)*z10+np.conj(z11)*z11+np.conj(z20)*z20
squareMat=np.abs(squareMat)
posVals=np.array(list(range(0,N)))

xValsAll=posVals.dot(squareMat)

drift=[elem-xValsAll[0] for elem in xValsAll]



outDir="./"+name+"Evo/"
Path(outDir).mkdir(parents=True,exist_ok=True)
outData=np.array([drift]).T

pdOut=pd.DataFrame(data=outData,columns=["drift"])

pdOut.to_csv(outDir+name+"drift.csv",index=False)



####plot init and final

initAmplitude=np.abs(z00[:,0])**2+np.abs(z10[:,0])**2+np.abs(z11[:,0])**2+np.abs(z20[:,0])**2
finalAmplitude=np.abs(z00[:,-1])**2+np.abs(z10[:,-1])**2+np.abs(z11[:,-1])**2+np.abs(z20[:,-1])**2

plt.figure()
plt.plot(range(0,N),initAmplitude,color="black")
plt.title("init")
plt.savefig(outDir+"init.png")
plt.close()

plt.figure()
plt.plot(range(0,N),finalAmplitude,color="black")
plt.title("final")
plt.savefig(outDir+"final.png")
plt.close()