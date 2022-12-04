import numpy as np
import pandas as pd
from scipy.linalg import expm
from multiprocessing import Pool
from datetime import datetime
from pathlib import Path

r=1
theta0=np.pi/3

T=1e3


def jcExpression(tau):
    return r*np.sin(theta0)

def jdExpression(tau):
    return r*np.sin(theta0)

def dtaujcExpression(tau):
    return 0

def dtaujdExpression(tau):
    return 0

def jb1Expression(tau):
    return r*np.cos(theta0)*np.cos(2*np.pi*tau)

def jb2Expression(tau):
    return r*np.cos(theta0)*np.sin(2*np.pi*tau)

def jbExpression(k,tau):
    return jb1Expression(tau)+np.exp(1j*k)*jb2Expression(tau)

def dtaujb1Expression(tau):
    return -2*np.pi*r*np.cos(theta0)*np.sin(2*np.pi*tau)

def dtaujb2Expression(tau):
    return 2*np.pi*r*np.cos(theta0)*np.cos(2*np.pi*tau)


def dtaujbExpression(k,tau):
    return dtaujb1Expression(tau)+np.exp(1j*k)*dtaujb2Expression(tau)

def DeltaExpression(k,tau):
    jbFunc=jb1Expression(tau)+np.exp(1j*k)*jb2Expression(tau)
    jcFunc=jcExpression(tau)
    jdFunc=jdExpression(tau)
    return np.sqrt(np.abs(jbFunc)**2+jcFunc**2+jdFunc**2)

def deltaExpression(tau):
    return np.sqrt(jcExpression(tau)**2+jdExpression(tau)**2)

def dtauDeltaExpression(k,tau):
    jbFunc=jbExpression(k,tau)
    dtaujbFunc=dtaujbExpression(k,tau)
    jcFunc=jcExpression(tau)
    dtaujcFunc=dtaujcExpression(tau)
    jdFunc=jdExpression(tau)
    dtaujdFunc=dtaujdExpression(tau)
    DFunc=DeltaExpression(k,tau)
    return (np.conj(jbFunc)*dtaujbFunc+jbFunc*np.conj(dtaujbFunc)+2*jcFunc*dtaujcFunc+2*jdFunc*dtaujdFunc)/(2*DFunc)

def dtaudeltaExpression(tau):
    jcFunc = jcExpression(tau)
    jdFunc=jdExpression(tau)
    dFunc=deltaExpression(tau)
    dtaujcFunc = dtaujcExpression(tau)
    dtaujdFunc = dtaujdExpression(tau)
    return (jcFunc*dtaujcFunc+jdFunc*dtaujdFunc)/dFunc

def GMat(k,tau):
    jb1Func=jb1Expression(tau)
    jb2Func=jb2Expression(tau)
    jbFunc = jbExpression(k, tau)
    dtaujbFunc = dtaujbExpression(k, tau)
    jcFunc = jcExpression(tau)
    dtaujcFunc = dtaujcExpression(tau)
    jdFunc = jdExpression(tau)
    dtaujdFunc = dtaujdExpression(tau)
    DFunc = DeltaExpression(k, tau)
    dtauDFunc=dtauDeltaExpression(k,tau)
    dFunc=deltaExpression(tau)

    G0000 = 1 / (2 * DFunc ** 4) * (-np.exp(1j * k) * jb2Func * np.conj(dtaujbFunc) * DFunc ** 2
                                    - np.exp(-1j * k) * jb2Func * dtaujbFunc * DFunc ** 2
                                    + np.exp(1j * k) * jb2Func * np.conj(jbFunc) * DFunc * dtauDFunc
                                    + np.exp(-1j * k) * jb2Func * jbFunc * DFunc * dtauDFunc
                                    + 1j * jbFunc * jb1Func * jb2Func * np.sin(k) * np.conj(dtaujbFunc)
                                    - 1j * np.conj(jbFunc) * jb1Func * jb2Func * np.sin(k) * dtaujbFunc)

    G1010 = 0

    G1011 = 1 / (dFunc ** 2 * DFunc ** 3) * (np.exp(1j * k) * jb2Func * jcFunc * dtaujdFunc * DFunc ** 2
                                             - np.exp(1j * k) * jb2Func * jdFunc * dtaujcFunc * DFunc ** 2
                                             - 1j * jbFunc * jb1Func * jb2Func * jcFunc * dtaujdFunc * np.sin(k)
                                             + 1j * jbFunc * jb1Func * jb2Func * jdFunc * dtaujcFunc * np.sin(k))

    G1110 = 1 / (dFunc ** 2 * DFunc ** 3) * (np.exp(-1j * k) * jb2Func * jcFunc * dtaujdFunc * DFunc ** 2
                                             - np.exp(-1j * k) * jb2Func * jdFunc * dtaujcFunc * DFunc ** 2
                                             + 1j * np.conj(jbFunc) * jb1Func * jb2Func * jcFunc * dtaujdFunc * np.sin(k)
                                             - 1j * np.conj(jbFunc) * jb1Func * jb2Func * jdFunc * dtaujcFunc * np.sin(k))

    G1111 = 1 / DFunc ** 4 * (np.exp(-1j * k) * jb2Func * dtaujbFunc * DFunc ** 2
                              + np.exp(1j * k) * jb2Func * np.conj(dtaujbFunc) * DFunc ** 2
                              - np.exp(-1j * k) * jb2Func * jbFunc * DFunc * dtauDFunc
                              - np.exp(1j * k) * jb2Func * np.conj(jbFunc) * DFunc * dtauDFunc
                              + 1j * np.conj(jbFunc) * jb1Func * jb2Func * dtaujbFunc * np.sin(k)
                              - 1j * jbFunc * jb1Func * jb2Func * np.conj(dtaujbFunc) * np.sin(k))

    G2020 = 1 / (2 * DFunc ** 4) * (-np.exp(1j * k) * jb2Func * np.conj(dtaujbFunc) * DFunc ** 2
                                    - np.exp(-1j * k) * jb2Func * dtaujbFunc * DFunc ** 2
                                    + np.exp(1j * k) * jb2Func * np.conj(jbFunc) * DFunc * dtauDFunc
                                    + np.exp(-1j * k) * jb2Func * jbFunc * DFunc * dtauDFunc
                                    + 1j * jbFunc * jb1Func * jb2Func * np.conj(dtaujbFunc) * np.sin(k)
                                    - 1j * np.conj(jbFunc) * jb1Func * jb2Func * dtaujbFunc * np.sin(k))

    return np.array([
        [G0000,0,0,0],
        [0,G1010,G1011,0],
        [0,G1110,G1111,0],
        [0,0,0,G2020]
    ])




def SMat(k,tau):
    jb1Func = jb1Expression(tau)
    jb2Func = jb2Expression(tau)
    jbFunc = jbExpression(k, tau)
    # dtaujbFunc = dtaujbExpression(k, tau)
    jcFunc = jcExpression(tau)
    dtaujcFunc = dtaujcExpression(tau)
    jdFunc = jdExpression(tau)
    dtaujdFunc = dtaujdExpression(tau)
    DFunc = DeltaExpression(k, tau)
    # dtauDFunc = dtauDeltaExpression(k, tau)
    dFunc = deltaExpression(tau)

    S1010 = 0

    S1011 = 1 / (dFunc ** 2 * DFunc ** 3) * (jbFunc * jdFunc * dtaujcFunc * jb1Func * jb2Func * np.cos(k)
                                             + jbFunc * jdFunc * dtaujcFunc * jb2Func ** 2
                                             - jbFunc * jcFunc * dtaujdFunc * jb1Func * jb2Func * np.cos(k)
                                             - jbFunc * jcFunc * dtaujdFunc * jb2Func ** 2)

    S1110 = 1 / (dFunc ** 2 * DFunc ** 3) * (np.conj(jbFunc) * jdFunc * dtaujcFunc * jb1Func * jb2Func * np.cos(k)
                                             + np.conj(jbFunc) * jdFunc * dtaujcFunc * jb2Func ** 2
                                             - np.conj(jbFunc) * jcFunc * dtaujdFunc * jb1Func * jb2Func * np.cos(k)
                                             - np.conj(jbFunc) * jcFunc * dtaujdFunc * jb2Func ** 2)

    S1111 = 0

    return np.array([
        [0,0,0,0],
        [0,S1010,S1011,0],
        [0,S1110,S1111,0],
        [0,0,0,0]
    ])


def FMat(k,tau):
    jb1Func = jb1Expression(tau)
    jb2Func = jb2Expression(tau)
    DFunc = DeltaExpression(k, tau)

    tmp=jb1Func*jb2Func*np.sin(k)/DFunc
    return np.diag([tmp,0,0,-tmp])



def LMat(k,tau=0):
    # jb1Func = jb1Expression(tau)
    # jb2Func = jb2Expression(tau)
    jbFunc = jbExpression(k, tau)
    dtaujbFunc = dtaujbExpression(k, tau)
    jcFunc = jcExpression(tau)
    dtaujcFunc = dtaujcExpression(tau)
    jdFunc = jdExpression(tau)
    dtaujdFunc = dtaujdExpression(tau)
    DFunc = DeltaExpression(k, tau)
    # dtauDFunc = dtauDeltaExpression(k, tau)
    dFunc = deltaExpression(tau)
    dtaudFunc=dtaudeltaExpression(tau)
    L0010 = 1j / (np.sqrt(2) * DFunc ** 2 * dFunc) * (-jdFunc * dtaujcFunc + jcFunc * dtaujdFunc)

    L0011 = 1j / (np.sqrt(2) * DFunc ** 3) * (-jbFunc * dtaudFunc + dFunc * dtaujbFunc)

    L0020 = 1j / (8 * DFunc ** 3) * (np.conj(jbFunc) * dtaujbFunc - jbFunc * np.conj(dtaujbFunc))

    L1000 = 1j / (np.sqrt(2) * DFunc ** 2 * dFunc) * (jcFunc * dtaujdFunc - jdFunc * dtaujcFunc)

    L1020 = 1j / (np.sqrt(2) * DFunc ** 2 * dFunc) * (jdFunc * dtaujcFunc - jcFunc * dtaujdFunc)

    L1100 = 1j / (np.sqrt(2) * DFunc ** 3) * (dFunc * np.conj(dtaujbFunc) - np.conj(jbFunc) * dtaudFunc)

    L1120 = 1j / (np.sqrt(2) * DFunc ** 3) * (np.conj(jbFunc) * dtaudFunc - dFunc * np.conj(dtaujbFunc))

    L2000 = 1j / (8 * DFunc ** 3) * (jbFunc * np.conj(dtaujbFunc) - np.conj(jbFunc) * dtaujbFunc)

    L2010 = 1j / (np.sqrt(2) * DFunc ** 2 * dFunc) * (jdFunc * dtaujcFunc - jcFunc * dtaujdFunc)

    L2011 = 1j / (np.sqrt(2) * DFunc ** 3) * (jbFunc * dtaudFunc - dFunc * dtaujbFunc)



    return np.array([
        [0,L0010,L0011,L0020],
        [L1000,0,0,L1020],
        [L1100,0,0,L1120],
        [L2000,L2010,L2011,0]
    ])




def AMat(k,tau):
    # jb1Func = jb1Expression(tau)
    # jb2Func = jb2Expression(tau)
    jbFunc = jbExpression(k, tau)
    dtaujbFunc = dtaujbExpression(k, tau)
    jcFunc = jcExpression(tau)
    dtaujcFunc = dtaujcExpression(tau)
    jdFunc = jdExpression(tau)
    dtaujdFunc = dtaujdExpression(tau)
    DFunc = DeltaExpression(k, tau)
    # dtauDFunc = dtauDeltaExpression(k, tau)
    dFunc = deltaExpression(tau)

    A0000 = 1 / (4 * DFunc ** 2) * (np.conj(jbFunc) * dtaujbFunc - jbFunc * np.conj(dtaujbFunc))

    A1010 = 0

    A1011 = jbFunc / (dFunc ** 2 * DFunc) * (jcFunc * dtaujdFunc - jdFunc * dtaujcFunc)

    A1110 = np.conj(jbFunc) / (dFunc ** 2 * DFunc) * (jdFunc * dtaujcFunc - jcFunc * dtaujdFunc)

    A1111 = 1 / (2 * DFunc ** 2) * (jbFunc * np.conj(dtaujbFunc) - np.conj(jbFunc) * dtaujbFunc)

    A2020 = 1 / (4 * DFunc ** 2) * (np.conj(jbFunc) * dtaujbFunc - jbFunc * np.conj(dtaujbFunc))


    return np.array([
        [A0000,0,0,0],
        [0,A1010,A1011,0],
        [0,A1110,A1111,0],
        [0,0,0,A2020]
    ])


def Theta0(k,tau):
    return GMat(k,tau)+SMat(k,tau)-T*FMat(k,tau)


Q=30
M=10
J=10

N=50#momentum values
kValsAll=np.array([2*np.pi/N*n for n in range(0,N)])

def tauFunction(q,m,j):
    return 1/Q*q+1/(Q*M)*m+1/(Q*M*J)*j

def Aexp(nqmj):
    """

    :param nqmj:
    :return:
    """
    n,q,m,j=nqmj
    k=kValsAll[n]
    tauTmp=tauFunction(q,m,j)
    ATmp=AMat(k,tauTmp)
    return [n,q,m,j,expm(ATmp/(Q*M*J))]

#AExpTensor holds the exponentials of A

AExpTensor=np.zeros((N,Q,M,J,4,4),dtype=complex)

inIndsAExp=[[n,q,m,j] for n in range(0,N) for q in range(0,Q) for m in range(0,M) for j in range(1,J+1)]

procNum=48
tAExpStart=datetime.now()

pool0=Pool(procNum)
retAEx=pool0.map(Aexp,inIndsAExp)

tAExpEnd=datetime.now()

print("A exp time: ",tAExpEnd-tAExpStart)

for item in retAEx:
    n,q,m,j,AExpTmp=item
    jm1=j-1
    AExpTensor[n,q,m,jm1,:,:]=np.copy(AExpTmp)

def VMat(nqm):
    n,q,m=nqm
    retMat=np.eye(4,dtype=complex)
    for jm1 in range(0,J)[::-1]:
        retMat=retMat@AExpTensor[n,q,m,jm1,:,:]

    return [n,q,m,retMat]


VTensor=np.zeros((N,Q,M,4,4),dtype=complex)
inVinds=[[n,q,m] for n in range(0,N) for q in range(0,Q) for m in range(0,M)]

procNum=48

tVProdStart=datetime.now()
pool1=Pool(procNum)
retVProds=pool1.map(VMat,inVinds)
tVProdEnd=datetime.now()

print("V prods time: ",tVProdEnd-tVProdStart)

for item in retVProds:
    n,q,m,VTmp=item
    VTensor[n,q,m,:,:]=np.copy(VTmp)

WTensor=np.zeros((N,Q,4,4),dtype=complex)

def Wk(n):
    WkTensor=np.zeros((Q,4,4),dtype=complex)
    WkTensor[0,:,:]=np.eye(4)
    for q in range(1,Q):
        prodTmp=np.copy(WkTensor[q-1,:,:])
        for m in range(0,M):
            prodTmp=VTensor[n,q-1,m,:,:]@prodTmp
        WkTensor[q,:,:]=np.copy(prodTmp)
    return [n,WkTensor]


tWkTensorStart=datetime.now()
procNum=48
pool2=Pool(procNum)
retWk=pool2.map(Wk,range(0,N))

tWkTensorEnd=datetime.now()
print("Wk tensor time: ", tWkTensorEnd-tWkTensorStart)

for item in retWk:
    n,WkTensor=item
    WTensor[n,:,:,:]=np.copy(WkTensor)


UTensor=np.zeros((N,Q,M,4,4),dtype=complex)

#initialize for m=0
for n in range(0,N):
    for q in range(0,Q):
        UTensor[n,q,0,:,:]=np.copy(WTensor[n,q,:,:])

#initialize for m>0

def oneU(nqm):
    #m=1,2,...,M-1
    n,q,m=nqm
    prodTmp=np.copy(WTensor[n,q,:,:])
    for j in range(0,m):
        prodTmp=VTensor[n,q,j,:,:]@prodTmp
    return [n,q,m,prodTmp]



inUinds=[[n,q,m] for n in range(0,N) for q in range(0,Q) for m in range(1,M)]

procNum=48
pool3=Pool(procNum)
tOneUStart=datetime.now()
retU=pool3.map(oneU,inUinds)
tOneUEnd=datetime.now()

print("U time: ",tOneUEnd-tOneUStart)
for item in retU:
    n,q,m,prodTmp=item
    UTensor[n,q,m,:,:]=np.copy(prodTmp)


PTensor=np.zeros((N,Q,M,4,4),dtype=complex)

def oneP(nqm):
    n,q,m=nqm
    kTmp=kValsAll[n]
    Theta0Mat=Theta0(kTmp,1/Q*q+1/(Q*M)*m)
    U0Tmp=UTensor[n,q,m,:,:]

    prodTmp=np.conj(U0Tmp.T)@Theta0Mat@U0Tmp

    return [n,q,m,prodTmp]


inPinds=[[n,q,m] for n in range(0,N) for q in range(0,Q) for m in range(0,M)]
tPStart=datetime.now()
procNum=48
pool4=Pool(procNum)
retP=pool4.map(oneP,inPinds)

tPEnd=datetime.now()
print("P time: ",tPEnd-tPStart)

for item in retP:
    n,q,m,prodTmp=item
    PTensor[n,q,m,:,:]=np.copy(prodTmp)


#the  integral of matrix p on time grid points q=0,1,...,Q
intPTensor=np.zeros((N,Q+1,4,4),dtype=complex)

def oneIntP(nq):
    # q =0,1,...,Q
    n,q=nq
    #q=0
    if q==0:
        return [n,q,np.zeros((4,4),dtype=complex)]
    #q>0
    sumTmp=np.zeros((4,4),dtype=complex)
    for m in range(0,M):
        sumTmp=sumTmp+PTensor[n,q-1,m,:,:]*1/(Q*M)

    return [n,q,sumTmp]

inIntPInds=[[n,q] for n in range(0,N) for q in range(0,Q+1)]
procNum=48
pool5=Pool(procNum)
tIntPStart=datetime.now()
retIntP=pool5.map(oneIntP,inIntPInds)
tIntPEnd=datetime.now()
print("int P time: ",tIntPEnd-tIntPStart)
for item in retIntP:
    n,q,sumTmp=item
    intPTensor[n,q,:,:]=np.copy(sumTmp)
#integral of F at each time grid q=0,1,...,Q

deltaDTensor=np.zeros((N,Q+1,4,4),dtype=complex)

def sumF(nq):
    #q>=1
    n,q,=nq
    kTmp=kValsAll[n]
    sumTmp=np.zeros((4,4),dtype=complex)
    for m in range(0,M):
        sumTmp=sumTmp+FMat(kTmp,1/Q*(q-1)+1/(Q*M)*m)*1/(Q*M)
    return [n,q,sumTmp]


indeltaDInds=[[n,q] for n in range(0,N) for q in range(1,Q+1)]
procNum=48
pool6=Pool(procNum)
tSumFStart=datetime.now()
retSumF=pool6.map(sumF,indeltaDInds)
tSumFEnd=datetime.now()
print("sum F time: ",tSumFEnd-tSumFStart)
for item in retSumF:
    n,q,sumTmp=item
    deltaDTensor[n,q,:,:]=np.copy(sumTmp)
#initial coefficients

cTensor=np.ones((N,4),dtype=complex)
cTensor[:,0]*=-2
cTensor[:,1:]*=0
#norm
# nm=np.sqrt(np.real(np.sum(np.conj(cTensor)*cTensor)))
nm=np.linalg.norm(cTensor,"fro")
cTensor/=nm

####################

#cumulative integral of P
cumulativeP=np.zeros((N,Q+1,4,4),dtype=complex)
tCMPStart=datetime.now()
for n in range(0,N):
    sumPTmp=np.copy(intPTensor[n,0,:,:])
    cumulativeP[n,0,:,:]=np.copy(sumPTmp)
    for q in range(1,Q+1):
        sumPTmp+=intPTensor[n,q,:,:]
        cumulativeP[n,q,:,:]=np.copy(sumPTmp)
tCMPEnd=datetime.now()
print("cumulative P time: ",tCMPEnd-tCMPStart)
#cumulative integral of F

cumulativeF=np.zeros((N,Q+1,4,4),dtype=complex)
tCMFStart=datetime.now()
for n in range(0,N):
    sumdDTmp=np.copy(deltaDTensor[n,0,:,:])
    cumulativeF[n,0,:,:]=np.copy(sumdDTmp)
    for q in range(1,Q+1):
        sumdDTmp+=deltaDTensor[n,q,:,:]
        cumulativeF[n,q,:,:]=np.copy(sumdDTmp)

tCMFEnd=datetime.now()

print("cumulative F time: ",tCMFEnd-tCMFStart)

###############
positionTopologicalTensor=np.zeros((N,Q+1))

for n in range(0,N):
    cTmp=cTensor[n,:]
    for q in range(0,Q+1):
        prodTmp=np.conj(cTmp).dot(cumulativeP[n,q,:,:]).dot(cTmp)
        positionTopologicalTensor[n,q]=np.real(prodTmp)

positionTopologicalAtEachInstant=np.sum(positionTopologicalTensor,axis=0)


ibcTensor=np.zeros((N,Q+1))
for n in range(0,N):
    cTmp=cTensor[n,:]
    kTmp=kValsAll[n]
    LTmp=LMat(kTmp,0)
    for q in range(0,Q+1):
        ibcTmp=-2*np.real(np.conj(cTmp).dot(cumulativeF[n,q,:,:]).dot(LTmp).dot(cTmp))
        ibcTensor[n,q]=ibcTmp


ibcAtEachInstant=np.sum(ibcTensor,axis=0)

outDir="./JceqJd/"

Path(outDir).mkdir(parents=True,exist_ok=True)

outData=np.array([positionTopologicalAtEachInstant,ibcAtEachInstant]).T

pdOut=pd.DataFrame(data=outData,columns=["topological","ibc"])

pdOut.to_csv(outDir+"position.csv",index=False)