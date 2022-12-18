import numpy as np
import pandas as pd
from scipy.linalg import expm
from multiprocessing import Pool
from datetime import datetime
from pathlib import Path

#this script computes drift without resorting to ibc expressions

r=1

alpha1=np.pi/4
alpha2=np.pi*np.sqrt(2)
T=2*1e3

name="example6"

def jcExpression(tau):
    return r*np.cos(2*np.pi*tau)

def jdExpression(tau):
    return r*np.sin(2*np.pi*tau)*np.cos(alpha1+2*np.pi*tau)

def dtaujcExpression(tau):
    return -2*np.pi*r*np.sin(2*np.pi*tau)

def dtaujdExpression(tau):
    return 2*np.pi*r*np.cos(2*np.pi*tau)*np.cos(alpha1+2*np.pi*tau)\
           -2*np.pi*r*np.sin(2*np.pi*tau)*np.sin(alpha1+2*np.pi*tau)

def jb1Expression(tau):
    return r*np.sin(2*np.pi*tau)*np.sin(alpha1+2*np.pi*tau)*np.cos(alpha2+2*np.pi*tau)

def jb2Expression(tau):
    return r*np.sin(2*np.pi*tau)*np.sin(alpha1+2*np.pi*tau)*np.sin(alpha2+2*np.pi*tau)

def jbExpression(k,tau):
    return jb1Expression(tau)+np.exp(1j*k)*jb2Expression(tau)

def dtaujb1Expression(tau):
    return 2*np.pi*r*np.cos(2*np.pi*tau)*np.sin(alpha1+2*np.pi*tau)*np.cos(alpha2+2*np.pi*tau)\
           +2*np.pi*r*np.sin(2*np.pi*tau)*np.cos(alpha1+2*np.pi*tau)*np.cos(alpha2+2*np.pi*tau)\
           -2*np.pi*r*np.sin(2*np.pi*tau)*np.sin(alpha1+2*np.pi*tau)*np.sin(alpha2+2*np.pi*tau)


def dtaujb2Expression(tau):
    return 2*np.pi*r*np.cos(2*np.pi*tau)*np.sin(alpha1+2*np.pi*tau)*np.sin(alpha2+2*np.pi*tau)\
           +2*np.pi*r*np.sin(2*np.pi*tau)*np.cos(alpha1+2*np.pi*tau)*np.sin(alpha2+2*np.pi*tau)\
           +2*np.pi*r*np.sin(2*np.pi*tau)*np.sin(alpha1+2*np.pi*tau)*np.cos(alpha2+2*np.pi*tau)


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


def vec00(k,tau):
    jbFunc = jbExpression(k, tau)
    jcFunc = jcExpression(tau)
    jdFunc = jdExpression(tau)
    DFunc = DeltaExpression(k, tau)
    # dtauDFunc = dtauDeltaExpression(k, tau)
    # dFunc = deltaExpression(tau)
    vec00Tmp=np.array([-1/np.sqrt(2),np.conj(jbFunc)/(np.sqrt(2)*DFunc),
             jcFunc/(np.sqrt(2)*DFunc),jdFunc/(np.sqrt(2)*DFunc)])

    return vec00Tmp

def vec10(k,tau):
    # jbFunc = jbExpression(k, tau)
    # dtaujbFunc = dtaujbExpression(k, tau)
    jcFunc = jcExpression(tau)
    # dtaujcFunc = dtaujcExpression(tau)
    jdFunc = jdExpression(tau)
    # dtaujdFunc = dtaujdExpression(tau)
    # DFunc = DeltaExpression(k, tau)
    # dtauDFunc = dtauDeltaExpression(k, tau)
    dFunc = deltaExpression(tau)

    vec10Tmp=np.array([0,0,-jdFunc/dFunc,jcFunc/dFunc])

    return vec10Tmp



def vec11(k,tau):
    jbFunc = jbExpression(k, tau)
    # dtaujbFunc = dtaujbExpression(k, tau)
    jcFunc = jcExpression(tau)
    # dtaujcFunc = dtaujcExpression(tau)
    jdFunc = jdExpression(tau)
    # dtaujdFunc = dtaujdExpression(tau)
    DFunc = DeltaExpression(k, tau)
    # dtauDFunc = dtauDeltaExpression(k, tau)
    dFunc = deltaExpression(tau)

    vec11Tmp=np.array([0,dFunc/DFunc,-jbFunc*jcFunc/(dFunc*DFunc),-jbFunc*jdFunc/(dFunc*DFunc)])
    return vec11Tmp



def vec20(k,tau):
    jbFunc = jbExpression(k, tau)
    # dtaujbFunc = dtaujbExpression(k, tau)
    jcFunc = jcExpression(tau)
    # dtaujcFunc = dtaujcExpression(tau)
    jdFunc = jdExpression(tau)
    # dtaujdFunc = dtaujdExpression(tau)
    DFunc = DeltaExpression(k, tau)
    # dtauDFunc = dtauDeltaExpression(k, tau)
    # dFunc = deltaExpression(tau)

    vec20Tmp=np.array([1/np.sqrt(2),np.conj(jbFunc)/(np.sqrt(2)*DFunc),
               jcFunc/(np.sqrt(2)*DFunc),jdFunc/(np.sqrt(2)*DFunc)])


    return vec20Tmp



def dtauvec00(k,tau):
    jbFunc = jbExpression(k, tau)
    dtaujbFunc = dtaujbExpression(k, tau)
    jcFunc = jcExpression(tau)
    dtaujcFunc = dtaujcExpression(tau)
    jdFunc = jdExpression(tau)
    dtaujdFunc = dtaujdExpression(tau)
    DFunc = DeltaExpression(k, tau)
    dtauDFunc = dtauDeltaExpression(k, tau)
    # dFunc = deltaExpression(tau)
    dtauvec00Tmp=np.array([0,1/(np.sqrt(2)*DFunc**2)*(np.conj(dtaujbFunc)*DFunc-np.conj(jbFunc)*dtauDFunc),
                  1/(np.sqrt(2)*DFunc**2)*(dtaujcFunc*DFunc-jcFunc*dtauDFunc),
                  1/(np.sqrt(2)*DFunc**2)*(dtaujdFunc*DFunc-jdFunc*dtauDFunc)])

    return dtauvec00Tmp



def dtauvec10(k,tau):
    # jbFunc = jbExpression(k, tau)
    # dtaujbFunc = dtaujbExpression(k, tau)
    jcFunc = jcExpression(tau)
    dtaujcFunc = dtaujcExpression(tau)
    jdFunc = jdExpression(tau)
    dtaujdFunc = dtaujdExpression(tau)
    # DFunc = DeltaExpression(k, tau)
    # dtauDFunc = dtauDeltaExpression(k, tau)
    dFunc = deltaExpression(tau)
    dtaudFunc=dtaudeltaExpression(tau)
    dtauvec10Tmp=np.array([0,0,-1/dFunc**2*(dtaujdFunc*dFunc-jdFunc*dtaudFunc),1/dFunc**2*(dtaujcFunc*dFunc-jcFunc*dtaudFunc)])

    return dtauvec10Tmp


def dtauvec11(k,tau):
    jbFunc = jbExpression(k, tau)
    dtaujbFunc = dtaujbExpression(k, tau)
    jcFunc = jcExpression(tau)
    dtaujcFunc = dtaujcExpression(tau)
    jdFunc = jdExpression(tau)
    dtaujdFunc = dtaujdExpression(tau)
    DFunc = DeltaExpression(k, tau)
    dtauDFunc = dtauDeltaExpression(k, tau)
    dFunc = deltaExpression(tau)
    dtaudFunc = dtaudeltaExpression(tau)

    dtauvec11Tmp=np.array([0,1/DFunc**2*(dtaudFunc*DFunc-dFunc*dtauDFunc),
                  -1/(dFunc**2*DFunc**2)*(dtaujbFunc*jcFunc*dFunc*DFunc+jbFunc*dtaujcFunc*dFunc*DFunc-jbFunc*jcFunc*dtaudFunc*DFunc-jbFunc*jcFunc*dFunc*dtauDFunc),
                  -1/(dFunc**2*DFunc**2)*(dtaujbFunc*jdFunc*dFunc*DFunc+jbFunc*dtaujdFunc*dFunc*DFunc-jbFunc*jdFunc*dtaudFunc*DFunc-jbFunc*jdFunc*dFunc*dtauDFunc)])


    return dtauvec11Tmp


def dtauvec20(k,tau):
    jbFunc = jbExpression(k, tau)
    dtaujbFunc = dtaujbExpression(k, tau)
    jcFunc = jcExpression(tau)
    dtaujcFunc = dtaujcExpression(tau)
    jdFunc = jdExpression(tau)
    dtaujdFunc = dtaujdExpression(tau)
    DFunc = DeltaExpression(k, tau)
    dtauDFunc = dtauDeltaExpression(k, tau)
    # dFunc = deltaExpression(tau)
    # dtaudFunc = dtaudeltaExpression(tau)

    dtauvec20Tmp=np.array([0,1/(np.sqrt(2)*DFunc**2)*(np.conj(dtaujbFunc)*DFunc-np.conj(jbFunc)*dtauDFunc),
                  1/(np.sqrt(2)*DFunc**2)*(dtaujcFunc*DFunc-jcFunc*dtauDFunc),
                  1/(np.sqrt(2)*DFunc**2)*(dtaujdFunc*DFunc-jdFunc*dtauDFunc)])


    return dtauvec20Tmp


Q=30
M=10
J=10
N=50#momentum values
kValsAll=np.array([2*np.pi/N*n for n in
                   range(0,N)])

def tauFunction(q,m,j):
    return 1/Q*q+1/(Q*M)*m+1/(Q*M*J)*j

#######################################################
#1.

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


assembleU0Start=datetime.now()
U0Tensor=np.zeros((N,Q+1,M+1,4,4),dtype=complex)

#initialization
for n in range(0,N):
    U0Tensor[n,0,0,:,:]=np.eye(4)


for n in range(0,N):
    for q in range(0,Q):
        for m in range(0,M):
            U0Tensor[n,q,m+1,:,:]=VTensor[n,q,m,:,:]@U0Tensor[n,q,m,:,:]
        U0Tensor[n,q+1,0,:,:]=np.copy(U0Tensor[n,q,M,:,:])


assembleU0End=datetime.now()

print("assembling U0Tensor time: ", assembleU0End-assembleU0Start)

######################################## 1end
########################################
#2.
def En1(n1,n,q,m,j):
    k=kValsAll[n]
    tau=tauFunction(q,m,j)
    jbFunc = jbExpression(k, tau)
    # dtaujbFunc = dtaujbExpression(k, tau)
    jcFunc = jcExpression(tau)
    # dtaujcFunc = dtaujcExpression(tau)
    jdFunc = jdExpression(tau)
    # dtaujdFunc = dtaujdExpression(tau)
    # DFunc = DeltaExpression(k, tau)
    # dtauDFunc = dtauDeltaExpression(k, tau)
    DFunc = np.abs(np.sqrt(jbFunc * np.conj(jbFunc) + jcFunc ** 2 + jdFunc ** 2))
    if n1==0:
        E0 = -DFunc
        return E0
    elif n1==1:
        E1 = 0
        return E1

    else:
        E2 = DFunc
        return E2



def intEn1Wrappter(n1nqm):
    #computes each element of tensor intEn1
    n1,n,q,m=n1nqm

    sumTmp=0
    for j in range(0,J):
        sumTmp+=En1(n1,n,q,m,j)
    sumTmp*=1/(Q*M*J)
    return [n1,n,q,m,sumTmp]

tintEn1Start=datetime.now()
intEn1Tensor=np.zeros((3,N,Q,M))

intEn1inInds=[[n1,n,q,m] for n1 in range(0,3) for n in range(0,N) for q in range(0,Q) for m in range(0,M)]

procNum=48
pool2=Pool(procNum)
retintEn1=pool2.map(intEn1Wrappter,intEn1inInds)


for item in retintEn1:
    n1,n,q,m,sumTmp=item
    intEn1Tensor[n1,n,q,m]=sumTmp

tintEn1End=datetime.now()

print("intEn1 time: ", tintEn1End-tintEn1Start)

tsTensorStart=datetime.now()
sTensor=np.zeros((3,N,Q+1,M+1))
for n1 in range(0,3):
    for n in range(0,N):
        for q in range(0,Q):
            for m in range(0,M):
                sTensor[n1,n,q,m+1]=sTensor[n1,n,q,m]+intEn1Tensor[n1,n,q,m]
            sTensor[n1,n,q+1,0]=sTensor[n1,n,q,M]


tsTensorEnd=datetime.now()

print("sTensor time: ",tsTensorEnd-tsTensorStart)



def oneB(nqm):
    n,q,m=nqm
    k=kValsAll[n]
    tau=tauFunction(q,m,0)
    vec00Tmp=vec00(k,tau)
    vec10Tmp=vec10(k,tau)
    vec11Tmp=vec11(k,tau)
    vec20Tmp=vec20(k,tau)
    dtauvec00Tmp=dtauvec00(k,tau)
    dtauvec10Tmp=dtauvec10(k,tau)
    dtauvec11Tmp=dtauvec11(k,tau)
    dtauvec20Tmp=dtauvec20(k,tau)
    B0010=-np.exp(1j*T*(sTensor[0,n,q,m]-sTensor[1,n,q,m]))*\
        np.conj(vec00Tmp).dot(dtauvec10Tmp)
    B0011=-np.exp(1j*T*(sTensor[0,n,q,m]-sTensor[1,n,q,m]))*\
        np.conj(vec00Tmp).dot(dtauvec11Tmp)
    B0020=-np.exp(1j*T*(sTensor[0,n,q,m]-sTensor[2,n,q,m]))*\
        np.conj(vec00Tmp).dot(dtauvec20Tmp)

    B1000=-np.exp(1j*T*(sTensor[1,n,q,m]-sTensor[0,n,q,m]))*\
        np.conj(vec10Tmp).dot(dtauvec00Tmp)

    B1020=-np.exp(1j*T*(sTensor[1,n,q,m]-sTensor[2,n,q,m]))*\
        np.conj(vec10Tmp).dot(dtauvec20Tmp)
    B1100=-np.exp(1j*T*(sTensor[1,n,q,m]-sTensor[0,n,q,m]))*\
        np.conj(vec11Tmp).dot(dtauvec00Tmp)

    B1120=-np.exp(1j*T*(sTensor[1,n,q,m]-sTensor[2,n,q,m]))*\
        np.conj(vec11Tmp).dot(dtauvec20Tmp)

    B2000=-np.exp(1j*T*(sTensor[2,n,q,m]-sTensor[0,n,q,m]))*\
        np.conj(vec20Tmp).dot(dtauvec00Tmp)

    B2010=-np.exp(1j*T*(sTensor[2,n,q,m]-sTensor[1,n,q,m]))*\
        np.conj(vec20Tmp).dot(dtauvec10Tmp)

    B2011=-np.exp(1j*T*(sTensor[2,n,q,m]-sTensor[1,n,q,m]))*\
        np.conj(vec20Tmp).dot(dtauvec11Tmp)

    BTmp=np.array([
        [0,B0010,B0011,B0020],
        [B1000,0,0,B1020],
        [B1100,0,0,B1120],
        [B2000,B2010,B2011,0]
    ])

    return [n,q,m,BTmp]



BTensor=np.zeros((N,Q,M,4,4),dtype=complex)

inBTensorInds=[[n,q,m] for n in range(0,N) for q in range(0,Q) for m in range(0,M)]

tBTensorStart=datetime.now()
procNum=48

pool3=Pool(procNum)

retinBTensor=pool3.map(oneB,inBTensorInds)
for item in retinBTensor:
    n,q,m,BTmp=item
    BTensor[n,q,m,:,:]=np.copy(BTmp)



tBTensorEnd=datetime.now()

print("BTensor time: ",tBTensorEnd-tBTensorStart)
########################################## 2 end

#########################################
#3.
tTheta1TensorStart=datetime.now()
Theta1Tensor=np.zeros((N,Q+1,4,4),dtype=complex)

for n in range(0,N):
    for q in range(1,Q+1):
        sumTmp=np.zeros((4,4),dtype=complex)
        for m in range(0,M):
            sumTmp+=np.conj(U0Tensor[n,q-1,m,:,:]).T@BTensor[n,q-1,m,:,:]@U0Tensor[n,q-1,m,:,:]
        sumTmp*=1/(Q*M)
        Theta1Tensor[n,q,:,:]=np.copy(sumTmp)

tTheta1TensorEnd=datetime.now()

print("Theta1Tensor time: ",tTheta1TensorEnd-tTheta1TensorStart)

#########################3 end

########################
#4.
tU1TensorStart=datetime.now()
U1Tensor=np.zeros((N,Q+1,4,4),dtype=complex)

for n in range(0,N):
    for q in range(1,Q+1):
        sumTmp=np.zeros((4,4),dtype=complex)
        for j in range(1,q+1):
            sumTmp+=Theta1Tensor[n,j,:,:]
        U1Tensor[n,q,:,:]=U0Tensor[n,q,0,:,:]@sumTmp


tU1TensorEnd=datetime.now()
print("U1Tensor time: ",tU1TensorEnd-tU1TensorStart)
######################## 4 end
############################
#5.
tcStart=datetime.now()

cInitTensor=np.ones((N,4),dtype=complex)
cInitTensor[:,1]*=0
cInitTensor[:,2]*=0
R0=N/2
for n in range(0,N):
    cInitTensor[n,:]*=np.exp(-1j*kValsAll[n]*R0)
nm=np.linalg.norm(cInitTensor,"fro")
cInitTensor/=nm

c0Tensor=np.zeros((N,Q+1,4),dtype=complex)
c1Tensor=np.zeros((N,Q+1,4),dtype=complex)

for n in range(0,N):
    for q in range(0,Q+1):
        c0Tensor[n,q,:]=U0Tensor[n,q,0,:,:].dot(cInitTensor[n,:])
        c1Tensor[n,q,:]=U1Tensor[n,q,0,:,:].dot(cInitTensor[n,:])

tcEnd=datetime.now()

print("c time: ",tcStart-tcEnd)
################# 5 end

##########################
#6.

def y0vec(nq):
    n,q=nq
    k=kValsAll[n]
    tau=tauFunction(q,0,0)
    vec00Tmp = vec00(k, tau)
    vec10Tmp = vec10(k, tau)
    vec11Tmp = vec11(k, tau)
    vec20Tmp = vec20(k, tau)
    c0Tmp=c0Tensor[n,q,:]

    y0VecTmp=c0Tmp[0]*np.exp(-1j*T*sTensor[0,n,q,0])*vec00Tmp+\
        c0Tmp[1]*np.exp(-1j*T*sTensor[1,n,q,0])*vec10Tmp+\
        c0Tmp[2]*np.exp(-1j*T*sTensor[1,n,q,0])*vec11Tmp+\
        c0Tmp[3]*np.exp(-1j*T*sTensor[2,n,q,0])*vec20Tmp

    return [n,q,y0VecTmp]


def y1Vec(nq):
    #q=0,1,...,Q
    n,q=nq
    k = kValsAll[n]
    tau = tauFunction(q, 0, 0)
    vec00Tmp = vec00(k, tau)
    vec10Tmp = vec10(k, tau)
    vec11Tmp = vec11(k, tau)
    vec20Tmp = vec20(k, tau)
    c1Tmp=c1Tensor[n,q,:]

    y1VecTmp = c1Tmp[0] * np.exp(-1j * T * sTensor[0, n, q, 0]) * vec00Tmp + \
               c1Tmp[1] * np.exp(-1j * T * sTensor[1, n, q, 0]) * vec10Tmp + \
               c1Tmp[2] * np.exp(-1j * T * sTensor[1, n, q, 0]) * vec11Tmp + \
               c1Tmp[3] * np.exp(-1j * T * sTensor[2, n, q, 0]) * vec20Tmp

    return [n, q, y1VecTmp]

ty0y1Start=datetime.now()
inyTensorInds=[[n,q] for n in range(0,N) for q in range(0,Q+1)]


y0Tensor=np.zeros((N,Q+1,4),dtype=complex)
y1Tensor=np.zeros((N,Q+1,4),dtype=complex)
procNum=48

pool4=Pool(procNum)
rety0=pool4.map(y0vec,inyTensorInds)

for item in rety0:
    n,q,y0Tmp=item
    y0Tensor[n,q,:]=np.copy(y0Tmp)

pool5=Pool(procNum)
rety1=pool5.map(y1Vec,inyTensorInds)

for item in rety1:
    n,q,y1Tmp=item
    y1Tensor[n,q,:]=np.copy(y1Tmp)

ty0y1End=datetime.now()
print("y0 y1 time: ",ty0y1End-ty0y1Start)
######################6 end

######################
#7.
tToRealStart=datetime.now()
solTensor=np.zeros((2,4,N,Q+1),dtype=complex)
# for solTensor, index 0 refers to order 0 or order 1, index 1 (=0,1,2,3) refers to 4 components of y0 and y1
for r in range(0,4):
    solTensor[0,r,:,:]=np.copy(y0Tensor[:,:,r])
    solTensor[1,r,:,:]=np.copy(y1Tensor[:,:,r])


xiTensor=np.zeros((2,4,N,Q+1),dtype=complex)
for b in range(0,2):
    for r in range(0,4):
        for q in range(0,Q+1):
            xiTensor[b,r,:,q]=np.fft.fft(solTensor[b,r,:,q],norm="ortho")


tToRealEnd=datetime.now()

print("to real time: ",tToRealEnd-tToRealStart)

###########################7 end

#################################
#8.
z0Tensor=np.zeros((Q+1))
z1Tensor=np.zeros((Q+1),dtype=complex)
z1StartTensor=np.zeros((Q+1),dtype=complex)

x0Squared=