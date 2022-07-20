import numpy as np
import quadpy
from scipy.linalg import  expm

#this script verifies numerical and perturbative solution from paper
mu=1
a0=1
T=4000
N=10

def omega0(m,n):
    return (m**2-n**2)*np.pi**2/(2*mu*a0**2)


def U1mn(m,n,tau):
    def f(tau1):
        return 1/(1+tau1)*np.exp(1j*T*omega0(m,n)*(tau)/(1+tau1))
    val,err=quadpy.quad(f,0,tau,epsabs=0.1, epsrel=0.1, limit=1000000)
    # print(err)
    return (-1)**(m+n+1)*2*m*n/(m**2-n**2)*val

ID=np.eye(N,dtype=complex)
U1=np.zeros((N,N),dtype=complex)

tau=0.8
for m in range(0,N):
    for n in range(0,N):
        if n==m:
            continue
        else:
            U1[m,n]=U1mn(m,n,tau)


U=ID+U1

psi0=np.zeros(N,dtype=complex)
psi0[0]=1
psi0[1]=1
psi0[2]=1
psi0/=np.linalg.norm(psi0,2)

psitau=U@psi0


t=tau*T
Q=10000
dt=t/Q
def K(t):
    retK=np.zeros((N,N),dtype=complex)
    for m in range(0,N):
        for n in range(0,N):
            if n==m:
                continue
            else:
                MM=m
                NN=n
                retK[m,n]=np.exp(1j*omega0(MM,NN)*T*t/(T+t))*(-1)**(MM+NN)/(T+t)*2*MM*NN/(MM**2-NN**2)
    return retK


def V(j):
    tj=j*dt
    return expm(dt*K(tj))

psit=psi0[:]
for j in range(1,Q+1):
    psit=V(j)@psit

err=np.linalg.norm(psit-psitau,2)
print(psit)
print(psitau)
print(err)