import numpy as np
import pandas as pd
from scipy.linalg import expm
from multiprocessing import Pool


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


