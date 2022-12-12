from sympy import *

#symbolic computation for example from paper

Jb1,Jb2,Jc,Jd,Delta,delta=symbols("Jb1,Jb2,Jc,Jd,Delta,delta",cls=Function,real=True)

Jb=symbols("Jb",cls=Function)

k,tau=symbols("k,tau",cls=Symbol,real=True)

# jcFunc=Jc(tau)
jcFunc=sin(tau)
dtaujcFunc=diff(jcFunc,tau)

# jdFunc=Jd(tau)
jdFunc=(1+tau**2)*(cos(tau)+1+tau**3*sin(tau))
dtaujdFunc=diff(jdFunc,tau)

# jbFunc=Jb(tau,k)
#
# jb1Func=Jb1(tau)
# jb2Func=Jb2(tau)
# jbFunc=Jb1(tau)+exp(I*k)*Jb2(tau)

jb1Func=tau**3*cos(tau)**3
jb2Func=(1+tau**2)*(cos(tau)**2+2)
jbFunc=jb1Func+exp(I*k)*jb2Func
dtaujbFunc=diff(jbFunc,tau)
dtaujb2Func=diff(jb2Func,tau)

# DFunc=Delta(tau,k)
DFunc=sqrt(jbFunc*conjugate(jbFunc)+jcFunc**2+jdFunc**2)
dtauDFunc=diff(DFunc,tau)

dtauDExpression=(conjugate(jbFunc)*dtaujbFunc+jbFunc*conjugate(dtaujbFunc)+2*jcFunc*dtaujcFunc+2*jdFunc*dtaujdFunc)/(2*DFunc)


# dFunc=delta(tau)
dFunc=sqrt(jcFunc**2+jdFunc**2)
dtaudFunc=diff(dFunc,tau)

dtaudExpression=(jcFunc*dtaujcFunc+jdFunc*dtaujdFunc)/dFunc


vec00=Matrix([-1/sqrt(2),conjugate(jbFunc)/(sqrt(2)*DFunc),
             jcFunc/(sqrt(2)*DFunc),jdFunc/(sqrt(2)*DFunc)])

vec10=Matrix([0,0,-jdFunc/dFunc,jcFunc/dFunc])

vec11=Matrix([0,dFunc/DFunc,-jbFunc*jcFunc/(dFunc*DFunc),-jbFunc*jdFunc/(dFunc*DFunc)])

vec20=Matrix([1/sqrt(2),conjugate(jbFunc)/(sqrt(2)*DFunc),
               jcFunc/(sqrt(2)*DFunc),jdFunc/(sqrt(2)*DFunc)])


##################################A matrix
# A0000in=-conjugate(vec00).dot(diff(vec00,tau))
#
A0000=1/(4*DFunc**2)*(conjugate(jbFunc)*dtaujbFunc-jbFunc*conjugate(dtaujbFunc))

# A1010in=-conjugate(vec10).dot(diff(vec10,tau))
#
A1010=0

# A1011in=-conjugate(vec10).dot(diff(vec11,tau))
#
A1011=jbFunc/(dFunc**2*DFunc)*(jcFunc*dtaujdFunc-jdFunc*dtaujcFunc)

# A1110in=-conjugate(vec11).dot(diff(vec10,tau))
#
A1110=conjugate(jbFunc)/(dFunc**2*DFunc)*(jdFunc*dtaujcFunc-jcFunc*dtaujdFunc)

# A1111in=-conjugate(vec11).dot(diff(vec11,tau))
#
A1111=1/(2*DFunc**2)*(jbFunc*conjugate(dtaujbFunc)-conjugate(jbFunc)*dtaujbFunc)


# A2020in=-conjugate(vec20).dot(diff(vec20,tau))

A2020=1/(4*DFunc**2)*(conjugate(jbFunc)*dtaujbFunc-jbFunc*conjugate(dtaujbFunc))


##############################################


####################################################dtau vec
# dtauvec00in=diff(vec00,tau)
dtauvec00=Matrix([0,1/(sqrt(2)*DFunc**2)*(conjugate(dtaujbFunc)*DFunc-conjugate(jbFunc)*dtauDFunc),
                  1/(sqrt(2)*DFunc**2)*(dtaujcFunc*DFunc-jcFunc*dtauDFunc),
                  1/(sqrt(2)*DFunc**2)*(dtaujdFunc*DFunc-jdFunc*dtauDFunc)])

# dtauvec10in=diff(vec10,tau)
dtauvec10=Matrix([0,0,-1/dFunc**2*(dtaujdFunc*dFunc-jdFunc*dtaudFunc),1/dFunc**2*(dtaujcFunc*dFunc-jcFunc*dtaudFunc)])


# dtauvec11in=diff(vec11,tau)
dtauvec11=Matrix([0,1/DFunc**2*(dtaudFunc*DFunc-dFunc*dtauDFunc),
                  -1/(dFunc**2*DFunc**2)*(dtaujbFunc*jcFunc*dFunc*DFunc+jbFunc*dtaujcFunc*dFunc*DFunc-jbFunc*jcFunc*dtaudFunc*DFunc-jbFunc*jcFunc*dFunc*dtauDFunc),
                  -1/(dFunc**2*DFunc**2)*(dtaujbFunc*jdFunc*dFunc*DFunc+jbFunc*dtaujdFunc*dFunc*DFunc-jbFunc*jdFunc*dtaudFunc*DFunc-jbFunc*jdFunc*dFunc*dtauDFunc)])

# dtauvec20in=diff(vec20,tau)

dtauvec20=Matrix([0,1/(sqrt(2)*DFunc**2)*(conjugate(dtaujbFunc)*DFunc-conjugate(jbFunc)*dtauDFunc),
                  1/(sqrt(2)*DFunc**2)*(dtaujcFunc*DFunc-jcFunc*dtauDFunc),
                  1/(sqrt(2)*DFunc**2)*(dtaujdFunc*DFunc-jdFunc*dtauDFunc)])

#########################################################
#######################################################dk vec
# dkvec00in=diff(vec00,k)
dkvec00=Matrix([0,1/(sqrt(2)*DFunc**3)*(-I*exp(-I*k)*jb2Func*DFunc**2+conjugate(jbFunc)*jb1Func*jb2Func*sin(k)),
                1/(sqrt(2)*DFunc**3)*jcFunc*jb1Func*jb2Func*sin(k),
                1/(sqrt(2)*DFunc**3)*jdFunc*jb1Func*jb2Func*sin(k)])
# dkvec10in=diff(vec10,k)
dkvec10=Matrix([0,0,0,0])



# dkvec11in=diff(vec11,k)
dkvec11=Matrix([0,dFunc/DFunc**3*jb1Func*jb2Func*sin(k),
                -jcFunc/(dFunc*DFunc**3)*(I*exp(I*k)*jb2Func*DFunc**2+jbFunc*jb1Func*jb2Func*sin(k)),
                -jdFunc/(dFunc*DFunc**3)*(I*exp(I*k)*jb2Func*DFunc**2+jbFunc*jb1Func*jb2Func*sin(k))])


# dkvec20in=diff(vec20,k)

dkvec20=Matrix([0,1/(sqrt(2)*DFunc**3)*(-I*exp(-I*k)*jb2Func*DFunc**2+conjugate(jbFunc)*jb1Func*jb2Func*sin(k)),
                1/(sqrt(2)*DFunc**3)*jcFunc*jb1Func*jb2Func*sin(k),
                1/(sqrt(2)*DFunc**3)*jdFunc*jb1Func*jb2Func*sin(k)])


###############################

############################# G matrix elements

# G0000in=I*conjugate(dkvec00).dot(dtauvec00)-I*conjugate(dtauvec00).dot(dkvec00)


G0000=1/(2*DFunc**4)*(-exp(I*k)*jb2Func*conjugate(dtaujbFunc)*DFunc**2
                        -exp(-I*k)*jb2Func*dtaujbFunc*DFunc**2
                        +exp(I*k)*jb2Func*conjugate(jbFunc)*DFunc*dtauDFunc
                        +exp(-I*k)*jb2Func*jbFunc*DFunc*dtauDFunc
                        +I*jbFunc*jb1Func*jb2Func*sin(k)*conjugate(dtaujbFunc)
                        -I*conjugate(jbFunc)*jb1Func*jb2Func*sin(k)*dtaujbFunc)


# G1010in=I*conjugate(dkvec10).dot(dtauvec10)-I*conjugate(dtauvec10).dot(dkvec10)

G1010=0


# G1011in=I*conjugate(dkvec10).dot(dtauvec11)-I*conjugate(dtauvec10).dot(dkvec11)

G1011=1/(dFunc**2*DFunc**3)*(exp(I*k)*jb2Func*jcFunc*dtaujdFunc*DFunc**2
                             -exp(I*k)*jb2Func*jdFunc*dtaujcFunc*DFunc**2
                             -I*jbFunc*jb1Func*jb2Func*jcFunc*dtaujdFunc*sin(k)
                             +I*jbFunc*jb1Func*jb2Func*jdFunc*dtaujcFunc*sin(k))

# G1110in=I*conjugate(dkvec11).dot(dtauvec10)-I*conjugate(dtauvec11).dot(dkvec10)

G1110=1/(dFunc**2*DFunc**3)*(exp(-I*k)*jb2Func*jcFunc*dtaujdFunc*DFunc**2
                             -exp(-I*k)*jb2Func*jdFunc*dtaujcFunc*DFunc**2
                             +I*conjugate(jbFunc)*jb1Func*jb2Func*jcFunc*dtaujdFunc*sin(k)
                             -I*conjugate(jbFunc)*jb1Func*jb2Func*jdFunc*dtaujcFunc*sin(k))


# G1111in=I*conjugate(dkvec11).dot(dtauvec11)-I*conjugate(dtauvec11).dot(dkvec11)

G1111=1/DFunc**4*(exp(-I*k)*jb2Func*dtaujbFunc*DFunc**2
                  +exp(I*k)*jb2Func*conjugate(dtaujbFunc)*DFunc**2
                  -exp(-I*k)*jb2Func*jbFunc*DFunc*dtauDFunc
                  -exp(I*k)*jb2Func*conjugate(jbFunc)*DFunc*dtauDFunc
                  +I*conjugate(jbFunc)*jb1Func*jb2Func*dtaujbFunc*sin(k)
                  -I*jbFunc*jb1Func*jb2Func*conjugate(dtaujbFunc)*sin(k))


# G2020in=I*conjugate(dkvec20).dot(dtauvec20)-I*conjugate(dtauvec20).dot(dkvec20)

G2020=1/(2*DFunc**4)*(-exp(I*k)*jb2Func*conjugate(dtaujbFunc)*DFunc**2
                      -exp(-I*k)*jb2Func*dtaujbFunc*DFunc**2
                      +exp(I*k)*jb2Func*conjugate(jbFunc)*DFunc*dtauDFunc
                      +exp(-I*k)*jb2Func*jbFunc*DFunc*dtauDFunc
                      +I*jbFunc*jb1Func*jb2Func*conjugate(dtaujbFunc)*sin(k)
                      -I*conjugate(jbFunc)*jb1Func*jb2Func*dtaujbFunc*sin(k))





##############################################Gamma matrix
Gamma0000=-(jb1Func*jb2Func*cos(k)+jb2Func**2)/(2*DFunc**2)
# Gm0000in=-I*conjugate(vec00).dot(dkvec00)




Gamma1010=0

# Gm1010in=-I*conjugate(vec10).dot(dkvec10)

Gamma1011=0
# Gm1011in=-I*conjugate(vec10).dot(dkvec11)





Gamma1110=0

# Gm1110in=-I*conjugate(vec11).dot(dkvec10)


Gamma1111=(jb1Func*jb2Func*cos(k)+jb2Func**2)/DFunc**2
# Gm1111in=-I*conjugate(vec11).dot(dkvec11)


Gamma2020=-(jb1Func*jb2Func*cos(k)+jb2Func**2)/(2*DFunc**2)

# Gm2020in=-I*conjugate(vec20).dot(dkvec20)



###################################################


#########################################S matrix=[Gamma,A]


GammaMatrix=Matrix([[Gamma0000,0,0,0],
                    [0,Gamma1010,Gamma1011,0],
                    [0,Gamma1110,Gamma1111,0],
                    [0,0,0,Gamma2020]])

AMatrix=Matrix([[A0000,0,0,0],
                [0,A1010,A1011,0],
                [0,A1110,A1111,0],
                [0,0,0,A2020]])


SMatrix=GammaMatrix*AMatrix-AMatrix*GammaMatrix

S1010=0
# S1010in=-A1011*Gamma1110+A1110*Gamma1011
# S1010in=SMatrix[1,1]


S1011=1/(dFunc**2*DFunc**3)*(jbFunc*jdFunc*dtaujcFunc*jb1Func*jb2Func*cos(k)
                             +jbFunc*jdFunc*dtaujcFunc*jb2Func**2
                             -jbFunc*jcFunc*dtaujdFunc*jb1Func*jb2Func*cos(k)
                             -jbFunc*jcFunc*dtaujdFunc*jb2Func**2)

# S1011in=-A1011*Gamma1111
# S1011in=SMatrix[1,2]


S1110=1/(dFunc**2*DFunc**3)*(conjugate(jbFunc)*jdFunc*dtaujcFunc*jb1Func*jb2Func*cos(k)
                             +conjugate(jbFunc)*jdFunc*dtaujcFunc*jb2Func**2
                             -conjugate(jbFunc)*jcFunc*dtaujdFunc*jb1Func*jb2Func*cos(k)
                             -conjugate(jbFunc)*jcFunc*dtaujdFunc*jb2Func**2)

# S1110in=SMatrix[2,1]
# S1110in=A1110*Gamma1111


S1111=0
# S1111in=SMatrix[2,2]
###################################

##########################################matrix L elems

E0=-DFunc
E1=0
E2=DFunc

L0010=I/(sqrt(2)*DFunc**2*dFunc)*(-jdFunc*dtaujcFunc+jcFunc*dtaujdFunc)

L0010=L0010.subs([(tau,0)])

# L0010in=(I*conjugate(vec00).dot(dtauvec10)/(E0-E1)).subs([(tau,0)])




L0011=I/(sqrt(2)*DFunc**3)*(dFunc*dtaujbFunc-jbFunc*dtaudFunc)

L0011=L0011.subs([(tau,0)])

# L0011in=(I*conjugate(vec00).dot(dtauvec11)/(E0-E1)).subs([(tau,0)])



L0020=I/(8*DFunc**3)*(conjugate(jbFunc)*dtaujbFunc-jbFunc*conjugate(dtaujbFunc))

L0020=L0020.subs([(tau,0)])
# L0020in=(I*conjugate(vec00).dot(dtauvec20)/(E0-E2)).subs([(tau,0)])




L1000=I/(sqrt(2)*DFunc**2*dFunc)*(jcFunc*dtaujdFunc-jdFunc*dtaujcFunc)
L1000=L1000.subs([(tau,0)])
# L1000in=(I/(E1-E0)*conjugate(vec10).dot(dtauvec00)).subs([(tau,0)])





L1020=I/(sqrt(2)*DFunc**2*dFunc)*(jdFunc*dtaujcFunc-jcFunc*dtaujdFunc)
L1020=L1020.subs([(tau,0)])
# L1020in=(I/(E1-E2)*conjugate(vec10).dot(dtauvec20)).subs([(tau,0)])





L1100=I/(sqrt(2)*DFunc**3)*(dFunc*conjugate(dtaujbFunc)-conjugate(jbFunc)*dtaudFunc)
L1100=L1100.subs([(tau,0)])
# L1100in=(I/(E1-E0)*conjugate(vec11).dot(dtauvec00)).subs([(tau,0)])


L1120=I/(sqrt(2)*DFunc**3)*(conjugate(jbFunc)*dtaudFunc-dFunc*conjugate(dtaujbFunc))
L1120=L1120.subs([(tau,0)])
# L1120in=(I/(E1-E2)*conjugate(vec11).dot(dtauvec20)).subs([(tau,0)])




L2000=I/(8*DFunc**3)*(jbFunc*conjugate(dtaujbFunc)-conjugate(jbFunc)*dtaujbFunc)
L2000=L2000.subs([(tau,0)])
# L2000in=(I/(E2-E0)*conjugate(vec20).dot(dtauvec00)).subs([(tau,0)])




L2010=I/(sqrt(2)*DFunc**2*dFunc)*(jdFunc*dtaujcFunc-jcFunc*dtaujdFunc)
L2010=L2010.subs([(tau,0)])
# L2010in=(I/(E2-E1)*conjugate(vec20).dot(dtauvec10)).subs([(tau,0)])



L2011=I/(sqrt(2)*DFunc**3)*(jbFunc*dtaudFunc-dFunc*dtaujbFunc)
L2011=L2011.subs([(tau,0)])
# L2011in=(I/(E2-E1)*conjugate(vec20).dot(dtauvec11)).subs([(tau,0)])

##############################################3

########################################dk A

dkA0000=1/(4*DFunc**2)*(
    (-I*exp(-I*k)*jb2Func*dtaujbFunc+I*exp(I*k)*conjugate(jbFunc)*dtaujb2Func-I*exp(I*k)*jb2Func*conjugate(dtaujbFunc)
     +I*exp(-I*k)*jbFunc*dtaujb2Func))\
    +1/(2*DFunc**4)*(conjugate(jbFunc)*dtaujbFunc-jbFunc*conjugate(dtaujbFunc))*jb1Func*jb2Func*sin(k)


# # dkA000in=diff(A0000,k)
#
# dkA1010=0
#
# dkA1011=1/(dFunc**2*DFunc**3)*(I*exp(I*k)*jb2Func*DFunc**2+jbFunc*jb1Func*jb2Func*sin(k))*(jcFunc*dtaujdFunc-jdFunc*dtaujcFunc)
#
# # dkA1011in=diff(A1011,k)
#
# dkA1110=1/(dFunc**2*DFunc**3)*(-I*exp(-I*k)*jb2Func*DFunc**2+conjugate(jbFunc)*jb1Func*jb2Func*sin(k))*(-jcFunc*dtaujdFunc+jdFunc*dtaujcFunc)
#
# # dkA1110in=diff(A1110,k)
#
#
# dkA1111=1/(2*DFunc**2)*(I*exp(I*k)*jb2Func*conjugate(dtaujbFunc)-I*exp(-I*k)*jbFunc*dtaujb2Func
#                         +I*exp(-I*k)*jb2Func*dtaujbFunc-I*exp(I*k)*conjugate(jbFunc)*dtaujb2Func)\
#     +1/(DFunc**4)*(jbFunc*conjugate(dtaujbFunc)-conjugate(jbFunc)*dtaujbFunc)*jb1Func*jb2Func*sin(k)
#
#
# # dkA1111in=diff(A1111,k)
#
#
# dkA2020=1/(4*DFunc**2)*(
#     (-I*exp(-I*k)*jb2Func*dtaujbFunc+I*exp(I*k)*conjugate(jbFunc)*dtaujb2Func-I*exp(I*k)*jb2Func*conjugate(dtaujbFunc)
#      +I*exp(-I*k)*jbFunc*dtaujb2Func))\
#     +1/(2*DFunc**4)*(conjugate(jbFunc)*dtaujbFunc-jbFunc*conjugate(dtaujbFunc))*jb1Func*jb2Func*sin(k)
#
# # dkA2020in=diff(A2020,k)
#
