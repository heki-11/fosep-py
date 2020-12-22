import scipy
import math
import numpy
import pandas as pd
from scipy import signal
from scipy.signal import TransferFunction
from scipy.optimize import linprog

def main():
    N = 2000   # Define parameters
    b_ratio = 6
    b_deg = 5
    R = 5
    vmax = 50
    amax = 10e3
    jmax = 5e6
    emax = 13e-3
    Ts = 0.001
    wn = 2*math.pi*25
    zeta = 0.1


    n = math.ceil(N/b_ratio) # Unchanged parameters
    theta_end = 2*math.pi*R
    pre = numpy.zeros((N,1))
    num = [wn**2]
    den = [1, 2*zeta*wn, wn**2]
    Gx = signal.TransferFunction(num,den)
    #print(isinstance(Gx,scipy.signal.lti))
    #Gx = scipy.signal.cont2discrete(Gx, dt=Ts)
    Gy = Gx
    Ux = pd.read_excel('Ux.xls',header=None)
    Utx = pd.read_excel('Utx.xls',header=None)
    #print(Ux)
    Uy = Ux
    Uty = Utx
    Dbase = numpy.ones( (N-1,N) )
    Dconcat = numpy.tril(Dbase,1)-2*numpy.tril(Dbase,0)+numpy.tril(Dbase,-1)
    Dbase2 = numpy.ones((N-2,N-1))
    Dconcat2 = numpy.tril(Dbase2,1)-2*numpy.tril(Dbase2,0)+numpy.tril(Dbase2,-1)
    Dbase3 = numpy.ones((N-3,N-2))
    Dconcat3 = numpy.tril(Dbase3,1)-2*numpy.tril(Dbase3,0)+numpy.tril(Dbase3,-1)
    xTBI = R*numpy.cos(pre/R)
    yTBI = R*numpy.sin(pre/R)-R
    gradx = -1*numpy.sin(pre/R)
    grady = 1*numpy.cos(pre/R)
    #print(numpy.linalg.pinv(Utx).shape)
    #print(Utx.shape)
    f1 = numpy.dot(Utx,numpy.linalg.pinv(Utx))
    f2 = numpy.dot(Uty,numpy.linalg.pinv(Uty))
    amaxvec = amax*numpy.ones((N-2,1))
    jmaxvec = jmax*numpy.ones((N-3,1))
    emaxvec = emax*numpy.ones((N,1))

    Aineq_mono = -Dconcat
    Aineq_v = Dconcat/Ts
    #print(gradx.shape)
    #print(numpy.diagflat(gradx).shape)
    Aineq_a = numpy.vstack((numpy.dot(Dconcat2,numpy.dot(Dconcat,numpy.diagflat(gradx))),
                           -numpy.dot(Dconcat2,numpy.dot(Dconcat,numpy.diagflat(gradx))),
                           numpy.dot(Dconcat2,numpy.dot(Dconcat,numpy.diagflat(grady))),
                           -numpy.dot(Dconcat2,numpy.dot(Dconcat,numpy.diagflat(grady)))))/Ts**2
    #print(Aineq_a.shape)
    Aineq_j = numpy.vstack((numpy.dot(Dconcat3,numpy.dot(Dconcat2,numpy.dot(Dconcat,numpy.diagflat(gradx)))),
                           -numpy.dot(Dconcat3,numpy.dot(Dconcat2,numpy.dot(Dconcat,numpy.diagflat(gradx)))),
                           numpy.dot(Dconcat3,numpy.dot(Dconcat2,numpy.dot(Dconcat,numpy.diagflat(grady)))),
                           -numpy.dot(Dconcat3,numpy.dot(Dconcat2,numpy.dot(Dconcat,numpy.diagflat(grady))))))/Ts**3
    #print(Aineq_j.shape)
    Aineq_end = numpy.identity(N)
    #print(N)
    #print(Aineq_end.shape)
    #print(f1.shape)
    #print(Ux.shape)
    Aineq_cont = numpy.vstack((numpy.multiply(numpy.identity(N)-f1,gradx),
                              numpy.multiply(numpy.identity(N)-f2,grady),
                              -numpy.multiply(numpy.identity(N)-f1,gradx),
                              -numpy.multiply(numpy.identity(N)-f2,grady)))
    #print(Aineq_cont.shape)
    Aineq = numpy.vstack((Aineq_mono,Aineq_cont,Aineq_end,
                         Aineq_a,Aineq_v,Aineq_j))
    #print(Aineq.shape)
    Aineq = numpy.dot(Aineq,Ux)

    bineq_mono = numpy.zeros((N-1,1))
    bineq_v = vmax*numpy.ones((N-1,1))
    bineq_end = theta_end*numpy.ones((N,1))
    bineq_a = numpy.vstack((numpy.dot(Dconcat2,numpy.dot(Dconcat,(numpy.multiply(gradx,pre)-xTBI)/Ts**2))+amaxvec,
                           -numpy.dot(Dconcat2,numpy.dot(Dconcat,(numpy.multiply(gradx,pre)-xTBI)/Ts**2))+amaxvec,
                           numpy.dot(Dconcat2,numpy.dot(Dconcat,(numpy.multiply(grady,pre)-yTBI)/Ts**2))+amaxvec,
                           -numpy.dot(Dconcat2,numpy.dot(Dconcat,(numpy.multiply(grady,pre)-yTBI)/Ts**2))+amaxvec))
    bineq_j = numpy.vstack((numpy.dot(Dconcat3,numpy.dot(Dconcat2,numpy.dot(Dconcat,(numpy.multiply(gradx,pre)-xTBI))))/Ts**3+jmaxvec,
                           -numpy.dot(Dconcat3,numpy.dot(Dconcat2,numpy.dot(Dconcat,(numpy.multiply(gradx,pre)-xTBI))))/Ts**3+jmaxvec,
                           numpy.dot(Dconcat3,numpy.dot(Dconcat2,numpy.dot(Dconcat,(numpy.multiply(grady,pre)-yTBI))))/Ts**3+jmaxvec,
                           -numpy.dot(Dconcat3,numpy.dot(Dconcat2,numpy.dot(Dconcat,(numpy.multiply(grady,pre)-yTBI))))/Ts**3+jmaxvec))
    bineq_cont = numpy.vstack((numpy.dot((numpy.identity(N)-f1),(numpy.multiply(pre,gradx)-xTBI))+emaxvec,
                              numpy.dot((numpy.identity(N)-f2),(numpy.multiply(pre,grady)-yTBI))+emaxvec,
                              -numpy.dot((numpy.identity(N)-f1),(numpy.multiply(pre,gradx)-xTBI))+emaxvec,
                              -numpy.dot((numpy.identity(N)-f2),(numpy.multiply(pre,grady)-yTBI))+emaxvec))
    #print(bineq_cont.shape)
    #print(bineq_j.shape)
    #print(bineq_a.shape)
    bineq = numpy.vstack((bineq_mono,bineq_cont,bineq_end,
                         bineq_a,bineq_v,bineq_j))

    Aeq = numpy.zeros((1,N))
    Aeq[0,0] = 1
    Aeq = numpy.dot(Aeq,Ux)
    beq = [0]

    f = -numpy.dot(numpy.ones((1,N)),Ux)
    opt = linprog(c=f, A_ub=Aineq, b_ub=bineq,
        A_eq = Aeq, b_eq = beq,options=dict(autoscale=True,presolve=False,disp=True))
    print(opt)


if __name__ == "__main__":
    main()
