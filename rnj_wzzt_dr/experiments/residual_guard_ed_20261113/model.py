"""Small history-only output calibration; frozen candidate scores are unchanged."""
import numpy as np
from ...observations import load


def calibrate(score,observed):
    H=np.asarray([p['H'] for p in score['parameters']]);o=load(observed)
    P=np.column_stack((o.p[64:],o.q[64:]));Y=o.y[64:]
    mean=np.einsum('tc,knc->ktn',P,H);residual=Y[None]-mean
    # One multiplicative correction per voltage output, shrunk towards one by
    # one residual-variance scale. No impedance or topology is refitted.
    c=1+np.sum(mean*residual,axis=1)/(np.sum(mean*mean,axis=1)+len(P)*score['sigma2'])
    return c,c[:,:,None]*H


def coefficients(A,B,K,F,scale,sigma2,qratio=.35):
    """Expected A/B evidence under mean KP, minus the no-action evidence."""
    n=A.shape[0];D=A-B;E=K-A;normal=np.sqrt(2*sigma2)
    M=(D[:,:n]+qratio*D[:,n:])*scale/normal
    N=(E[:,:n]+qratio*E[:,n:])*scale/normal
    b=F@D.T/normal;e=F@E.T/normal
    Q=4*(M.T@M+N.T@M+M.T@N)
    a=(b.reshape(24,4,n).sum(1)@M+e.reshape(24,4,n).sum(1)@M+
       b.reshape(24,4,n).sum(1)@N)
    return (Q+Q.T)/2,a


def direct(A,B,K,F,shift,sigma2):
    D=A-B;E=K-A;b=F@D.T;d=shift@D.T;e=F@E.T;v=shift@E.T
    return float(np.sum(2*b*d+d*d+2*e*d+2*v*b+2*v*d)/(2*sigma2))


class Evidence:
    def __init__(self,base,means=None):
        self.base=base;self.shape=base.shape;self.dimension=base.dimension
        self.Q=list(base.Q);self.a=list(base.a);self.pairs=[]
        if means is not None:
            for j in base.competitors:
                for i,k in ((base.target,j),(j,base.target)):
                    A,B,K=base.H[i],base.H[k],means[i]
                    Q,a=coefficients(A,B,K,base.forecast,base.scale,base.sigma2,base.qratio)
                    self.Q.append(Q);self.a.append(a);self.pairs.append((i,k,K))
        self.Q=np.array(self.Q);self.a=np.array(self.a)

    def thresholds(self,gamma):return np.r_[np.full(len(self.base.M),gamma),np.zeros(len(self.pairs))]

    def values(self,x):
        x=np.asarray(x).reshape(self.shape)
        return np.einsum('hi,kij,hj->k',x,self.Q,x)+2*np.einsum('khi,hi->k',self.a,x)

    def direct(self,x):
        shift=self.base.shift(x);original=self.base.direct_gains(x)
        extra=[direct(self.base.H[i],self.base.H[j],K,self.base.forecast,shift,self.base.sigma2)
               for i,j,K in self.pairs]
        return np.r_[original,extra]


def along(evidence,direction,required,alpha):
    """Intersect feasible positive scaling intervals; handles indefinite guards."""
    d=np.asarray(direction).reshape(evidence.shape);maximum=float(np.max(abs(d)))
    if maximum==0:return None
    high=alpha/maximum;low=0.
    quadratic=np.einsum('hi,kij,hj->k',d,evidence.Q,d)
    linear=2*np.einsum('khi,hi->k',evidence.a,d)
    for q,l,g in zip(quadratic,linear,required,strict=True):
        if g==0:
            if q>0:low=max(low,max(0.,-l/q))
            elif q<0:
                if l<=0:return None
                high=min(high,-l/q)
            elif l<0:return None
        elif q>0:
            root=np.sqrt(l*l+4*q*g)
            low=max(low,2*g/(root+l) if l>=0 else (root-l)/(2*q))
        elif q==0:
            if l<=0:return None
            low=max(low,g/l)
        else:
            discriminant=l*l+4*q*g
            if l<=0 or discriminant<0:return None
            root=np.sqrt(discriminant);low=max(low,2*g/(l+root));high=min(high,(l+root)/(-2*q))
    if low>high+1e-12*max(1.,high):return None
    scale=min(high,low*(1+2e-8)+1e-12)
    point=d*scale
    return point if np.min(evidence.values(point)-required)>=-1e-8 else None
