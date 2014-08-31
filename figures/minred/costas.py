#! /usr/bin/env python
import numpy as n, aipy as a

def reps_from_file(f):
    """
    read in pre-solved costas arrays
    only the y-axis values are stored
    """
    if type(f) is str: f = open(f)
    return n.array([[int(i) for i in L.split()] for L in f.readlines()]) - 1

def from_rep(r, N=None, cycle=0):
    """
    makes the 2d representation from the file data
    """
    if N is None: N = len(r)
    d = n.zeros((N,N))
    i = (n.arange(len(r)) + cycle) % len(r)
    d[i,r] = 1
    return d

def welch1(p, alpha):
    """
    creates a costas array from a prime p 
    again only creates the y axis 
    """
    N = p - 1
    i = [1]
    while len(i) < N: i.append((i[-1] * alpha) % p)
    i = n.array(i) - 1
    assert(not n.any(i[1:] == i[0]))
    return i

def welch2(p, alpha):
    return welch1(p,alpha)[1:] - 1

def rconv2(x,y):
    """
    convolution of two arrays 
    """
    return n.fft.irfft2(n.fft.rfft2(x) * n.conj(n.fft.rfft2(y)))

def arconv2(x):
    """
    convolution of array with self 
    """
    return n.fft.irfft2(n.abs(n.fft.rfft2(x))**2)

def recenter(x, null_auto=True):
    if null_auto: x[0,0] = 0
    return a.img.recenter(x, n.array(x.shape)/2)

if __name__ == '__main__':
    import sys
    import pylab
    cmap = pylab.get_cmap('gist_yarg')
    prms = {'cmap':pylab.get_cmap('gist_yarg'), 'interpolation':'nearest', 'origin':'lower'}
    if False: # if True will read from file given in command line arg
        rs = reps_from_file(sys.argv[-1])
    else:
        p = 13
        #p = 37
        #p = 71
        rs = [] # set of represenatations
        for alpha in range(2,p):
            try: rs.append(welch1(p,alpha)); print alpha
            except(AssertionError): pass
        rs = n.array(rs)
        print 'Welch method generated %d solutions for p=%d' % (rs.shape[0], p)
    N = rs.shape[-1] 
    _d = []
    ds = []
    flag = 0
    pylab.ion()
    for cnt, r in enumerate(rs):
        for c1 in range(N):
            d = from_rep(r, 2*N, cycle=c1) # 2d array 
            uv = n.round(n.abs(arconv2(d))) # uv sampling
            _uv = uv.copy(); _uv[0,0] = 0 # gets rid of all N correllations with self
            if _uv.max() > 1: continue # checks for non-redundancy
            d1 = rconv2(d, uv)[:N,:N] # convolve array with uv sampling -- array where non-zero entries only generate redundant samples
            vacant = n.where(n.round(d1) == 0) # sets real rounding errors to 0
            if len(vacant[0]) > 0:
            #if True:
                print p, cnt, c1, vacant
                _d.append(d.copy())
                d[vacant] = 1
                ds.append(d)
                uv = arconv2(_d[-1])
                _uv = uv.copy(); _uv[0,0] = 0
                d1 = rconv2(_d[-1], uv)[:N,:N]
                _uv2 = arconv2(d); _uv2[0,0] = 0 # new sampling pattern
                print flag, _uv.sum(), _uv2.sum(), uv.size
                if flag == 0:
                    pylab.subplot(231)
                    plt0 = pylab.imshow(from_rep(r, N, 0), **prms)
                    pylab.grid()
                    pylab.subplot(232)
                    plt1 = pylab.imshow(_d[-1][:N,:N], **prms)
                    pylab.grid()
                    pylab.subplot(233)
                    plt2 = pylab.imshow(recenter(uv), extent=(-N,N,-N,N), **prms)
                    pylab.subplot(234)
                    plt3 = pylab.imshow(1-d1.clip(0,1), vmin=0, vmax=1, **prms)
                    pylab.grid()
                    pylab.subplot(235)
                    plt4 = pylab.imshow(d[:N,:N], **prms)
                    pylab.grid()
                    pylab.subplot(236)
                    plt5 = pylab.imshow(recenter(arconv2(d)), extent=(-N-1,N,-N-1,N), **prms)
                else:
                    print 'setting data'
                    plt0.set_data(from_rep(r, N, 0))
                    plt1.set_data(_d[-1][:N,:N])
                    plt2.set_data(recenter(uv))
                    plt3.set_data(1-d1.clip(0,1))
                    plt4.set_data(d[:N,:N])
                    plt5.set_data(recenter(arconv2(d)))
                flag += 1
                pylab.draw()
                #import time; time.sleep(2)
    print n.where(ds[-1])
    pylab.ioff()
    pylab.show()
    _d = n.array(_d)
    d = n.array(ds)
    if False:
        pylab.subplot(221)
        pylab.imshow(_d[0,:N,:N], **prms)
        pylab.subplot(222)
        uv = arconv2(_d[0])
        pylab.imshow(recenter(uv), **prms)
        pylab.subplot(223)
        d1 = rconv2(_d[0], uv)[:N,:N]
        pylab.imshow(d1.clip(0,1), vmin=0, vmax=1, **prms)
        pylab.subplot(224)
        pylab.imshow(d[0,:N,:N], **prms)
    else:
        m1 = int(n.ceil(n.sqrt(float(d.shape[0]))))
        m2 = int(n.ceil(d.shape[0]/float(m1)))
        for i in range(d.shape[0]):
            uv = arconv2(d[i])
            d1 = rconv2(d[i], uv)[:N,:N]
            pylab.subplot(m2,m1,i+1)
            #pylab.imshow(d1.clip(0,1), vmin=0, vmax=1, **prms)
            pylab.imshow(uv, **prms)
    pylab.show()
