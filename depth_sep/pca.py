import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from experiments import read_data
from result_show import print_params
from librf import optReLUSampler


def pca(dataset,N,m,gamma):
    X_tr = read_data(dataset)[0]
    X = Xtr[:m]
    d = len(X[0])
    k_RFF = np.random.randn(d,N) * np.sqrt(gamma)
    b_RFF = np.random.rand(N) * np.pi
    X_RFF = np.cos(X.dot(k_RFF) + b_RFF)
    k_RRF = np.random.randn(d+1,N)
    k_RRF = k_RRF / np.linalg.norm(k_RRF,axis=0)
    X_ext = np.concatenate((X,np.ones((m,1))),axis=1)
    X_RRF = np.maximum(X_ext.dot(k_RRF),0)
    _,s_RFF,_ = np.linalg.svd(X_RFF)
    s_RFF = s_RFF / np.max(s_RFF)
    _,s_RRF,_ = np.linalg.svd(X_RRF)
    s_RRF = s_RRF / np.max(s_RRF)
    # _,s_origin,_ = np.linalg.svd(X)
    # s_origin = s_origin / np.max(s_origin)
    fig = plt.figure()
    plt.plot(s_RFF,label='Fourier')
    plt.plot(s_RRF,label='ReLU')
    # plt.plot(np.log(s_origin),label='original')
    plt.legend(loc=1)
    plt.savefig('image/pca-'+dataset+'-reweight.eps')
    plt.close(fig)

if __name__ == '__main__':
    dataset = 'sine1-10'
    F_gamma,F_rate,R_rate = print_params(dataset)
    pca(dataset,20,500,10**F_gamma)
