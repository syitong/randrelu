import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def plot_learning_rate():
    tags = ['accuracy','sparsity','traintime','testtime']
    units = ['-','-','s','s']
    for idx in range(4):
        tag = tags[idx]
        unit = units[idx]
        x_labels = np.arange(-2.,3.,0.5)
        orfsvm = np.zeros((10,len(x_labels)))
        urfsvm = np.zeros((10,len(x_labels)))
        for prefix in range(1,11,1):
            orfsvm[prefix-1,:] = np.loadtxt(
                'result/covtype_{0:s}{2:s}{1:s}'.format(
                'Gaussian',str(prefix), 'layer 2'))[:,idx]
            urfsvm[prefix-1,:] = np.loadtxt(
                'result/covtype_{0:s}{2:s}{1:s}'.format(
                    'ReLU',str(prefix),'layer 2'))[:,idx]

        orfmean = np.mean(orfsvm,axis=0)
        urfmean = np.mean(urfsvm,axis=0)
        orfstd = np.std(orfsvm,axis=0)
        urfstd = np.std(urfsvm,axis=0)

        # opt vs unif feature selection
        # plt.title("opt vs unif feature selection on MNIST")
        # plt.xlabel('sample size (k)')
        # plt.ylabel('accuracy')
        # plt.xticks(samplesize/1000)
        # plt.errorbar(samplesize/1000,orfmean,yerr=orfstd,fmt='bs--',label='opt',fillstyle='none')
        # plt.errorbar(samplesize/1000,urfmean,yerr=urfstd,fmt='gx:',label='unif')
        # plt.legend(loc=4)
        # plt.savefig('image/opt_vs_unif.eps')

        # Gaussian vs ReLU random features
        fig = plt.figure()
        plt.title("Fourier vs ReLU Feature {} on Covtype".format(tag))
        plt.xlabel('log opt rate (-)')
        plt.ylabel('{0} ({1})'.format(tag,unit))
        plt.xticks(x_labels)
        plt.errorbar(x_labels,orfmean,yerr=orfstd,fmt='bs--',label='Fourier',fillstyle='none')
        plt.errorbar(x_labels,urfmean,yerr=urfstd,fmt='gx:',label='ReLU')
        plt.legend(loc=4)
        plt.savefig('image/covtype_Fourier_vs_ReLU_{}.eps'.format(tag))
        plt.close(fig)

def plot_params(dataset):
    with open('result/'+dataset+'-alloc','r') as f:
        result = eval(f.read())
    result_trim = [row[1:] for row in result[1:]]
    result_trim = np.array(result_trim)
    F_result = np.max(result_trim[:,:-1],axis=0)
    x_labels = result[0][1:-1]
    R_result = [max(result_trim[:,-1])] * len(x_labels)
    fig = plt.figure()
    plt.title("Random Features Methods on "+dataset)
    plt.xlabel('log(Gamma)')
    plt.ylabel('accuracy')
    plt.xticks(x_labels)
    plt.ylim((0,1.01))
    plt.plot(x_labels,F_result,'x--',label='Fourier')
    plt.plot(x_labels,R_result,':',label='ReLU')
    plt.legend(loc=1)
    plt.savefig('image/{}-gamma.eps'.format(dataset))
    plt.close(fig)

def print_params(dataset):
    with open('result/'+dataset+'-alloc','r') as f:
        result = eval(f.read())
    for row in result:
        if type(row[0]) == str:
            print('{:^20}'.format(row[0]),end='')
        else:
            print('{:^ 20}'.format(row[0]),end='')
        for item in row[1:]:
            if type(item) == str:
                print('{:>7}'.format(item),end='')
            else:
                print('{:>7.2f}'.format(item),end='')
        print('')
    F_result = [row[1:-1] for row in result[1:]]
    F_result = np.array(F_result)
    x,y = np.unravel_index(np.argmax(F_result),
        F_result.shape)
    F_gamma = result[0][y+1]
    F_rate = result[x+1][0]
    R_result = np.array([row[-1] for row in result[1:]])
    x = np.argmax(R_result)
    R_rate = result[x+1][0]
    print('best log(Gamma): ',F_gamma)
    print('best log(rate) for Fourier features: ',F_rate)
    print('best log(rate) for ReLU features: ',R_rate)
    return F_gamma,F_rate,R_rate

def _dict_print(dictx,loc=0):
    for key,value in dictx.items():
        print(' '*loc,end='')
        print(key,end='')
        if type(value) == dict:
            print('')
            _dict_print(value,loc+5)
        else:
            print(': {}'.format(value))
    return 1

def print_test_results(dataset):
    with open('result/'+dataset+'_test-alloc','r') as f:
        _dict_print(eval(f.read()))

if __name__ == '__main__':
    # plot_params('adult')
    # print_params('adult')
    print_test_results('covtype')
