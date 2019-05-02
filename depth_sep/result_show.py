import numpy as np
from numpy import array
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import argparse

# def plot_learning_rate():
#     tags = ['accuracy','sparsity','traintime','testtime']
#     units = ['-','-','s','s']
#     for idx in range(4):
#         tag = tags[idx]
#         unit = units[idx]
#         x_labels = np.arange(-2.,3.,0.5)
#         orfsvm = np.zeros((10,len(x_labels)))
#         urfsvm = np.zeros((10,len(x_labels)))
#         for prefix in range(1,11,1):
#             orfsvm[prefix-1,:] = np.loadtxt(
#                 'result/covtype_{0:s}{2:s}{1:s}'.format(
#                 'Gaussian',str(prefix), 'layer 2'))[:,idx]
#             urfsvm[prefix-1,:] = np.loadtxt(
#                 'result/covtype_{0:s}{2:s}{1:s}'.format(
#                     'ReLU',str(prefix),'layer 2'))[:,idx]
#
#         orfmean = np.mean(orfsvm,axis=0)
#         urfmean = np.mean(urfsvm,axis=0)
#         orfstd = np.std(orfsvm,axis=0)
#         urfstd = np.std(urfsvm,axis=0)

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
        # fig = plt.figure()
        # plt.title("Fourier vs ReLU Feature {} on Covtype".format(tag))
        # plt.xlabel('log opt rate (-)')
        # plt.ylabel('{0} ({1})'.format(tag,unit))
        # plt.xticks(x_labels)
        # plt.errorbar(x_labels,orfmean,yerr=orfstd,fmt='bs--',label='Fourier',fillstyle='none')
        # plt.errorbar(x_labels,urfmean,yerr=urfstd,fmt='gx:',label='ReLU')
        # plt.legend(loc=4)
        # plt.savefig('image/covtype_Fourier_vs_ReLU_{}.eps'.format(tag))
        # plt.close(fig)

def _extract_xy(dataset,feature):
    filename = 'result/{0:s}-{1:s}-screen-'.format(dataset,feature)
    with open(filename+'alloc','r') as f:
        result,params = eval(f.read())
    result_trim = [row[1:] for row in result[1:]]
    result_trim = np.array(result_trim)
    y = np.max(result_trim[:,:],axis=0)
    x = result[0][1:]
    return x,y

def plot_params(dataset):
    x,y1 = _extract_xy(dataset,'ReLU')
    _,y2 = _extract_xy(dataset,'Gaussian')
    fig = plt.figure()
    plt.title("Random Features Methods on "+dataset)
    plt.xlabel('log(Gamma)')
    plt.ylabel('accuracy')
    # plt.xticks(x)
    plt.ylim((0,1.01))
    plt.plot(x,y1,'x--',label='ReLU')
    plt.plot(x,y2,'o:',label='Fourier')
    plt.legend(loc=2)
    plt.savefig('image/{}-gamma.eps'.format(dataset))
    plt.close(fig)

def print_params(filename):
    with open(filename,'r') as f:
        result, params = eval(f.read())
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
    F_result = [row[1:] for row in result[1:]]
    F_result = np.array(F_result)
    x,y = np.unravel_index(np.argmax(F_result),
        F_result.shape)
    logGamma = result[0][y+1]
    lograte = result[x+1][0]
    print('best log(Gamma): ',logGamma)
    print('best log(rate): ',lograte)
    _dict_print(params)
    return logGamma,lograte,params

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

def print_test_results(filename):
    with open(filename,'r') as f:
        result,_,model_params,fit_params = eval(f.read())
    _dict_print(result)
    _dict_print(model_params)
    _dict_print(fit_params)
    return result

# For RF, 1-hidden-NN and 2-hidden-NN comparison
def plot_test_results():
    rootNN1 = './result/eldan-NN-H1-test/'
    rootNN2 = './result/eldan-NN-H2-test/'
    rootRF = './result/eldan-RF-test/'
    N_list_NN2 = [7,11,17,25,37,54,77,110,157]
    N_list_RF = N_list_NN1 = [20,40,80,160,320,640,1280,2560,5120]
    yNN1 = []
    yerrNN1 = []
    yNN2 = []
    yerrNN2 = []
    yRF = []
    yerrRF = []
    for idx in range(9):
        filename_NN1 = rootNN1 + 'eldan-NN-test-N' + str(N_list_RF[idx]) + '-ep100/results/output-alloc'
        result = print_test_results(filename_NN1)
        yNN1 += [result['accuracy']['mean']]
        yerrNN1 += [result['accuracy']['std']]

        filename_NN2 = rootNN2 + 'eldan-NN-test-N' + str(N_list_NN2[idx]) + '-ep100/results/output-alloc'
        result = print_test_results(filename_NN2)
        yNN2 += [result['accuracy']['mean']]
        yerrNN2 += [result['accuracy']['std']]

        filename_RF = rootRF + 'eldan-RF-test-N' + str(N_list_RF[idx]) + '-ep100/results/output-alloc'
        result = print_test_results(filename_RF)
        yRF += [result['accuracy']['mean']]
        yerrRF += [result['accuracy']['std']]

    plt.title("Performance of Shallow and Deep Models")
    plt.xlabel('log(# of parameters)')
    plt.ylabel('accuracy')
    plt.xticks(range(9))
    plt.errorbar(range(9),yRF,yerr=yerrRF,fmt='rs--',label='RF',fillstyle='none')
    plt.errorbar(range(9),yNN1,yerr=yerrNN1,fmt='gx:',label='NN1',fillstyle='none')
    plt.errorbar(range(9),yNN2,yerr=yerrNN2,fmt='bo-.',label='NN2',fillstyle='none')
    plt.legend(loc=4)
    plt.savefig('fig/depth_sep.eps')

if __name__ == '__main__':
    ## parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--file', type=str, help='name of result file')
    args = parser.parse_args()
    # print_params(args.file)
    # print_test_results(args.file)
    plot_test_results()
