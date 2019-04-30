import numpy as np
import os
import librf
import libnn
import time
from lib.utils import mkdir
from datetime import datetime
from log import log
import argparse
from sklearn.preprocessing import StandardScaler
from libmnist import get_train_test_data
from multiprocessing import Pool
from functools import partial
from result_show import print_params
from sklearn.svm import SVC
DATA_PATH = 'data/'

def read_params(filename='params'):
    with open(filename,'r') as f:
        params = eval(f.read())
    return params

def _read_data(filename):
    data = np.load(DATA_PATH + filename)
    return data

def _unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_data(dataset):
    if dataset == 'mnist':
        Xtr,Ytr,Xts,Yts = get_train_test_data()
    elif dataset == 'cifar':
        X = []
        Y = []
        for idx in range(1): # set to 5 for the complete data set
            X.append(unpickle('data/cifar-10/data_batch_'+idx)[b'data'])
            Y.append(unpickle('data/cifar-10/data_batch_'+idx)[b'labels'])
        Xtr = np.concatenate(X,axis=0)
        Ytr = np.concatenate(Y,axis=0)
        Xts = unpickle('data/cifar-10/test_batch')[b'data']
        Yts = unpickle('data/cifar-10/test_batch')[b'labels']
    else:
        Xtr = _read_data(dataset+'-train-data.npy')
        Ytr = _read_data(dataset+'-train-label.npy')
        Xts = _read_data(dataset+'-test-data.npy')
        Yts = _read_data(dataset+'-test-label.npy')
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xts = scaler.transform(Xts)
    return Xtr,Ytr,Xts,Yts

def _validate(data,labels,folds,model_type,model_params,fit_params,index):
    kfolds_data = np.split(data,folds)
    kfolds_labels = np.split(labels,folds)
    Xts = kfolds_data.pop(index)
    Yts = kfolds_labels.pop(index)
    Xtr = np.concatenate(kfolds_data)
    Ytr = np.concatenate(kfolds_labels)
    clf = model_type(**model_params)
    clf.fit(Xtr,Ytr,**fit_params)
    score = clf.score(Xts,Yts)
    return score

def validate(data,labels,val_size,model_type,model_params,fit_params,folds=5,holdout=True):
    rand_list = np.random.permutation(len(data))
    X = data[rand_list[:val_size]]
    Y = labels[rand_list[:val_size]]
    f = partial(_validate,X,Y,folds,model_type,model_params,fit_params)
    if holdout:
        score = f(0)
        return score
    else:
        # with Pool() as p:
        #     score_list = p.map(f,range(folds))
        score_list = []
        for idx in range(folds):
            score_list.append(f(idx))
        return sum(score_list) / folds

def _train_and_test(Xtr,Ytr,Xts,Yts,model_type,model_params,fit_params):
    clf = model_type(**model_params)
    t1 = time.process_time()
    clf.fit(Xtr,Ytr,**fit_params)
    t2 = time.process_time()
    Ypr,_,sparsity = clf.predict(Xts)
    t3 = time.process_time()
    score = sum(Ypr == Yts) / len(Yts)
    return score,sparsity,t2-t1,t3-t2

def params_process(model, logGamma, lograte, params, tbdir, d):
    model_params = {}
    fit_params = {}
    if model == 'NN':
        model_params = {
            'dim':d,
            'width':params['N'],
            'depth':params['H'],
        }
        model_type = libnn.fullnn
    elif model == 'RF':
        model_params = {
            'n_old_features':d,
            'n_new_features':params['N'],
            'loss_fn':params['loss_fn'],
            'feature':params['feature'],
            'Gamma':10. ** logGamma
        }
        fit_params = {
            'bd':params['bd']
        }
        model_type = librf.RF
    fit_params['n_epoch'] = params['n_epoch']
    fit_params['opt_rate'] = 10. ** lograte
    fit_params['opt_method'] = 'adam'
    model_params['tbdir'] = tbdir
    model_params['classes'] = params['classes']
    model_params['task'] = params['task']
    model_params['gpu'] = params['gpu']
    return model_params, fit_params, model_type

def train_and_test(dataset,model='RF',params='auto',
    prefix='0'):
    # If params is a string, treat it as the allocated
    # params screen results. And choose the best params
    # config.
    if type(params) == 'str':
        logGamma,lograte,params = print_params(params)
    else:
        logGamma = params['logGamma']
        lograte = params['lograte']
    model = params['model']

    dirname = '{0:s}-{1:s}-test-N{2:d}-ep{3:d}'.format(dataset,model,params['N'],params['n_epoch'])
    _, resdir, _, tbdir = mkdir(dirname, prefix)

    Xtrain,Ytrain,Xtest,Ytest = read_data(dataset)
    model_params, fit_params, model_type = params_process(
        model, logGamma, lograte, params, tbdir)

    # only write log file for trial 0
    if prefix == '0':
        logfile = log('log/experiments.log','train and test')
        logfile.record(str(datetime.now()))
        logfile.record('{0} = {1}'.format('dataset',dataset))
        for key,val in model_params.items():
            logfile.record('{0} = {1}'.format(key,val))
        for key,val in fit_params.items():
            logfile.record('{0} = {1}'.format(key,val))
        logfile.save()

    score1,sparsity1,traintime1,testtime1 = _train_and_test(Xtrain,
        Ytrain,Xtest,Ytest,model_type,model_params,fit_params
        )
    output = {
            'accuracy':score1,
            'sparsity':sparsity1,
            'traintime':traintime1,
            'testtime':testtime1
        }
    finalop = [output,dataset,model_params,fit_params]
    filename = resdir + 'output-' + prefix
    with open(filename,'w') as f:
        f.write(str(finalop))

def screen_params(params,prefix='0'):
    model = params['model']
    dataset = params['dataset']
    val_size = params['val_size']
    folds = params['folds']
    task = params['task']
    N = params['N']
    lograte = params['lograte'][int(prefix)]
    if prefix == '0':
        # only write log file for trial 0
        logfile = log('log/experiments.log','screen params')
        logfile.record(str(datetime.now()))
        for key,val in params.items():
            logfile.record('{0} = {1}'.format(key,val))
        logfile.save()
    Xtrain,Ytrain,_,_ = read_data(dataset)

    results = []
    dirname = '{0:s}-{1:s}-N{2:d}-screen'.format(dataset,model,N)
    if model == 'RF':
        for logGamma in params['logGamma']:
            Gamma = 10**logGamma
            _, resdir, _, tbdir = mkdir(dirname, '{:.1f}-{:.1f}'.format(lograte,logGamma))
            model_params, fit_params, model_type = params_process(
                model, logGamma, lograte, params, tbdir, len(Xtrain[0]))

            model_params['Gamma'] = Gamma
            score = validate(Xtrain,Ytrain,val_size,model_type,
                model_params, fit_params, folds)
            results.append({'Gamma':Gamma,'score':score})
    else:
        logGamma = -100
        _, resdir, _, tbdir = mkdir(dirname, '{:.1f}-{:.1f}'.format(lograte,logGamma))
        model_params, fit_params, model_type = params_process(
            model, logGamma, lograte, params, tbdir, len(Xtrain[0]))
        score = validate(Xtrain,Ytrain,val_size,model_type,
            model_params, fit_params, folds)
        results.append({'Gamma':-100,'score':score})
    filename = resdir + 'output-' + prefix
    with open(filename,'w') as f:
        f.write(str(results))

def N_selecting(dataset, N, prefix='0'):
    params = read_params(dataset)
    params['N'] = N
    screen_params(params, prefix)

if __name__ == '__main__':
    ## parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--N', default=20, type=int,
            help='width of layer')
    parser.add_argument('--trial', default=0, type=str,
            help='index of learning rate')
    parser.add_argument('--file', default='eldan-params',
            type=str, help='file name of params')
    parser.add_argument('--seed', default=0, type=int,
            help='random seed')
    args = parser.parse_args()
    np.random.seed(args.seed)
    N_selecting(args.file, args.N, args.trial)
