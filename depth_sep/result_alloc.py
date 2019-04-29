import numpy as np
from numpy import array
import argparse

def read_params(filename='params'):
    with open(filename,'r') as f:
        params = eval(f.read())
    return params

def screen_params_alloc(params):
    dataset = params['dataset']
    feature = params['feature']
    lograte = params['lograte']
    logGamma = params['logGamma']
    N = params['N']
    filename = 'result/{0:s}-{1:d}-screen-'.format(dataset,N)
    row = ['log(rate)\log(Gamma)']
    row.extend(logGamma)
    output = [row]
    for idx,rate in enumerate(lograte):
        row = []
        try:
            with open(filename + str(idx),'r') as f:
                result = eval(f.read())
        except:
            print('lograte list is not run out')
            break
        row.append(rate)
        for item in result:
            row.append(item['score'])
        output.append(row)
    finalop = [output,params]
    with open(filename+'alloc','w') as f:
        f.write(str(finalop))

def screen_params_append(params):
    dataset = params['dataset']
    feature = params['feature']
    lograte = params['lograte']
    logGamma = params['logGamma']
    N = params['N']
    filename = 'result/{0:s}-{1:d}-screen-'.format(dataset,N)
    with open(filename+'alloc','r') as fr:
        result,params = eval(fr.read())
    result[0].extend(logGamma)
    for idx,rate in enumerate(lograte):
        try:
            with open(filename+str(idx),'r') as fr:
                new_result = eval(fr.read())
        except:
            print('lograte list is not run out')
            break
        for item in new_result:
            result[idx+1].append(item['score'])
    sortidx = np.argsort(result[0][1:]) + 1
    updated = []
    for row in result:
        newrow = [row[0]]
        for idx in sortidx:
            newrow.append(row[idx])
        updated.append(newrow)
    finalop = [updated,params]
    with open(filename+'alloc','w') as f:
        f.write(str(finalop))

def train_and_test_alloc(dataset,feature,trials):
    filename = 'result/{0:s}-{1:s}-test-'.format(dataset,feature)
    tags = ['accuracy','sparsity','traintime','testtime']
    alloc = {}
    for idx in range(4):
        tag = tags[idx]
        result = np.zeros(trials)
        for prefix in range(trials):
            with open(filename+str(prefix),'r') as fr:
                dict1,_,model_params,fit_params = eval(fr.read())
            result[prefix] = dict1[tag]
        mean = np.mean(result)
        std = np.std(result)
        alloc[tag] = {'mean':mean,'std':std}
    finalop = [alloc,dataset,model_params,fit_params]
    with open(filename+'alloc','w') as fw:
        fw.write(str(finalop))

if __name__ == '__main__':
    ## parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--N', default=20, type=int,
            help='width of layer')
    parser.add_argument('--file', default='eldan-params',
            type=str, help='file name of params')
    args = parser.parse_args()
    params = read_params('eldan-params')
    params['N'] = args.N
    screen_params_alloc(params)