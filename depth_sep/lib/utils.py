import os

def mkdir(name, prefix=None):
    root_dir = './result/'
    dirname = root_dir + name
    if not os.path.exists(os.path.dirname(dirname)):
        os.makedirs(os.path.dirname(dirname))
    plotdir = dirname + '/plots/'
    if not os.path.exists(os.path.dirname(plotdir)):
        os.makedirs(os.path.dirname(plotdir))
    resdir = dirname + '/results/'
    if not os.path.exists(os.path.dirname(resdir)):
        os.makedirs(os.path.dirname(resdir))
    if prefix != None:
        tbdir = dirname + '/tf/tensorboard-' + prefix + '/'
        if not os.path.exists(os.path.dirname(tbdir)):
            os.makedirs(os.path.dirname(tbdir))
        modeldir = dirname + '/tf/model-'+ prefix + '/'
        if not os.path.exists(os.path.dirname(modeldir)):
            os.makedirs(os.path.dirname(modeldir))
    else:
        modeldir = tbdir = resdir
    print("Results saved in " + dirname)
    return plotdir, resdir, modeldir, tbdir
