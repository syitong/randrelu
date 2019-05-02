import os

def mkdir(name, prefix=None):
    root_dir = './result/'
    dirname = root_dir + name
    if not os.path.exists(os.path.dirname(dirname)):
        os.makedirs(os.path.dirname(dirname), exist_ok=True)
    plotdir = dirname + '/plots/'
    if not os.path.exists(os.path.dirname(plotdir)):
        os.makedirs(os.path.dirname(plotdir), exist_ok=True)
    resdir = dirname + '/results/'
    if not os.path.exists(os.path.dirname(resdir)):
        os.makedirs(os.path.dirname(resdir), exist_ok=True)
    if prefix != None:
        tbdir = dirname + '/tf/tensorboard-' + prefix + '/'
        if not os.path.exists(os.path.dirname(tbdir)):
            os.makedirs(os.path.dirname(tbdir), exist_ok=True)
        modeldir = dirname + '/tf/model-'+ prefix + '/'
        if not os.path.exists(os.path.dirname(modeldir)):
            os.makedirs(os.path.dirname(modeldir), exist_ok=True)
    else:
        modeldir = tbdir = resdir
    print("Results saved in " + dirname)
    return plotdir, resdir, modeldir, tbdir
