import numpy as np
# from joblib import Parallel, delayed
import tensorflow as tf

class fullnn:
    counter = 0
    """
    This class is used to generate a fully L-layer and W-node-per-layer
    connected neural network that can generate and predict binary labels
    and be trained using SGD with minibatch.
    The nonlinear node simply use ReLU.
    """
    def __init__(self,dim,width,depth,classes,
            task='classification',gpu=-1):
        # Use the times of calls of class as random seed
        tf.set_random_seed(type(self).counter)
        type(self).counter += 1
        self._task = task
        self._dim = dim
        self._width = width
        self._depth = depth
        self._classes = classes
        self._learn_rate = 0.
        self._n_classes = len(classes)
        self._total_iter = 0
        self._total_epoch = 0
        if gpu >= 0:
            with tf.device('/gpu:'+str(gpu)):
                self._graph = tf.Graph()
        else:
            self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        self._model_fn()

    @property
    def dim(self):
        return self._dim
    @property
    def width(self):
        return self._width
    @property
    def depth(self):
        return self._depth
    @property
    def classes(self):
        return self._classes
    @property
    def total_iter(self):
        return self._total_iter
    @property
    def total_epoch(self):
        return self._total_epoch
    @property
    def trainable_params(self):
        var_list = {}
        with self._graph.as_default():
            for var in tf.trainable_variables():
                var_list[var.name] = self._sess.run(var)
        return var_list

    def _model_fn(self):
        with self._graph.as_default():
            x = tf.placeholder(dtype=tf.float32,
                shape=[None,self._dim],name='features')
            y = tf.placeholder(dtype=tf.uint8,
                shape=[None],name='labels')
            hl = x
            initializer = tf.glorot_normal_initializer()
            for idx in range(self._depth):
                hl_name = 'Hidden_Layer' + str(idx)
                hl = tf.layers.dense(inputs=hl,units=self._width,
                    kernel_initializer=initializer,
                    activation=tf.nn.relu,
                    name=hl_name)

            if self._task == 'classification':
                logits = tf.layers.dense(inputs=hl,units=self._n_classes,
                # use_bias=False,
                kernel_initializer=initializer,
                name='Logits')
                probabs = tf.nn.softmax(logits)
                tf.add_to_collection("Output",probabs)
            elif self._task == 'regression':
                logits = tf.layers.dense(inputs=hl,units=1,
                        use_bias=False,
                        kernel_initializer=initializer,
                        name='Logits')
                logits = tf.reshape(logits,shape=[-1])
            tf.add_to_collection("Output",logits)
            if self._task == 'classification':
                onehot_labels = tf.one_hot(indices=y,depth=self._n_classes)
                log_loss = tf.losses.softmax_cross_entropy(
                    onehot_labels=onehot_labels,logits=logits
                )
                tf.add_to_collection('Loss',log_loss)
            elif self._task == 'regression':
                loss = tf.losses.mean_squared_error(labels=y,
                    predictions=logits)
                tf.add_to_collection('Loss',loss)
            self._sess.run(tf.global_variables_initializer())

    def predict(self,data,batch_size=50):
        with self._graph.as_default():
            if self._task == 'classification':
                logits,probabs = tf.get_collection('Output')
                predictions = {
                    'indices':tf.argmax(input=logits,axis=1),
                    'probabilities':probabs
                }
            elif self._task == 'regression':
                logits = tf.get_collection('Output')[0]
                predictions = {
                    'labels':logits
                }
        classes = []
        probabilities = []
        idx = 0
        while idx < len(data):
            t = idx + batch_size
            batch = data[idx:t,:]
            batch.reshape(len(batch),-1)
            idx = t
            feed_dict = {'features:0':batch}
            results = self._sess.run(predictions,feed_dict=feed_dict)
            if self._task == 'classification':
                classes.extend([self._classes[index] for index in results['indices']])
                probabilities.extend(results['probabilities'])
            elif self._task == 'regression':
                classes.extend(results['labels'])
        return classes,probabilities

    def score(self,data,labels):
        predictions,_ = self.predict(data)
        s = 0.
        for idx in range(len(data)):
            if self._task == 'classification':
                s += (predictions[idx]==labels[idx])
            elif self._task == 'regression':
                s += (predictions[idx] - labels[idx])**2
        accuracy = s / len(data)
        return accuracy

    def fit(self,data,labels,opt_rate=1.,opt_method='sgd',batch_size=200,n_epoch=5):
        self._learn_rate = opt_rate
        with self._graph.as_default():
            loss = tf.get_collection('Loss')[0]
            if self._task == 'classification':
                label_idx = [self._classes.index(label) for label in labels]
                label_idx = np.array(label_idx)
            elif self._task == 'regression':
                label_idx = np.array(labels)
            if opt_method == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learn_rate)
            elif opt_method == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self._learn_rate)
            global_step = tf.Variable(0,trainable=False,name='global')
            train_op = optimizer.minimize(loss=loss,
                global_step=global_step,name='Train_op')
            self._sess.run(tf.global_variables_initializer())
        for idx in range(n_epoch):
            rand_indices = np.random.permutation(len(data)) - 1
            for jdx in range(len(data)//batch_size):
                batch_indices = rand_indices[jdx*batch_size:(jdx+1)*batch_size]
                feed_dict = {
                    'features:0':data[batch_indices],
                    'labels:0':label_idx[batch_indices]
                }
                if jdx % 100 == 1:
                    print('epoch: {2:d}, iter: {0:d}, loss: {1:.4f}'.format(
                        self.total_iter, self._sess.run(loss,feed_dict), self.total_epoch))
                self._sess.run(train_op,feed_dict)
                self._total_iter += 1
            self._total_epoch += 1

    def get_params(self,deep=False):
        params = {
            'dim': self._dim,
            'width': self._width,
            'depth': self._depth,
            'classes': self._classes,
            'learn_rate': self._learn_rate
        }
        return params

    def __del__(self):
        self._sess.close()
        print('Session is closed.')
