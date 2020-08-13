import tensorflow as tf

from keras.optimizers import Optimizer
from keras import backend as K
from keras.legacy import interfaces

import numpy as np
import zmq
from ast import literal_eval as make_tuple
from py_rmpe_server.py_rmpe_data_iterator import RawDataIterator

import six
if six.PY3:
  buffer_ = memoryview
else:
  buffer_ = buffer  # noqa


class MultiSGD(Optimizer):
    """
    Modified SGD with added support for learning multiplier for kernels and biases
    as suggested in: https://github.com/fchollet/keras/issues/5920
    Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, lr_mult=None, **kwargs):
        super(MultiSGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, dtype='float32', name='lr')
            self.momentum = K.variable(momentum, dtype='float32', name='momentum')
            self.decay = K.variable(decay, dtype='float32', name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.lr_mult = lr_mult

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):

            if p.name in self.lr_mult:
                multiplied_lr = lr * self.lr_mult[p.name]
            else:
                multiplied_lr = lr

            v = self.momentum * m - multiplied_lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - multiplied_lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(MultiSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



## Data Iterator

class DataIteratorBase:

    def __init__(self, batch_size = 10):

        self.batch_size = batch_size

        self.split_point = 38
        self.vec_num = 38
        self.heat_num = 19

        self.keypoints = [None]*self.batch_size #this is not passed to NN, will be accessed by accuracy calculation


    def gen_raw(self): # this function used for test purposes in py_rmpe_server

        while True:
            yield tuple(self._recv_arrays())

    def gen(self):
        batches_x, batches_x1, batches_x2, batches_y1, batches_y2 = \
            [None]*self.batch_size, [None]*self.batch_size, [None]*self.batch_size, \
            [None]*self.batch_size, [None]*self.batch_size

        sample_idx = 0

        for foo in self.gen_raw():

            if len(foo)==4:
                data_img, mask_img, label, kpts = foo
            else:
                data_img, mask_img, label = foo
                kpts = None

            # image
            dta_img = np.transpose(data_img, (1, 2, 0))
            batches_x[sample_idx]=dta_img[np.newaxis, ...]

            # mask - the same for vec_weights, heat_weights
            vec_weights = np.repeat(mask_img[:,:,np.newaxis], self.vec_num, axis=2)
            heat_weights = np.repeat(mask_img[:,:,np.newaxis], self.heat_num, axis=2)

            batches_x1[sample_idx]=vec_weights[np.newaxis, ...]
            batches_x2[sample_idx]=heat_weights[np.newaxis, ...]

            # label
            vec_label = label[:self.split_point, :, :]
            vec_label = np.transpose(vec_label, (1, 2, 0))
            heat_label = label[self.split_point:, :, :]
            heat_label = np.transpose(heat_label, (1, 2, 0))

            batches_y1[sample_idx]=vec_label[np.newaxis, ...]
            batches_y2[sample_idx]=heat_label[np.newaxis, ...]

            self.keypoints[sample_idx] = kpts

            sample_idx += 1

            if sample_idx == self.batch_size:
                sample_idx = 0

                batch_x = np.concatenate(batches_x)
                batch_x1 = np.concatenate(batches_x1)
                batch_x2 = np.concatenate(batches_x2)
                batch_y1 = np.concatenate(batches_y1)
                batch_y2 = np.concatenate(batches_y2)

                yield [batch_x, batch_x1,  batch_x2], \
                       [batch_y1, batch_y2,
                        batch_y1, batch_y2,
                        batch_y1, batch_y2,
                        batch_y1, batch_y2,
                        batch_y1, batch_y2,
                        batch_y1, batch_y2]

                self.keypoints = [None] * self.batch_size

    def keypoints(self):
        return self.keypoints


class DataGeneratorClient(DataIteratorBase):

    def __init__(self, host, port, hwm=20, batch_size=10, limit=None):

        super(DataGeneratorClient, self).__init__(batch_size)

        self.limit = limit
        self.records = 0

        """
        :param host:
        :param port:
        :param hwm:, optional
          The `ZeroMQ high-water mark (HWM)
          <http://zguide.zeromq.org/page:all#High-Water-Marks>`_ on the
          sending socket. Increasing this increases the buffer, which can be
          useful if your data preprocessing times are very random.  However,
          it will increase memory usage. There is no easy way to tell how
          many batches will actually be queued with a particular HWM.
          Defaults to 10. Be sure to set the corresponding HWM on the
          receiving end as well.
        :param batch_size:
        :param shuffle:
        :param seed:
        """
        self.host = host
        self.port = port
        self.hwm = hwm
        self.socket = None

        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)
        self.socket.set_hwm(self.hwm)
        self.socket.connect("tcp://{}:{}".format(self.host, self.port))


    def _recv_arrays(self):
        """Receive a list of NumPy arrays.
        Parameters
        ----------
        socket : :class:`zmq.Socket`
        The socket to receive the arrays on.
        Returns
        -------
        list
        A list of :class:`numpy.ndarray` objects.
        Raises
        ------
        StopIteration
        If the first JSON object received contains the key `stop`,
        signifying that the server has finished a single epoch.
        """

        if self.limit is not None and self.records > self.limit:
            raise StopIteration

        headers = self.socket.recv_json()
        if 'stop' in headers:
            raise StopIteration
        arrays = []

        for header in headers:
            data = self.socket.recv()
            buf = buffer_(data)
            array = np.frombuffer(buf, dtype=np.dtype(header['descr']))
            array.shape = make_tuple(header['shape']) if isinstance(header['shape'], str) else header['shape']
            # this need for comparability with C++ code, for some reasons it is string here, not tuple

            if header['fortran_order']:
                array.shape = header['shape'][::-1]
                array = array.transpose()
            arrays.append(array)

        self.records += 1
        return arrays


class DataIterator(DataIteratorBase):

    def __init__(self, file, shuffle=True, augment=True, batch_size=10, limit=None):

        super(DataIterator, self).__init__(batch_size)

        self.limit = limit
        self.records = 0

        self.raw_data_iterator = RawDataIterator(file, shuffle=shuffle, augment=augment)
        self.generator = self.raw_data_iterator.gen()


    def _recv_arrays(self):

        while True:

            if self.limit is not None and self.records > self.limit:
                raise StopIteration

            tpl = next(self.generator, None)
            if tpl is not None:
                self.records += 1
                return tpl

            if self.limit is None or self.records < self.limit:
                print("Staring next generator loop cycle")
                self.generator = self.raw_data_iterator.gen()
            else:
                raise StopIteration


# For testing

import numpy as np
from io import StringIO
import PIL.Image
from IPython.display import Image, display

def showBGRimage(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    a[:,:,[0,2]] = a[:,:,[2,0]] # for B,G,R order
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

def showmap(a, fmt='png'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

#def checkparam(param):
#    octave = param['octave']
#    starting_range = param['starting_range']
#    ending_range = param['ending_range']
#    assert starting_range <= ending_range, 'starting ratio should <= ending ratio'
#    assert octave >= 1, 'octave should >= 1'
#    return starting_range, ending_range, octave

def getJetColor(v, vmin, vmax):
    c = np.zeros((3))
    if (v < vmin):
        v = vmin
    if (v > vmax):
        v = vmax
    dv = vmax - vmin
    if (v < (vmin + 0.125 * dv)): 
        c[0] = 256 * (0.5 + (v * 4)) #B: 0.5 ~ 1
    elif (v < (vmin + 0.375 * dv)):
        c[0] = 255
        c[1] = 256 * (v - 0.125) * 4 #G: 0 ~ 1
    elif (v < (vmin + 0.625 * dv)):
        c[0] = 256 * (-4 * v + 2.5)  #B: 1 ~ 0
        c[1] = 255
        c[2] = 256 * (4 * (v - 0.375)) #R: 0 ~ 1
    elif (v < (vmin + 0.875 * dv)):
        c[1] = 256 * (-4 * v + 3.5)  #G: 1 ~ 0
        c[2] = 255
    else:
        c[2] = 256 * (-4 * v + 4.5) #R: 1 ~ 0.5                      
    return c

def colorize(gray_img):
    out = np.zeros(gray_img.shape + (3,))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y,x,:] = getJetColor(gray_img[y,x], 0, 1)
    return out

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad