import numpy as np
import CNN_utils


# ref ：https://blog.csdn.net/Daycym/article/details/83826222
class Convolution:
    def __init__(self, ch_in, ch_out, kernel_size=(5, 5), stride=1, pad=0):
        # Init parameters
        self.W = np.random.random(size=(ch_out, ch_in, kernel_size[0], kernel_size[1])) - 0.5
        self.b = np.zeros((1, 1, 1, ch_out), dtype=np.float32)
        # get stride and pad
        self.stride = stride
        self.pad = pad
        # temp data, used in backword
        self.x = None
        self.feature_col = None
        self.W_col = None
        # gradient of W and b
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x

        kernel_nb, C, FH, FW = self.W.shape
        # size of input data
        N, C, H, W = x.shape
        # size of output feature
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)
        # transform array to col (im2col)
        self.feature_col = CNN_utils.im2col(x, FH, FW, self.stride, self.pad)
        # transform kernel to col vec (6, c*h*w)' ==> (c*h*w, 6)
        self.W_col = self.W.reshape(kernel_nb, -1).T
        # forward pass
        out = np.dot(self.feature_col, self.W_col) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        kernel_nb, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, kernel_nb)
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.feature_col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(kernel_nb, C, FH, FW)
        dcol = np.dot(dout, self.W_col.T)
        # reverse transform of im2col
        dx = CNN_utils.col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        # 展开
        col = CNN_utils.im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        # 最大值
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        # 转换
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        self.x = x
        self.arg_max = arg_max
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = CNN_utils.col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx


class FCLayer:
    def __init__(self, inplanes, outplanes, preweight=None):
        # Load pre weight
        if preweight is None:
            self.weight = np.random.rand(inplanes,
                                         outplanes) - 0.5  # self.weight = numpy.random.normal(inplanes, outplanes)
            self.bias = np.random.rand(outplanes) - 0.5
        else:
            self.weight, self.bias = preweight
        self.input = None
        self.output = None
        self.wgrad = np.zeros(self.weight.shape)
        self.bgrad = np.zeros(self.bias.shape)

    def forward(self, x):
        self.input = x
        self.output = np.dot(self.input, self.weight) + self.bias
        return self.output

    def backword(self, grad):
        # grad (10,1) expected
        self.bgrad = np.mean(grad, axis=0)
        self.wgrad = np.dot(self.input.T, grad)  # use direate of mat mul
        grad = np.dot(grad, self.weight.T)  # 666!
        return grad


class CEloss:
    def __init__(self):
        self.input = None
        self.q = None
        self.p = None
        self.grad = None
        self.target = None

    def forward(self, input, target=None):
        # input is the output of NN [64,10]
        self.input = input
        self.p = target

        temp = np.sum(np.exp(self.input - np.max(self.input, axis=1, keepdims=True)), axis=1, keepdims=True)
        self.q = np.exp(self.input - np.max(self.input, axis=1, keepdims=True)) / temp  # ==> [64,1]

        if self.p is not None:
            loss = -np.sum(np.sum(self.p * np.log(self.q + 1e-3), axis=1)) / self.p.shape[0]  # avg of all examples
            return self.q, loss
        else:
            return self.q

    def backword(self):
        self.grad = (self.q - self.p) / self.p.shape[0]
        return self.grad


class ReLU:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x
        x[self.input <= 0] *= 0
        self.output = x
        return self.output

    def backword(self, grad):
        grad[self.input > 0] *= 1
        grad[self.input <= 0] *= 0
        return grad
