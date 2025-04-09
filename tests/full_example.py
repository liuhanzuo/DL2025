import numpy as np

from module import Zeros, XavierUniform, Activation, module

class Linear(module):
    def __init__(self, d_in, d_out, w_init=XavierUniform(), b_init=Zeros()):

        super().__init__()

        self.initializers = {
            "weight": w_init,
            'bias': b_init,
        }

        self.input = None
        self.d_in = d_in
        self.d_out = d_out

        if d_in:
            self.shapes = {
                "weight": [d_in, d_out],
                "bias": [d_out]
            }

            self._init_params()

    def _forward(self, inputs):
        if not self.is_init:
            d_in = inputs.shape[-1]
            self.shapes = {
                "weight": [d_in, self.d_out],
                "bias": [self.d_out]
            }
            self.d_in = d_in
            self._init_params()

        # `@` is the matrix multiplication operator in NumPy
        out = inputs @ self.params['weight'] + self.params['bias']
        self.input = inputs
        return out

    def _backward(self, grad):
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        self.grads = {
            "weight": np.zeros((self.d_in, self.d_out)),
            "bias": np.zeros(self.d_out)
        }
        self.grads['weight'] = self.input.reshape(-1,1) @ grad.reshape(1,-1)
        self.grads['bias'] = grad
        return grad @ self.params['weight'].T
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

    @property
    def param_names(self):
        return ('weight', 'bias')

    @property
    def weight(self):
        return self.params['weight']

    @property
    def bias(self):
        return self.params['bias']
class Conv2D(module):
    """
    Implement 2D convolution layer
    :param kernel: A list/tuple of int that has length 4 (in_channels, height, width, out_channels)
    :param stride: A list/tuple of int that has length 2 (height, width)
    :param padding: String ["SAME", "VALID"]
    :param w_init: weight initializer
    :param b_init: bias initializer
    """

    def __init__(self,
                 kernel,
                 stride=(1, 1),
                 padding="SAME",
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__()

        self.kernel_shape = kernel
        self.stride = stride
        self.initializers = {"weight": w_init, "bias": b_init}
        self.shapes = {"weight": self.kernel_shape,
                       "bias": self.kernel_shape[-1]}

        self.padding_mode = padding
        assert padding in ['SAME', 'VALID']
        if padding == 'SAME' and stride != (1, 1):
            raise RunTimeError(
                "padding='SAME' is not supported for strided convolutions.")
        self.padding = None

        self._init_params()

    def _forward(self, inputs):
        """
        :param inputs:  shape (batch_size, in_c, in_h, in_w)
        :return outputs: shape (batch_size, out_c, out_h, out_w)
        where batch size is the number of images
        """
        assert len(
            inputs.shape) == 4, 'Expected shape of inputs is (batch_size, in_c, in_h, in_w).'
        in_c, k_h, k_w, out_c = self.kernel_shape
        s_h, s_w = self.stride
        X = self._inputs_preprocess(inputs)
        bsz, _, h, w = X.shape

        out_h = (h - k_h) // s_h + 1
        out_w = (w - k_w) // s_w + 1
        Y = np.zeros([bsz, out_c, out_h, out_w])
        for in_c_i in range(in_c):
            for out_c_i in range(out_c):
                kernel = self.params['weight'][in_c_i, :, :, out_c_i]
                for r in range(out_h):
                    r_start = r * s_h
                    for c in range(out_w):
                        c_start = c * s_w
                        patch = X[:, in_c_i, r_start: r_start +
                                  k_h, c_start: c_start+k_w] * kernel
                        Y[:, out_c_i, r,
                            c] += patch.reshape(bsz, -1).sum(axis=-1)
        self.input = inputs
        return Y + self.params['bias'].reshape(1, -1, 1, 1)
    
    def _backward(self, grad):
        """
        Compute gradients w.r.t layer parameters and backward gradients.
        :param grad: gradients from previous layer 
            with shape (batch_size, out_c, out_h, out_w)
        :return d_in: gradients to next layers 
            with shape (batch_size, in_c, in_h, in_w)
        """
        assert len(
            grad.shape) == 4, 'Expected shape of upstream gradient is (batch_size, out_c, out_h, out_w)'
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        # YOUR CODE HERE
        in_c, k_h, k_w, out_c = self.kernel_shape
        s_h, s_w = self.stride
        X = self._inputs_preprocess(self.input)
        bsz, _, h, w = X.shape
        out_h = (h - k_h) // s_h + 1
        out_w = (w - k_w) // s_w + 1

        bsz, _, h, w = self.input.shape
        self.grads= {
            "weight": np.zeros((in_c, k_h, k_w, out_c)),
            "bias": np.zeros(out_c)
        }

        for in_c_i in range(in_c):
            for out_c_i in range(out_c):
                for r in range(out_h):
                    r_start = r * s_h
                    for c in range(out_w):
                        c_start = c * s_w
                        for i in range(bsz):
                            self.grads['weight'][in_c_i, :, :, out_c_i] +=  grad[i, out_c_i, r, c] * X[i, in_c_i, r_start: r_start + k_h, c_start: c_start + k_w]
        self.grads['bias'] = np.sum(grad,axis=(0, 2, 3))

        Cgrad = np.zeros(self.input.shape)
        for in_c_i in range(in_c):
            for out_c_i in range(out_c):
                kernel = self.params['weight'][in_c_i, :, :, out_c_i]
                for r in range(out_h):
                    r_start = r * s_h
                    for c in range(out_w):
                        c_start = c * s_w
                        for bs in range(bsz):
                            for rt in range(k_h):
                                for ct in range(k_w):
                                    if r_start + rt - self.padding[2][0] in range(h) and c_start + ct - self.padding[3][0] in range(w):
                                        Cgrad[bs, in_c_i, r_start + rt - self.padding[2][0], c_start + ct - self.padding[3][0]] += grad[bs, out_c_i, r, c] * kernel[rt, ct]
        return Cgrad
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

    def _inputs_preprocess(self, inputs):
        _, _, in_h, in_w = inputs.shape
        _, k_h, k_w, _ = self.kernel_shape
        # padding calculation
        if self.padding is None:
            self.padding = self.get_padding_2d(
                (in_h, in_w), (k_h, k_w), self.stride, self.padding_mode)
        return np.pad(inputs, pad_width=self.padding, mode="constant")

    def get_padding_2d(self, in_shape, k_shape, stride, mode):

        def get_padding_1d(w, k, s):
            if mode == "SAME":
                pads = s * (w - 1) + k - w
                half = pads // 2
                padding = (half, half) if pads % 2 == 0 else (half, half + 1)
            else:
                padding = (0, 0)
            return padding

        h_pad = get_padding_1d(in_shape[0], k_shape[0], stride[0])
        w_pad = get_padding_1d(in_shape[1], k_shape[1], stride[1])
        return (0, 0), (0, 0), h_pad, w_pad

    @property
    def param_names(self):
        return "weight", "bias"

    @property
    def weight(self):
        return self.params['weight']

    @property
    def bias(self):
        return self.params['bias']
class MaxPool2D(module):

    def __init__(self, pool_size, stride, padding="VALID"):
        """
        Implement 2D max-pooling layer
        :param pool_size: A list/tuple of 2 integers (pool_height, pool_width)
        :param stride: A list/tuple of 2 integers (stride_height, stride_width)
        :param padding: A string ("SAME", "VALID")
        """
        super().__init__()
        self.kernel_shape = pool_size
        self.stride = stride

        self.padding_mode = padding
        self.padding = None

    def _forward(self, inputs):
        """
        :param inputs:  shape (batch_size, in_c, in_h, in_w)
        """
        s_h, s_w = self.stride
        k_h, k_w = self.kernel_shape
        batch_sz, in_c, in_h, in_w = inputs.shape

        # zero-padding
        if self.padding is None:
            self.padding = self.get_padding_2d(
                (in_h, in_w), (k_h, k_w), self.stride, self.padding_mode)
        X = np.pad(inputs, pad_width=self.padding, mode="constant")
        padded_h, padded_w = X.shape[2:4]

        out_h = (padded_h - k_h) // s_h + 1
        out_w = (padded_w - k_w) // s_w + 1

        # construct output matrix and argmax matrix
        max_pool = np.empty(shape=(batch_sz, in_c, out_h, out_w))
        argmax = np.empty(shape=(batch_sz, in_c, out_h, out_w), dtype=int)
        for r in range(out_h):
            r_start = r * s_h
            for c in range(out_w):
                c_start = c * s_w
                pool = X[:, :, r_start: r_start+k_h, c_start: c_start+k_w]
                pool = pool.reshape((batch_sz, in_c, -1))

                _argmax = np.argmax(pool, axis=2)[:, :, np.newaxis]
                argmax[:, :, r, c] = _argmax.squeeze(axis=2)

                # get max elements
                _max_pool = np.take_along_axis(
                    pool, _argmax, axis=2).squeeze(axis=2)
                max_pool[:, :, r, c] = _max_pool

        self.X_shape = X.shape
        self.out_shape = (out_h, out_w)
        self.argmax = argmax
        self.input = inputs
        return max_pool

    def _backward(self, grad):
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        # YOUR CODE HERE
        s_h, s_w = self.stride
        k_h, k_w = self.kernel_shape
        batch_sz, in_c, in_h, in_w = self.input.shape

        out_h, out_w = self.out_shape

        Pgrad = np.zeros(self.input.shape)
        for r in range(out_h) :
            r_start = r * s_h
            for c in range(out_w) :
                c_start = c * s_w
                for bsz in range(batch_sz) :
                    for ch in range(in_c) :
                        rt = self.argmax[bsz, ch, r, c] // k_w
                        ct = self.argmax[bsz, ch, r, c] % k_w
                        Pgrad[bsz, ch, r_start + rt, c_start + ct] = grad[bsz, ch, r, c]
        return Pgrad
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

    def get_padding_2d(self, in_shape, k_shape, stride, mode):

        def get_padding_1d(w, k, s):
            if mode == "SAME":
                pads = s * (w - 1) + k - w
                half = pads // 2
                padding = (half, half) if pads % 2 == 0 else (half, half + 1)
            else:
                padding = (0, 0)
            return padding

        h_pad = get_padding_1d(in_shape[0], k_shape[0], stride[0])
        w_pad = get_padding_1d(in_shape[1], k_shape[1], stride[1])
        return (0, 0), (0, 0), h_pad, w_pad
class Reshape(module):

    def __init__(self, *output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.input_shape = None

    def _forward(self, inputs):
        self.input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], *self.output_shape)

    def _backward(self, grad):
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        # YOUR CODE HERE
        return grad.reshape(self.input_shape)
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
class ReLU(Activation):

    def func(self, x):
        return np.maximum(x, 0.0)

    def _backward(self, grad):
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        # YOUR CODE HERE
        return grad * (self.inputs > 0)
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
class Tanh(Activation):

    def func(self, x):
        return np.tanh(x)

    def _backward(self, grad):
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        # YOUR CODE HERE
        Tgrad = np.zeros_like(self.inputs)
        for index, x in np.ndenumerate(self.inputs):
            Tgrad[index] = grad[index] * (1 - np.tanh(x)**2)
        return Tgrad
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################