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
        # YOUR CODE HERE
        # self.grads['bias'] = grad
        # input_shape = self.input.shape
        # grad_shape = grad.shape
        # #consider the batch_size, need to reshape with batch_size condition
        # input_shape = input_shape + (1,)
        # grad_shape = grad_shape[:-1] + (1,) + (grad_shape[-1],)
        # # print(input_shape)
        # # print(grad_shape)
        # self.grads['weight'] = self.input.reshape(input_shape) @ grad.reshape(grad_shape)
        if len(grad.shape) == 1:
            self.grads['bias'] = grad
            self.grads['weight'] = np.outer(self.input, grad)
            return self.params['weight'] @ grad.T
        else:
            self.grads['bias'] = np.sum(grad, axis=0)
            self.grads['weight'] = self.input.T @ grad
            #grad has shape (batch_size, d_out), weight has shape (d_in, d_out)
            grad_in=np.zeros([grad.shape[0],self.d_in])
            for b in range(grad.shape[0]):
                for i in range(self.d_in):
                    for o in range(self.d_out):
                        grad_in[b][i] += (grad[b][o] * self.params['weight'][i][o]).item()
            return grad_in
        
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
        '''
        Calculate bias and init weight
        '''
        self.grads['bias'] = grad.sum(axis=(0, 2, 3))
        X_padded = self._inputs_preprocess(self.input)
        _, in_c, h_pad, w_pad = X_padded.shape
        _, out_c, out_h, out_w = grad.shape
        k_h, k_w = self.kernel_shape[1], self.kernel_shape[2]
        s_h, s_w = self.stride
        self.grads['weight'] = np.zeros_like(self.params['weight'])
        '''
        Calculate weight
        '''
        for i in range(in_c):
            for o in range(out_c):
                for kh in range(k_h):
                    for kw in range(k_w):
                        x_slice = X_padded[:, i, kh:kh+out_h*s_h:s_h, kw:kw+out_w*s_w:s_w]
                        
                        self.grads['weight'][i, kh, kw, o] = np.sum(x_slice * grad[:, o, :, :])
        '''
        Calculate d_in_padded
        '''
        d_in_padded = np.zeros_like(X_padded)
            #     for in_c_i in range(in_c):
            # for out_c_i in range(out_c):
            #     kernel = self.params['weight'][in_c_i, :, :, out_c_i]
            #     for r in range(out_h):
            #         r_start = r * s_h
            #         for c in range(out_w):
            #             c_start = c * s_w
            #             patch = X[:, in_c_i, r_start: r_start +
            #                       k_h, c_start: c_start+k_w] * kernel
            #             Y[:, out_c_i, r,
            #                 c] += patch.reshape(bsz, -1).sum(axis=-1)
        # for o in range(out_c):
        #     for i in range(in_c):
        #         kernel = self.params['weight'][i, :, :, o]

        #         for kh in range(k_h):
        #             for kw in range(k_w):
        #                 grad_region = grad[:, o, :, :] * kernel[kh, kw]

        #                 h_start = kh
        #                 w_start = kw
        #                 h_end = h_start + out_h * s_h
        #                 w_end = w_start + out_w * s_w

        #                 d_in_padded[:, i, h_start:h_end:s_h, w_start:w_end:s_w] += grad_region
        for in_c_i in range(in_c):
            for out_c_i in range(out_c):
                kernel = self.params['weight'][in_c_i, :, :, out_c_i]
                for r in range(out_h):
                    r_start = r * s_h
                    for c in range(out_w):
                        c_start = c * s_w
                        # r and c are the indices of the output
                        # r_start and c_start are the indices of the input
                        grad_region = grad[:, out_c_i, r, c]
                        for kw in range(k_w):
                            for kh in range(k_h):
                                d_in_padded[:, in_c_i, r_start+kh, c_start+kw] += kernel[kh, kw] * grad_region
        '''
        Calculate d_in
        '''
        pad_h = self.padding[2]
        pad_w = self.padding[3]

        if pad_h != (0, 0) or pad_w != (0, 0):
            d_in = d_in_padded[:, :, 
                               pad_h[0]:-pad_h[1], 
                               pad_w[0]:-pad_w[1]]
        else:
            d_in = d_in_padded

        return d_in
        
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
        self.input = inputs
        ##############################################################################
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
                #pool has shape (batch_sz, in_c, k_h*k_w)

                _argmax = np.argmax(pool, axis=2)[:, :, np.newaxis]
                argmax[:, :, r, c] = _argmax.squeeze(axis=2)

                # get max elements
                _max_pool = np.take_along_axis(
                    pool, _argmax, axis=2).squeeze(axis=2)
                max_pool[:, :, r, c] = _max_pool

        self.X_shape = X.shape
        self.out_shape = (out_h, out_w)
        self.argmax = argmax
        return max_pool

    def _backward(self, grad):
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        # YOUR CODE HERE
        #grad has shape (batch_sz, in_c, out_h, out_w)
        batch_sz, in_c, out_h, out_w = grad.shape
        s_h, s_w = self.stride
        k_h, k_w = self.kernel_shape
        grad_pool = np.zeros(self.X_shape)
        h_pad, w_pad = self.padding[2], self.padding[3]
        # print("padding shape is",self.padding)
        max_h = self.argmax // k_w
        max_w = self.argmax % k_w
        # b_idx = np.arange(batch_sz)[:]
        for r in range(out_h):
            r_start = r * s_h
            for in_c_i in range(in_c):
                for c in range(out_w):
                    c_start = c * s_w
                    for b in range(batch_sz):
                        grad_pool[b, in_c_i, r_start + max_h[b, in_c_i, r, c], c_start + max_w[b, in_c_i, r, c]] += grad[b, in_c_i, r, c]
                    # grad_pool[b_idx, :, r_start + max_h[b_idx, :, r, c], c_start + max_w[b_idx, :, r, c]] = grad[b_idx, :, r, c]
                
        '''
        Deal with padding
        '''
        if h_pad != (0, 0) or w_pad != (0, 0):
            grad_pool = grad_pool[:, :, h_pad[0]:-h_pad[1], w_pad[0]:-w_pad[1]]
        else:
            grad_pool = grad_pool
        return grad_pool
                
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
        # print(f'input shape{self.input_shape}, output shape{self.output_shape}')
        return grad.reshape(self.input_shape)
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
class ReLU(Activation):

    def func(self, x):
        self.input = x
        return np.maximum(x, 0.0)

    def _backward(self, grad):
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        # YOUR CODE HERE
        return grad * (self.input > 0)
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
class Tanh(Activation):

    def func(self, x):
        self.input = x
        return np.tanh(x)

    def _backward(self, grad):
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        # YOUR CODE HERE
        return grad * (1 - np.tanh(self.input) ** 2)
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################