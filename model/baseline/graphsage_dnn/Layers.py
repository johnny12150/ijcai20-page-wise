import tensorflow as tf


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', True)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            #            if self.logging and not self.sparse_inputs:
            #                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class custom_Dense(Layer):
    """Dense layer."""

    def __init__(self, w0, w1, w2, b0, b1, b2, dropout=0.,
                 act=tf.nn.relu, placeholders=None, bias=True, featureless=False,
                 sparse_inputs=False, **kwargs):
        super().__init__(**kwargs)

        self.dropout = dropout

        self.act = act
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.sparse_inputs = sparse_inputs
        if sparse_inputs:
            self.num_features_nonzero = placeholders['num_features_nonzero']
        '''用tf.variable_scope才可以重複使用變量'''
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = w0
            self.vars['weights_1'] = w1
            self.vars['weights_2'] = w2
            self.vars['bias'] = b0
            self.vars['bias_1'] = b1
            self.vars['bias_2'] = b2

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # transform
        hidden_1 = tf.matmul(x, self.vars['weights'])
        hidden_1 += self.vars['bias']
        hidden_1 = self.act(hidden_1)
        hidden_2 = tf.matmul(hidden_1, self.vars['weights_1'])
        hidden_2 += self.vars['bias_1']
        hidden_2 = self.act(hidden_2)
        hidden_3 = tf.matmul(hidden_2, self.vars['weights_2'])
        hidden_3 += self.vars['bias_2']
        # output = tf.nn.sigmoid(hidden_3)
        output = hidden_3
        return output



