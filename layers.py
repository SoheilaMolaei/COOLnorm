import tensorflow as tf
from tensorflow.python.keras.utils.tf_utils import smart_cond
from tensorflow.python.keras.backend import learning_phase
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras.engine.input_spec import InputSpec

from absl import flags
FLAGS = flags.FLAGS


class GraphConvolution( tf.keras.layers.Layer ):
    """Graph Convolutional Layer."""

    def __init__( self,
                  input_rows,
                  input_dim,
                  output_dim,
                  support,        # input_rows x input_rows
                  activation=None,
                  use_bias=False,
                  kernel_initializer='glorot_uniform',
                  bias_initializer='zeros',
                  kernel_regularizer=None,
                  bias_regularizer=None,
                  kernel_constraint=None,
                  bias_constraint=None,
                  dropout=0.,
                  sparse_inputs=False,
                  featureless=False,
                  model='COOL',
                  **kwargs ):
        super( GraphConvolution, self ).__init__( **kwargs )

        self.input_rows = input_rows
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.support    = support

        self.activation = activation
        self.use_bias   = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer   = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer   = regularizers.get(bias_regularizer)
        self.kernel_constraint  = constraints.get(kernel_constraint)
        self.bias_constraint    = constraints.get(bias_constraint)

        self.dropout       = dropout
        self.sparse_inputs = sparse_inputs
        self.featureless   = featureless
        self.model         = model

        self.supports_masking = True
        self.input_spec = InputSpec( min_ndim=2 )

    def build( self, input_shape ):
        self.input_spec = InputSpec( min_ndim=2, axes={ -1: self.input_dim } )

        self.kernel = []
        for i in range( len(self.support) ):
            self.kernel.append( self.add_weight(
                      'kernel{}'.format(i),
                       shape=[self.input_dim, self.output_dim],
                       initializer=self.kernel_initializer,
                       regularizer=self.kernel_regularizer,
                       constraint=self.kernel_constraint,
                       dtype=self.dtype,
                       trainable=True ) )
        if self.use_bias:
            self.bias = self.add_weight(
                        'bias',
                        shape=[ self.output_dim, ],
                        initializer=self.bias_initializer,
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint,
                        dtype=self.dtype,
                        trainable=True )
        else:
            self.bias = None
        self.built = True

    def call( self, x, idx=0 ):
        if self.sparse_inputs:
            random_tensor = 1-self.dropout
            random_tensor += tf.random_uniform( tf.shape(x.values) )
            dropout_mask = tf.cast( tf.floor(random_tensor), dtype=tf.bool )
            x = tf.sparse_retain(x, dropout_mask) * (1./(1-self.dropout))

        else:
            x = tf.nn.dropout( x, rate=self.dropout )

        supports = list()
        for _ker, _sup in zip( self.kernel, self.support ):
            if self.featureless:
                pre_sup = _ker
            elif self.sparse_inputs:
                pre_sup = tf.sparse_tensor_dense_matmul( x, _ker )
            else:
                pre_sup = tf.matmul( x, _ker )

            if self.model in (  'COOL', 'COOLnorm' ):
                    support = tf.sparse_tensor_dense_matmul( _sup, pre_sup )
                    supports.append( support )

            else:
                    raise RuntimeError( 'unknown model' )



        output = tf.add_n( supports )

        if self.use_bias: output += self.bias

        if self.activation is not None:
            output = self.activation( output )

        return output
