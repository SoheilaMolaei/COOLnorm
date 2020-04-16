from tensorflow.keras.layers import Dense, Dropout
from layers import  GraphConvolution
from metrics import *

from absl import flags
FLAGS = flags.FLAGS

class Model( object ):
    def __init__(self, **kwargs):
        allowed_kwargs = { 'name', 'logging', 'input_rows' }
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations=[]

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN( Model ):
    def __init__( self, placeholders, **kwargs ):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_rows, self.input_dim = self.inputs.get_shape().as_list()
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.activations=[]
        self.optimizer = tf.train.AdamOptimizer( learning_rate=FLAGS.learning_rate )

        self.build()

    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)
    def _loss( self ):

        for layer in self.layers:
                for _ker in layer.kernel:
                    self.loss += FLAGS.weight_decay * tf.nn.l2_loss( _ker )


        # Cross entropy error

        self.loss += masked_softmax_cross_entropy( self.outputs,
                                                       self.placeholders['labels'],
                                                       self.placeholders['labels_mask'] )

    def _accuracy( self ):
        self.accuracy = masked_accuracy( self.outputs, self.placeholders['labels'],
                                             self.placeholders['labels_mask'] )

    def _build( self ):
        dims = [ self.input_dim, *FLAGS.hidden, self.output_dim ]

        for l, (din, dout) in enumerate( zip( dims, dims[1:] ) ):
            if l == 0:
                sparse_inputs = True
            else:
                sparse_inputs = False

            if l == len(dims)-2:
                activation = None
            else:
                activation = tf.nn.relu

            self.layers.append(
                GraphConvolution( input_rows=self.input_rows,
                                  input_dim=din,
                                  output_dim=dout,
                                  support=self.placeholders['support'],
                                  dropout=self.placeholders['dropout'],
                                  sparse_inputs=sparse_inputs,
                                  activation=activation,
                                  model=FLAGS.model
                                 ) )

    def predict( self ):
        return tf.nn.softmax( self.outputs )
