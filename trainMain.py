from __future__ import division, absolute_import, print_function
from copy import deepcopy
import networkx as nx
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import os, gc, random, time, itertools
import tensorflow as tf
if tf.__version__.startswith('2'):
    tf.disable_v2_behavior()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
import sys
import scipy.sparse as sp
import numpy as np

from utils import sparse_to_tuple,construct_feed_dict, \
                  preprocess_features, preprocess_adj, preprocess_high_order_adj,load_data, input_data
from models import GCN

from absl import app, flags
FLAGS = flags.FLAGS
flags.DEFINE_enum(  'dataset', 'cora', input_data , 'Dataset' )
flags.DEFINE_enum(  'model', 'COOLnorm',
                   [ 'COOL', 'COOLnorm' ],
                     'Model' )

flags.DEFINE_float(   'learning_rate', 0.01, 'initial learning rate.' )
flags.DEFINE_float(   'dropout', 0.5, 'Dropout rate (1 - keep probability).' )
flags.DEFINE_integer( 'epochs', 500, 'Number of epochs to train.' )
flags.DEFINE_list(    'hidden', ['64',], 'size of hidden layer(s)' )
flags.DEFINE_float(   'weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.' )
flags.DEFINE_integer( 'early_stoping', 100, 'Tolerance for early stopping (# of epochs).' )        # 0: no stop 1: simple early stop 2: more strict conditions
flags.DEFINE_integer( 'max_degree', 3, 'Maximum Chebyshev polynomial degree.' )
flags.DEFINE_integer( 'repeat', 1, 'number of repeats' )
flags.DEFINE_integer( 'alfa', 0, 'number of repeats' )
flags.DEFINE_integer( 'beta', 1, 'number of repeats' )
# for COOLnorm
flags.DEFINE_integer( 'order', 5, 'order of high-order GCN' )
flags.DEFINE_float(   'threshold', 1e-4, 'A threshold to apply nodes filtering on random walk matrix.' )


def build_model( adj, features, n_classes ):
    placeholders = {
        'features': tf.sparse_placeholder( tf.float32, shape=tf.constant(features[2],dtype=tf.int64) ),
        'labels': tf.placeholder(tf.float32, shape=(None, n_classes)),
        'labels_mask': tf.placeholder(tf.int32),
        'noise': tf.placeholder( tf.float32, shape=() ),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'alfa': tf.placeholder( tf.float32, shape=() ),
        'beta': tf.placeholder( tf.float32, shape=() ),
    }

    if FLAGS.model == 'COOL':
        support = [ sparse_to_tuple( preprocess_adj(adj) ) ]
        model_func = GCN

    elif FLAGS.model == 'COOLnorm':
        support = [ sparse_to_tuple( preprocess_high_order_adj( adj, FLAGS.order, FLAGS.threshold ) ) ]
        model_func = GCN


    else:
        raise ValueError( 'Invalid argument for model: ' + str(FLAGS.model) )

    placeholders['support'] = [ tf.sparse_placeholder(tf.float32) for _ in support ]

    model = model_func( placeholders )
    return model, support, placeholders


def evaluate(sess,model, features, support, labels, placeholders, mask , dropout=0.,alfa=0,beta=1 ):
    feed_dict_val = construct_feed_dict( features, support, labels, mask, placeholders, dropout ,alfa,beta)
    outs_val = sess.run( [model.loss, model.accuracy], feed_dict=feed_dict_val )
    return outs_val[0], outs_val[1] 


def train_Model( dataset, data_seed, init_seed ):
    print( '{} Model on {}'.format( FLAGS.model, dataset ) )

    tf.reset_default_graph()
    adj, features, labels, train_mask, val_mask, test_mask = load_data( dataset, data_seed )

    #Feature Selection part 
    y_train = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_train = np.argmax(y_train, axis=1)
#     clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(features[train_mask], y_train[train_mask])
#     model = SelectFromModel(clf, prefit=True)
#     features = model.transform(features)
    
    #alfa A+ beta A' (Clique Finding) 
    graphMain=nx.from_numpy_matrix(adj.todense())
    listClique=list(nx.find_cliques(graphMain))
    tmp=deepcopy(np.matrix(adj.todense()))
    for i in listClique:
            for j in i:
                for k in i:
                    if j!=k:
                        adj[j,k]=len(i)-1
                        adj[k,j]=len(i)-1
    adj=FLAGS.alfa*np.matrix(adj.todense())+FLAGS.beta*tmp 

    features = preprocess_features( features )

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    #config.log_device_placement = True
    config.gpu_options.allow_growth = True

    train_loss = []
    train_acc  = []
    valid_loss = []
    valid_acc  = []
    with tf.Graph().as_default():
            random.seed( init_seed )
            np.random.seed( init_seed )
            tf.set_random_seed( init_seed )

            sess = tf.Session(config=config)

            model, support, placeholders = build_model( adj, features, labels.shape[1] )
            sess.run( tf.global_variables_initializer() )

            start_t = time.time()
            for epoch in range( FLAGS.epochs ):
                feed_dict = construct_feed_dict( features, support, labels, train_mask, placeholders,
                                                  FLAGS.dropout,FLAGS.alfa,FLAGS.beta )
                feed_dict.update( {tf.keras.backend.learning_phase(): 1} )
                outs = sess.run( [model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict )
                train_loss.append( outs[1] )
                train_acc.append( outs[2] )

                # Validation
                outs = evaluate(sess,model,features, support, labels, placeholders, val_mask )
                valid_loss.append( outs[0] )
                valid_acc.append( outs[1] )

                if epoch > FLAGS.early_stoping \
                        and np.mean( valid_loss[-10:] ) > np.mean( valid_loss[-100:] ) \
                        and np.mean( valid_acc[-10:] ) < np.mean( valid_acc[-100:] ):
                            print( "Early stopping at epoch {}...".format( epoch ) )
                            break


            test_loss, test_acc = evaluate(sess,model,features, support, labels, placeholders, test_mask )
            print( "Test set results:", "loss=", "{:.5f}".format(test_loss),
                      "accuracy=", "{:.5f}".format(test_acc))

    tf.reset_default_graph()
    from importlib import reload
    import scipy.io as sio
    sio.savemat('train_lossCoolClique.mat', {'train_loss_GOOLnorm':train_loss})
    sio.savemat('train_accCoolClique.mat', {'train_loss_GOOLnorm':train_acc})
    sio.savemat('valid_lossCoolClique.mat', {'train_loss_GOOLnorm':valid_loss},{'valid_acc':valid_acc})
    sio.savemat('valid_accgCoolClique.mat', {'train_loss_GOOLnorm':valid_acc})

    return {'train_loss': train_loss,'train_acc':  train_acc,'valid_loss': valid_loss,'valid_acc':  valid_acc,'test_loss':  test_loss,'test_acc':   test_acc,}

def main( argv ):
    FLAGS.hidden = [int(h) for h in FLAGS.hidden ]
    data_seeds = [None]
    seed=2019
    init_seeds = range( seed, seed+FLAGS.repeat )
    result = []
    for _data_seed, _init_seed in itertools.product( data_seeds, init_seeds ):
        result.append( train_Model( FLAGS.dataset, _data_seed, _init_seed ) )

        # compute and print the final scores
    final_result = np.array( [ ( r['train_loss'][-1], r['train_acc'][-1],
                               r['valid_loss'][-1], r['valid_acc'][-1],
                               r['test_loss'], r['test_acc'] )
                               for r in result ] )
    mean = np.mean( final_result, axis=0 )

    print('******************loss  acc')
    print( '     Train Mean', np.round( mean[0], 2 ), np.round( mean[1], 2 ) )
    print( 'Validation Mean', np.round( mean[2], 2 ), np.round( mean[3], 2 ) )
    print( '      Test Mean', np.round( mean[4], 2 ), np.round( mean[5], 2 ) )

if __name__ == '__main__':
    app.run( main )
