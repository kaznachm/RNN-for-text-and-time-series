==========
What did we do in python:
==========

#17.05

import sys
import csv
import numpy as np
import theano
import theano.tensor as T
import lasagne
#import cPickle as pickle
import logging
from lasagne.layers import get_output
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
import os
import linecache
from lasagne.layers import InputLayer, ExpressionLayer, EmbeddingLayer


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
results_directory = 'Results_multivariate_text/'

def load_parameters2_mvrt_text(params_csvfile, line_num): 
    """
    reads the params for the new generated params
    """
    params={}
    values = linecache.getline(params_csvfile,line_num)[:-1].split(',')
    params['num_units'] = int(values[0])
    params['num_att_units'] = int(values[1])
    data_type = str(values[2]) 
    params['data_name'] = str(values[3])
    model = str(values[4]) 
    if params['data_name'] == 'coservit':
        params['windowise'] = 288
        params['horizon'] = 27
        params['word_dim'] = 200
        params['vocab_size'] = 155564
    if data_type == 'orig':
        params['data_type'] = 'original'
        if params['data_name'] == 'coservit':
            #params['time_data'] = '../../../Data' + params['data_name'] + 
            #params['text_data'] = '../../../Data' + params['data_name'] + 
            #params['data_file'] =  '../../../Data/' + params['data_name'] + 'ticket2time.pkl'
            params['data_file'] =  '../../../Data/coservit/' + 'x_enc.pkl'
            params['text_file_wv'] =  '../../../Data/coservit/' + 'x_dec_wv.dat' ##params['text_file_wv'] =  '../../../Data/coservit/' + 'tickets_wv_pad_mtx.dat'
            params['text_file_w'] =  '../../../Data/coservit/' + 'x_dec_w.dat' #params['text_file_w'] =  '../../../Data/coservit/' + 'tickets_w_pad_mtx.dat'
            params['metr_dec_file'] =  '../../../Data/coservit/' + 'metr_for_dec.pkl' #list of lists with metric ids corresponding to aligned tickets. Ex.: t1,t1,t2,t3,t3,t3->[[3432, 4657], [3442], [6567, 4657, 7855]]
        else:
            #TODO
            pass
    else:
        pass
        """params['data_type'] = 'missing'
        #params['data_file'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/'+ params['data_name'] + '_'+ data_type +'.pkl'
        params['data_file'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/'+ params['data_name'] + '_'+ data_type +'_intr_linear.pkl'
        params['data_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/' + params['data_name'] + '_mask_' + data_type + '.pkl'
        params['missing_percent'] = data_type
        params['original_data_file'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'.pkl'
        params['original_data_mask_file'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'_mask.pkl' 
    #TODO:
    if params['data_name']  == 'pse':
        params['data2_name'] = 'weather_data'
        params['data2_file'] = '../../../Data/' + params['data2_name'] + '/missing_gaps_v2/'+ params['data2_name'] + '_'+ data_type +'.pkl'
        params['data2_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/' + params['data_name'] + '_mask_' + data_type + '.pkl'
        params['data2_variables'] = ['mintmp', 'maxtmp', 'precip'] 
    elif params['data_name']  == 'consseason':
        params['data2_name'] = 'consseason'
        params['data2_file'] = params['data_file']
        params['data2_file_missing_mask'] = params['data_file_missing_mask']
        params['data2_variables'] = ['Global_reactive_power', 'Voltage', 'Global_intensity'] 
    elif params['data_name']  == 'airq':
        params['data2_name'] = 'airq'
        params['data2_file'] = params['data_file']
        params['data2_file_missing_mask'] = params['data_file_missing_mask']
        params['data2_variables'] = ['NO2(GT)', 'CO(GT)', 'NOx(GT)'] """
    if model == 'RNN':
        params['attention'] = False
    else:
        params['attention'] = True
        if model == 'RNN-ATT':
            params['att_type'] = 'original'
        elif model == 'ATT-lambda':
            params['att_type'] = 'lambda'
        elif model == 'ATT-lambda-mu':
            params['att_type'] = 'lambda_mu' 
        elif model == 'ATT-lambda-mu-alt':
            params['att_type'] = 'lambda_mu_alt'
        elif model == 'ATT-dist':
            params['att_type'] = 'adist'        
    params['alg'] = 'adam'
    params['lambda_regularization'] = 0.0001
    params['eta_decay'] = 0.95
    params['early_stop_patience'] = 10000
    params['num_layers'] = 1
    params['regularization_type'] = 'l2'
    params['early_stop'] = True
    params['grad_clipping'] = 100
    params['train_test_ratio'] = 0.75
    params['train_val_ratio'] = 0.75
    params['patience_increase'] = 2 
    params['epsilon_improvement'] = True 
    params['lr_decay_after_n_epoch'] = 50
    params['interpolation'] = True #!!!!!!
    params['max_num_epochs'] = 25
    params['epsilon'] = 0.01
    params['learning_rate'] = 0.001
    params['batch_size'] = 64
    params['padding'] = True 
    params['learning_rate_decay'] = False
    params['interpolation_type'] = 'None'
    return params
	
	
params = load_parameters2_mvrt_text('params/fuzzy.param.text', 1)
params


X_enc_sym = T.dtensor3('x_enc_sym')
X_dec_sym = T.dtensor3('x_dec_sym')
y_sym = T.lmatrix('y_sym') # indexes of 1hot words, for loss #y_sym = T.ftensor3('y_sym') #y_sym = T.matrix('y_sym')
Emb_mtx_sym = T.dmatrix('emb_mtx_sym') #(155564, 200)
eta = theano.shared(np.array(params['learning_rate'], dtype=theano.config.floatX))

word_dim = params['word_dim']
vocab_size = params['vocab_size']
grad_clip = params['grad_clipping']
hidden_size = params['num_units']
pred_len = 27
hidden_size=64
grad_clip = 100
vocab_size=155564
word_dim=200

#l_out, l_out_val = model_seq2seq_GRU_text(X_enc_sym, X_enc_sym2, X_dec_sym, params['windowise'], params['horizon'],  hidden_size = params['num_units'], 
#                                                            grad_clip = params['grad_clipping'], vocab_size = params['vocab_size'], word_dim = params['word_dim'])
														
														
l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym)
l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(), W_hid=lasagne.init.Uniform(), 		W_cell=lasagne.init.Uniform()), updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(), W_hid=lasagne.init.Uniform(), W_cell=lasagne.init.Uniform()), 		hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(), W_hid=lasagne.init.Uniform(), W_cell=lasagne.init.Uniform()), grad_clipping=grad_clip, only_return_final=True)
l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(), W_hid=lasagne.init.Uniform(), 		W_cell=lasagne.init.Uniform()), updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(), W_hid=lasagne.init.Uniform(), W_cell=lasagne.init.Uniform()), grad_clipping=grad_clip, 		only_return_final=True, backwards=True)
l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1)
dec_units = 2*1 +1
l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, word_dim),input_var=X_dec_sym) #(None, 27, 200)
s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))	#(None, 1, 200)
h_init = lasagne.layers.ConcatLayer([l_forward, l_enc], axis=1)
l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(), W_hid=lasagne.init.Uniform(), 		W_cell=lasagne.init.Uniform()), updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(), W_hid=lasagne.init.Uniform(), W_cell=lasagne.init.Uniform()), 		hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(), W_hid=lasagne.init.Uniform(), W_cell=lasagne.init.Uniform()), learn_init=False, hid_init=h_init, 		grad_clipping=grad_clip, only_return_final=True )
										 
r_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_resetgate, W_hid=l_dec.W_hid_to_resetgate, b=l_dec.b_resetgate)
u_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_updategate, W_hid=l_dec.W_hid_to_updategate, b=l_dec.b_updategate)
h_update = lasagne.layers.Gate(W_in=l_dec.W_in_to_hidden_update, W_hid=l_dec.W_hid_to_hidden_update, b=l_dec.b_hidden_update)
l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 64)
l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, nonlinearity=lasagne.nonlinearities.softmax)

w_dense = l_out.W
b_dense = l_out.b
l_out_loop = l_out #(None, 155563)
l_out_loop_val = l_out #(None, 155563)
l_out = lasagne.layers.ReshapeLayer(l_out, ([0], 1, [1])) #3d (None, 1, 155563)
l_out_val = l_out #3d (None, 1, 155563)
h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1) # (None, 320)
h_init_val = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1) #(None, 320) 

for i in range(1,pred_len): 
	#i=1
	s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1) #(None, 200)
  	s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #(None, 1, 200)
	l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, learn_init=False,	 #(None, 320)
                                         hid_init=h_init, grad_clipping=grad_clip, only_return_final=True, 
                                         resetgate=r_gate, updategate=u_gate, hidden_update=h_update)
	l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 64)
    
	#for val and train sets
	coord_max = ExpressionLayer(l_out_loop_val, lambda X: X.argmax(-1).astype('int32'), output_shape='auto') #(None,)
	emb_slice = ExpressionLayer(coord_max, lambda x: Emb_mtx_sym[x,:], output_shape=(None,word_dim)) #(None, 200)
	
	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: X.argmax(-1).astype('int32'), output_shape='auto') #(None,)
	#pred_d = ExpressionLayer(coord_max, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X], 1), output_shape='auto')
	#pred_emb = lasagne.layers.DenseLayer(pred_d, num_units=word_dim, W=Emb_mtx_sym, nonlinearity=lasagne.nonlinearities.linear) #no biases #(None, 200)
	#pred_emb.params[pred_emb.W].remove("trainable")
	"""
	def argm(l_row):
		return T.argmax(l_row)
	
	coord_max = theano.scan(fn=argm, outputs_info=None, sequences=l_out_loop_val, non_sequences=None)

	def get_row(a_coord, emb_mtx):
		return 1
	
	pred_d, updates = theano.scan(fn=get_row, outputs_info=None, sequences=coord_max, non_sequences=Emb_mtx_sym)
	func = theano.function(inputs=[coord_max], outputs=pred_d)
	"""
	
	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X.argmax(-1).astype('int32')], 1), output_shape='auto')
	#hot = lasagne.layers.DenseLayer(coord_max, num_units=word_dim, nonlinearity=lasagne.nonlinearities.linear)
	
	pred = lasagne.layers.ReshapeLayer(emb_slice, ([0], 1, [1])) #######pred = lasagne.layers.ReshapeLayer(pred_emb, ([0], 1, [1])) #(None, 1, 200)
	
	l_dec_val = lasagne.layers.GRULayer(pred, num_units=hidden_size*dec_units, learn_init=False,
                                         hid_init=h_init_val, grad_clipping=grad_clip, only_return_final=True,
                                         resetgate=r_gate, updategate=u_gate, hidden_update=h_update) #(None, 320)
	l_dec_val_hid_state = lasagne.layers.SliceLayer(l_dec_val, indices=slice(0,hidden_size)) #(None, 64)
	l_out_loop = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
	l_out_loop = lasagne.layers.ReshapeLayer(l_out_loop, ([0], 1, [1])) ####### #(None, 1, 155563)
	l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop], axis=1) #(None, 2, 155563)
	l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
	l_out_loop_val_d = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1])) #(None, 1, 155563)
	l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val_d], axis=1) #(None, 2, 155563)
	#l_out_loop_val = lasagne.layers.FlattenLayer(l_out_loop_val, 2)
	h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1) #(None, 320)
	h_init_val = lasagne.layers.ConcatLayer([l_dec_val_hid_state, l_enc], axis=1) #(None, 320)

#l_out.output_shape #shape = (None, 27, 155563)
#l_out_val.output_shape # shape = (None, 27, 155563)

#l_out = lasagne.layers.ReshapeLayer(l_out, (-1, hidden_size)) #(None, 64)
#l_out_val = lasagne.layers.ReshapeLayer(l_out_val, (-1, hidden_size)) #(None, 64)




===================================

l_out_conc = lasagne.layers.ConcatLayer([l_out, l_out_loop])
l_out_val_conc = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val], axis=1)

l_out = lasagne.layers.ReshapeLayer(l_out, ([0], 1, [1]))
l_out_loop = lasagne.layers.ReshapeLayer(l_out_loop, ([0], 1, [1]))



===================================

aa = np.array([[[1,2,3], [4,5,6], [0,2,1], [0,1,2]], [[6,7,8], [3,6,0], [4,3,2], [1,1,2]]])
a = theano.shared(aa, 'a')
bb = np.array([[1,1,1,1], [1,1,1,1]])
b = theano.shared(bb, 'b')

def categorical_crossentropy_3d(coding_dist, true_dist):
	# Zero out the false probabilities and sum the remaining true probabilities to remove the third dimension.
	indexes = theano.tensor.arange(coding_dist.shape[2])
	mask = theano.tensor.neq(indexes, true_dist.reshape((true_dist.shape[0], true_dist.shape[1], 1)))
	pred_probs = theano.tensor.set_subtensor(coding_dist[theano.tensor.nonzero(mask)], 0.).sum(axis=2)
	pred_probs_log = T.log(pred_probs)
	pred_probs_per_sample = -pred_probs_log.sum(axis=1)
	return pred_probs, pred_probs_log, pred_probs_per_sample

s, ss, sss = categorical_crossentropy_3d(a, b)
s, ss, sss = s.eval(), ss.eval(), sss.eval()
print(s)
print(ss)
print(sss)

===================================

#elementwise multiplication of 3d tensors
aa = np.array([[[1,2,3], [4,5,6], [0,2,1], [0,1,2]], [[6,7,8], [3,6,0], [4,3,2], [1,1,2]]], dtype=np.float32)
bb = np.array([[[0,0,2], [6,7,1], [0,-1,9], [0,9,-2]], [[6,7,8], [3,6,0], [4,3,2], [1,1,2]]], dtype=np.float32)

x = T.ftensor3('x')
y = T.ftensor3('y')
z = x * y
f1 = theano.function([x, y], z)
print(f1(aa,bb))

#try the same but with cross_entr function

a = theano.shared(aa, 'a')
b = theano.shared(bb, 'b')

def crossentropy_3d(coding_dist, true_dist):
	pred_probs = coding_dist * true_dist
	pred_probs_per_sample = pred_probs.sum(axis=2).sum(axis=1)
	return -theano.tensor.log(pred_probs_per_sample)
	
z = crossentropy_3d(a, b)
result = z.eval()
print(result)


===================================

Traceback (most recent call last):
  File "prediction_rnn_mvrt_intrl_GRU.py", line 2056, in <module>
    main()  
  File "prediction_rnn_mvrt_intrl_GRU.py", line 1989, in main
    train_loss, validation_loss = train_and_test_text(params, results_directory)
  File "prediction_rnn_mvrt_intrl_GRU.py", line 1865, in train_and_test_text
    val_loss = f_val(X_val, X_val2, X_val_dec, y_val_w) #val_loss = f_val(X_val, X_val_mask, X_val2, X_val_mask2, X_val3, X_val_mask3, X_val4, X_val_mask4, X_val_dec, X_val_mask_dec, y_val)
  File "/home/ama/kaznachm/.local/lib/python2.7/site-packages/theano/compile/function_module.py", line 898, in __call__
    storage_map=getattr(self.fn, 'storage_map', None))
  File "/home/ama/kaznachm/.local/lib/python2.7/site-packages/theano/gof/link.py", line 325, in raise_with_op
    reraise(exc_type, exc_value, exc_trace)
  File "/home/ama/kaznachm/.local/lib/python2.7/site-packages/theano/compile/function_module.py", line 884, in __call__
    self.fn() if output_subset is None else\
ValueError: Shape mismatch: x has 155563 cols (and 260 rows) but y has 200 rows (and 1920 cols)
Apply node that caused the error: Dot22(SoftmaxWithBias.0, Join.0)
Toposort index: 285
Inputs types: [TensorType(float64, matrix), TensorType(float64, matrix)]
Inputs shapes: [(260, 155563), (200, 1920)]
Inputs strides: [(1244504, 8), (15360, 8)]
Inputs values: ['not shown', 'not shown']
Outputs clients: [[Reshape{2}(Dot22.0, MakeVector{dtype='int64'}.0)]]

HINT: Re-running with most Theano optimization disabled could give you a back-trace of when this node was created. This can be done with by setting the Theano flag 'optimizer=fast_compile'. If that does not work, Theano optimizations can be disabled with 'optimizer=None'.
HINT: Use the Theano flag 'exception_verbosity=high' for a debugprint and storage map footprint of this apply node.

===================================


for i in range(1,pred_len): 
	s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1) #(None, 200)
  	s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #(None, 1, 200)
	l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, learn_init=False,	 #(None, 320)
                                         hid_init=h_init, grad_clipping=grad_clip, only_return_final=True, 
                                         resetgate=r_gate, updategate=u_gate, hidden_update=h_update)
	l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 64)
	pred = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1])) #(None, 1, 155563)
	pred_2 = lasagne.layers.DenseLayer(pred, num_units=word_dim, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
	pred_2 = lasagne.layers.ReshapeLayer(pred_2, ([0], 1, [1])) ####### #(None, 1, 155563)
    
	#ind_word = T.argmax(pred, axis=0)
	l_dec_val = lasagne.layers.GRULayer(pred_2, num_units=hidden_size*dec_units, learn_init=False,
                                         hid_init=h_init_val, grad_clipping=grad_clip, only_return_final=True,
                                         resetgate=r_gate, updategate=u_gate, hidden_update=h_update) #(None, 320)
	l_dec_val_hid_state = lasagne.layers.SliceLayer(l_dec_val, indices=slice(0,hidden_size)) #(None, 64)
	l_out_loop = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
	l_out_loop = lasagne.layers.ReshapeLayer(l_out_loop, ([0], 1, [1])) ####### #(None, 1, 155563)
	l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop], axis=1) #(None, 2, 155563)
	l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
	l_out_loop_val_d = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1])) #(None, 1, 155563)
	l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val_d], axis=1) #(None, 2, 155563)
	#l_out_loop_val = lasagne.layers.FlattenLayer(l_out_loop_val, 2)
	h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1) #(None, 320)
	h_init_val = lasagne.layers.ConcatLayer([l_dec_val_hid_state, l_enc], axis=1) #(None, 320)
	
	
	
==============================================================================================================================================================================================================

01.06.17 Order matters

import lasagne
import numpy as np
import theano
import theano.tensor as T
from theano import shared
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan
from lasagne.layers.base import Layer, MergeLayer
from lasagne.layers.input import InputLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers import helper, SliceLayer
from lasagne.layers.recurrent import Gate
import logging
import sys
import csv
from lasagne.layers import get_output
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
import os
import linecache
from lasagne.layers import InputLayer, ExpressionLayer, EmbeddingLayer
#            
def load_parameters2_mvrt_text(params_csvfile, line_num): 
    """
    reads the params for the new generated params
    """
    params={}
    values = linecache.getline(params_csvfile,line_num)[:-1].split(',')
    params['num_units'] = int(values[0]) #128
    params['num_att_units'] = int(values[1]) #512
    data_type = str(values[2]) 
    params['data_name'] = str(values[3])
    model = str(values[4]) 
    if params['data_name'] == 'coservit':
        params['windowise'] = 288
        params['horizon'] = 27
        params['word_dim'] = 202 # 200
        params['vocab_size'] = 22 ###### 155564
        
        #params['time_data'] = '../../../Data' + params['data_name'] + 
        #params['text_data'] = '../../../Data' + params['data_name'] + 
        #params['data_file'] =  '../../../Data/' + params['data_name'] + 'ticket2time.pkl'
        params['data_file'] =  '../../../Data/coservit/' + 'x_enc.pkl'
        params['text_file_wv'] =  '../../../Data/coservit/' + 'x_dec_wv.dat' ##params['text_file_wv'] =  '../../../Data/coservit/' + 'tickets_wv_pad_mtx.dat'
        params['text_file_w'] =  '../../../Data/coservit/' + 'x_dec_w.dat' #params['text_file_w'] =  '../../../Data/coservit/' + 'tickets_w_pad_mtx.dat'
        params['metr_dec_file'] =  '../../../Data/coservit/' + 'metr_for_dec.pkl' #list of lists with metric ids corresponding to aligned tickets. Ex.: t1,t1,t2,t3,t3,t3->[[3432, 4657], [3442], [6567, 4657, 7855]]
        params['emb_file'] =  '../../../Data/coservit/' + 'lda_emb_mtx.dat'#'emb_mtx.dat'
    else:
        pass
        """params['data_type'] = 'missing'
        #params['data_file'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/'+ params['data_name'] + '_'+ data_type +'.pkl'
        params['data_file'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/'+ params['data_name'] + '_'+ data_type +'_intr_linear.pkl'
        params['data_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/' + params['data_name'] + '_mask_' + data_type + '.pkl'
        params['missing_percent'] = data_type
        params['original_data_file'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'.pkl'
        params['original_data_mask_file'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'_mask.pkl' 
    #TODO:
    if params['data_name']  == 'pse':
        params['data2_name'] = 'weather_data'
        params['data2_file'] = '../../../Data/' + params['data2_name'] + '/missing_gaps_v2/'+ params['data2_name'] + '_'+ data_type +'.pkl'
        params['data2_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/' + params['data_name'] + '_mask_' + data_type + '.pkl'
        params['data2_variables'] = ['mintmp', 'maxtmp', 'precip'] 
    elif params['data_name']  == 'consseason':
        params['data2_name'] = 'consseason'
        params['data2_file'] = params['data_file']
        params['data2_file_missing_mask'] = params['data_file_missing_mask']
        params['data2_variables'] = ['Global_reactive_power', 'Voltage', 'Global_intensity'] 
    elif params['data_name']  == 'airq':
        params['data2_name'] = 'airq'
        params['data2_file'] = params['data_file']
        params['data2_file_missing_mask'] = params['data_file_missing_mask']
        params['data2_variables'] = ['NO2(GT)', 'CO(GT)', 'NOx(GT)'] """
            
    if data_type == 'orig':
        params['data_type'] = 'original'
    else:
        #TODO
        pass
        
    if model == 'RNN':
        params['attention'] = False
    else:
        params['attention'] = True
        if model == 'RNN-ATT':
            params['att_type'] = 'original'
        elif model == 'ATT-lambda':
            params['att_type'] = 'lambda'
        elif model == 'ATT-lambda-mu':
            params['att_type'] = 'lambda_mu' 
        elif model == 'ATT-lambda-mu-alt':
            params['att_type'] = 'lambda_mu_alt'
        elif model == 'ATT-dist':
            params['att_type'] = 'adist'
        elif model == 'RNN-text-set':
            params['att_type'] = 'set_enc'
            params['num_metrics'] = 4
            params['set_steps'] = 5 #number of iterations in set_enc part
    params['alg'] = 'adam'
    params['lambda_regularization'] = 0.0001
    params['eta_decay'] = 0.95
    params['early_stop_patience'] = 10000
    params['num_layers'] = 1
    params['regularization_type'] = 'l2'
    params['early_stop'] = True
    params['grad_clipping'] = 100
    params['train_test_ratio'] = 0.75
    params['train_val_ratio'] = 0.75
    params['patience_increase'] = 2 
    params['epsilon_improvement'] = True 
    params['lr_decay_after_n_epoch'] = 50
    params['interpolation'] = True #!!!!!!
    params['max_num_epochs'] = 25
    params['epsilon'] = 0.01
    params['learning_rate'] = 0.001
    params['batch_size'] = 64
    params['padding'] = True 
    params['learning_rate_decay'] = False
    params['interpolation_type'] = 'None'
    return params

params = load_parameters2_mvrt_text('params/fuzzy.param.text', 2)
params
#
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
results_directory = 'Results_multivariate_text_OM/'
#
att_type = params['att_type']
att_size = params['num_att_units']
word_dim = params['word_dim']
vocab_size = params['vocab_size']
grad_clip = params['grad_clipping']
hidden_size = params['num_units']
pred_len = 27
num_metrics = params['num_metrics']
set_steps = params['set_steps']
grad_clip = 100
#
#Gate
class Gate_setenc(object):
    def __init__(self, W_in=init.GlorotNormal(), W_hid=init.GlorotNormal(),
                 W_cell=init.Normal(0.1), b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        if W_in is not None:
            self.W_in = W_in
        self.W_hid = W_hid
        # Don't store a cell weight vector when cell is None
        if W_cell is not None:
            self.W_cell = W_cell
        self.b = b
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

#
#Layer
class GRULayer_setenc(lasagne.layers.Layer): #it is not mvrt lasagne.layers.Layer
    def __init__(self, incoming, num_units, 
                 resetgate=Gate_setenc(W_in=None,W_cell=None), 
                 updategate=Gate_setenc(W_in=None,W_cell=None),
                 hidden_update=Gate_setenc(W_in=None, W_cell=None),
                 nonlinearity=nonlinearities.tanh,
                 hid_init=init.Constant(0.),
                 set_steps=5,
                 att_num_units = 64,
                 W_hid_to_att = init.GlorotNormal(),
                 W_ctx_to_att = init.GlorotNormal(),
                 W_att = init.Normal(0.1), #---
                 backwards=False,
                 learn_init=True, #--???????
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=True, #-- ???????
                 **kwargs):
        #         
        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.  We will just provide the
        # layer input as incomings, unless a mask input was provided.
        #incomings = [incoming]
        #if mask_input is not None:
            #incomings.append(mask_input) #-----
        #
        # Initialize parent layer
        super(GRULayer_setenc, self).__init__(incoming, **kwargs) #super(GRULayer_setenc, self).__init__(incomings, **kwargs)
        #
        self.learn_init = learn_init
        self.num_units = num_units #128
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.set_steps = set_steps
        self.att_num_units = att_num_units #512
        self.only_return_final = only_return_final
        #
        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")
        #
        # Retrieve the dimensionality of the incoming layer
        #
        if unroll_scan and self.input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")
        #
        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(self.input_shape[2:]) #4
        print(num_inputs)
        #print(self.input_shape) #(None, 256, 4)
        #print(num_inputs) #4
        #
        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            #self.add_param(gate.W_in, (num_inputs, num_units),
                                   #name="W_in_to_{}".format(gate_name))
            return (self.add_param(gate.W_hid, (num_units, num_units), #128
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)
        #
        # Add in all parameters from gates, nonlinearities will be sigmas, look Gate_setenc
        """(self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')
        #
        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')"""
        (self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')
        #
        (self.W_hid_to_hidden_update, self.b_hidden_update, 
         self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')
        #
        #attention Weights 
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        #
        # Initialize hidden state
        #self.hid_init = hid_init ######
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)
        """if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError(
                    "When hid_init is provided as a TensorVariable, it should "
                    "have 2 dimensions and have shape (num_batch, num_units)")
            self.hid_init = hid_init
        else: # here
            self.hid_init = self.add_param(  #--????????#--????????#--????????#--????????#--????????#--????????#--????????
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False) #not trainable"""
        #print(self.hid_init, type(self.hid_init)) #(hid_init, <class 'theano.tensor.sharedvar.TensorSharedVariable'>
#
    def get_output_shape_for(self, input_shape):                        #(None, 256, 128)
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shp = self.input_shape[0]
        # PRINTS
        if self.only_return_final:
            return self.input_shape[0], self.num_units #(None, 128)
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return self.input_shape[0], self.input_shape[1], self.num_units
#
    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        #print('uiop') # PRINTS
        # Retrieve the layer input
        input = inputs
        #print(input) #<lasagne.layers.merge.ConcatLayer object at 0x7fe2e1f7fb10>
        #print(input.ndim)
        # Retrieve the mask when it is supplied
        #mask = inputs[1] if len(inputs) > 1 else None # -> mask=None
#
        #print(type(inputs)) #<class 'lasagne.layers.merge.ConcatLayer'>
        # Treat all dimensions after the second as flattened feature dimensions
        #if input.ndim > 3:
            #input = T.flatten(input, 3)
#
        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        #input = lasagne.layers.ReshapeLayer(input, ([1], [0], [2]))#
        input = input.dimshuffle(1, 0, 2)#
        seq_len, num_batch, _ = input.shape #256,None -- no 4 #(256, None) #seq_len, num_batch, _ = input.output_shape
        print(seq_len, num_batch) #(256, None)
        #print(input.output_shape) #(256, None, 4)
#
        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        """W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)"""
#
        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)
#
        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)
#
        #if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            #input = T.dot(input, W_in_stacked) + b_stacked
#
        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
   #     
        def plain_et_step(x_snp, o_t0):#def plain_et_step(self, x_snp, o_t0)
            #reading from memory steps
            bs, seq_len_m, _ = x_snp.shape
            m_in = x_snp.dimshuffle(1, 0, 2)#----replace
            e_qt = T.dot(o_t0, self.W_hid_to_att)#---
            e_m = T.dot(m_in, self.W_ctx_to_att)#----
            e_q = T.tile(e_qt, (seq_len_m, 1, 1)) #e_q = T.tile(e_qt, (self.seq_len_m, 1, 1))
            et_p = T.tanh(e_m + e_q)
            et = T.dot(et_p, self.W_att)
            alpha = T.exp(et)
            alpha /= T.sum(alpha, axis=0)
            mt = x_snp.dimshuffle(2, 1, 0)
            mult = T.mul(mt, alpha)
            rt = T.sum(mult, axis=1)
            return rt.T
#
        def step(hid_previous, *args): #W_hid_stacked, #W_in_stacked, b_stacked):
            #x_snp = incom
            #print(x_snp.output_shape)
            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            print("qwerty")
            hid_input = T.dot(hid_previous, W_hid_stacked) + b_stacked ####for r, z, h tilde WHAT ABOUT THE SIZES WHEN MULT??????
#       
            print("tyui")
            if self.grad_clipping is not False:
                #input_n = theano.gradient.grad_clip(
                    #input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)
                print('d')
#
           # if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                #input_n = T.dot(input_n, W_in_stacked) + b_stacked
#
            # Reset and update gates
            resetgate = slice_w(hid_input, 0) #+ slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) #+ slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)
#
            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            #hidden_update_in = slice_w(input_n, 2)
            ####hidden_update_hid = slice_w(hid_input, 2) #h tilde
            ####hidden_update = resetgate*hidden_update_hid #hidden_update = hidden_update_in + resetgate*hidden_update_hid
            hidden_update = slice_w(hid_input, 2)
            #
            if self.grad_clipping is not False:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update) #here it is sigma, but in encoder.py this is tanh ????????????
#
            print('d')
            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid0 = (np.float32(1.0) - updategate)*hid_previous + updategate*hidden_update
            rt = plain_et_step(input, hid0)
            h_t = T.concatenate([hid0,rt], axis=1)
    #        
            return h_t #------??????
        """def step(x_snp, snp_mask, m_snp_count):
            r_t = T.nnet.sigmoid(T.dot(hr_tm1, self.Wo_hh_r) + self.bo_r)
            z_t = T.nnet.sigmoid(T.dot(hr_tm1, self.W_hh_z) + self.b_z)
            h_tilde = T.tanh(T.dot(r_t * hr_tm1, self.W_hh) + self.b_hh)
            o_t = (np.float32(1.0) - z_t) * hr_tm1 + z_t * h_tilde
            #o_t: GRU hidden state
            #concatanation of hidden state and reading from memory
            rt = self.plain_et_step(x_snp, o_t)
            h_t = T.concatenate([o_t0,rt], axis=1)---------
            return h_t"""
     #       
#
        """def step_masked(input_n, mask_n, hid_previous, W_hid_stacked,
                        W_in_stacked, b_stacked):
#
            hid = step(input_n, hid_previous, W_hid_stacked, W_in_stacked,
                       b_stacked)
#
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            not_mask = 1 - mask_n
            hid = hid*mask_n + hid_previous*not_mask
#
            return hid"""
#
        """if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
            print('d')
        else:
            #sequences = input #[input]
            step_fun = step
            print("step")"""
        step_fun = step
#
        """if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)"""
#
        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked, b_stacked] #non_seqs = [W_hid_stacked] #non_seqs = [input, W_hid_stacked]
        #non_seqs += [W_ctx_stacked] #
        non_seqs += [self.W_hid_to_att, self.W_ctx_to_att, self.W_att, input] #
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        #if not self.precompute_input:
            #non_seqs += [b_stacked]#[W_in_stacked, b_stacked]
        # theano.scan only allows for positional arguments, so when
        # self.precompute_input is True, we need to supply fake placeholder
        # arguments for the input weights and biases.
        #else:
            #non_seqs += [(), ()]
#
        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else: #here
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            #hid_out, _ = theano.scan(fn=step_fun, outputs_info=o_enc_info, non_sequences=incomings[0], n_steps=self.set_steps)#self.num_set_iter) #----??????
            print('d')
            hid_out,_ = theano.scan(fn=step_fun, outputs_info=self.hid_init, non_sequences=non_seqs, n_steps=self.set_steps)#self.num_set_iter)
        """hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]"""
#
        # dimshuffle back to (n_batch, n_time_steps, n_features))
        #######hid_out = hid_out.dimshuffle(1, 0, 2) #done below
#
        # if scan is backward reverse the output
        #if self.backwards:-------- ?????????
            #hid_out = hid_out[:, ::-1, :]-------??????????? #done below
         #
        #copied from layers.py
        """if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)
            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]     """
        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        #return theano.shared(np.random.randn(3,4))
#
        print(hid_out.shape)
        return hid_out

X_enc_sym = T.dtensor3('x_enc_sym') 
X_dec_sym = T.dtensor3('x_dec_sym') ##X_dec_sym = T.ftensor3('x_dec_sym')
y_sym = T.lmatrix('y_sym') # indexes of 1hot words, for loss 
Emb_mtx_sym = T.dmatrix('emb_mtx_sym')
eta = theano.shared(np.array(params['learning_rate'], dtype=theano.config.floatX))

#in model
#X = X_enc_sym[:,:,0:1]
#l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
l_in_enc = lasagne.layers.InputLayer(shape=(None, None, num_metrics), input_var=X_enc_sym) #(None, None, 4) #l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=0, axis=2) #(None, None)
l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #(None, None, 1)
    
l_forward = lasagne.layers.GRULayer(l_in_slice, num_units=hidden_size, #(None, 128)
                                                resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                grad_clipping=grad_clip, only_return_final=True)
l_backward = lasagne.layers.GRULayer(l_in_slice, num_units=hidden_size, #(None, 128)
                                                resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None),
                                                grad_clipping=grad_clip, only_return_final=True, backwards=True)
l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
l_enc_conc = lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1)) #(None, 256, 1)
#    
resetgate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_resetgate, W_hid=l_forward.W_hid_to_resetgate, W_cell = None, b=l_forward.b_resetgate)
updategate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_updategate, W_hid=l_forward.W_hid_to_updategate, W_cell = None, b=l_forward.b_updategate)
hidden_update_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_hidden_update, W_hid=l_forward.W_hid_to_hidden_update, W_cell=None, b=l_forward.b_hidden_update)
#
resetgate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_resetgate, W_hid=l_backward.W_hid_to_resetgate, W_cell= None, b=l_backward.b_resetgate)
updategate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_updategate, W_hid=l_backward.W_hid_to_updategate, W_cell=None, b=l_backward.b_updategate)
hidden_update_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_hidden_update, W_hid=l_backward.W_hid_to_hidden_update, W_cell=None, b=l_backward.b_hidden_update)
                                      
for i in range(1, num_metrics):
    #X = X_enc_sym[:,:,i:(i+1)]
    #l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
    l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=i, axis=2) #(None, None)
    l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], 1, [1]))
        
    l_forward = lasagne.layers.GRULayer(l_in_slice, num_units=hidden_size, #(None, 128) l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                                    resetgate=resetgate_f, 
                                                    updategate=updategate_f, 
                                                    hidden_update=hidden_update_f, 
                                                    grad_clipping=grad_clip, only_return_final=True)
    l_backward = lasagne.layers.GRULayer(l_in_slice, num_units=hidden_size, #(None, 128) l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size,
                                                    resetgate=resetgate_b, 
                                                    updategate=updategate_b, 
                                                    hidden_update=hidden_update_b,
                                                    grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
    l_enc_conc = lasagne.layers.ConcatLayer([l_enc_conc, lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1))], axis=2) #(None, 256, 4)

#l_enc_conc = lasagne.layers.ReshapeLayer(l_enc_conc, ([1], [0], [2])) #(256, None, 4)

#model
#l_q = theano.shared(value=np.zeros((hidden_size,1), dtype='float32'))
#l_q = lasagne.init.Constant(0.)
#l_input = lasagne.layers.InputLayer(shape=(None, 256, 4), name=l_enc_conc)
#l_setenc = GRULayer_setenc(incoming=l_enc_conc, num_units=hidden_size, learn_init=False, set_steps=5, att_num_units=att_size, grad_clipping=grad_clip, 
                           nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)  
#lasagne.layers.get_output(l_setenc)


l_out, l_out_val = model_seq2seq_GRU_setenc(X_enc_sym, X_dec_sym, Emb_mtx_sym, params['horizon']-1, params['num_metrics'], params['set_steps'], hidden_size = params['num_units'], 
                    grad_clip = params['grad_clipping'], att_size = params['num_att_units'], vocab_size = params['vocab_size'], word_dim = params['word_dim'])
                                            
network_output, network_output_val = lasagne.layers.get_output([l_out, l_out_val])

def categorical_crossentropy_3d(coding_dist, true_dist, lengths=None):
    #http://stackoverflow.com/questions/30225633/cross-entropy-for-batch-with-theano
    
    # Zero out the false probabilities and sum the remaining true probabilities to remove the third dimension.
    indexes = theano.tensor.arange(coding_dist.shape[2])
    mask = theano.tensor.neq(indexes, true_dist.reshape((true_dist.shape[0], true_dist.shape[1], 1)))
    pred_probs = theano.tensor.set_subtensor(coding_dist[theano.tensor.nonzero(mask)], 0.).sum(axis=2)
    pred_probs_log = T.log(pred_probs)
    pred_probs_per_sample = -pred_probs_log.sum(axis=1)
    return pred_probs_per_sample

weights = lasagne.layers.get_all_params(l_out,trainable=True)
if params['regularization_type'] == 'l1':
    reg_loss = lasagne.regularization.regularize_network_params(l_out, l1) * params['lambda_regularization']
else:
    reg_loss = lasagne.regularization.regularize_network_params(l_out, l2) * params['lambda_regularization']

loss_T = categorical_crossentropy_3d(network_output, y_sym).mean() + reg_loss
loss_val_T = categorical_crossentropy_3d(network_output_val, y_sym).mean() 
loss_test = categorical_crossentropy_3d(network_output_val, y_sym).mean() 
#metric_probs = get_metric_probs(network_output, y_sym) #####             

updates = lasagne.updates.adam(loss_T, weights, learning_rate=eta)

f_train = theano.function([X_enc_sym, X_dec_sym, y_sym], loss_T, updates=updates, allow_input_downcast=True)           
                                            
    




l_setenc.output_shape #(None, 256, 128) #(None, 128)
l_setenc.get_output_for(l_enc_conc)
   
ll = lasagne.layers.GRULayer(incoming=l_forward, num_units=hidden_size, resetgate=lasagne.layers.Gate(W_cell=None), \
                            updategate=lasagne.layers.Gate(W_cell=None), hidden_update=lasagne.layers.Gate(W_cell=None))
                             
    
ll = lasagne.layers.GRULayer(incoming=None, num_units=hidden_size, resetgate=lasagne.layers.Gate(W_in=None, W_cell=None), \
                            updategate=lasagne.layers.Gate(W_in=None, W_cell=None), hidden_update=lasagne.layers.Gate(W_in=None, W_cell=None))
                            
                            
08.06.17==============================================================================================================================================================================================================     
List            
                            
import lasagne
import numpy as np
import theano
import theano.tensor as T
from theano import shared
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan
from lasagne.layers.base import Layer, MergeLayer
from lasagne.layers.input import InputLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers import helper, SliceLayer
from lasagne.layers.recurrent import Gate
import logging
import sys
import csv
from lasagne.layers import get_output
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
import os
import linecache
from lasagne.layers import InputLayer, ExpressionLayer, EmbeddingLayer
#            
def load_parameters2_mvrt_text(params_csvfile, line_num): 
    """
    reads the params for the new generated params
    """
    params={}
    values = linecache.getline(params_csvfile,line_num)[:-1].split(',')
    params['num_units'] = int(values[0]) #128
    params['num_att_units'] = int(values[1]) #512
    data_type = str(values[2]) 
    params['data_name'] = str(values[3])
    model = str(values[4]) 
    if params['data_name'] == 'coservit':
        params['windowise'] = 288
        params['horizon'] = 27
        params['word_dim'] = 202 # 200
        params['vocab_size'] = 22 ###### 155564
        
        #params['time_data'] = '../../../Data' + params['data_name'] + 
        #params['text_data'] = '../../../Data' + params['data_name'] + 
        #params['data_file'] =  '../../../Data/' + params['data_name'] + 'ticket2time.pkl'
        params['data_file'] =  '../../../Data/coservit/' + 'x_enc.pkl'
        params['text_file_wv'] =  '../../../Data/coservit/' + 'x_dec_wv.dat' ##params['text_file_wv'] =  '../../../Data/coservit/' + 'tickets_wv_pad_mtx.dat'
        params['text_file_w'] =  '../../../Data/coservit/' + 'x_dec_w.dat' #params['text_file_w'] =  '../../../Data/coservit/' + 'tickets_w_pad_mtx.dat'
        params['metr_dec_file'] =  '../../../Data/coservit/' + 'metr_for_dec.pkl' #list of lists with metric ids corresponding to aligned tickets. Ex.: t1,t1,t2,t3,t3,t3->[[3432, 4657], [3442], [6567, 4657, 7855]]
        params['emb_file'] =  '../../../Data/coservit/' + 'lda_emb_mtx.dat'#'emb_mtx.dat'
    else:
        pass
        """params['data_type'] = 'missing'
        #params['data_file'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/'+ params['data_name'] + '_'+ data_type +'.pkl'
        params['data_file'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/'+ params['data_name'] + '_'+ data_type +'_intr_linear.pkl'
        params['data_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/' + params['data_name'] + '_mask_' + data_type + '.pkl'
        params['missing_percent'] = data_type
        params['original_data_file'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'.pkl'
        params['original_data_mask_file'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'_mask.pkl' 
    #TODO:
    if params['data_name']  == 'pse':
        params['data2_name'] = 'weather_data'
        params['data2_file'] = '../../../Data/' + params['data2_name'] + '/missing_gaps_v2/'+ params['data2_name'] + '_'+ data_type +'.pkl'
        params['data2_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/' + params['data_name'] + '_mask_' + data_type + '.pkl'
        params['data2_variables'] = ['mintmp', 'maxtmp', 'precip'] 
    elif params['data_name']  == 'consseason':
        params['data2_name'] = 'consseason'
        params['data2_file'] = params['data_file']
        params['data2_file_missing_mask'] = params['data_file_missing_mask']
        params['data2_variables'] = ['Global_reactive_power', 'Voltage', 'Global_intensity'] 
    elif params['data_name']  == 'airq':
        params['data2_name'] = 'airq'
        params['data2_file'] = params['data_file']
        params['data2_file_missing_mask'] = params['data_file_missing_mask']
        params['data2_variables'] = ['NO2(GT)', 'CO(GT)', 'NOx(GT)'] """
            
    if data_type == 'orig':
        params['data_type'] = 'original'
    else:
        #TODO
        pass
        
    if model == 'RNN':
        params['attention'] = False
    else:
        params['attention'] = True
        if model == 'RNN-ATT':
            params['att_type'] = 'original'
        elif model == 'ATT-lambda':
            params['att_type'] = 'lambda'
        elif model == 'ATT-lambda-mu':
            params['att_type'] = 'lambda_mu' 
        elif model == 'ATT-lambda-mu-alt':
            params['att_type'] = 'lambda_mu_alt'
        elif model == 'ATT-dist':
            params['att_type'] = 'adist'
        elif model == 'RNN-text-set':
            params['att_type'] = 'set_enc'
            params['num_metrics'] = 4
            params['set_steps'] = 5 #number of iterations in set_enc part
    params['alg'] = 'adam'
    params['lambda_regularization'] = 0.0001
    params['eta_decay'] = 0.95
    params['early_stop_patience'] = 10000
    params['num_layers'] = 1
    params['regularization_type'] = 'l2'
    params['early_stop'] = True
    params['grad_clipping'] = 100
    params['train_test_ratio'] = 0.75
    params['train_val_ratio'] = 0.75
    params['patience_increase'] = 2 
    params['epsilon_improvement'] = True 
    params['lr_decay_after_n_epoch'] = 50
    params['interpolation'] = True #!!!!!!
    params['max_num_epochs'] = 25
    params['epsilon'] = 0.01
    params['learning_rate'] = 0.001
    params['batch_size'] = 64
    params['padding'] = True 
    params['learning_rate_decay'] = False
    params['interpolation_type'] = 'None'
    return params

params = load_parameters2_mvrt_text('params/fuzzy.param.text', 2)
params
#
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
results_directory = 'Results_multivariate_text_OM/'
#
att_type = params['att_type']
att_size = params['num_att_units']
word_dim = params['word_dim']
vocab_size = params['vocab_size']
grad_clip = params['grad_clipping']
hidden_size = params['num_units']
pred_len = 27
num_metrics = params['num_metrics']
set_steps = params['set_steps']
grad_clip = 100
#
#Gate
class Gate_setenc(object):
    def __init__(self, W_in=init.GlorotNormal(), W_hid=init.GlorotNormal(),
                 W_cell=init.Normal(0.1), b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        if W_in is not None:
            self.W_in = W_in
        self.W_hid = W_hid
        # Don't store a cell weight vector when cell is None
        if W_cell is not None:
            self.W_cell = W_cell
        self.b = b
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

#
#Layer
class GRULayer_setenc(lasagne.layers.Layer): #it is not mvrt lasagne.layers.Layer
    def __init__(self, incoming, num_units, 
                 resetgate=Gate_setenc(W_in=None,W_cell=None), 
                 updategate=Gate_setenc(W_in=None,W_cell=None),
                 hidden_update=Gate_setenc(W_in=None, W_cell=None),
                 nonlinearity=nonlinearities.tanh,
                 hid_init=init.Constant(0.),
                 set_steps=5,
                 att_num_units = 64,
                 W_hid_to_att = init.GlorotNormal(),
                 W_ctx_to_att = init.GlorotNormal(),
                 W_att = init.Normal(0.1), #---
                 backwards=False,
                 learn_init=True, #--???????
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=True, #-- ???????
                 **kwargs):
        #         
        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.  We will just provide the
        # layer input as incomings, unless a mask input was provided.
        #incomings = [incoming]
        #if mask_input is not None:
            #incomings.append(mask_input) #-----
        #
        # Initialize parent layer
        super(GRULayer_setenc, self).__init__(incoming, **kwargs) #super(GRULayer_setenc, self).__init__(incomings, **kwargs)
        #
        self.learn_init = learn_init
        self.num_units = num_units #128
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.set_steps = set_steps
        self.att_num_units = att_num_units #512
        self.only_return_final = only_return_final
        #
        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")
        #
        # Retrieve the dimensionality of the incoming layer
        #
        if unroll_scan and self.input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")
        #
        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(self.input_shape[2:]) #4
        print(num_inputs)
        #print(self.input_shape) #(None, 256, 4)
        #print(num_inputs) #4
        #
        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            #self.add_param(gate.W_in, (num_inputs, num_units),
                                   #name="W_in_to_{}".format(gate_name))
            return (self.add_param(gate.W_hid, (num_units, num_units), #128
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)
        #
        # Add in all parameters from gates, nonlinearities will be sigmas, look Gate_setenc
        """(self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')
        #
        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')"""
        (self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')
        #
        (self.W_hid_to_hidden_update, self.b_hidden_update, 
         self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')
        #
        #attention Weights 
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        #
        # Initialize hidden state
        #self.hid_init = hid_init ######
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)
        """if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError(
                    "When hid_init is provided as a TensorVariable, it should "
                    "have 2 dimensions and have shape (num_batch, num_units)")
            self.hid_init = hid_init
        else: # here
            self.hid_init = self.add_param(  #--????????#--????????#--????????#--????????#--????????#--????????#--????????
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False) #not trainable"""
        #print(self.hid_init, type(self.hid_init)) #(hid_init, <class 'theano.tensor.sharedvar.TensorSharedVariable'>
#
    def get_output_shape_for(self, input_shape):                        #(None, 256, 128)
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shp = self.input_shape[0]
        # PRINTS
        if self.only_return_final:
            return self.input_shape[0], self.num_units #(None, 128)
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return self.input_shape[0], self.input_shape[1], self.num_units
#
    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        #print('uiop') # PRINTS
        # Retrieve the layer input
        input = inputs
        #print(input) #<lasagne.layers.merge.ConcatLayer object at 0x7fe2e1f7fb10>
        #print(input.ndim)
        # Retrieve the mask when it is supplied
        #mask = inputs[1] if len(inputs) > 1 else None # -> mask=None
#
        #print(type(inputs)) #<class 'lasagne.layers.merge.ConcatLayer'>
        # Treat all dimensions after the second as flattened feature dimensions
        #if input.ndim > 3:
            #input = T.flatten(input, 3)
#
        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        #input = lasagne.layers.ReshapeLayer(input, ([1], [0], [2]))#
        input = input.dimshuffle(1, 0, 2)#
        seq_len, num_batch, _ = input.shape #256,None -- no 4 #(256, None) #seq_len, num_batch, _ = input.output_shape
        print(seq_len, num_batch) #(256, None)
        #print(input.output_shape) #(256, None, 4)
#
        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        """W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)"""
#
        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)
#
        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)
#
        #if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            #input = T.dot(input, W_in_stacked) + b_stacked
#
        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
   #     
        def plain_et_step(x_snp, o_t0):#def plain_et_step(self, x_snp, o_t0)
            #reading from memory steps
            bs, seq_len_m, _ = x_snp.shape
            m_in = x_snp.dimshuffle(1, 0, 2)#----replace
            e_qt = T.dot(o_t0, self.W_hid_to_att)#---
            e_m = T.dot(m_in, self.W_ctx_to_att)#----
            e_q = T.tile(e_qt, (seq_len_m, 1, 1)) #e_q = T.tile(e_qt, (self.seq_len_m, 1, 1))
            et_p = T.tanh(e_m + e_q)
            et = T.dot(et_p, self.W_att)
            alpha = T.exp(et)
            alpha /= T.sum(alpha, axis=0)
            mt = x_snp.dimshuffle(2, 1, 0)
            mult = T.mul(mt, alpha)
            rt = T.sum(mult, axis=1)
            return rt.T
#
        def step(hid_previous, *args): #W_hid_stacked, #W_in_stacked, b_stacked):
            #x_snp = incom
            #print(x_snp.output_shape)
            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked) + b_stacked ####for r, z, h tilde WHAT ABOUT THE SIZES WHEN MULT??????
#       
            print("tyui")
            if self.grad_clipping is not False:
                #input_n = theano.gradient.grad_clip(
                    #input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)
                print('d')
#
           # if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                #input_n = T.dot(input_n, W_in_stacked) + b_stacked
#
            # Reset and update gates
            resetgate = slice_w(hid_input, 0) #+ slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) #+ slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)
#
            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            #hidden_update_in = slice_w(input_n, 2)
            ####hidden_update_hid = slice_w(hid_input, 2) #h tilde
            ####hidden_update = resetgate*hidden_update_hid #hidden_update = hidden_update_in + resetgate*hidden_update_hid
            hidden_update = slice_w(hid_input, 2)
            #
            if self.grad_clipping is not False:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update) #here it is sigma, but in encoder.py this is tanh ????????????
#
            print('d')
            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid0 = (np.float32(1.0) - updategate)*hid_previous + updategate*hidden_update
            rt = plain_et_step(input, hid0)
            h_t = T.concatenate([hid0,rt], axis=1)
    #        
            return h_t #------??????
        """def step(x_snp, snp_mask, m_snp_count):
            r_t = T.nnet.sigmoid(T.dot(hr_tm1, self.Wo_hh_r) + self.bo_r)
            z_t = T.nnet.sigmoid(T.dot(hr_tm1, self.W_hh_z) + self.b_z)
            h_tilde = T.tanh(T.dot(r_t * hr_tm1, self.W_hh) + self.b_hh)
            o_t = (np.float32(1.0) - z_t) * hr_tm1 + z_t * h_tilde
            #o_t: GRU hidden state
            #concatanation of hidden state and reading from memory
            rt = self.plain_et_step(x_snp, o_t)
            h_t = T.concatenate([o_t0,rt], axis=1)---------
            return h_t"""
     #       
#
        """def step_masked(input_n, mask_n, hid_previous, W_hid_stacked,
                        W_in_stacked, b_stacked):
#
            hid = step(input_n, hid_previous, W_hid_stacked, W_in_stacked,
                       b_stacked)
#
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            not_mask = 1 - mask_n
            hid = hid*mask_n + hid_previous*not_mask
#
            return hid"""
#
        """if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
            print('d')
        else:
            #sequences = input #[input]
            step_fun = step
            print("step")"""
        step_fun = step
#
        """if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)"""
#
        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked, b_stacked] #non_seqs = [W_hid_stacked] #non_seqs = [input, W_hid_stacked]
        #non_seqs += [W_ctx_stacked] #
        non_seqs += [self.W_hid_to_att, self.W_ctx_to_att, self.W_att, input] #
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        #if not self.precompute_input:
            #non_seqs += [b_stacked]#[W_in_stacked, b_stacked]
        # theano.scan only allows for positional arguments, so when
        # self.precompute_input is True, we need to supply fake placeholder
        # arguments for the input weights and biases.
        #else:
            #non_seqs += [(), ()]
#
        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else: #here
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            #hid_out, _ = theano.scan(fn=step_fun, outputs_info=o_enc_info, non_sequences=incomings[0], n_steps=self.set_steps)#self.num_set_iter) #----??????
            print('d')
            hid_out,_ = theano.scan(fn=step_fun, outputs_info=self.hid_init, non_sequences=non_seqs, n_steps=self.set_steps)#self.num_set_iter)
        """hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]"""
#
        # dimshuffle back to (n_batch, n_time_steps, n_features))
        #######hid_out = hid_out.dimshuffle(1, 0, 2) #done below
#
        # if scan is backward reverse the output
        #if self.backwards:-------- ?????????
            #hid_out = hid_out[:, ::-1, :]-------??????????? #done below
         #
        #copied from layers.py
        """if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)
            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]     """
        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        #return theano.shared(np.random.randn(3,4))
#
        print(hid_out.shape)
        return hid_out

def categorical_crossentropy_3d(coding_dist, true_dist, lengths=None):
    #http://stackoverflow.com/questions/30225633/cross-entropy-for-batch-with-theano
    
    # Zero out the false probabilities and sum the remaining true probabilities to remove the third dimension.
    indexes = theano.tensor.arange(coding_dist.shape[2])
    mask = theano.tensor.neq(indexes, true_dist.reshape((true_dist.shape[0], true_dist.shape[1], 1)))
    pred_probs = theano.tensor.set_subtensor(coding_dist[theano.tensor.nonzero(mask)], 0.).sum(axis=2)
    pred_probs_log = T.log(pred_probs)
    pred_probs_per_sample = -pred_probs_log.sum(axis=1)
    return pred_probs_per_sample

#in network_text()
#X_enc_sym = T.dtensor3('x_enc_sym') 
X_dec_sym = T.dtensor3('x_dec_sym') ##X_dec_sym = T.ftensor3('x_dec_sym')
y_sym = T.lmatrix('y_sym') # indexes of 1hot words, for loss 
Emb_mtx_sym = T.dmatrix('emb_mtx_sym')
eta = theano.shared(np.array(params['learning_rate'], dtype=theano.config.floatX))

import theano.typed_list
X_enc_sym_list = theano.typed_list.TypedListType(T.dtensor3)() # 2d or 3d?
name = 'x_enc_sym'
for i in range(1, num_metrics+1):
    X_enc_sym_list = theano.typed_list.append(X_enc_sym_list, T.dtensor3('x_enc_sym'+str(i)))

theano.typed_list.basic.length(X_enc_sym_list) #Length.0

#in model
#X = X_enc_sym[:,:,0:1]
#l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=theano.typed_list.basic.getitem(X_enc_sym_list,0)) #(None, None, 4) #l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
#l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=0, axis=2) #(None, None)
#l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #(None, None, 1)
    
l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                grad_clipping=grad_clip, only_return_final=True)
l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None),
                                                grad_clipping=grad_clip, only_return_final=True, backwards=True)
l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
l_enc_conc = lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1)) #(None, 256, 1)
#    
resetgate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_resetgate, W_hid=l_forward.W_hid_to_resetgate, W_cell = None, b=l_forward.b_resetgate)
updategate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_updategate, W_hid=l_forward.W_hid_to_updategate, W_cell = None, b=l_forward.b_updategate)
hidden_update_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_hidden_update, W_hid=l_forward.W_hid_to_hidden_update, W_cell=None, b=l_forward.b_hidden_update)
#
resetgate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_resetgate, W_hid=l_backward.W_hid_to_resetgate, W_cell= None, b=l_backward.b_resetgate)
updategate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_updategate, W_hid=l_backward.W_hid_to_updategate, W_cell=None, b=l_backward.b_updategate)
hidden_update_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_hidden_update, W_hid=l_backward.W_hid_to_hidden_update, W_cell=None, b=l_backward.b_hidden_update)
                                      
for i in range(1, num_metrics):
    #X = X_enc_sym[:,:,i:(i+1)]
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=theano.typed_list.basic.getitem(X_enc_sym_list,i))
    #l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=i, axis=2) #(None, None)
    #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], 1, [1]))
        
    l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                                    resetgate=resetgate_f, 
                                                    updategate=updategate_f, 
                                                    hidden_update=hidden_update_f, 
                                                    grad_clipping=grad_clip, only_return_final=True)
    l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size,
                                                    resetgate=resetgate_b, 
                                                    updategate=updategate_b, 
                                                    hidden_update=hidden_update_b,
                                                    grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
    l_enc_conc = lasagne.layers.ConcatLayer([l_enc_conc, lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1))], axis=2) #(None, 256, 4)

#l_enc_conc = lasagne.layers.ReshapeLayer(l_enc_conc, ([1], [0], [2])) #(256, None, 4)

#add what above to model
#l_q = theano.shared(value=np.zeros((hidden_size,1), dtype='float32'))
#l_q = lasagne.init.Constant(0.)
#l_input = lasagne.layers.InputLayer(shape=(None, 256, 4), name=l_enc_conc)
#l_setenc = GRULayer_setenc(incoming=l_enc_conc, num_units=hidden_size, learn_init=False, set_steps=5, att_num_units=att_size, grad_clipping=grad_clip, 
                           nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)  
#lasagne.layers.get_output(l_setenc)


l_out, l_out_val = model_seq2seq_GRU_setenc(X_enc_sym_list, X_dec_sym, Emb_mtx_sym, params['horizon']-1, params['num_metrics'], params['set_steps'], hidden_size = params['num_units'], 
                    grad_clip = params['grad_clipping'], att_size = params['num_att_units'], vocab_size = params['vocab_size'], word_dim = params['word_dim'])
                                            
network_output, network_output_val = lasagne.layers.get_output([l_out, l_out_val])



weights = lasagne.layers.get_all_params(l_out,trainable=True)
if params['regularization_type'] == 'l1':
    reg_loss = lasagne.regularization.regularize_network_params(l_out, l1) * params['lambda_regularization']
else:
    reg_loss = lasagne.regularization.regularize_network_params(l_out, l2) * params['lambda_regularization']

loss_T = categorical_crossentropy_3d(network_output, y_sym).mean() + reg_loss
loss_val_T = categorical_crossentropy_3d(network_output_val, y_sym).mean() 
loss_test = categorical_crossentropy_3d(network_output_val, y_sym).mean() 
#metric_probs = get_metric_probs(network_output, y_sym) #####             

updates = lasagne.updates.adam(loss_T, weights, learning_rate=eta)

f_train = theano.function([X_enc_sym_list, X_dec_sym, y_sym], loss_T, updates=updates, allow_input_downcast=True)

08.06.17************************************************************************************************************************************************************************************************************************************************
By hand sumbolic matrices X_enc_sym1, ...
                            
import lasagne
import numpy as np
import theano
import theano.tensor as T
from theano import shared
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan
from lasagne.layers.base import Layer, MergeLayer
from lasagne.layers.input import InputLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers import helper, SliceLayer
from lasagne.layers.recurrent import Gate
import logging
import sys
import csv
from lasagne.layers import get_output
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
import os
import linecache
from lasagne.layers import InputLayer, ExpressionLayer, EmbeddingLayer
#            
def load_parameters2_mvrt_text(params_csvfile, line_num): 
    """
    reads the params for the new generated params
    """
    params={}
    values = linecache.getline(params_csvfile,line_num)[:-1].split(',')
    params['num_units'] = int(values[0]) #128
    params['num_att_units'] = int(values[1]) #512
    data_type = str(values[2]) 
    params['data_name'] = str(values[3])
    model = str(values[4]) 
    if params['data_name'] == 'coservit':
        params['windowise'] = 288
        params['horizon'] = 27
        params['word_dim'] = 202 # 200
        params['vocab_size'] = 22 ###### 155564
        
        #params['time_data'] = '../../../Data' + params['data_name'] + 
        #params['text_data'] = '../../../Data' + params['data_name'] + 
        #params['data_file'] =  '../../../Data/' + params['data_name'] + 'ticket2time.pkl'
        params['data_file'] =  '../../../Data/coservit/' + 'x_enc.pkl'
        params['text_file_wv'] =  '../../../Data/coservit/' + 'x_dec_wv.dat' ##params['text_file_wv'] =  '../../../Data/coservit/' + 'tickets_wv_pad_mtx.dat'
        params['text_file_w'] =  '../../../Data/coservit/' + 'x_dec_w.dat' #params['text_file_w'] =  '../../../Data/coservit/' + 'tickets_w_pad_mtx.dat'
        params['metr_dec_file'] =  '../../../Data/coservit/' + 'metr_for_dec.pkl' #list of lists with metric ids corresponding to aligned tickets. Ex.: t1,t1,t2,t3,t3,t3->[[3432, 4657], [3442], [6567, 4657, 7855]]
        params['emb_file'] =  '../../../Data/coservit/' + 'lda_emb_mtx.dat'#'emb_mtx.dat'
    else:
        pass
        """params['data_type'] = 'missing'
        #params['data_file'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/'+ params['data_name'] + '_'+ data_type +'.pkl'
        params['data_file'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/'+ params['data_name'] + '_'+ data_type +'_intr_linear.pkl'
        params['data_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/' + params['data_name'] + '_mask_' + data_type + '.pkl'
        params['missing_percent'] = data_type
        params['original_data_file'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'.pkl'
        params['original_data_mask_file'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'_mask.pkl' 
    #TODO:
    if params['data_name']  == 'pse':
        params['data2_name'] = 'weather_data'
        params['data2_file'] = '../../../Data/' + params['data2_name'] + '/missing_gaps_v2/'+ params['data2_name'] + '_'+ data_type +'.pkl'
        params['data2_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/' + params['data_name'] + '_mask_' + data_type + '.pkl'
        params['data2_variables'] = ['mintmp', 'maxtmp', 'precip'] 
    elif params['data_name']  == 'consseason':
        params['data2_name'] = 'consseason'
        params['data2_file'] = params['data_file']
        params['data2_file_missing_mask'] = params['data_file_missing_mask']
        params['data2_variables'] = ['Global_reactive_power', 'Voltage', 'Global_intensity'] 
    elif params['data_name']  == 'airq':
        params['data2_name'] = 'airq'
        params['data2_file'] = params['data_file']
        params['data2_file_missing_mask'] = params['data_file_missing_mask']
        params['data2_variables'] = ['NO2(GT)', 'CO(GT)', 'NOx(GT)'] """
            
    if data_type == 'orig':
        params['data_type'] = 'original'
    else:
        #TODO
        pass
        
    if model == 'RNN':
        params['attention'] = False
    else:
        params['attention'] = True
        if model == 'RNN-ATT':
            params['att_type'] = 'original'
        elif model == 'ATT-lambda':
            params['att_type'] = 'lambda'
        elif model == 'ATT-lambda-mu':
            params['att_type'] = 'lambda_mu' 
        elif model == 'ATT-lambda-mu-alt':
            params['att_type'] = 'lambda_mu_alt'
        elif model == 'ATT-dist':
            params['att_type'] = 'adist'
        elif model == 'RNN-text-set':
            params['att_type'] = 'set_enc'
            params['num_metrics'] = 4
            params['set_steps'] = 5 #number of iterations in set_enc part
    params['alg'] = 'adam'
    params['lambda_regularization'] = 0.0001
    params['eta_decay'] = 0.95
    params['early_stop_patience'] = 10000
    params['num_layers'] = 1
    params['regularization_type'] = 'l2'
    params['early_stop'] = True
    params['grad_clipping'] = 100
    params['train_test_ratio'] = 0.75
    params['train_val_ratio'] = 0.75
    params['patience_increase'] = 2 
    params['epsilon_improvement'] = True 
    params['lr_decay_after_n_epoch'] = 50
    params['interpolation'] = True #!!!!!!
    params['max_num_epochs'] = 25
    params['epsilon'] = 0.01
    params['learning_rate'] = 0.001
    params['batch_size'] = 64
    params['padding'] = True 
    params['learning_rate_decay'] = False
    params['interpolation_type'] = 'None'
    return params

params = load_parameters2_mvrt_text('params/fuzzy.param.text', 2)
params
#
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
results_directory = 'Results_multivariate_text_OM/'
#
att_type = params['att_type']
att_size = params['num_att_units']
word_dim = params['word_dim']
vocab_size = params['vocab_size']
grad_clip = params['grad_clipping']
hidden_size = params['num_units']
pred_len = 27
num_metrics = params['num_metrics']
set_steps = params['set_steps']
grad_clip = 100
#
#Gate
class Gate_setenc(object):
    def __init__(self, W_in=init.GlorotNormal(), W_hid=init.GlorotNormal(),
                 W_cell=init.Normal(0.1), b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        if W_in is not None:
            self.W_in = W_in
        self.W_hid = W_hid
        # Don't store a cell weight vector when cell is None
        if W_cell is not None:
            self.W_cell = W_cell
        self.b = b
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

#
#Layer
class GRULayer_setenc(lasagne.layers.Layer): #it is not mvrt lasagne.layers.Layer
    def __init__(self, incoming, num_units, 
                 resetgate=Gate_setenc(W_in=None,W_cell=None), 
                 updategate=Gate_setenc(W_in=None,W_cell=None),
                 hidden_update=Gate_setenc(W_in=None, W_cell=None),
                 nonlinearity=nonlinearities.tanh,
                 hid_init=init.Constant(0.),
                 set_steps=5,
                 att_num_units = 64,
                 W_hid_to_att = init.GlorotNormal(),
                 W_ctx_to_att = init.GlorotNormal(),
                 W_att = init.Normal(0.1), #---
                 backwards=False,
                 learn_init=True, #--???????
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=True, #-- ???????
                 **kwargs):
        #         
        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.  We will just provide the
        # layer input as incomings, unless a mask input was provided.
        #incomings = [incoming]
        #if mask_input is not None:
            #incomings.append(mask_input) #-----
        #
        # Initialize parent layer
        super(GRULayer_setenc, self).__init__(incoming, **kwargs) #super(GRULayer_setenc, self).__init__(incomings, **kwargs)
        #
        self.learn_init = learn_init
        self.num_units = num_units #128
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.set_steps = set_steps
        self.att_num_units = att_num_units #512
        self.only_return_final = only_return_final
        #
        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")
        #
        # Retrieve the dimensionality of the incoming layer
        #
        if unroll_scan and self.input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")
        #
        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(self.input_shape[2:]) #4
        print(num_inputs)
        #print(self.input_shape) #(None, 256, 4)
        #print(num_inputs) #4
        #
        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            #self.add_param(gate.W_in, (num_inputs, num_units),
                                   #name="W_in_to_{}".format(gate_name))
            return (self.add_param(gate.W_hid, (num_units, num_units), #128
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)
        #
        # Add in all parameters from gates, nonlinearities will be sigmas, look Gate_setenc
        """(self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')
        #
        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')"""
        (self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')
        #
        (self.W_hid_to_hidden_update, self.b_hidden_update, 
         self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')
        #
        #attention Weights 
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        #
        # Initialize hidden state
        #self.hid_init = hid_init ######
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)
        """if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError(
                    "When hid_init is provided as a TensorVariable, it should "
                    "have 2 dimensions and have shape (num_batch, num_units)")
            self.hid_init = hid_init
        else: # here
            self.hid_init = self.add_param(  #--????????#--????????#--????????#--????????#--????????#--????????#--????????
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False) #not trainable"""
        #print(self.hid_init, type(self.hid_init)) #(hid_init, <class 'theano.tensor.sharedvar.TensorSharedVariable'>
#
    def get_output_shape_for(self, input_shape):                        #(None, 256, 128)
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shp = self.input_shape[0]
        # PRINTS
        if self.only_return_final:
            return self.input_shape[0], self.num_units #(None, 128)
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return self.input_shape[0], self.input_shape[1], self.num_units
#
    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        #print('uiop') # PRINTS
        # Retrieve the layer input
        input = inputs
        #print(input) #<lasagne.layers.merge.ConcatLayer object at 0x7fe2e1f7fb10>
        #print(input.ndim)
        # Retrieve the mask when it is supplied
        #mask = inputs[1] if len(inputs) > 1 else None # -> mask=None
#
        #print(type(inputs)) #<class 'lasagne.layers.merge.ConcatLayer'>
        # Treat all dimensions after the second as flattened feature dimensions
        #if input.ndim > 3:
            #input = T.flatten(input, 3)
#
        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        #input = lasagne.layers.ReshapeLayer(input, ([1], [0], [2]))#
        input = input.dimshuffle(1, 0, 2)#
        seq_len, num_batch, _ = input.shape #256,None -- no 4 #(256, None) #seq_len, num_batch, _ = input.output_shape
        print(seq_len, num_batch) #(256, None)
        #print(input.output_shape) #(256, None, 4)
#
        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        """W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)"""
#
        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)
#
        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)
#
        #if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            #input = T.dot(input, W_in_stacked) + b_stacked
#
        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
   #     
        def plain_et_step(o_t0):#def plain_et_step(self, x_snp, o_t0)
            #reading from memory steps
            bs, seq_len_m, _ = input.shape
            m_in = x_snp.dimshuffle(1, 0, 2)#----replace
            e_qt = T.dot(o_t0, self.W_hid_to_att)#---
            e_m = T.dot(m_in, self.W_ctx_to_att)#----
            e_q = T.tile(e_qt, (seq_len_m, 1, 1)) #e_q = T.tile(e_qt, (self.seq_len_m, 1, 1))
            et_p = T.tanh(e_m + e_q)
            et = T.dot(et_p, self.W_att)
            alpha = T.exp(et)
            alpha /= T.sum(alpha, axis=0)
            mt = x_snp.dimshuffle(2, 1, 0)
            mult = T.mul(mt, alpha)
            rt = T.sum(mult, axis=1)
            return rt.T
#
        def step(hid_previous, *args): #W_hid_stacked, #W_in_stacked, b_stacked):
            #x_snp = incom
            #print(x_snp.output_shape)
            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked) + b_stacked ####for r, z, h tilde WHAT ABOUT THE SIZES WHEN MULT??????
#       
            print("tyui")
            if self.grad_clipping is not False:
                #input_n = theano.gradient.grad_clip(
                    #input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)
                print('d')
#
           # if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                #input_n = T.dot(input_n, W_in_stacked) + b_stacked
#
            # Reset and update gates
            resetgate = slice_w(hid_input, 0) #+ slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) #+ slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)
#
            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            #hidden_update_in = slice_w(input_n, 2)
            ####hidden_update_hid = slice_w(hid_input, 2) #h tilde
            ####hidden_update = resetgate*hidden_update_hid #hidden_update = hidden_update_in + resetgate*hidden_update_hid
            hidden_update = slice_w(hid_input, 2)
            #
            if self.grad_clipping is not False:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update) #here it is sigma, but in encoder.py this is tanh ????????????
#
            print('d')
            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid0 = (np.float32(1.0) - updategate)*hid_previous + updategate*hidden_update
            rt = plain_et_step(hid0)
            h_t = T.concatenate([hid0,rt], axis=1)
    #        
            return h_t #------??????
        """def step(x_snp, snp_mask, m_snp_count):
            r_t = T.nnet.sigmoid(T.dot(hr_tm1, self.Wo_hh_r) + self.bo_r)
            z_t = T.nnet.sigmoid(T.dot(hr_tm1, self.W_hh_z) + self.b_z)
            h_tilde = T.tanh(T.dot(r_t * hr_tm1, self.W_hh) + self.b_hh)
            o_t = (np.float32(1.0) - z_t) * hr_tm1 + z_t * h_tilde
            #o_t: GRU hidden state
            #concatanation of hidden state and reading from memory
            rt = self.plain_et_step(x_snp, o_t)
            h_t = T.concatenate([o_t0,rt], axis=1)---------
            return h_t"""
     #       
#
        """def step_masked(input_n, mask_n, hid_previous, W_hid_stacked,
                        W_in_stacked, b_stacked):
#
            hid = step(input_n, hid_previous, W_hid_stacked, W_in_stacked,
                       b_stacked)
#
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            not_mask = 1 - mask_n
            hid = hid*mask_n + hid_previous*not_mask
#
            return hid"""
#
        """if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
            print('d')
        else:
            #sequences = input #[input]
            step_fun = step
            print("step")"""
        step_fun = step
#
        """if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)"""
#
        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked, b_stacked] #non_seqs = [W_hid_stacked] #non_seqs = [input, W_hid_stacked]
        #non_seqs += [W_ctx_stacked] #
        non_seqs += [self.W_hid_to_att, self.W_ctx_to_att, self.W_att, input] #
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        #if not self.precompute_input:
            #non_seqs += [b_stacked]#[W_in_stacked, b_stacked]
        # theano.scan only allows for positional arguments, so when
        # self.precompute_input is True, we need to supply fake placeholder
        # arguments for the input weights and biases.
        #else:
            #non_seqs += [(), ()]
#
        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else: #here
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            #hid_out, _ = theano.scan(fn=step_fun, outputs_info=o_enc_info, non_sequences=incomings[0], n_steps=self.set_steps)#self.num_set_iter) #----??????
            print('d')
            hid_out, _ = theano.scan(fn=step_fun, outputs_info=self.hid_init, non_sequences=non_seqs, n_steps=self.set_steps, strict=True)#self.num_set_iter)
        """hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]"""
#
        # dimshuffle back to (n_batch, n_time_steps, n_features))
        #######hid_out = hid_out.dimshuffle(1, 0, 2) #done below
#
        # if scan is backward reverse the output
        #if self.backwards:-------- ?????????
            #hid_out = hid_out[:, ::-1, :]-------??????????? #done below
         #
        #copied from layers.py
        """if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)
            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]     """
        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        #return theano.shared(np.random.randn(3,4))
#
        print(hid_out.shape)
        return hid_out

def categorical_crossentropy_3d(coding_dist, true_dist, lengths=None):
    #http://stackoverflow.com/questions/30225633/cross-entropy-for-batch-with-theano
    
    # Zero out the false probabilities and sum the remaining true probabilities to remove the third dimension.
    indexes = theano.tensor.arange(coding_dist.shape[2])
    mask = theano.tensor.neq(indexes, true_dist.reshape((true_dist.shape[0], true_dist.shape[1], 1)))
    pred_probs = theano.tensor.set_subtensor(coding_dist[theano.tensor.nonzero(mask)], 0.).sum(axis=2)
    pred_probs_log = T.log(pred_probs)
    pred_probs_per_sample = -pred_probs_log.sum(axis=1)
    return pred_probs_per_sample

#in network_text()
X_enc_sym1 = T.dtensor3('x_enc_sym1')
X_enc_sym2 = T.dtensor3('x_enc_sym2')
X_enc_sym3 = T.dtensor3('x_enc_sym3') 
X_dec_sym = T.dtensor3('x_dec_sym') ##X_dec_sym = T.ftensor3('x_dec_sym')
y_sym = T.lmatrix('y_sym') # indexes of 1hot words, for loss 
Emb_mtx_sym = T.dmatrix('emb_mtx_sym')
eta = theano.shared(np.array(params['learning_rate'], dtype=theano.config.floatX))

#import theano.typed_list
#X_enc_sym_list = theano.typed_list.TypedListType(T.dtensor3)() # 2d or 3d?
#name = 'x_enc_sym'
#for i in range(1, num_metrics+1):
#    X_enc_sym_list = theano.typed_list.append(X_enc_sym_list, T.dtensor3('x_enc_sym'+str(i)))

#theano.typed_list.basic.length(X_enc_sym_list) #Length.0

#in model
#X = X_enc_sym[:,:,0:1]
#l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym1) #(None, None, 4) #l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
#l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=0, axis=2) #(None, None)
#l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #(None, None, 1)
    
l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                grad_clipping=grad_clip, only_return_final=True)
l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None),
                                                grad_clipping=grad_clip, only_return_final=True, backwards=True)
l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
l_enc_conc = lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1)) #(None, 256, 1)
#    
resetgate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_resetgate, W_hid=l_forward.W_hid_to_resetgate, W_cell = None, b=l_forward.b_resetgate)
updategate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_updategate, W_hid=l_forward.W_hid_to_updategate, W_cell = None, b=l_forward.b_updategate)
hidden_update_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_hidden_update, W_hid=l_forward.W_hid_to_hidden_update, W_cell=None, b=l_forward.b_hidden_update)
#
resetgate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_resetgate, W_hid=l_backward.W_hid_to_resetgate, W_cell= None, b=l_backward.b_resetgate)
updategate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_updategate, W_hid=l_backward.W_hid_to_updategate, W_cell=None, b=l_backward.b_updategate)
hidden_update_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_hidden_update, W_hid=l_backward.W_hid_to_hidden_update, W_cell=None, b=l_backward.b_hidden_update)
          
"""                              
for i in range(1, num_metrics):
    #X = X_enc_sym[:,:,i:(i+1)]
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=theano.typed_list.basic.getitem(X_enc_sym_list,i))
    #l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=i, axis=2) #(None, None)
    #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], 1, [1]))
        
    l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                                    resetgate=resetgate_f, 
                                                    updategate=updategate_f, 
                                                    hidden_update=hidden_update_f, 
                                                    grad_clipping=grad_clip, only_return_final=True)
    l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size,
                                                    resetgate=resetgate_b, 
                                                    updategate=updategate_b, 
                                                    hidden_update=hidden_update_b,
                                                    grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
    l_enc_conc = lasagne.layers.ConcatLayer([l_enc_conc, lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1))], axis=2) #(None, 256, 4)
"""
#X_enc_sym2
l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym2)
l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                                    resetgate=resetgate_f, updategate=updategate_f, hidden_update=hidden_update_f, grad_clipping=grad_clip, only_return_final=True)
l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size,
                                                    resetgate=resetgate_b, updategate=updategate_b, hidden_update=hidden_update_b, grad_clipping=grad_clip, only_return_final=True, backwards=True)
l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
l_enc_conc = lasagne.layers.ConcatLayer([l_enc_conc, lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1))], axis=2) #(None, 256, 4)
#X_enc_sym3
l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym3)
l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                                    resetgate=resetgate_f, updategate=updategate_f, hidden_update=hidden_update_f, grad_clipping=grad_clip, only_return_final=True)
l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size,
                                                    resetgate=resetgate_b, updategate=updategate_b, hidden_update=hidden_update_b, grad_clipping=grad_clip, only_return_final=True, backwards=True)
l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
l_enc_conc = lasagne.layers.ConcatLayer([l_enc_conc, lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1))], axis=2) #(None, 256, 4)

#add what above to model
#l_q = theano.shared(value=np.zeros((hidden_size,1), dtype='float32'))
#l_q = lasagne.init.Constant(0.)
#l_input = lasagne.layers.InputLayer(shape=(None, 256, 4), name=l_enc_conc)
#l_setenc = GRULayer_setenc(incoming=l_enc_conc, num_units=hidden_size, learn_init=False, set_steps=5, att_num_units=att_size, grad_clipping=grad_clip, 
                           nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)  
#lasagne.layers.get_output(l_setenc)


def model_seq2seq_GRU_setenc(X_enc_sym1, X_enc_sym2, X_enc_sym3, X_dec_sym, Emb_mtx_sym, pred_len, num_metrics, set_steps, hidden_size=64, grad_clip = 100, att_size=64, vocab_size=22, word_dim=202): ##def model_seq2seq_GRU_text(X_enc_sym, X_enc_sym2, X_dec_sym, max_len, pred_len, hidden_size=64, grad_clip = 100, vocab_size=155563, word_dim=200):#(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, max_len, pred_len, hidden_size=64, grad_clip = 100) # max_len=288 - windowise, pred_len=27 - horizon
    #copy of model_seq2seq_mvrt4_GRU_text
    # model for order matters article 31.05.17
    #
    #collect last hiddens for each metric 
    #X = X_enc_sym[:,:,0:1]
    #l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym1) #(None, None, 4) #l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
    #l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=0, axis=2) #(None, None)
    #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #(None, None, 1)
    
    l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    grad_clipping=grad_clip, only_return_final=True)
    l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None),
                                                    grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
    l_enc_conc = lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1)) #(None, 256, 1)
    #    
    resetgate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_resetgate, W_hid=l_forward.W_hid_to_resetgate, W_cell = None, b=l_forward.b_resetgate)
    updategate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_updategate, W_hid=l_forward.W_hid_to_updategate, W_cell = None, b=l_forward.b_updategate)
    hidden_update_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_hidden_update, W_hid=l_forward.W_hid_to_hidden_update, W_cell=None, b=l_forward.b_hidden_update)
    #
    resetgate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_resetgate, W_hid=l_backward.W_hid_to_resetgate, W_cell= None, b=l_backward.b_resetgate)
    updategate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_updategate, W_hid=l_backward.W_hid_to_updategate, W_cell=None, b=l_backward.b_updategate)
    hidden_update_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_hidden_update, W_hid=l_backward.W_hid_to_hidden_update, W_cell=None, b=l_backward.b_hidden_update)
    #                                  
    #X_enc_sym2
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym2)
    l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                                        resetgate=resetgate_f, updategate=updategate_f, hidden_update=hidden_update_f, grad_clipping=grad_clip, only_return_final=True)
    l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size,
                                                        resetgate=resetgate_b, updategate=updategate_b, hidden_update=hidden_update_b, grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
    l_enc_conc = lasagne.layers.ElemwiseSumLayer([l_enc_conc, lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1))], axis=2) #(None, 256, 4)
    #X_enc_sym3
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym3)
    l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                                        resetgate=resetgate_f, updategate=updategate_f, hidden_update=hidden_update_f, grad_clipping=grad_clip, only_return_final=True)
    l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size,
                                                        resetgate=resetgate_b, updategate=updategate_b, hidden_update=hidden_update_b, grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
    l_enc_conc = lasagne.layers.ElemwiseSumLayer([l_enc_conc, lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1))], axis=2) #(None, 256, 4)
    l_setenc = lasagne.layers.ElemwiseSumLayer([l_enc_conc], axis=2) #(None, 256)
    # 
    # Set-encoder part
    #l_q = lasagne.init.Constant(0.)
    #for i in range(set_steps):
    #l_setenc = GRULayer_setenc(incoming=l_enc_conc, num_units=hidden_size, learn_init=False, set_steps=set_steps, att_num_units=att_size, grad_clipping=grad_clip, 
    #                            nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True) 
    #l_q = lasagne.layers.SliceLayer(l_setenc, -1, 1)               ###
    #
    dec_units = 2*1 +1 #dec_units = 4*2 +1 #nr_encoder *2 + dec -> bi-directional so *2 !!important to set it right
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, word_dim),input_var=X_dec_sym)#pred_len=27 #l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len,1),input_var=X_dec_sym)
    #l_mask_dec = lasagne.layers.InputLayer(shape=(None, pred_len), input_var=mask_dec)
    #    
    s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
    s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #SHOULD WE RESHAPE IT???
    #s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    #s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    #h_init = lasagne.layers.ConcatLayer([l_forward, l_q], axis=1) #?????????????
    #
    l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, 
                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)),
                                    learn_init=False, grad_clipping=grad_clip, only_return_final=True ) #l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, mask_input=s_lmask_dec1, learn_init=False, 
                                         #hid_init=h_init, grad_clipping=grad_clip, only_return_final=True )
    r_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_resetgate, W_hid=l_dec.W_hid_to_resetgate, b=l_dec.b_resetgate)
    u_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_updategate, W_hid=l_dec.W_hid_to_updategate, b=l_dec.b_updategate)
    h_update = lasagne.layers.Gate(W_in=l_dec.W_in_to_hidden_update, W_hid=l_dec.W_hid_to_hidden_update, b=l_dec.b_hidden_update)                                
    #
    l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size))
    #TO CHANGE BACK BELOW
    l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, nonlinearity=lasagne.nonlinearities.softmax)  #l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=1, nonlinearity=lasagne.nonlinearities.linear)
    #
    w_dense = l_out.W
    b_dense = l_out.b
    l_out_loop = l_out
    l_out_loop_val = l_out  
    l_out = lasagne.layers.ReshapeLayer(l_out, ([0], 1, [1]))
    l_out_val = l_out
    #h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_q], axis=1) #---
    #h_init_val = lasagne.layers.ConcatLayer([l_dec_hid_state, l_q], axis=1) #---
#
    for i in range(1,pred_len): #comments in this cycle are for the first iteration
        s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1) #(None, 200)
        s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #(None, 1, 200)
        l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, learn_init=False,	 #(None, 320)
                                            grad_clipping=grad_clip, only_return_final=True, 
                                            resetgate=r_gate, updategate=u_gate, hidden_update=h_update)
        l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 64)
        #
        #------ ARGMAX for validation(->forecast) parts in network via embedding matrix 18.05 ---------------------------
    	#for val and train sets
    	coord_max = ExpressionLayer(l_out_loop_val, lambda x: x.argmax(-1).astype('int32'), output_shape='auto') #(None,)
    	emb_slice = ExpressionLayer(coord_max, lambda x: Emb_mtx_sym[x,:], output_shape=(None,word_dim)) #(None, 200)
	#
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: X.argmax(-1).astype('int32'), output_shape='auto') #(None,)
    	#pred_d = ExpressionLayer(coord_max, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X], 1), output_shape='auto')
    	#pred_emb = lasagne.layers.DenseLayer(pred_d, num_units=word_dim, W=Emb_mtx_sym, nonlinearity=lasagne.nonlinearities.linear) #no biases #(None, 200)
    	#pred_emb.params[pred_emb.W].remove("trainable")
        #
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X.argmax(-1).astype('int32')], 1), output_shape='auto')
    	#hot = lasagne.layers.DenseLayer(coord_max, num_units=word_dim, nonlinearity=lasagne.nonlinearities.linear)
        #
        pred = lasagne.layers.ReshapeLayer(emb_slice, ([0], 1, [1])) #######pred = lasagne.layers.ReshapeLayer(pred_emb, ([0], 1, [1])) #(None, 1, 200)
        #
        l_dec_val = lasagne.layers.GRULayer(pred, num_units=hidden_size*dec_units, learn_init=False,
                                             grad_clipping=grad_clip, only_return_final=True,#hid_init=h_init_val, grad_clipping=grad_clip, only_return_final=True,
                                             resetgate=r_gate, updategate=u_gate, hidden_update=h_update) #(None, 320)
    	l_dec_val_hid_state = lasagne.layers.SliceLayer(l_dec_val, indices=slice(0,hidden_size)) #(None, 64)
    	l_out_loop = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
    	l_out_loop = lasagne.layers.ReshapeLayer(l_out_loop, ([0], 1, [1])) ####### #(None, 1, 155563)
    	l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop], axis=1) #(None, 2, 155563)
    	l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
    	l_out_loop_val_d = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1])) #(None, 1, 155563)
    	l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val_d], axis=1) #(None, 2, 155563)
    	#l_out_loop_val = lasagne.layers.FlattenLayer(l_out_loop_val, 2)
    	h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1)  #h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1) #(None, 320)
    	h_init_val = lasagne.layers.ConcatLayer([l_dec_val_hid_state, l_enc], axis=1) #(None, 320)
    #l_out = lasagne.layers.ReshapeLayer(l_out, (-1, hidden_size))
    #l_out_val = lasagne.layers.ReshapeLayer(l_out_val, (-1, hidden_size))
    return (l_out, l_out_val)

l_out, l_out_val = model_seq2seq_GRU_setenc(X_enc_sym1, X_enc_sym3, X_enc_sym3, X_dec_sym, Emb_mtx_sym, params['horizon']-1, params['num_metrics'], params['set_steps'], hidden_size = params['num_units'], 
                    grad_clip = params['grad_clipping'], att_size = params['num_att_units'], vocab_size = params['vocab_size'], word_dim = params['word_dim'])
                                            
network_output, network_output_val = lasagne.layers.get_output([l_out, l_out_val])



weights = lasagne.layers.get_all_params(l_out,trainable=True)
if params['regularization_type'] == 'l1':
    reg_loss = lasagne.regularization.regularize_network_params(l_out, l1) * params['lambda_regularization']
else:
    reg_loss = lasagne.regularization.regularize_network_params(l_out, l2) * params['lambda_regularization']

loss_T = categorical_crossentropy_3d(network_output, y_sym).mean() + reg_loss
loss_val_T = categorical_crossentropy_3d(network_output_val, y_sym).mean() 
loss_test = categorical_crossentropy_3d(network_output_val, y_sym).mean() 
#metric_probs = get_metric_probs(network_output, y_sym) #####             

updates = lasagne.updates.adam(loss_T, weights, learning_rate=eta)

f_train = theano.function([X_enc_sym1, X_enc_sym2, X_enc_sym3, X_dec_sym, y_sym], loss_T, updates=updates, allow_input_downcast=True)



08.06.17 - 20:00------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
By hand sumbolic matrices X_enc_sym1, ... with Yagmur
                            
import lasagne
import numpy as np
import theano
import theano.tensor as T
from theano import shared
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan
from lasagne.layers.base import Layer, MergeLayer
from lasagne.layers.input import InputLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers import helper, SliceLayer
from lasagne.layers.recurrent import Gate
import logging
import sys
import csv
from lasagne.layers import get_output
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
import os
import linecache
from lasagne.layers import InputLayer, ExpressionLayer, EmbeddingLayer
#            
def load_parameters2_mvrt_text(params_csvfile, line_num): 
    """
    reads the params for the new generated params
    """
    params={}
    values = linecache.getline(params_csvfile,line_num)[:-1].split(',')
    params['num_units'] = int(values[0]) #128
    params['num_att_units'] = int(values[1]) #512
    data_type = str(values[2]) 
    params['data_name'] = str(values[3])
    model = str(values[4]) 
    if params['data_name'] == 'coservit':
        params['windowise'] = 288
        params['horizon'] = 27
        params['word_dim'] = 202 # 200
        params['vocab_size'] = 22 ###### 155564
        
        #params['time_data'] = '../../../Data' + params['data_name'] + 
        #params['text_data'] = '../../../Data' + params['data_name'] + 
        #params['data_file'] =  '../../../Data/' + params['data_name'] + 'ticket2time.pkl'
        params['data_file'] =  '../../../Data/coservit/' + 'x_enc.pkl'
        params['text_file_wv'] =  '../../../Data/coservit/' + 'x_dec_wv.dat' ##params['text_file_wv'] =  '../../../Data/coservit/' + 'tickets_wv_pad_mtx.dat'
        params['text_file_w'] =  '../../../Data/coservit/' + 'x_dec_w.dat' #params['text_file_w'] =  '../../../Data/coservit/' + 'tickets_w_pad_mtx.dat'
        params['metr_dec_file'] =  '../../../Data/coservit/' + 'metr_for_dec.pkl' #list of lists with metric ids corresponding to aligned tickets. Ex.: t1,t1,t2,t3,t3,t3->[[3432, 4657], [3442], [6567, 4657, 7855]]
        params['emb_file'] =  '../../../Data/coservit/' + 'lda_emb_mtx.dat'#'emb_mtx.dat'
    else:
        pass
        """params['data_type'] = 'missing'
        #params['data_file'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/'+ params['data_name'] + '_'+ data_type +'.pkl'
        params['data_file'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/'+ params['data_name'] + '_'+ data_type +'_intr_linear.pkl'
        params['data_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/' + params['data_name'] + '_mask_' + data_type + '.pkl'
        params['missing_percent'] = data_type
        params['original_data_file'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'.pkl'
        params['original_data_mask_file'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'_mask.pkl' 
    #TODO:
    if params['data_name']  == 'pse':
        params['data2_name'] = 'weather_data'
        params['data2_file'] = '../../../Data/' + params['data2_name'] + '/missing_gaps_v2/'+ params['data2_name'] + '_'+ data_type +'.pkl'
        params['data2_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/' + params['data_name'] + '_mask_' + data_type + '.pkl'
        params['data2_variables'] = ['mintmp', 'maxtmp', 'precip'] 
    elif params['data_name']  == 'consseason':
        params['data2_name'] = 'consseason'
        params['data2_file'] = params['data_file']
        params['data2_file_missing_mask'] = params['data_file_missing_mask']
        params['data2_variables'] = ['Global_reactive_power', 'Voltage', 'Global_intensity'] 
    elif params['data_name']  == 'airq':
        params['data2_name'] = 'airq'
        params['data2_file'] = params['data_file']
        params['data2_file_missing_mask'] = params['data_file_missing_mask']
        params['data2_variables'] = ['NO2(GT)', 'CO(GT)', 'NOx(GT)'] """
            
    if data_type == 'orig':
        params['data_type'] = 'original'
    else:
        #TODO
        pass
        
    if model == 'RNN':
        params['attention'] = False
    else:
        params['attention'] = True
        if model == 'RNN-ATT':
            params['att_type'] = 'original'
        elif model == 'ATT-lambda':
            params['att_type'] = 'lambda'
        elif model == 'ATT-lambda-mu':
            params['att_type'] = 'lambda_mu' 
        elif model == 'ATT-lambda-mu-alt':
            params['att_type'] = 'lambda_mu_alt'
        elif model == 'ATT-dist':
            params['att_type'] = 'adist'
        elif model == 'RNN-text-set':
            params['att_type'] = 'set_enc'
            params['num_metrics'] = 4
            params['set_steps'] = 5 #number of iterations in set_enc part
    params['alg'] = 'adam'
    params['lambda_regularization'] = 0.0001
    params['eta_decay'] = 0.95
    params['early_stop_patience'] = 10000
    params['num_layers'] = 1
    params['regularization_type'] = 'l2'
    params['early_stop'] = True
    params['grad_clipping'] = 100
    params['train_test_ratio'] = 0.75
    params['train_val_ratio'] = 0.75
    params['patience_increase'] = 2 
    params['epsilon_improvement'] = True 
    params['lr_decay_after_n_epoch'] = 50
    params['interpolation'] = True #!!!!!!
    params['max_num_epochs'] = 25
    params['epsilon'] = 0.01
    params['learning_rate'] = 0.001
    params['batch_size'] = 64
    params['padding'] = True 
    params['learning_rate_decay'] = False
    params['interpolation_type'] = 'None'
    return params

params = load_parameters2_mvrt_text('params/fuzzy.param.text', 2)
params
#
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
results_directory = 'Results_multivariate_text_OM/'
#
att_type = params['att_type']
att_size = params['num_att_units']
word_dim = params['word_dim']
vocab_size = params['vocab_size']
grad_clip = params['grad_clipping']
hidden_size = params['num_units']
pred_len = 27
num_metrics = params['num_metrics']
set_steps = params['set_steps']
grad_clip = 100
#
#Gate
class Gate_setenc(object):
    def __init__(self, W_in=init.GlorotNormal(), W_hid=init.GlorotNormal(),
                 W_cell=init.Normal(0.1), b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        if W_in is not None:
            self.W_in = W_in
        self.W_hid = W_hid
        # Don't store a cell weight vector when cell is None
        if W_cell is not None:
            self.W_cell = W_cell
        self.b = b
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

#
#Layer
class GRULayer_setenc(lasagne.layers.Layer): #it is not mvrt lasagne.layers.Layer
    def __init__(self, incoming, num_units, 
                 resetgate=Gate_setenc(W_in=None,W_cell=None), 
                 updategate=Gate_setenc(W_in=None,W_cell=None),
                 hidden_update=Gate_setenc(W_in=None, W_cell=None),
                 nonlinearity=nonlinearities.tanh,
                 hid_init=init.Constant(0.),
                 set_steps=5,
                 att_num_units = 64,
                 W_hid_to_att = init.GlorotNormal(),
                 W_ctx_to_att = init.GlorotNormal(),
                 W_att = init.Normal(0.1), #---
                 backwards=False,
                 learn_init=True, #--???????
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=True, #-- ???????
                 **kwargs):
        #         
        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.  We will just provide the
        # layer input as incomings, unless a mask input was provided.
        #incomings = [incoming]
        #if mask_input is not None:
            #incomings.append(mask_input) #-----
        #
        # Initialize parent layer
        super(GRULayer_setenc, self).__init__(incoming, **kwargs) #super(GRULayer_setenc, self).__init__(incomings, **kwargs)
        #
        self.learn_init = learn_init
        self.num_units = num_units #128
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.set_steps = set_steps
        self.att_num_units = att_num_units #512
        self.only_return_final = only_return_final
        #
        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")
        #
        # Retrieve the dimensionality of the incoming layer
        #
        if unroll_scan and self.input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")
        #
        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(self.input_shape[2:]) #4
        print(num_inputs)
        #print(self.input_shape) #(None, 256, 4)
        #print(num_inputs) #4
        #
        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            #self.add_param(gate.W_in, (num_inputs, num_units),
                                   #name="W_in_to_{}".format(gate_name))
            return (self.add_param(gate.W_hid, (num_units, num_units), #128
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)
        #
        # Add in all parameters from gates, nonlinearities will be sigmas, look Gate_setenc
        """(self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')
        #
        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')"""
        (self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')
        #
        (self.W_hid_to_hidden_update, self.b_hidden_update, 
         self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')
        #
        #attention Weights 
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        #
        # Initialize hidden state
        #self.hid_init = hid_init ######
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False) #???????
        """if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError(
                    "When hid_init is provided as a TensorVariable, it should "
                    "have 2 dimensions and have shape (num_batch, num_units)")
            self.hid_init = hid_init
        else: # here
            self.hid_init = self.add_param(  #--????????#--????????#--????????#--????????#--????????#--????????#--????????
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False) #not trainable"""
        #print(self.hid_init, type(self.hid_init)) #(hid_init, <class 'theano.tensor.sharedvar.TensorSharedVariable'>
#
    def get_output_shape_for(self, input_shape):                        #(None, 256, 128)
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shp = self.input_shape[0]
        # PRINTS
        if self.only_return_final:
            return self.input_shape[0], 3*self.num_units #(None, 128)
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return self.input_shape[0], self.input_shape[1], 3*self.num_units
#
    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        #print('uiop') # PRINTS
        # Retrieve the layer input
        input = inputs
        #print(input) #<lasagne.layers.merge.ConcatLayer object at 0x7fe2e1f7fb10>
        #print(input.ndim)
        # Retrieve the mask when it is supplied
        #mask = inputs[1] if len(inputs) > 1 else None # -> mask=None
#
        #print(type(inputs)) #<class 'lasagne.layers.merge.ConcatLayer'>
        # Treat all dimensions after the second as flattened feature dimensions
        #if input.ndim > 3:
            #input = T.flatten(input, 3)
#
        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        #input = lasagne.layers.ReshapeLayer(input, ([1], [0], [2]))#
        input = input.dimshuffle(1, 0, 2)#
        seq_len, num_batch, _ = input.shape #256,None -- no 4 #(256, None) #seq_len, num_batch, _ = input.output_shape
        print(seq_len, num_batch) #(256, None)
        #print(input.output_shape) #(256, None, 4)
#
        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        """W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)"""
#
        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)
#
        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)
#
        #if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            #input = T.dot(input, W_in_stacked) + b_stacked
#
        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
   #     
        def plain_et_step(o_t0):#def plain_et_step(self, x_snp, o_t0)
            #reading from memory steps
            bs, seq_len_m, _ = input.shape
            m_in = input.dimshuffle(1, 0, 2)#----replace
            e_qt = T.dot(o_t0, self.W_hid_to_att)#---
            e_m = T.dot(m_in, self.W_ctx_to_att)#----
            e_q = T.tile(e_qt, (seq_len_m, 1, 1)) #e_q = T.tile(e_qt, (self.seq_len_m, 1, 1))
            et_p = T.tanh(e_m + e_q)
            et = T.dot(et_p, self.W_att)
            alpha = T.exp(et)
            alpha /= T.sum(alpha, axis=0)
            mt = input.dimshuffle(2, 1, 0)
            mult = T.mul(mt, alpha)
            rt = T.sum(mult, axis=1)
            #print(input.shape, rt.shape)
            return rt.T
#
        def step(hid_previous, *args): #W_hid_stacked, #W_in_stacked, b_stacked):
            #x_snp = incom
            #print(x_snp.output_shape)
            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked) + b_stacked ####for r, z, h tilde WHAT ABOUT THE SIZES WHEN MULT??????
#       
            print("tyui")
            if self.grad_clipping is not False:
                #input_n = theano.gradient.grad_clip(
                    #input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)
                print('d')
#
           # if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                #input_n = T.dot(input_n, W_in_stacked) + b_stacked
#
            # Reset and update gates
            resetgate = slice_w(hid_input, 0) #+ slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) #+ slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)
#
            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            #hidden_update_in = slice_w(input_n, 2)
            ####hidden_update_hid = slice_w(hid_input, 2) #h tilde
            ####hidden_update = resetgate*hidden_update_hid #hidden_update = hidden_update_in + resetgate*hidden_update_hid
            hidden_update = slice_w(hid_input, 2)
            #
            if self.grad_clipping is not False:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update) #here it is sigma, but in encoder.py this is tanh ????????????
#
            print('d')
            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid0 = (np.float32(1.0) - updategate)*hid_previous + updategate*hidden_update
            rt = plain_et_step(hid0)
            h_t = T.concatenate([hid0,rt], axis=1)
    #        
            return h_t #------??????
        """def step(x_snp, snp_mask, m_snp_count):
            r_t = T.nnet.sigmoid(T.dot(hr_tm1, self.Wo_hh_r) + self.bo_r)
            z_t = T.nnet.sigmoid(T.dot(hr_tm1, self.W_hh_z) + self.b_z)
            h_tilde = T.tanh(T.dot(r_t * hr_tm1, self.W_hh) + self.b_hh)
            o_t = (np.float32(1.0) - z_t) * hr_tm1 + z_t * h_tilde
            #o_t: GRU hidden state
            #concatanation of hidden state and reading from memory
            rt = self.plain_et_step(x_snp, o_t)
            h_t = T.concatenate([o_t0,rt], axis=1)---------
            return h_t"""
     #       
#
        """def step_masked(input_n, mask_n, hid_previous, W_hid_stacked,
                        W_in_stacked, b_stacked):
#
            hid = step(input_n, hid_previous, W_hid_stacked, W_in_stacked,
                       b_stacked)
#
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            not_mask = 1 - mask_n
            hid = hid*mask_n + hid_previous*not_mask
#
            return hid"""
#
        """if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
            print('d')
        else:
            #sequences = input #[input]
            step_fun = step
            print("step")"""
        step_fun = step
#
        """if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)"""
#
        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked, b_stacked] #non_seqs = [W_hid_stacked] #non_seqs = [input, W_hid_stacked]
        #non_seqs += [W_ctx_stacked] #
        non_seqs += [self.W_hid_to_att, self.W_ctx_to_att, self.W_att, input] #
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        #if not self.precompute_input:
            #non_seqs += [b_stacked]#[W_in_stacked, b_stacked]
        # theano.scan only allows for positional arguments, so when
        # self.precompute_input is True, we need to supply fake placeholder
        # arguments for the input weights and biases.
        #else:
            #non_seqs += [(), ()]
#
        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else: #here
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            #hid_out, _ = theano.scan(fn=step_fun, outputs_info=o_enc_info, non_sequences=incomings[0], n_steps=self.set_steps)#self.num_set_iter) #----??????
            print('d')
            hid_out, _ = theano.scan(fn=step_fun, outputs_info=self.hid_init, non_sequences=non_seqs, n_steps=self.set_steps, strict=True)#self.num_set_iter)
        """hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]"""
#
        # dimshuffle back to (n_batch, n_time_steps, n_features))
        #######hid_out = hid_out.dimshuffle(1, 0, 2) #done below
#
        # if scan is backward reverse the output
        #if self.backwards:-------- ?????????
            #hid_out = hid_out[:, ::-1, :]-------??????????? #done below
         #
        #copied from layers.py
        """if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)
            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]     """
        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        #return theano.shared(np.random.randn(3,4))
#
        print(hid_out.shape)
        return hid_out

def categorical_crossentropy_3d(coding_dist, true_dist, lengths=None):
    #http://stackoverflow.com/questions/30225633/cross-entropy-for-batch-with-theano
    
    # Zero out the false probabilities and sum the remaining true probabilities to remove the third dimension.
    indexes = theano.tensor.arange(coding_dist.shape[2])
    mask = theano.tensor.neq(indexes, true_dist.reshape((true_dist.shape[0], true_dist.shape[1], 1)))
    pred_probs = theano.tensor.set_subtensor(coding_dist[theano.tensor.nonzero(mask)], 0.).sum(axis=2)
    pred_probs_log = T.log(pred_probs)
    pred_probs_per_sample = -pred_probs_log.sum(axis=1)
    return pred_probs_per_sample

#in network_text()
X_enc_sym1 = T.dtensor3('x_enc_sym1')
X_enc_sym2 = T.dtensor3('x_enc_sym2')
X_enc_sym3 = T.dtensor3('x_enc_sym3') 
X_dec_sym = T.dtensor3('x_dec_sym') ##X_dec_sym = T.ftensor3('x_dec_sym')
y_sym = T.lmatrix('y_sym') # indexes of 1hot words, for loss 
Emb_mtx_sym = T.dmatrix('emb_mtx_sym')
eta = theano.shared(np.array(params['learning_rate'], dtype=theano.config.floatX))

#import theano.typed_list
#X_enc_sym_list = theano.typed_list.TypedListType(T.dtensor3)() # 2d or 3d?
#name = 'x_enc_sym'
#for i in range(1, num_metrics+1):
#    X_enc_sym_list = theano.typed_list.append(X_enc_sym_list, T.dtensor3('x_enc_sym'+str(i)))

#theano.typed_list.basic.length(X_enc_sym_list) #Length.0

#in model
#X = X_enc_sym[:,:,0:1]
#l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym1) #(None, None, 4) #l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
#l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=0, axis=2) #(None, None)
#l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #(None, None, 1)
    
l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                grad_clipping=grad_clip, only_return_final=True)
l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None),
                                                grad_clipping=grad_clip, only_return_final=True, backwards=True)
l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
l_enc_conc = lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1)) #(None, 256, 1)
#    
resetgate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_resetgate, W_hid=l_forward.W_hid_to_resetgate, W_cell = None, b=l_forward.b_resetgate)
updategate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_updategate, W_hid=l_forward.W_hid_to_updategate, W_cell = None, b=l_forward.b_updategate)
hidden_update_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_hidden_update, W_hid=l_forward.W_hid_to_hidden_update, W_cell=None, b=l_forward.b_hidden_update)
#
resetgate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_resetgate, W_hid=l_backward.W_hid_to_resetgate, W_cell= None, b=l_backward.b_resetgate)
updategate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_updategate, W_hid=l_backward.W_hid_to_updategate, W_cell=None, b=l_backward.b_updategate)
hidden_update_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_hidden_update, W_hid=l_backward.W_hid_to_hidden_update, W_cell=None, b=l_backward.b_hidden_update)

#X_enc_sym2
l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym2)
l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                                    resetgate=resetgate_f, updategate=updategate_f, hidden_update=hidden_update_f, grad_clipping=grad_clip, only_return_final=True)
l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size,
                                                    resetgate=resetgate_b, updategate=updategate_b, hidden_update=hidden_update_b, grad_clipping=grad_clip, only_return_final=True, backwards=True)
l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
l_enc_conc = lasagne.layers.ConcatLayer([l_enc_conc, lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1))], axis=2) #(None, 256, 4)
#X_enc_sym3
l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym3)
l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                                    resetgate=resetgate_f, updategate=updategate_f, hidden_update=hidden_update_f, grad_clipping=grad_clip, only_return_final=True)
l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size,
                                                    resetgate=resetgate_b, updategate=updategate_b, hidden_update=hidden_update_b, grad_clipping=grad_clip, only_return_final=True, backwards=True)
l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
l_enc_conc = lasagne.layers.ConcatLayer([l_enc_conc, lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1))], axis=2) #(None, 256, 4)

#add what above to model
#l_q = theano.shared(value=np.zeros((hidden_size,1), dtype='float32'))
#l_q = lasagne.init.Constant(0.)
#l_input = lasagne.layers.InputLayer(shape=(None, 256, 4), name=l_enc_conc)
#l_setenc = GRULayer_setenc(incoming=l_enc_conc, num_units=hidden_size, learn_init=False, set_steps=5, att_num_units=att_size, grad_clipping=grad_clip, 
 #                          nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)  #(None, 128)
l_setenc = lasagne.layers.ConcatLayer([lasagne.layers.SliceLayer(l_enc_conc, indices=0, axis=2), \
            lasagne.layers.SliceLayer(lasagne.layers.SliceLayer(l_enc_conc, indices=0, axis=2), indices=slice(0,hidden_size))],axis=1) #(None, 384)
#lasagne.layers.get_output(l_setenc)

#DECODER

dec_units = 3*1 +1 #dec_units = 4*2 +1 #nr_encoder *2 + dec -> bi-directional so *2 !!important to set it right
    #decoder
l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, word_dim),input_var=X_dec_sym)#pred_len=27 #l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len,1),input_var=X_dec_sym)
    #l_mask_dec = lasagne.layers.InputLayer(shape=(None, pred_len), input_var=mask_dec)
    #    
s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #SHOULD WE RESHAPE IT???
    #s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    #s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    #h_init = lasagne.layers.ConcatLayer([T.alloc(0., (l_forward.output_shape[0], hidden_size)), l_setenc], axis=1) #
h_init = lasagne.layers.ConcatLayer([l_forward, l_setenc], axis=1) #????????????? #(None, 512)
    #
l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, hid_init=h_init,
                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)),
                                    learn_init=False, grad_clipping=grad_clip, only_return_final=True ) #l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, mask_input=s_lmask_dec1, learn_init=False, 
                                                                             #hid_init=h_init, grad_clipping=grad_clip, only_return_final=True )
r_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_resetgate, W_hid=l_dec.W_hid_to_resetgate, b=l_dec.b_resetgate)
u_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_updategate, W_hid=l_dec.W_hid_to_updategate, b=l_dec.b_updategate)
h_update = lasagne.layers.Gate(W_in=l_dec.W_in_to_hidden_update, W_hid=l_dec.W_hid_to_hidden_update, b=l_dec.b_hidden_update)                                
#
l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 128)
#TO CHANGE BACK BELOW
l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, nonlinearity=lasagne.nonlinearities.softmax)  #l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=1, nonlinearity=lasagne.nonlinearities.linear)
#
w_dense = l_out.W
b_dense = l_out.b
l_out_loop = l_out
l_out_loop_val = l_out  
l_out = lasagne.layers.ReshapeLayer(l_out, ([0], 1, [1]))
l_out_val = l_out
h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1) #--- ##(None, 512)
h_init_val = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1) #--- ##(None, 512)

i=0
s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1) #(None, 200) ##(None, 202)
s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #(None, 1, 200) ##(None, 1, 202)
l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, hid_init=h_init, learn_init=False,	 #(None, 320)
                                            grad_clipping=grad_clip, only_return_final=True, 
                                            resetgate=r_gate, updategate=u_gate, hidden_update=h_update) ##(None, 512)
l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 64) ##(None, 128)
        #
        #------ ARGMAX for validation(->forecast) parts in network via embedding matrix 18.05 ---------------------------
    	#for val and train sets
coord_max = ExpressionLayer(l_out_loop_val, lambda x: x.argmax(-1).astype('int32'), output_shape='auto') #(None,)
emb_slice = ExpressionLayer(coord_max, lambda x: Emb_mtx_sym[x,:], output_shape=(None,word_dim)) #(None, 200)
	#
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: X.argmax(-1).astype('int32'), output_shape='auto') #(None,)
    	#pred_d = ExpressionLayer(coord_max, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X], 1), output_shape='auto')
    	#pred_emb = lasagne.layers.DenseLayer(pred_d, num_units=word_dim, W=Emb_mtx_sym, nonlinearity=lasagne.nonlinearities.linear) #no biases #(None, 200)
    	#pred_emb.params[pred_emb.W].remove("trainable")
        #
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X.argmax(-1).astype('int32')], 1), output_shape='auto')
    	#hot = lasagne.layers.DenseLayer(coord_max, num_units=word_dim, nonlinearity=lasagne.nonlinearities.linear)
        #
pred = lasagne.layers.ReshapeLayer(emb_slice, ([0], 1, [1])) #######pred = lasagne.layers.ReshapeLayer(pred_emb, ([0], 1, [1])) #(None, 1, 200)
        #
l_dec_val = lasagne.layers.GRULayer(pred, num_units=hidden_size*dec_units, learn_init=False, hid_init=h_init,
                                             grad_clipping=grad_clip, only_return_final=True,#hid_init=h_init_val, grad_clipping=grad_clip, only_return_final=True,
                                             resetgate=r_gate, updategate=u_gate, hidden_update=h_update) #(None, 320)
l_dec_val_hid_state = lasagne.layers.SliceLayer(l_dec_val, indices=slice(0,hidden_size)) #(None, 64)
l_out_loop = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
l_out_loop = lasagne.layers.ReshapeLayer(l_out_loop, ([0], 1, [1])) ####### #(None, 1, 155563)
l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop], axis=1) #(None, 2, 155563)
l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
l_out_loop_val_d = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1])) #(None, 1, 155563)
l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val_d], axis=1) #(None, 2, 155563)
    	#l_out_loop_val = lasagne.layers.FlattenLayer(l_out_loop_val, 2)
h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1)  #h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1) #(None, 320)
h_init_val = lasagne.layers.ConcatLayer([l_dec_val_hid_state, l_setenc], axis=1) #(None, 320)


def model_seq2seq_GRU_setenc(X_enc_sym1, X_enc_sym2, X_enc_sym3, X_dec_sym, Emb_mtx_sym, pred_len, num_metrics, set_steps, hidden_size=64, grad_clip = 100, att_size=64, vocab_size=22, word_dim=202): ##def model_seq2seq_GRU_text(X_enc_sym, X_enc_sym2, X_dec_sym, max_len, pred_len, hidden_size=64, grad_clip = 100, vocab_size=155563, word_dim=200):#(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, max_len, pred_len, hidden_size=64, grad_clip = 100) # max_len=288 - windowise, pred_len=27 - horizon
    #copy of model_seq2seq_mvrt4_GRU_text
    # model for order matters article 31.05.17
    #
    #collect last hiddens for each metric 
    #X = X_enc_sym[:,:,0:1]
    #l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym1) #(None, None, 4) #l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
    #l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=0, axis=2) #(None, None)
    #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #(None, None, 1)
    
    l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    grad_clipping=grad_clip, only_return_final=True)
    l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None),
                                                    grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
    l_enc_conc = lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1)) #(None, 256, 1)
    #    
    resetgate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_resetgate, W_hid=l_forward.W_hid_to_resetgate, W_cell = None, b=l_forward.b_resetgate)
    updategate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_updategate, W_hid=l_forward.W_hid_to_updategate, W_cell = None, b=l_forward.b_updategate)
    hidden_update_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_hidden_update, W_hid=l_forward.W_hid_to_hidden_update, W_cell=None, b=l_forward.b_hidden_update)
    #
    resetgate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_resetgate, W_hid=l_backward.W_hid_to_resetgate, W_cell= None, b=l_backward.b_resetgate)
    updategate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_updategate, W_hid=l_backward.W_hid_to_updategate, W_cell=None, b=l_backward.b_updategate)
    hidden_update_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_hidden_update, W_hid=l_backward.W_hid_to_hidden_update, W_cell=None, b=l_backward.b_hidden_update)
    #                                  
    #X_enc_sym2
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym2)
    l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                                        resetgate=resetgate_f, updategate=updategate_f, hidden_update=hidden_update_f, grad_clipping=grad_clip, only_return_final=True)
    l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size,
                                                        resetgate=resetgate_b, updategate=updategate_b, hidden_update=hidden_update_b, grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
    l_enc_conc = lasagne.layers.ConcatLayer([l_enc_conc, lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1))], axis=2) #(None, 256, 4)
    #X_enc_sym3
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym3)
    l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                                        resetgate=resetgate_f, updategate=updategate_f, hidden_update=hidden_update_f, grad_clipping=grad_clip, only_return_final=True)
    l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size,
                                                        resetgate=resetgate_b, updategate=updategate_b, hidden_update=hidden_update_b, grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
    l_enc_conc = lasagne.layers.ConcatLayer([l_enc_conc, lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1))], axis=2) #(None, 256, 4)
    #l_setenc = lasagne.layers.ElemwiseSumLayer([l_enc_conc], axis=2) #(None, 256)
    # 
    # Set-encoder part
    #l_q = lasagne.init.Constant(0.)
    #for i in range(set_steps):
    #l_setenc = GRULayer_setenc(incoming=l_enc_conc, num_units=hidden_size, learn_init=False, set_steps=set_steps, att_num_units=att_size, grad_clipping=grad_clip, 
     #                           nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True) #(None, 384)
    
    #l_q = lasagne.layers.SliceLayer(l_setenc, -1, 1)               ###
    l_setenc = lasagne.layers.ConcatLayer([lasagne.layers.SliceLayer(l_enc_conc, indices=0, axis=2), \
                lasagne.layers.SliceLayer(lasagne.layers.SliceLayer(l_enc_conc, indices=0, axis=2), indices=slice(0,hidden_size))],axis=1) #(None, 384)
    #
    dec_units = 3*1 +1 #dec_units = 4*2 +1 #nr_encoder *2 + dec -> bi-directional so *2 !!important to set it right
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, word_dim),input_var=X_dec_sym)#pred_len=27 #l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len,1),input_var=X_dec_sym)
    #l_mask_dec = lasagne.layers.InputLayer(shape=(None, pred_len), input_var=mask_dec)
    #    
    s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
    s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #SHOULD WE RESHAPE IT???
    #s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    #s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    #h_init = lasagne.layers.ConcatLayer([T.alloc(0., (l_forward.output_shape[0], hidden_size)), l_setenc], axis=1) #
    h_init = lasagne.layers.ConcatLayer([l_forward, l_setenc], axis=1) #????????????? ##(None, 512)
    #
    l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, hid_init=h_init,
                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)),
                                    learn_init=False, grad_clipping=grad_clip, only_return_final=True ) #l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, mask_input=s_lmask_dec1, learn_init=False, 
                                         #hid_init=h_init, grad_clipping=grad_clip, only_return_final=True )
    r_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_resetgate, W_hid=l_dec.W_hid_to_resetgate, b=l_dec.b_resetgate)
    u_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_updategate, W_hid=l_dec.W_hid_to_updategate, b=l_dec.b_updategate)
    h_update = lasagne.layers.Gate(W_in=l_dec.W_in_to_hidden_update, W_hid=l_dec.W_hid_to_hidden_update, b=l_dec.b_hidden_update)                                
    #
    l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 128)
    #TO CHANGE BACK BELOW
    l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, nonlinearity=lasagne.nonlinearities.softmax)  #l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=1, nonlinearity=lasagne.nonlinearities.linear)
    #
    w_dense = l_out.W
    b_dense = l_out.b
    l_out_loop = l_out
    l_out_loop_val = l_out  
    l_out = lasagne.layers.ReshapeLayer(l_out, ([0], 1, [1]))
    l_out_val = l_out
    h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1) #--- ##(None, 512)
    h_init_val = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1) #--- ##(None, 512)
#
    for i in range(1,pred_len): #comments in this cycle are for the first iteration
        s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1) #(None, 200) ##(None, 202)
        s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #(None, 1, 200) ##(None, 1, 202)
        l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, hid_init=h_init, learn_init=False,	 #(None, 320)
                                            grad_clipping=grad_clip, only_return_final=True, 
                                            resetgate=r_gate, updategate=u_gate, hidden_update=h_update) ##(None, 512)
        l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 64) ##(None, 128)
        #
        #------ ARGMAX for validation(->forecast) parts in network via embedding matrix 18.05 ---------------------------
    	#for val and train sets
    	coord_max = ExpressionLayer(l_out_loop_val, lambda x: x.argmax(-1).astype('int32'), output_shape='auto') #(None,)
    	emb_slice = ExpressionLayer(coord_max, lambda x: Emb_mtx_sym[x,:], output_shape=(None,word_dim)) #(None, 200)
	#
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: X.argmax(-1).astype('int32'), output_shape='auto') #(None,)
    	#pred_d = ExpressionLayer(coord_max, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X], 1), output_shape='auto')
    	#pred_emb = lasagne.layers.DenseLayer(pred_d, num_units=word_dim, W=Emb_mtx_sym, nonlinearity=lasagne.nonlinearities.linear) #no biases #(None, 200)
    	#pred_emb.params[pred_emb.W].remove("trainable")
        #
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X.argmax(-1).astype('int32')], 1), output_shape='auto')
    	#hot = lasagne.layers.DenseLayer(coord_max, num_units=word_dim, nonlinearity=lasagne.nonlinearities.linear)
        #
        pred = lasagne.layers.ReshapeLayer(emb_slice, ([0], 1, [1])) #######pred = lasagne.layers.ReshapeLayer(pred_emb, ([0], 1, [1])) #(None, 1, 200)
        #
        l_dec_val = lasagne.layers.GRULayer(pred, num_units=hidden_size*dec_units, learn_init=False, hid_init=h_init,
                                             grad_clipping=grad_clip, only_return_final=True,#hid_init=h_init_val, grad_clipping=grad_clip, only_return_final=True,
                                             resetgate=r_gate, updategate=u_gate, hidden_update=h_update) #(None, 320)
    	l_dec_val_hid_state = lasagne.layers.SliceLayer(l_dec_val, indices=slice(0,hidden_size)) #(None, 64)
    	l_out_loop = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
    	l_out_loop = lasagne.layers.ReshapeLayer(l_out_loop, ([0], 1, [1])) ####### #(None, 1, 155563)
    	l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop], axis=1) #(None, 2, 155563)
    	l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
    	l_out_loop_val_d = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1])) #(None, 1, 155563)
    	l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val_d], axis=1) #(None, 2, 155563)
    	#l_out_loop_val = lasagne.layers.FlattenLayer(l_out_loop_val, 2)
    	h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1)  #h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1) #(None, 320)
    	h_init_val = lasagne.layers.ConcatLayer([l_dec_val_hid_state, l_setenc], axis=1) #(None, 320)
    #l_out = lasagne.layers.ReshapeLayer(l_out, (-1, hidden_size))
    #l_out_val = lasagne.layers.ReshapeLayer(l_out_val, (-1, hidden_size))
    return (l_out, l_out_val)

l_out, l_out_val = model_seq2seq_GRU_setenc(X_enc_sym1, X_enc_sym2, X_enc_sym3, X_dec_sym, Emb_mtx_sym, params['horizon']-1, params['num_metrics'], params['set_steps'], hidden_size = params['num_units'], 
                    grad_clip = params['grad_clipping'], att_size = params['num_att_units'], vocab_size = params['vocab_size'], word_dim = params['word_dim'])
                                            
network_output, network_output_val = lasagne.layers.get_output([l_out, l_out_val])
#network_output = lasagne.layers.get_output(l_out)

weights = lasagne.layers.get_all_params(l_out,trainable=True)
if params['regularization_type'] == 'l1':
    reg_loss = lasagne.regularization.regularize_network_params(l_out, l1) * params['lambda_regularization']
else:
    reg_loss = lasagne.regularization.regularize_network_params(l_out, l2) * params['lambda_regularization']

loss_T = categorical_crossentropy_3d(network_output, y_sym).mean() + reg_loss
loss_val_T = categorical_crossentropy_3d(network_output_val, y_sym).mean() 
loss_test = categorical_crossentropy_3d(network_output_val, y_sym).mean() 
#metric_probs = get_metric_probs(network_output, y_sym) #####             

updates = lasagne.updates.adam(loss_T, weights, learning_rate=eta)

f_train = theano.function([X_enc_sym1, X_enc_sym2, X_enc_sym3, X_dec_sym, y_sym], loss_T, updates=updates, allow_input_downcast=True)
f_val = theano.function([X_enc_sym1, X_enc_sym2, X_enc_sym3, X_dec_sym, Emb_mtx_sym, y_sym], loss_val_T, allow_input_downcast=True)#, on_unused_input='ignore')
forecast = theano.function([X_enc_sym1, X_enc_sym2, X_enc_sym3, X_dec_sym, Emb_mtx_sym, y_sym], loss_test, allow_input_downcast=True)


>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
08.05.17 - 20:02 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
As with Yagmur, but lists
                            
import lasagne
import numpy as np
import theano
import theano.tensor as T
from theano import shared
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan
from lasagne.layers.base import Layer, MergeLayer
from lasagne.layers.input import InputLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers import helper, SliceLayer
from lasagne.layers.recurrent import Gate
import logging
import sys
import csv
from lasagne.layers import get_output
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
import os
import linecache
from lasagne.layers import InputLayer, ExpressionLayer, EmbeddingLayer
#            
def load_parameters2_mvrt_text(params_csvfile, line_num): 
    """
    reads the params for the new generated params
    """
    params={}
    values = linecache.getline(params_csvfile,line_num)[:-1].split(',')
    params['num_units'] = int(values[0]) #128
    params['num_att_units'] = int(values[1]) #512
    data_type = str(values[2]) 
    params['data_name'] = str(values[3])
    model = str(values[4]) 
    if params['data_name'] == 'coservit':
        params['windowise'] = 288
        params['horizon'] = 27
        params['word_dim'] = 202 # 200
        params['vocab_size'] = 22 ###### 155564
        
        #params['time_data'] = '../../../Data' + params['data_name'] + 
        #params['text_data'] = '../../../Data' + params['data_name'] + 
        #params['data_file'] =  '../../../Data/' + params['data_name'] + 'ticket2time.pkl'
        params['data_file'] =  '../../../Data/coservit/' + 'x_enc.pkl'
        params['text_file_wv'] =  '../../../Data/coservit/' + 'x_dec_wv.dat' ##params['text_file_wv'] =  '../../../Data/coservit/' + 'tickets_wv_pad_mtx.dat'
        params['text_file_w'] =  '../../../Data/coservit/' + 'x_dec_w.dat' #params['text_file_w'] =  '../../../Data/coservit/' + 'tickets_w_pad_mtx.dat'
        params['metr_dec_file'] =  '../../../Data/coservit/' + 'metr_for_dec.pkl' #list of lists with metric ids corresponding to aligned tickets. Ex.: t1,t1,t2,t3,t3,t3->[[3432, 4657], [3442], [6567, 4657, 7855]]
        params['emb_file'] =  '../../../Data/coservit/' + 'lda_emb_mtx.dat'#'emb_mtx.dat'
    else:
        pass
        """params['data_type'] = 'missing'
        #params['data_file'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/'+ params['data_name'] + '_'+ data_type +'.pkl'
        params['data_file'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/'+ params['data_name'] + '_'+ data_type +'_intr_linear.pkl'
        params['data_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/' + params['data_name'] + '_mask_' + data_type + '.pkl'
        params['missing_percent'] = data_type
        params['original_data_file'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'.pkl'
        params['original_data_mask_file'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'_mask.pkl' 
    #TODO:
    if params['data_name']  == 'pse':
        params['data2_name'] = 'weather_data'
        params['data2_file'] = '../../../Data/' + params['data2_name'] + '/missing_gaps_v2/'+ params['data2_name'] + '_'+ data_type +'.pkl'
        params['data2_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/' + params['data_name'] + '_mask_' + data_type + '.pkl'
        params['data2_variables'] = ['mintmp', 'maxtmp', 'precip'] 
    elif params['data_name']  == 'consseason':
        params['data2_name'] = 'consseason'
        params['data2_file'] = params['data_file']
        params['data2_file_missing_mask'] = params['data_file_missing_mask']
        params['data2_variables'] = ['Global_reactive_power', 'Voltage', 'Global_intensity'] 
    elif params['data_name']  == 'airq':
        params['data2_name'] = 'airq'
        params['data2_file'] = params['data_file']
        params['data2_file_missing_mask'] = params['data_file_missing_mask']
        params['data2_variables'] = ['NO2(GT)', 'CO(GT)', 'NOx(GT)'] """
            
    if data_type == 'orig':
        params['data_type'] = 'original'
    else:
        #TODO
        pass
        
    if model == 'RNN':
        params['attention'] = False
    else:
        params['attention'] = True
        if model == 'RNN-ATT':
            params['att_type'] = 'original'
        elif model == 'ATT-lambda':
            params['att_type'] = 'lambda'
        elif model == 'ATT-lambda-mu':
            params['att_type'] = 'lambda_mu' 
        elif model == 'ATT-lambda-mu-alt':
            params['att_type'] = 'lambda_mu_alt'
        elif model == 'ATT-dist':
            params['att_type'] = 'adist'
        elif model == 'RNN-text-set':
            params['att_type'] = 'set_enc'
            params['num_metrics'] = 4
            params['set_steps'] = 5 #number of iterations in set_enc part
    params['alg'] = 'adam'
    params['lambda_regularization'] = 0.0001
    params['eta_decay'] = 0.95
    params['early_stop_patience'] = 10000
    params['num_layers'] = 1
    params['regularization_type'] = 'l2'
    params['early_stop'] = True
    params['grad_clipping'] = 100
    params['train_test_ratio'] = 0.75
    params['train_val_ratio'] = 0.75
    params['patience_increase'] = 2 
    params['epsilon_improvement'] = True 
    params['lr_decay_after_n_epoch'] = 50
    params['interpolation'] = True #!!!!!!
    params['max_num_epochs'] = 25
    params['epsilon'] = 0.01
    params['learning_rate'] = 0.001
    params['batch_size'] = 64
    params['padding'] = True 
    params['learning_rate_decay'] = False
    params['interpolation_type'] = 'None'
    return params

params = load_parameters2_mvrt_text('params/fuzzy.param.text', 2)
params
#
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
results_directory = 'Results_multivariate_text_OM/'
#
att_type = params['att_type']
att_size = params['num_att_units']
word_dim = params['word_dim']
vocab_size = params['vocab_size']
grad_clip = params['grad_clipping']
hidden_size = params['num_units']
pred_len = 27
num_metrics = params['num_metrics']
set_steps = params['set_steps']
grad_clip = 100
#
#Gate
class Gate_setenc(object):
    def __init__(self, W_in=init.GlorotNormal(), W_hid=init.GlorotNormal(),
                 W_cell=init.Normal(0.1), b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        if W_in is not None:
            self.W_in = W_in
        self.W_hid = W_hid
        # Don't store a cell weight vector when cell is None
        if W_cell is not None:
            self.W_cell = W_cell
        self.b = b
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

#
#Layer
class GRULayer_setenc(lasagne.layers.Layer): #it is not mvrt lasagne.layers.Layer
    def __init__(self, incoming, num_units, 
                 resetgate=Gate_setenc(W_in=None,W_cell=None), 
                 updategate=Gate_setenc(W_in=None,W_cell=None),
                 hidden_update=Gate_setenc(W_in=None, W_cell=None),
                 nonlinearity=nonlinearities.tanh,
                 hid_init=init.Constant(0.),
                 set_steps=5,
                 att_num_units = 64,
                 W_hid_to_att = init.GlorotNormal(),
                 W_ctx_to_att = init.GlorotNormal(),
                 W_att = init.Normal(0.1), #---
                 backwards=False,
                 learn_init=True, #--???????
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=True, #-- ???????
                 **kwargs):
        #         
        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.  We will just provide the
        # layer input as incomings, unless a mask input was provided.
        #incomings = [incoming]
        #if mask_input is not None:
            #incomings.append(mask_input) #-----
        #
        # Initialize parent layer
        super(GRULayer_setenc, self).__init__(incoming, **kwargs) #super(GRULayer_setenc, self).__init__(incomings, **kwargs)
        #
        self.learn_init = learn_init
        self.num_units = num_units #128
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.set_steps = set_steps
        self.att_num_units = att_num_units #512
        self.only_return_final = only_return_final
        #
        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")
        #
        # Retrieve the dimensionality of the incoming layer
        #
        if unroll_scan and self.input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")
        #
        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(self.input_shape[2:]) #4
        print(num_inputs)
        print(self.input_shape) #(None, 256, 4)
        #print(num_inputs) #4
        #
        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            #self.add_param(gate.W_in, (num_inputs, num_units),
                                   #name="W_in_to_{}".format(gate_name))
            return (self.add_param(gate.W_hid, (num_units, num_units), #128
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,1),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)
        #
        # Add in all parameters from gates, nonlinearities will be sigmas, look Gate_setenc
        """(self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')
        #
        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')"""
        (self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')
        #
        (self.W_hid_to_hidden_update, self.b_hidden_update, 
         self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')
        #
        #attention Weights 
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        #
        # Initialize hidden state
        #self.hid_init = hid_init ######
        
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (input.input_shape[0], self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False) #???????
        """if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError(
                    "When hid_init is provided as a TensorVariable, it should "
                    "have 2 dimensions and have shape (num_batch, num_units)")
            self.hid_init = hid_init
        else: # here
            self.hid_init = self.add_param(  #--????????#--????????#--????????#--????????#--????????#--????????#--????????
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False) #not trainable"""
        #print(self.hid_init, type(self.hid_init)) #(hid_init, <class 'theano.tensor.sharedvar.TensorSharedVariable'>
#
    def get_output_shape_for(self, input_shape):                        #(None, 256, 128)
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shp = self.input_shape[0]
        # PRINTS
        if self.only_return_final:
            return self.input_shape[0], 3*self.num_units #(None, 128)
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return self.input_shape[0], self.input_shape[1], 3*self.num_units
#
    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        #print('uiop') # PRINTS
        # Retrieve the layer input
        input = inputs
        #print(input) #<lasagne.layers.merge.ConcatLayer object at 0x7fe2e1f7fb10>
        #print(input.ndim)
        # Retrieve the mask when it is supplied
        #mask = inputs[1] if len(inputs) > 1 else None # -> mask=None
#
        #print(type(inputs)) #<class 'lasagne.layers.merge.ConcatLayer'>
        # Treat all dimensions after the second as flattened feature dimensions
        #if input.ndim > 3:
            #input = T.flatten(input, 3)
#
        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        #input = lasagne.layers.ReshapeLayer(input, ([1], [0], [2]))#
        input = input.dimshuffle(1, 0, 2)#
        seq_len, num_batch, _ = input.shape #256,None -- no 4 #(256, None) #seq_len, num_batch, _ = input.output_shape
        print(seq_len, num_batch) #(256, None)
        #print(input.output_shape) #(256, None, 4)
#
        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        """W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)"""
#
        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)
#
        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)
#
        #if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            #input = T.dot(input, W_in_stacked) + b_stacked
#
        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
   #     
        def plain_et_step(o_t0):#def plain_et_step(self, x_snp, o_t0)
            #reading from memory steps
            bs, seq_len_m, _ = input.shape
            m_in = input.dimshuffle(1, 0, 2)#----replace
            e_qt = T.dot(o_t0, self.W_hid_to_att)#---
            e_m = T.dot(m_in, self.W_ctx_to_att)#----
            e_q = T.tile(e_qt, (seq_len_m, 1, 1)) #e_q = T.tile(e_qt, (self.seq_len_m, 1, 1))
            et_p = T.tanh(e_m + e_q)
            et = T.dot(et_p, self.W_att)
            alpha = T.exp(et)
            alpha /= T.sum(alpha, axis=0)
            mt = input.dimshuffle(2, 1, 0)
            mult = T.mul(mt, alpha)
            rt = T.sum(mult, axis=1)
            #print(input.shape, rt.shape)
            return rt.T
            #
           """bs, seq_len_m, num_metr = input.shape#####bs, seq_len_m, _ = input.shape
            m_in = input#####.dimshuffle(1, 0, 2)#----replace
            e_qt = T.dot(o_t0, self.W_hid_to_att)#---
            e_m = T.dot(m_in, self.W_ctx_to_att)#---- ??????????????
            e_q = T.tile(e_qt, (bs, 1, num_metr)) #####e_q = T.tile(e_qt, (seq_len_m, 1, 1)) #e_q = T.tile(e_qt, (self.seq_len_m, 1, 1))
            et_p = T.tanh(e_m + e_q)
            et = T.dot(et_p, self.W_att)
            alpha = T.exp(et)
            alpha /= T.sum(alpha, axis=0)
            mt = input#####.dimshuffle(2, 1, 0)
            mult = T.mul(mt, alpha)
            rt = T.sum(mult, axis=2)#####1)
            #print(input.shape, rt.shape)
            return rt#####.T"""
#
        def step(hid_previous, *args): #W_hid_stacked, #W_in_stacked, b_stacked):
            #x_snp = incom
            #print(x_snp.output_shape)
            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked) + b_stacked ####for r, z, h tilde WHAT ABOUT THE SIZES WHEN MULT??????
#       
            print("tyui")
            if self.grad_clipping is not False:
                #input_n = theano.gradient.grad_clip(
                    #input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)
                print('d')
#
           # if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                #input_n = T.dot(input_n, W_in_stacked) + b_stacked
#
            # Reset and update gates
            resetgate = slice_w(hid_input, 0) #+ slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) #+ slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)
#
            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            #hidden_update_in = slice_w(input_n, 2)
            ####hidden_update_hid = slice_w(hid_input, 2) #h tilde
            ####hidden_update = resetgate*hidden_update_hid #hidden_update = hidden_update_in + resetgate*hidden_update_hid
            hidden_update = slice_w(hid_input, 2)
            #
            if self.grad_clipping is not False:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update) #here it is sigma, but in encoder.py this is tanh ????????????
#
            print('d')
            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid0 = (np.float32(1.0) - updategate)*hid_previous + updategate*hidden_update
            hid0 = T.tile(hid0, (bs, 1, 1))
            rt = plain_et_step(hid0)
            h_t = T.concatenate([hid0,rt], axis=1)
    #        
            return h_t #------??????
        """def step(x_snp, snp_mask, m_snp_count):
            r_t = T.nnet.sigmoid(T.dot(hr_tm1, self.Wo_hh_r) + self.bo_r)
            z_t = T.nnet.sigmoid(T.dot(hr_tm1, self.W_hh_z) + self.b_z)
            h_tilde = T.tanh(T.dot(r_t * hr_tm1, self.W_hh) + self.b_hh)
            o_t = (np.float32(1.0) - z_t) * hr_tm1 + z_t * h_tilde
            #o_t: GRU hidden state
            #concatanation of hidden state and reading from memory
            rt = self.plain_et_step(x_snp, o_t)
            h_t = T.concatenate([o_t0,rt], axis=1)---------
            return h_t"""
     #       
#
        """def step_masked(input_n, mask_n, hid_previous, W_hid_stacked,
                        W_in_stacked, b_stacked):
#
            hid = step(input_n, hid_previous, W_hid_stacked, W_in_stacked,
                       b_stacked)
#
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            not_mask = 1 - mask_n
            hid = hid*mask_n + hid_previous*not_mask
#
            return hid"""
#
        """if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
            print('d')
        else:
            #sequences = input #[input]
            step_fun = step
            print("step")"""
        step_fun = step
#
        """if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)"""
#
        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked, b_stacked] #non_seqs = [W_hid_stacked] #non_seqs = [input, W_hid_stacked]
        #non_seqs += [W_ctx_stacked] #
        non_seqs += [self.W_hid_to_att, self.W_ctx_to_att, self.W_att, input] #
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        #if not self.precompute_input:
            #non_seqs += [b_stacked]#[W_in_stacked, b_stacked]
        # theano.scan only allows for positional arguments, so when
        # self.precompute_input is True, we need to supply fake placeholder
        # arguments for the input weights and biases.
        #else:
            #non_seqs += [(), ()]
#
        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else: #here
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            #hid_out, _ = theano.scan(fn=step_fun, outputs_info=o_enc_info, non_sequences=incomings[0], n_steps=self.set_steps)#self.num_set_iter) #----??????
            print('d')
            hid_out, _ = theano.scan(fn=step_fun, outputs_info=self.hid_init, non_sequences=non_seqs, n_steps=self.set_steps, strict=True)#self.num_set_iter)
        """hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]"""
#
        # dimshuffle back to (n_batch, n_time_steps, n_features))
        #######hid_out = hid_out.dimshuffle(1, 0, 2) #done below
#
        # if scan is backward reverse the output
        #if self.backwards:-------- ?????????
            #hid_out = hid_out[:, ::-1, :]-------??????????? #done below
         #
        #copied from layers.py
        """if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)
            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]     """
        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        #return theano.shared(np.random.randn(3,4))
#
        print(hid_out.shape)
        return hid_out

def categorical_crossentropy_3d(coding_dist, true_dist, lengths=None):
    #http://stackoverflow.com/questions/30225633/cross-entropy-for-batch-with-theano
    
    # Zero out the false probabilities and sum the remaining true probabilities to remove the third dimension.
    indexes = theano.tensor.arange(coding_dist.shape[2])
    mask = theano.tensor.neq(indexes, true_dist.reshape((true_dist.shape[0], true_dist.shape[1], 1)))
    pred_probs = theano.tensor.set_subtensor(coding_dist[theano.tensor.nonzero(mask)], 0.).sum(axis=2)
    pred_probs_log = T.log(pred_probs)
    pred_probs_per_sample = -pred_probs_log.sum(axis=1)
    return pred_probs_per_sample

#in network_text()
import theano.typed_list
X_enc_sym_list = theano.typed_list.TypedListType(T.dtensor3)()
name = 'x_enc_sym'
for i in range(1, num_metrics+1):
    X_enc_sym_list = theano.typed_list.append(X_enc_sym_list, T.dtensor3('x_enc_sym'+str(i)))

#theano.typed_list.basic.length(X_enc_sym_list) #Length.0
X_dec_sym = T.dtensor3('x_dec_sym') ##X_dec_sym = T.ftensor3('x_dec_sym')
y_sym = T.lmatrix('y_sym') # indexes of 1hot words, for loss 
Emb_mtx_sym = T.dmatrix('emb_mtx_sym')
eta = theano.shared(np.array(params['learning_rate'], dtype=theano.config.floatX))

#theano.typed_list.basic.length(X_enc_sym_list) #Length.0

#in model
#X = X_enc_sym[:,:,0:1]
#l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=theano.typed_list.basic.getitem(X_enc_sym_list,0)) #(None, None, 4) #l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
#l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=0, axis=2) #(None, None)
#l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #(None, None, 1)
    
l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                grad_clipping=grad_clip, only_return_final=True)
l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None),
                                                grad_clipping=grad_clip, only_return_final=True, backwards=True)
l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
l_enc_conc = lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1)) #(None, 256, 1)
#    
resetgate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_resetgate, W_hid=l_forward.W_hid_to_resetgate, W_cell = None, b=l_forward.b_resetgate)
updategate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_updategate, W_hid=l_forward.W_hid_to_updategate, W_cell = None, b=l_forward.b_updategate)
hidden_update_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_hidden_update, W_hid=l_forward.W_hid_to_hidden_update, W_cell=None, b=l_forward.b_hidden_update)
#
resetgate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_resetgate, W_hid=l_backward.W_hid_to_resetgate, W_cell= None, b=l_backward.b_resetgate)
updategate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_updategate, W_hid=l_backward.W_hid_to_updategate, W_cell=None, b=l_backward.b_updategate)
hidden_update_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_hidden_update, W_hid=l_backward.W_hid_to_hidden_update, W_cell=None, b=l_backward.b_hidden_update)

for i in range(1, num_metrics):
        #X = X_enc_sym[:,:,i:(i+1)]
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=theano.typed_list.basic.getitem(X_enc_sym_list,i))
        #l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=i, axis=2) #(None, None)
        #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], 1, [1]))
#        
    l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                                        resetgate=resetgate_f, 
                                                        updategate=updategate_f, 
                                                        hidden_update=hidden_update_f, 
                                                        grad_clipping=grad_clip, only_return_final=True)
    l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size,
                                                        resetgate=resetgate_b, 
                                                        updategate=updategate_b, 
                                                        hidden_update=hidden_update_b,
                                                        grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
    l_enc_conc = lasagne.layers.ConcatLayer([l_enc_conc, lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1))], axis=2) #(None, 256, 4)

#add what above to model
#l_q = theano.shared(value=np.zeros((hidden_size,1), dtype='float32'))
#l_q = lasagne.init.Constant(0.)
#l_input = lasagne.layers.InputLayer(shape=(None, 256, 4), name=l_enc_conc)
l_setenc = GRULayer_setenc(incoming=l_enc_conc, num_units=hidden_size, learn_init=False, set_steps=5, att_num_units=att_size, grad_clipping=grad_clip, 
                           nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)  #(None, 128)
#l_setenc = lasagne.layers.ConcatLayer([lasagne.layers.SliceLayer(l_enc_conc, indices=0, axis=2), \
 #           lasagne.layers.SliceLayer(lasagne.layers.SliceLayer(l_enc_conc, indices=0, axis=2), indices=slice(0,hidden_size))],axis=1) #(None, 384)
#lasagne.layers.get_output(l_setenc)

#DECODER

dec_units = 3*1 +1 #dec_units = 4*2 +1 #nr_encoder *2 + dec -> bi-directional so *2 !!important to set it right
    #decoder
l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, word_dim),input_var=X_dec_sym)#pred_len=27 #l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len,1),input_var=X_dec_sym)
    #l_mask_dec = lasagne.layers.InputLayer(shape=(None, pred_len), input_var=mask_dec)
    #    
s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #SHOULD WE RESHAPE IT???
    #s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    #s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    #h_init = lasagne.layers.ConcatLayer([T.alloc(0., (l_forward.output_shape[0], hidden_size)), l_setenc], axis=1) #
h_init = lasagne.layers.ConcatLayer([l_forward, l_setenc], axis=1) #????????????? #(None, 512)
    #
l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, hid_init=h_init,
                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)),
                                    learn_init=False, grad_clipping=grad_clip, only_return_final=True ) #l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, mask_input=s_lmask_dec1, learn_init=False, 
                                                                             #hid_init=h_init, grad_clipping=grad_clip, only_return_final=True )
r_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_resetgate, W_hid=l_dec.W_hid_to_resetgate, b=l_dec.b_resetgate)
u_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_updategate, W_hid=l_dec.W_hid_to_updategate, b=l_dec.b_updategate)
h_update = lasagne.layers.Gate(W_in=l_dec.W_in_to_hidden_update, W_hid=l_dec.W_hid_to_hidden_update, b=l_dec.b_hidden_update)                                
#
l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 128)
#TO CHANGE BACK BELOW
l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, nonlinearity=lasagne.nonlinearities.softmax)  #l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=1, nonlinearity=lasagne.nonlinearities.linear)
#
w_dense = l_out.W
b_dense = l_out.b
l_out_loop = l_out
l_out_loop_val = l_out  
l_out = lasagne.layers.ReshapeLayer(l_out, ([0], 1, [1]))
l_out_val = l_out
h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1) #--- ##(None, 512)
h_init_val = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1) #--- ##(None, 512)

i=1
s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1) #(None, 200) ##(None, 202)
s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #(None, 1, 200) ##(None, 1, 202)
l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, hid_init=h_init, learn_init=False,	 #(None, 320)
                                            grad_clipping=grad_clip, only_return_final=True, 
                                            resetgate=r_gate, updategate=u_gate, hidden_update=h_update) ##(None, 512)
l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 64) ##(None, 128)
        #
        #------ ARGMAX for validation(->forecast) parts in network via embedding matrix 18.05 ---------------------------
    	#for val and train sets
coord_max = ExpressionLayer(l_out_loop_val, lambda x: x.argmax(-1).astype('int32'), output_shape='auto') #(None,)
emb_slice = ExpressionLayer(coord_max, lambda x: Emb_mtx_sym[x,:], output_shape=(None,word_dim)) #(None, 200)
	#
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: X.argmax(-1).astype('int32'), output_shape='auto') #(None,)
    	#pred_d = ExpressionLayer(coord_max, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X], 1), output_shape='auto')
    	#pred_emb = lasagne.layers.DenseLayer(pred_d, num_units=word_dim, W=Emb_mtx_sym, nonlinearity=lasagne.nonlinearities.linear) #no biases #(None, 200)
    	#pred_emb.params[pred_emb.W].remove("trainable")
        #
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X.argmax(-1).astype('int32')], 1), output_shape='auto')
    	#hot = lasagne.layers.DenseLayer(coord_max, num_units=word_dim, nonlinearity=lasagne.nonlinearities.linear)
        #
pred = lasagne.layers.ReshapeLayer(emb_slice, ([0], 1, [1])) #######pred = lasagne.layers.ReshapeLayer(pred_emb, ([0], 1, [1])) #(None, 1, 200)
        #
l_dec_val = lasagne.layers.GRULayer(pred, num_units=hidden_size*dec_units, learn_init=False, hid_init=h_init,
                                             grad_clipping=grad_clip, only_return_final=True,#hid_init=h_init_val, grad_clipping=grad_clip, only_return_final=True,
                                             resetgate=r_gate, updategate=u_gate, hidden_update=h_update) #(None, 320)
l_dec_val_hid_state = lasagne.layers.SliceLayer(l_dec_val, indices=slice(0,hidden_size)) #(None, 64)
l_out_loop = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
l_out_loop = lasagne.layers.ReshapeLayer(l_out_loop, ([0], 1, [1])) ####### #(None, 1, 155563)
l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop], axis=1) #(None, 2, 155563)
l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
l_out_loop_val_d = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1])) #(None, 1, 155563)
l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val_d], axis=1) #(None, 2, 155563)
    	#l_out_loop_val = lasagne.layers.FlattenLayer(l_out_loop_val, 2)
h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1)  #h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1) #(None, 320)
h_init_val = lasagne.layers.ConcatLayer([l_dec_val_hid_state, l_setenc], axis=1) #(None, 320)


def model_seq2seq_GRU_setenc(X_enc_sym_list, X_dec_sym, Emb_mtx_sym, pred_len, num_metrics, set_steps, hidden_size=64, grad_clip = 100, att_size=64, vocab_size=22, word_dim=202): ##def model_seq2seq_GRU_text(X_enc_sym, X_enc_sym2, X_dec_sym, max_len, pred_len, hidden_size=64, grad_clip = 100, vocab_size=155563, word_dim=200):#(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, max_len, pred_len, hidden_size=64, grad_clip = 100) # max_len=288 - windowise, pred_len=27 - horizon
    #copy of model_seq2seq_mvrt4_GRU_text
    # model for order matters article 31.05.17
    #
    #collect last hiddens for each metric 
    #X = X_enc_sym[:,:,0:1]
    #l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=theano.typed_list.basic.getitem(X_enc_sym_list,0)) #(None, None, 4) #l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
    #l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=0, axis=2) #(None, None)
    #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #(None, None, 1)
    #
    l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    grad_clipping=grad_clip, only_return_final=True)
    l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None),
                                                    grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
    l_enc_conc = lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1)) #(None, 256, 1)
    #    
    resetgate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_resetgate, W_hid=l_forward.W_hid_to_resetgate, W_cell = None, b=l_forward.b_resetgate)
    updategate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_updategate, W_hid=l_forward.W_hid_to_updategate, W_cell = None, b=l_forward.b_updategate)
    hidden_update_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_hidden_update, W_hid=l_forward.W_hid_to_hidden_update, W_cell=None, b=l_forward.b_hidden_update)
    #
    resetgate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_resetgate, W_hid=l_backward.W_hid_to_resetgate, W_cell= None, b=l_backward.b_resetgate)
    updategate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_updategate, W_hid=l_backward.W_hid_to_updategate, W_cell=None, b=l_backward.b_updategate)
    hidden_update_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_hidden_update, W_hid=l_backward.W_hid_to_hidden_update, W_cell=None, b=l_backward.b_hidden_update)
    #                                  
    for i in range(1, num_metrics):
        #X = X_enc_sym[:,:,i:(i+1)]
        l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=theano.typed_list.basic.getitem(X_enc_sym_list,i))
        #l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=i, axis=2) #(None, None)
        #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], 1, [1]))
        #
        l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                                        resetgate=resetgate_f, 
                                                        updategate=updategate_f, 
                                                        hidden_update=hidden_update_f, 
                                                        grad_clipping=grad_clip, only_return_final=True)
        l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size,
                                                        resetgate=resetgate_b, 
                                                        updategate=updategate_b, 
                                                        hidden_update=hidden_update_b,
                                                        grad_clipping=grad_clip, only_return_final=True, backwards=True)
        l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
        l_enc_conc = lasagne.layers.ConcatLayer([l_enc_conc, lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1))], axis=2) #(None, 256, 4)
    #l_setenc = lasagne.layers.ElemwiseSumLayer([l_enc_conc], axis=2) #(None, 256)
    # 
    # Set-encoder part
    #l_q = lasagne.init.Constant(0.)
    #for i in range(set_steps):
    #l_setenc = GRULayer_setenc(incoming=l_enc_conc, num_units=hidden_size, learn_init=False, set_steps=set_steps, att_num_units=att_size, grad_clipping=grad_clip, 
     #                           nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True) #(None, 384)
    
    #l_q = lasagne.layers.SliceLayer(l_setenc, -1, 1)               ###
    l_setenc = lasagne.layers.ConcatLayer([lasagne.layers.SliceLayer(l_enc_conc, indices=0, axis=2), \
                lasagne.layers.SliceLayer(lasagne.layers.SliceLayer(l_enc_conc, indices=0, axis=2), indices=slice(0,hidden_size))],axis=1) #(None, 384)
    #
    dec_units = 3*1 +1 #dec_units = 4*2 +1 #nr_encoder *2 + dec -> bi-directional so *2 !!important to set it right
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, word_dim),input_var=X_dec_sym)#pred_len=27 #l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len,1),input_var=X_dec_sym)
    #l_mask_dec = lasagne.layers.InputLayer(shape=(None, pred_len), input_var=mask_dec)
    #    
    s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
    s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #SHOULD WE RESHAPE IT???
    #s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    #s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    #h_init = lasagne.layers.ConcatLayer([T.alloc(0., (l_forward.output_shape[0], hidden_size)), l_setenc], axis=1) #
    h_init = lasagne.layers.ConcatLayer([l_forward, l_setenc], axis=1) #????????????? ##(None, 512)
    #
    l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, hid_init=h_init,
                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)),
                                    learn_init=False, grad_clipping=grad_clip, only_return_final=True ) #l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, mask_input=s_lmask_dec1, learn_init=False, 
                                         #hid_init=h_init, grad_clipping=grad_clip, only_return_final=True )
    r_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_resetgate, W_hid=l_dec.W_hid_to_resetgate, b=l_dec.b_resetgate)
    u_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_updategate, W_hid=l_dec.W_hid_to_updategate, b=l_dec.b_updategate)
    h_update = lasagne.layers.Gate(W_in=l_dec.W_in_to_hidden_update, W_hid=l_dec.W_hid_to_hidden_update, b=l_dec.b_hidden_update)                                
    #
    l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 128)
    #TO CHANGE BACK BELOW
    l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, nonlinearity=lasagne.nonlinearities.softmax)  #l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=1, nonlinearity=lasagne.nonlinearities.linear)
    #
    w_dense = l_out.W
    b_dense = l_out.b
    l_out_loop = l_out
    l_out_loop_val = l_out  
    l_out = lasagne.layers.ReshapeLayer(l_out, ([0], 1, [1]))
    l_out_val = l_out
    h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1) #--- ##(None, 512)
    h_init_val = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1) #--- ##(None, 512)
#
    for i in range(1,pred_len): #comments in this cycle are for the first iteration
        s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1) #(None, 200) ##(None, 202)
        s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #(None, 1, 200) ##(None, 1, 202)
        l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, hid_init=h_init, learn_init=False,	 #(None, 320)
                                            grad_clipping=grad_clip, only_return_final=True, 
                                            resetgate=r_gate, updategate=u_gate, hidden_update=h_update) ##(None, 512)
        l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 64) ##(None, 128)
        #
        #------ ARGMAX for validation(->forecast) parts in network via embedding matrix 18.05 ---------------------------
    	#for val and train sets
    	coord_max = ExpressionLayer(l_out_loop_val, lambda x: x.argmax(-1).astype('int32'), output_shape='auto') #(None,)
    	emb_slice = ExpressionLayer(coord_max, lambda x: Emb_mtx_sym[x,:], output_shape=(None,word_dim)) #(None, 200)
	#
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: X.argmax(-1).astype('int32'), output_shape='auto') #(None,)
    	#pred_d = ExpressionLayer(coord_max, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X], 1), output_shape='auto')
    	#pred_emb = lasagne.layers.DenseLayer(pred_d, num_units=word_dim, W=Emb_mtx_sym, nonlinearity=lasagne.nonlinearities.linear) #no biases #(None, 200)
    	#pred_emb.params[pred_emb.W].remove("trainable")
        #
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X.argmax(-1).astype('int32')], 1), output_shape='auto')
    	#hot = lasagne.layers.DenseLayer(coord_max, num_units=word_dim, nonlinearity=lasagne.nonlinearities.linear)
        #
        pred = lasagne.layers.ReshapeLayer(emb_slice, ([0], 1, [1])) #######pred = lasagne.layers.ReshapeLayer(pred_emb, ([0], 1, [1])) #(None, 1, 200)
        #
        l_dec_val = lasagne.layers.GRULayer(pred, num_units=hidden_size*dec_units, learn_init=False, hid_init=h_init,
                                             grad_clipping=grad_clip, only_return_final=True,#hid_init=h_init_val, grad_clipping=grad_clip, only_return_final=True,
                                             resetgate=r_gate, updategate=u_gate, hidden_update=h_update) #(None, 320)
    	l_dec_val_hid_state = lasagne.layers.SliceLayer(l_dec_val, indices=slice(0,hidden_size)) #(None, 64)
    	l_out_loop = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
    	l_out_loop = lasagne.layers.ReshapeLayer(l_out_loop, ([0], 1, [1])) ####### #(None, 1, 155563)
    	l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop], axis=1) #(None, 2, 155563)
    	l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
    	l_out_loop_val_d = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1])) #(None, 1, 155563)
    	l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val_d], axis=1) #(None, 2, 155563)
    	#l_out_loop_val = lasagne.layers.FlattenLayer(l_out_loop_val, 2)
    	h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1)  #h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1) #(None, 320)
    	h_init_val = lasagne.layers.ConcatLayer([l_dec_val_hid_state, l_setenc], axis=1) #(None, 320)
    #l_out = lasagne.layers.ReshapeLayer(l_out, (-1, hidden_size))
    #l_out_val = lasagne.layers.ReshapeLayer(l_out_val, (-1, hidden_size))
    return (l_out, l_out_val)

l_out, l_out_val = model_seq2seq_GRU_setenc(X_enc_sym_list, X_dec_sym, Emb_mtx_sym, params['horizon']-1, params['num_metrics'], params['set_steps'], hidden_size = params['num_units'], 
                    grad_clip = params['grad_clipping'], att_size = params['num_att_units'], vocab_size = params['vocab_size'], word_dim = params['word_dim'])
                                            
network_output, network_output_val = lasagne.layers.get_output([l_out, l_out_val])
#network_output = lasagne.layers.get_output(l_out)

weights = lasagne.layers.get_all_params(l_out,trainable=True)
if params['regularization_type'] == 'l1':
    reg_loss = lasagne.regularization.regularize_network_params(l_out, l1) * params['lambda_regularization']
else:
    reg_loss = lasagne.regularization.regularize_network_params(l_out, l2) * params['lambda_regularization']

loss_T = categorical_crossentropy_3d(network_output, y_sym).mean() + reg_loss
loss_val_T = categorical_crossentropy_3d(network_output_val, y_sym).mean() 
loss_test = categorical_crossentropy_3d(network_output_val, y_sym).mean() 
#metric_probs = get_metric_probs(network_output, y_sym) #####             

updates = lasagne.updates.adam(loss_T, weights, learning_rate=eta)

f_train = theano.function([X_enc_sym_list, X_dec_sym, y_sym], loss_T, updates=updates, allow_input_downcast=True)
f_val = theano.function([X_enc_sym_list, X_dec_sym, Emb_mtx_sym, y_sym], loss_val_T, allow_input_downcast=True)#, on_unused_input='ignore')
forecast = theano.function([X_enc_sym_list, X_dec_sym, Emb_mtx_sym, y_sym], loss_test, allow_input_downcast=True)


00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

10.05.17 ........................................................................................................................................................................................................................................................................
Commented everything in step() so it just outputs zero matrix and does nothing with the inputs or computations in plan_et_step,
so I think concat error may be while trying to concatenating l_setenc with .. . OR we dont understand how scan works so there is a mistake inside scan
                            
import lasagne
import numpy as np
import theano
import theano.tensor as T
from theano import shared
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan
from lasagne.layers.base import Layer, MergeLayer
from lasagne.layers.input import InputLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers import helper, SliceLayer
from lasagne.layers.recurrent import Gate
import logging
import sys
import csv
from lasagne.layers import get_output
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
import os
import linecache
from lasagne.layers import InputLayer, ExpressionLayer, EmbeddingLayer
from theano.ifelse import ifelse
#            
def load_parameters2_mvrt_text(params_csvfile, line_num): 
    """
    reads the params for the new generated params
    """
    params={}
    values = linecache.getline(params_csvfile,line_num)[:-1].split(',')
    params['num_units'] = int(values[0]) #128
    params['num_att_units'] = int(values[1]) #512
    data_type = str(values[2]) 
    params['data_name'] = str(values[3])
    model = str(values[4]) 
    if params['data_name'] == 'coservit':
        params['windowise'] = 288
        params['horizon'] = 27
        params['word_dim'] = 202 # 200
        params['vocab_size'] = 22 ###### 155564
        
        #params['time_data'] = '../../../Data' + params['data_name'] + 
        #params['text_data'] = '../../../Data' + params['data_name'] + 
        #params['data_file'] =  '../../../Data/' + params['data_name'] + 'ticket2time.pkl'
        params['data_file'] =  '../../../Data/coservit/' + 'x_enc.pkl'
        params['text_file_wv'] =  '../../../Data/coservit/' + 'x_dec_wv.dat' ##params['text_file_wv'] =  '../../../Data/coservit/' + 'tickets_wv_pad_mtx.dat'
        params['text_file_w'] =  '../../../Data/coservit/' + 'x_dec_w.dat' #params['text_file_w'] =  '../../../Data/coservit/' + 'tickets_w_pad_mtx.dat'
        params['metr_dec_file'] =  '../../../Data/coservit/' + 'metr_for_dec.pkl' #list of lists with metric ids corresponding to aligned tickets. Ex.: t1,t1,t2,t3,t3,t3->[[3432, 4657], [3442], [6567, 4657, 7855]]
        params['emb_file'] =  '../../../Data/coservit/' + 'lda_emb_mtx.dat'#'emb_mtx.dat'
    else:
        pass
        """params['data_type'] = 'missing'
        #params['data_file'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/'+ params['data_name'] + '_'+ data_type +'.pkl'
        params['data_file'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/'+ params['data_name'] + '_'+ data_type +'_intr_linear.pkl'
        params['data_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/' + params['data_name'] + '_mask_' + data_type + '.pkl'
        params['missing_percent'] = data_type
        params['original_data_file'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'.pkl'
        params['original_data_mask_file'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'_mask.pkl' 
    #TODO:
    if params['data_name']  == 'pse':
        params['data2_name'] = 'weather_data'
        params['data2_file'] = '../../../Data/' + params['data2_name'] + '/missing_gaps_v2/'+ params['data2_name'] + '_'+ data_type +'.pkl'
        params['data2_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/' + params['data_name'] + '_mask_' + data_type + '.pkl'
        params['data2_variables'] = ['mintmp', 'maxtmp', 'precip'] 
    elif params['data_name']  == 'consseason':
        params['data2_name'] = 'consseason'
        params['data2_file'] = params['data_file']
        params['data2_file_missing_mask'] = params['data_file_missing_mask']
        params['data2_variables'] = ['Global_reactive_power', 'Voltage', 'Global_intensity'] 
    elif params['data_name']  == 'airq':
        params['data2_name'] = 'airq'
        params['data2_file'] = params['data_file']
        params['data2_file_missing_mask'] = params['data_file_missing_mask']
        params['data2_variables'] = ['NO2(GT)', 'CO(GT)', 'NOx(GT)'] """
            
    if data_type == 'orig':
        params['data_type'] = 'original'
    else:
        #TODO
        pass
        
    if model == 'RNN':
        params['attention'] = False
    else:
        params['attention'] = True
        if model == 'RNN-ATT':
            params['att_type'] = 'original'
        elif model == 'ATT-lambda':
            params['att_type'] = 'lambda'
        elif model == 'ATT-lambda-mu':
            params['att_type'] = 'lambda_mu' 
        elif model == 'ATT-lambda-mu-alt':
            params['att_type'] = 'lambda_mu_alt'
        elif model == 'ATT-dist':
            params['att_type'] = 'adist'
        elif model == 'RNN-text-set':
            params['att_type'] = 'set_enc'
            params['num_metrics'] = 4
            params['set_steps'] = 5 #number of iterations in set_enc part
    params['alg'] = 'adam'
    params['lambda_regularization'] = 0.0001
    params['eta_decay'] = 0.95
    params['early_stop_patience'] = 10000
    params['num_layers'] = 1
    params['regularization_type'] = 'l2'
    params['early_stop'] = True
    params['grad_clipping'] = 100
    params['train_test_ratio'] = 0.75
    params['train_val_ratio'] = 0.75
    params['patience_increase'] = 2 
    params['epsilon_improvement'] = True 
    params['lr_decay_after_n_epoch'] = 50
    params['interpolation'] = True #!!!!!!
    params['max_num_epochs'] = 25
    params['epsilon'] = 0.01
    params['learning_rate'] = 0.001
    params['batch_size'] = 64
    params['padding'] = True 
    params['learning_rate_decay'] = False
    params['interpolation_type'] = 'None'
    return params

params = load_parameters2_mvrt_text('params/fuzzy.param.text', 2)
params
#
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
results_directory = 'Results_multivariate_text_OM/'
#
att_type = params['att_type']
att_size = params['num_att_units']
word_dim = params['word_dim']
vocab_size = params['vocab_size']
grad_clip = params['grad_clipping']
hidden_size = params['num_units']
pred_len = 27
num_metrics = params['num_metrics']
set_steps = params['set_steps']
grad_clip = 100
#
#Gate
class Gate_setenc(object):
    def __init__(self, W_in=init.GlorotNormal(), W_hid=init.GlorotNormal(),
                 W_cell=init.Normal(0.1), b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        if W_in is not None:
            self.W_in = W_in
        self.W_hid = W_hid
        # Don't store a cell weight vector when cell is None
        if W_cell is not None:
            self.W_cell = W_cell
        self.b = b
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

#
#Layer
class GRULayer_setenc(lasagne.layers.Layer): #it is not mvrt lasagne.layers.Layer
    def __init__(self, incoming, num_units, 
                 resetgate=Gate_setenc(W_in=None,W_cell=None), 
                 updategate=Gate_setenc(W_in=None,W_cell=None),
                 hidden_update=Gate_setenc(W_in=None, W_cell=None),
                 nonlinearity=nonlinearities.tanh,
                 hid_init=init.Constant(0.),
                 set_steps=5,
                 att_num_units = 64,
                 W_hid_to_att = init.GlorotNormal(),
                 W_ctx_to_att = init.GlorotNormal(),
                 W_att = init.Normal(0.1), #---
                 backwards=False,
                 learn_init=True, #--???????
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=True, #-- ???????
                 **kwargs):
        #         
        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.  We will just provide the
        # layer input as incomings, unless a mask input was provided.
        #incomings = [incoming]
        #if mask_input is not None:
            #incomings.append(mask_input) #-----
        #
        # Initialize parent layer
        super(GRULayer_setenc, self).__init__(incoming, **kwargs) #super(GRULayer_setenc, self).__init__(incomings, **kwargs)
        #
        self.learn_init = learn_init
        self.num_units = num_units #128
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.set_steps = set_steps
        self.att_num_units = att_num_units #512
        self.only_return_final = only_return_final
        #
        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")
        #
        # Retrieve the dimensionality of the incoming layer
        #
        if unroll_scan and self.input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")
        #
        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(self.input_shape[2:]) #4
        print(num_inputs)
        print(self.input_shape) #(None, 256, 4)
        #print(num_inputs) #4
        #
        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            #self.add_param(gate.W_in, (num_inputs, num_units),
                                   #name="W_in_to_{}".format(gate_name))
            return (self.add_param(gate.W_hid, (3*num_units, num_units), #128
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,1),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)
        #
        # Add in all parameters from gates, nonlinearities will be sigmas, look Gate_setenc
        """(self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')
        #
        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')"""
        """(self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')
        #
        (self.W_hid_to_hidden_update, self.b_hidden_update, 
         self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')
        #
        #attention Weights 
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh"""
        #
        # Initialize hidden state
        #self.hid_init = hid_init ######
        
        """if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init", #!!!!
                trainable=learn_init, regularizable=False) #???????"""
        #self.hid_init = T.dmatrix() #>>>
        self.hid_init = T.zeros((1, self.num_units*3))
        #hid_init = T.specify_shape(hid_init, (1, self.num_units))
        #hid_init = T.tile(hid_init, (input.shape[0], 1, 1))
        #T.tile(hid0, (input.shape[0], 1, 1))
        """if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError(
                    "When hid_init is provided as a TensorVariable, it should "
                    "have 2 dimensions and have shape (num_batch, num_units)")
            self.hid_init = hid_init
        else: # here
            self.hid_init = self.add_param(  #--????????#--????????#--????????#--????????#--????????#--????????#--????????
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False) #not trainable"""
        #print(self.hid_init, type(self.hid_init)) #(hid_init, <class 'theano.tensor.sharedvar.TensorSharedVariable'>
#
    def get_output_shape_for(self, input_shape):                        #(None, 256, 128)
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shp = self.input_shape[0]
        # PRINTS
        if self.only_return_final:
            return self.input_shape[0], 3*self.num_units #(None, 128)
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return self.input_shape[0], self.input_shape[1], 3*self.num_units
#
    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        #print('uiop') # PRINTS
        # Retrieve the layer input
        input = inputs
        #print(input) #<lasagne.layers.merge.ConcatLayer object at 0x7fe2e1f7fb10>
        #print(input.ndim)
        # Retrieve the mask when it is supplied
        #mask = inputs[1] if len(inputs) > 1 else None # -> mask=None
#
        #print(type(inputs)) #<class 'lasagne.layers.merge.ConcatLayer'>
        # Treat all dimensions after the second as flattened feature dimensions
        #if input.ndim > 3:
            #input = T.flatten(input, 3)
#
        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        #input = lasagne.layers.ReshapeLayer(input, ([1], [0], [2]))#
        #####input = input.dimshuffle(1, 0, 2)#
        num_batch, seq_len, _ = input.shape #256,None -- no 4 #(256, None) #seq_len, num_batch, _ = input.output_shape
        print(num_batch, seq_len) #(256, None)
        #print(input.output_shape) #(256, None, 4)
#
        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        """W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)"""
#
        # Same for hidden weight matrices
        """W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)
#
        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)"""
#
        #if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            #input = T.dot(input, W_in_stacked) + b_stacked
#
        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
   #     
        def plain_et_step(o_t0):#def plain_et_step(self, x_snp, o_t0)
            #reading from memory steps
            print("i'm in plain")
            bs, num_hid2, num_metr = input.shape
            m_in = input.dimshuffle(0, 2, 1)#----replace
            e_qt = T.dot(o_t0, self.W_hid_to_att)#---
            e_m = T.dot(m_in, self.W_ctx_to_att)#----
            e_q = T.tile(e_qt, (num_metr, 1, 1)) #e_q = T.tile(e_qt, (self.seq_len_m, 1, 1))
            et_p = T.tanh(e_m + e_q)
            et = T.dot(et_p, self.W_att)
            alpha = T.exp(et)
            alpha /= T.sum(alpha, axis=0)
            mt = input.dimshuffle(1, 0, 2)
            mult = T.mul(mt, alpha)
            rt = T.sum(mult, axis=2)
            #print(input.shape, rt.shape)
            print("plain finished")
            return rt.T
            #
            """print("i'm in plain")
            bs, seq_len_m, num_metr = input.shape#####bs, seq_len_m, _ = input.shape
            m_in = input#####.dimshuffle(1, 0, 2)#----replace
            e_qt = T.dot(o_t0, self.W_hid_to_att)#---
            e_m = T.dot(m_in, self.W_ctx_to_att)#---- ??????????????
            e_q = T.tile(e_qt, (1, 1, num_metr)) #####e_q = T.tile(e_qt, (seq_len_m, 1, 1)) #e_q = T.tile(e_qt, (self.seq_len_m, 1, 1))
            et_p = T.tanh(e_m + e_q)
            et = T.dot(et_p, self.W_att)
            #et = et.dimshuffle([0,'x',1])
            alpha = T.exp(et)
            alpha /= T.sum(alpha, axis=1)#####2)0)
            mt = input.dimshuffle(0, 2, 1)
            mult = T.mul(mt, alpha)
            rt = T.sum(mult, axis=1)#####2)
            #print(input.shape, rt.shape)
            print("plain finished")
            return rt#####.T"""
#
        def step(hid_previous, *args): #W_hid_stacked, #W_in_stacked, b_stacked):
            #x_snp = incom
            #print(x_snp.output_shape)
            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            print("qwerty")
            h_t = T.zeros((input.shape[0], 3*self.num_units))
            """one = T.alloc(np.array(1,dtype='int64'))
            #zero_vec = T.zeros((1, 3*self.num_units))
            x = ifelse(T.eq(hid_previous.shape[0], one), input.shape[0], one)
            print("after ifelse")
            hid_previous = hid_previous.repeat(x, axis=0) #hid_previous = T.tile(hid_previous, (x, 1))
            #
            #hid_previous = ifelse(T.eq(hid_previous.shape[0],one), T.tile(hid_previous, (x, 1),ndim=2),hid_previous)
            #
            hid_input = T.dot(hid_previous, W_hid_stacked) + b_stacked.T #>> ####for r, z, h tilde WHAT ABOUT THE SIZES WHEN MULT??????
#       
            print("tyui")
            if self.grad_clipping is not False:
                #input_n = theano.gradient.grad_clip(
                    #input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)
                print('d')
#
           # if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                #input_n = T.dot(input_n, W_in_stacked) + b_stacked
#
            # Reset and update gates
            resetgate = slice_w(hid_input, 0) #+ slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) #+ slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)
#
            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            #hidden_update_in = slice_w(input_n, 2)
            ####hidden_update_hid = slice_w(hid_input, 2) #h tilde
            ####hidden_update = resetgate*hidden_update_hid #hidden_update = hidden_update_in + resetgate*hidden_update_hid
            hidden_update = slice_w(hid_input, 2)
            #
            if self.grad_clipping is not False:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update) #here it is sigma, but in encoder.py this is tanh ????????????
#
            print('d')
            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid0 = (np.float32(1.0) - updategate)*hid_previous + updategate*hidden_update
            #hid0 = T.tile(hid0, (input.shape[0], 1, 1))
            rt = plain_et_step(hid0)
            h_t = T.concatenate([hid0,rt], axis=1)"""
    #        
            print("before return h_t in step")
            #h_t = hid_previous
            return h_t #------??????
        """def step(x_snp, snp_mask, m_snp_count):
            r_t = T.nnet.sigmoid(T.dot(hr_tm1, self.Wo_hh_r) + self.bo_r)
            z_t = T.nnet.sigmoid(T.dot(hr_tm1, self.W_hh_z) + self.b_z)
            h_tilde = T.tanh(T.dot(r_t * hr_tm1, self.W_hh) + self.b_hh)
            o_t = (np.float32(1.0) - z_t) * hr_tm1 + z_t * h_tilde
            #o_t: GRU hidden state
            #concatanation of hidden state and reading from memory
            rt = self.plain_et_step(x_snp, o_t)
            h_t = T.concatenate([o_t0,rt], axis=1)---------
            return h_t"""
     #       
#
        """def step_masked(input_n, mask_n, hid_previous, W_hid_stacked,
                        W_in_stacked, b_stacked):
#
            hid = step(input_n, hid_previous, W_hid_stacked, W_in_stacked,
                       b_stacked)
#
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            not_mask = 1 - mask_n
            hid = hid*mask_n + hid_previous*not_mask
#
            return hid"""
#
        """if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
            print('d')
        else:
            #sequences = input #[input]
            step_fun = step
            print("step")"""
        step_fun = step
#
        """if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)"""
#
        # The hidden-to-hidden weight matrix is always used in step
        #non_seqs = [W_hid_stacked, b_stacked] #non_seqs = [W_hid_stacked] #non_seqs = [input, W_hid_stacked]
        #non_seqs += [W_ctx_stacked] #
        #non_seqs += [self.W_hid_to_att, self.W_ctx_to_att, self.W_att, input] #
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        #if not self.precompute_input:
            #non_seqs += [b_stacked]#[W_in_stacked, b_stacked]
        # theano.scan only allows for positional arguments, so when
        # self.precompute_input is True, we need to supply fake placeholder
        # arguments for the input weights and biases.
        #else:
            #non_seqs += [(), ()]
#
        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else: #here
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            #hid_out, _ = theano.scan(fn=step_fun, outputs_info=o_enc_info, non_sequences=incomings[0], n_steps=self.set_steps)#self.num_set_iter) #----??????
            print('d')
            hid_out, _ = theano.scan(fn=step_fun, outputs_info=self.hid_init, non_sequences=[input], n_steps=self.set_steps, strict=True)#self.num_set_iter)
        """hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]"""
#
        # dimshuffle back to (n_batch, n_time_steps, n_features))
        #######hid_out = hid_out.dimshuffle(1, 0, 2) #done below
#
        # if scan is backward reverse the output
        #if self.backwards:-------- ?????????
            #hid_out = hid_out[:, ::-1, :]-------??????????? #done below
         #
        #copied from layers.py
        """if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)
            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]     """
        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        #return theano.shared(np.random.randn(3,4))
#
        #print(hid_out.shape)
        return hid_out

def categorical_crossentropy_3d(coding_dist, true_dist, lengths=None):
    #http://stackoverflow.com/questions/30225633/cross-entropy-for-batch-with-theano
    
    # Zero out the false probabilities and sum the remaining true probabilities to remove the third dimension.
    indexes = theano.tensor.arange(coding_dist.shape[2])
    mask = theano.tensor.neq(indexes, true_dist.reshape((true_dist.shape[0], true_dist.shape[1], 1)))
    pred_probs = theano.tensor.set_subtensor(coding_dist[theano.tensor.nonzero(mask)], 0.).sum(axis=2)
    pred_probs_log = T.log(pred_probs)
    pred_probs_per_sample = -pred_probs_log.sum(axis=1)
    return pred_probs_per_sample

#in network_text()
import theano.typed_list
X_enc_sym_list = theano.typed_list.TypedListType(T.dtensor3)()
name = 'x_enc_sym'
for i in range(1, num_metrics+1):
    X_enc_sym_list = theano.typed_list.append(X_enc_sym_list, T.dtensor3('x_enc_sym'+str(i)))

#theano.typed_list.basic.length(X_enc_sym_list) #Length.0
X_dec_sym = T.dtensor3('x_dec_sym') ##X_dec_sym = T.ftensor3('x_dec_sym')
y_sym = T.lmatrix('y_sym') # indexes of 1hot words, for loss 
Emb_mtx_sym = T.dmatrix('emb_mtx_sym')
eta = theano.shared(np.array(params['learning_rate'], dtype=theano.config.floatX))

#theano.typed_list.basic.length(X_enc_sym_list) #Length.0

#in model
#X = X_enc_sym[:,:,0:1]
#l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=theano.typed_list.basic.getitem(X_enc_sym_list,0)) #(None, None, 4) #l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
#l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=0, axis=2) #(None, None)
#l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #(None, None, 1)

l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                grad_clipping=grad_clip, only_return_final=True)
l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None),
                                                grad_clipping=grad_clip, only_return_final=True, backwards=True)
l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
l_enc_conc = lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1)) #(None, 256, 1)
#    
resetgate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_resetgate, W_hid=l_forward.W_hid_to_resetgate, W_cell = None, b=l_forward.b_resetgate)
updategate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_updategate, W_hid=l_forward.W_hid_to_updategate, W_cell = None, b=l_forward.b_updategate)
hidden_update_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_hidden_update, W_hid=l_forward.W_hid_to_hidden_update, W_cell=None, b=l_forward.b_hidden_update)
#
resetgate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_resetgate, W_hid=l_backward.W_hid_to_resetgate, W_cell= None, b=l_backward.b_resetgate)
updategate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_updategate, W_hid=l_backward.W_hid_to_updategate, W_cell=None, b=l_backward.b_updategate)
hidden_update_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_hidden_update, W_hid=l_backward.W_hid_to_hidden_update, W_cell=None, b=l_backward.b_hidden_update)

for i in range(1, num_metrics):
        #X = X_enc_sym[:,:,i:(i+1)]
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=theano.typed_list.basic.getitem(X_enc_sym_list,i))
        #l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=i, axis=2) #(None, None)
        #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], 1, [1]))
#        
    l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                                        resetgate=resetgate_f, 
                                                        updategate=updategate_f, 
                                                        hidden_update=hidden_update_f, 
                                                        grad_clipping=grad_clip, only_return_final=True)
    l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size,
                                                        resetgate=resetgate_b, 
                                                        updategate=updategate_b, 
                                                        hidden_update=hidden_update_b,
                                                        grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
    l_enc_conc = lasagne.layers.ConcatLayer([l_enc_conc, lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1))], axis=2) #(None, 256, 4)

#add what above to model
#l_q = theano.shared(value=np.zeros((hidden_size,1), dtype='float32'))
#l_q = lasagne.init.Constant(0.)
#l_input = lasagne.layers.InputLayer(shape=(None, 256, 4), name=l_enc_conc)
l_setenc = GRULayer_setenc(incoming=l_enc_conc, num_units=hidden_size, learn_init=False, set_steps=5, att_num_units=att_size, grad_clipping=grad_clip, 
                           nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)  #(None, 128)
#l_setenc = lasagne.layers.ConcatLayer([lasagne.layers.SliceLayer(l_enc_conc, indices=0, axis=2), \
 #           lasagne.layers.SliceLayer(lasagne.layers.SliceLayer(l_enc_conc, indices=0, axis=2), indices=slice(0,hidden_size))],axis=1) #(None, 384)
#lasagne.layers.get_output(l_setenc)

#DECODER

dec_units = 3*1 +1 #dec_units = 4*2 +1 #nr_encoder *2 + dec -> bi-directional so *2 !!important to set it right
    #decoder
l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, word_dim),input_var=X_dec_sym)#pred_len=27 #l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len,1),input_var=X_dec_sym)
    #l_mask_dec = lasagne.layers.InputLayer(shape=(None, pred_len), input_var=mask_dec)
    #    
s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #SHOULD WE RESHAPE IT???
    #s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    #s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    #h_init = lasagne.layers.ConcatLayer([T.alloc(0., (l_forward.output_shape[0], hidden_size)), l_setenc], axis=1) #
h_init = lasagne.layers.ConcatLayer([l_forward, l_setenc], axis=1) #????????????? #(None, 512)
    #
l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, hid_init=h_init,
                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)),
                                    learn_init=False, grad_clipping=grad_clip, only_return_final=True ) #l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, mask_input=s_lmask_dec1, learn_init=False, 
                                                                             #hid_init=h_init, grad_clipping=grad_clip, only_return_final=True )
r_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_resetgate, W_hid=l_dec.W_hid_to_resetgate, b=l_dec.b_resetgate)
u_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_updategate, W_hid=l_dec.W_hid_to_updategate, b=l_dec.b_updategate)
h_update = lasagne.layers.Gate(W_in=l_dec.W_in_to_hidden_update, W_hid=l_dec.W_hid_to_hidden_update, b=l_dec.b_hidden_update)                                
#
l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 128)
#TO CHANGE BACK BELOW
l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, nonlinearity=lasagne.nonlinearities.softmax)  #l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=1, nonlinearity=lasagne.nonlinearities.linear)
#
w_dense = l_out.W
b_dense = l_out.b
l_out_loop = l_out
l_out_loop_val = l_out  
l_out = lasagne.layers.ReshapeLayer(l_out, ([0], 1, [1]))
l_out_val = l_out
h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1) #--- ##(None, 512)
h_init_val = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1) #--- ##(None, 512)

i=1
s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1) #(None, 200) ##(None, 202)
s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #(None, 1, 200) ##(None, 1, 202)
l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, hid_init=h_init, learn_init=False,	 #(None, 320)
                                            grad_clipping=grad_clip, only_return_final=True, 
                                            resetgate=r_gate, updategate=u_gate, hidden_update=h_update) ##(None, 512)
l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 64) ##(None, 128)
        #
        #------ ARGMAX for validation(->forecast) parts in network via embedding matrix 18.05 ---------------------------
    	#for val and train sets
coord_max = ExpressionLayer(l_out_loop_val, lambda x: x.argmax(-1).astype('int32'), output_shape='auto') #(None,)
emb_slice = ExpressionLayer(coord_max, lambda x: Emb_mtx_sym[x,:], output_shape=(None,word_dim)) #(None, 200)
	#
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: X.argmax(-1).astype('int32'), output_shape='auto') #(None,)
    	#pred_d = ExpressionLayer(coord_max, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X], 1), output_shape='auto')
    	#pred_emb = lasagne.layers.DenseLayer(pred_d, num_units=word_dim, W=Emb_mtx_sym, nonlinearity=lasagne.nonlinearities.linear) #no biases #(None, 200)
    	#pred_emb.params[pred_emb.W].remove("trainable")
        #
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X.argmax(-1).astype('int32')], 1), output_shape='auto')
    	#hot = lasagne.layers.DenseLayer(coord_max, num_units=word_dim, nonlinearity=lasagne.nonlinearities.linear)
        #
pred = lasagne.layers.ReshapeLayer(emb_slice, ([0], 1, [1])) #######pred = lasagne.layers.ReshapeLayer(pred_emb, ([0], 1, [1])) #(None, 1, 200)
        #
l_dec_val = lasagne.layers.GRULayer(pred, num_units=hidden_size*dec_units, learn_init=False, hid_init=h_init,
                                             grad_clipping=grad_clip, only_return_final=True,#hid_init=h_init_val, grad_clipping=grad_clip, only_return_final=True,
                                             resetgate=r_gate, updategate=u_gate, hidden_update=h_update) #(None, 320)
l_dec_val_hid_state = lasagne.layers.SliceLayer(l_dec_val, indices=slice(0,hidden_size)) #(None, 64)
l_out_loop = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
l_out_loop = lasagne.layers.ReshapeLayer(l_out_loop, ([0], 1, [1])) ####### #(None, 1, 155563)
l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop], axis=1) #(None, 2, 155563)
l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
l_out_loop_val_d = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1])) #(None, 1, 155563)
l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val_d], axis=1) #(None, 2, 155563)
    	#l_out_loop_val = lasagne.layers.FlattenLayer(l_out_loop_val, 2)
h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1)  #h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1) #(None, 320)
h_init_val = lasagne.layers.ConcatLayer([l_dec_val_hid_state, l_setenc], axis=1) #(None, 320)


def model_seq2seq_GRU_setenc(X_enc_sym_list, X_dec_sym, Emb_mtx_sym, pred_len, num_metrics, set_steps, hidden_size=64, grad_clip = 100, att_size=64, vocab_size=22, word_dim=202): ##def model_seq2seq_GRU_text(X_enc_sym, X_enc_sym2, X_dec_sym, max_len, pred_len, hidden_size=64, grad_clip = 100, vocab_size=155563, word_dim=200):#(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, max_len, pred_len, hidden_size=64, grad_clip = 100) # max_len=288 - windowise, pred_len=27 - horizon
    #copy of model_seq2seq_mvrt4_GRU_text
    # model for order matters article 31.05.17
    #
    #collect last hiddens for each metric 
    #X = X_enc_sym[:,:,0:1]
    #l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=theano.typed_list.basic.getitem(X_enc_sym_list,0)) #(None, None, 4) #l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
    #l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=0, axis=2) #(None, None)
    #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #(None, None, 1)
    #
    l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    grad_clipping=grad_clip, only_return_final=True)
    l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None),
                                                    grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
    l_enc_conc = lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1)) #(None, 256, 1)
    #    
    resetgate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_resetgate, W_hid=l_forward.W_hid_to_resetgate, W_cell = None, b=l_forward.b_resetgate)
    updategate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_updategate, W_hid=l_forward.W_hid_to_updategate, W_cell = None, b=l_forward.b_updategate)
    hidden_update_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_hidden_update, W_hid=l_forward.W_hid_to_hidden_update, W_cell=None, b=l_forward.b_hidden_update)
    #
    resetgate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_resetgate, W_hid=l_backward.W_hid_to_resetgate, W_cell= None, b=l_backward.b_resetgate)
    updategate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_updategate, W_hid=l_backward.W_hid_to_updategate, W_cell=None, b=l_backward.b_updategate)
    hidden_update_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_hidden_update, W_hid=l_backward.W_hid_to_hidden_update, W_cell=None, b=l_backward.b_hidden_update)
    #                                  
    for i in range(1, num_metrics):
        #X = X_enc_sym[:,:,i:(i+1)]
        l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=theano.typed_list.basic.getitem(X_enc_sym_list,i))
        #l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=i, axis=2) #(None, None)
        #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], 1, [1]))
        #
        l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                                        resetgate=resetgate_f, 
                                                        updategate=updategate_f, 
                                                        hidden_update=hidden_update_f, 
                                                        grad_clipping=grad_clip, only_return_final=True)
        l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size,
                                                        resetgate=resetgate_b, 
                                                        updategate=updategate_b, 
                                                        hidden_update=hidden_update_b,
                                                        grad_clipping=grad_clip, only_return_final=True, backwards=True)
        l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
        l_enc_conc = lasagne.layers.ConcatLayer([l_enc_conc, lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1))], axis=2) #(None, 256, 4)
    #l_setenc = lasagne.layers.ElemwiseSumLayer([l_enc_conc], axis=2) #(None, 256)
    # 
    # Set-encoder part
    #l_q = lasagne.init.Constant(0.)
    #for i in range(set_steps):
    l_setenc = GRULayer_setenc(incoming=l_enc_conc, num_units=hidden_size, learn_init=False, set_steps=set_steps, att_num_units=att_size, grad_clipping=grad_clip, 
                                nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True) #(None, 384)
    
    #l_q = lasagne.layers.SliceLayer(l_setenc, -1, 1)               ###
    #l_setenc = lasagne.layers.ConcatLayer([lasagne.layers.SliceLayer(l_enc_conc, indices=0, axis=2), \
     #           lasagne.layers.SliceLayer(lasagne.layers.SliceLayer(l_enc_conc, indices=0, axis=2), indices=slice(0,hidden_size))],axis=1) #(None, 384)
    #
    dec_units = 3*1 +1 #dec_units = 4*2 +1 #nr_encoder *2 + dec -> bi-directional so *2 !!important to set it right
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, word_dim),input_var=X_dec_sym)#pred_len=27 #l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len,1),input_var=X_dec_sym)
    #l_mask_dec = lasagne.layers.InputLayer(shape=(None, pred_len), input_var=mask_dec)
    #    
    s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
    s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #SHOULD WE RESHAPE IT???
    #s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    #s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    #h_init = lasagne.layers.ConcatLayer([T.alloc(0., (l_forward.output_shape[0], hidden_size)), l_setenc], axis=1) #
    h_init = lasagne.layers.ConcatLayer([l_forward, l_setenc], axis=1) #????????????? ##(None, 512)
    #
    l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, hid_init=h_init,
                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)),
                                    learn_init=False, grad_clipping=grad_clip, only_return_final=True ) #l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, mask_input=s_lmask_dec1, learn_init=False, 
                                         #hid_init=h_init, grad_clipping=grad_clip, only_return_final=True )
    r_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_resetgate, W_hid=l_dec.W_hid_to_resetgate, b=l_dec.b_resetgate)
    u_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_updategate, W_hid=l_dec.W_hid_to_updategate, b=l_dec.b_updategate)
    h_update = lasagne.layers.Gate(W_in=l_dec.W_in_to_hidden_update, W_hid=l_dec.W_hid_to_hidden_update, b=l_dec.b_hidden_update)                                
    #
    l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 128)
    #TO CHANGE BACK BELOW
    l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, nonlinearity=lasagne.nonlinearities.softmax)  #l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=1, nonlinearity=lasagne.nonlinearities.linear)
    #
    w_dense = l_out.W
    b_dense = l_out.b
    l_out_loop = l_out
    l_out_loop_val = l_out  
    l_out = lasagne.layers.ReshapeLayer(l_out, ([0], 1, [1]))
    l_out_val = l_out
    h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1) #--- ##(None, 512)
    h_init_val = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1) #--- ##(None, 512)
#
    for i in range(1,pred_len): #comments in this cycle are for the first iteration
        s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1) #(None, 200) ##(None, 202)
        s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #(None, 1, 200) ##(None, 1, 202)
        l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, hid_init=h_init, learn_init=False,	 #(None, 320)
                                            grad_clipping=grad_clip, only_return_final=True, 
                                            resetgate=r_gate, updategate=u_gate, hidden_update=h_update) ##(None, 512)
        l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 64) ##(None, 128)
        #
        #------ ARGMAX for validation(->forecast) parts in network via embedding matrix 18.05 ---------------------------
    	#for val and train sets
    	coord_max = ExpressionLayer(l_out_loop_val, lambda x: x.argmax(-1).astype('int32'), output_shape='auto') #(None,)
    	emb_slice = ExpressionLayer(coord_max, lambda x: Emb_mtx_sym[x,:], output_shape=(None,word_dim)) #(None, 200)
	#
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: X.argmax(-1).astype('int32'), output_shape='auto') #(None,)
    	#pred_d = ExpressionLayer(coord_max, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X], 1), output_shape='auto')
    	#pred_emb = lasagne.layers.DenseLayer(pred_d, num_units=word_dim, W=Emb_mtx_sym, nonlinearity=lasagne.nonlinearities.linear) #no biases #(None, 200)
    	#pred_emb.params[pred_emb.W].remove("trainable")
        #
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X.argmax(-1).astype('int32')], 1), output_shape='auto')
    	#hot = lasagne.layers.DenseLayer(coord_max, num_units=word_dim, nonlinearity=lasagne.nonlinearities.linear)
        #
        pred = lasagne.layers.ReshapeLayer(emb_slice, ([0], 1, [1])) #######pred = lasagne.layers.ReshapeLayer(pred_emb, ([0], 1, [1])) #(None, 1, 200)
        #
        l_dec_val = lasagne.layers.GRULayer(pred, num_units=hidden_size*dec_units, learn_init=False, hid_init=h_init,
                                             grad_clipping=grad_clip, only_return_final=True,#hid_init=h_init_val, grad_clipping=grad_clip, only_return_final=True,
                                             resetgate=r_gate, updategate=u_gate, hidden_update=h_update) #(None, 320)
    	l_dec_val_hid_state = lasagne.layers.SliceLayer(l_dec_val, indices=slice(0,hidden_size)) #(None, 64)
    	l_out_loop = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
    	l_out_loop = lasagne.layers.ReshapeLayer(l_out_loop, ([0], 1, [1])) ####### #(None, 1, 155563)
    	l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop], axis=1) #(None, 2, 155563)
    	l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
    	l_out_loop_val_d = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1])) #(None, 1, 155563)
    	l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val_d], axis=1) #(None, 2, 155563)
    	#l_out_loop_val = lasagne.layers.FlattenLayer(l_out_loop_val, 2)
    	h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1)  #h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1) #(None, 320)
    	h_init_val = lasagne.layers.ConcatLayer([l_dec_val_hid_state, l_setenc], axis=1) #(None, 320)
    #l_out = lasagne.layers.ReshapeLayer(l_out, (-1, hidden_size))
    #l_out_val = lasagne.layers.ReshapeLayer(l_out_val, (-1, hidden_size))
    return (l_out, l_out_val)

l_out, l_out_val = model_seq2seq_GRU_setenc(X_enc_sym_list, X_dec_sym, Emb_mtx_sym, params['horizon']-1, params['num_metrics'], params['set_steps'], hidden_size = params['num_units'], 
                    grad_clip = params['grad_clipping'], att_size = params['num_att_units'], vocab_size = params['vocab_size'], word_dim = params['word_dim'])
                                            
                                            
                                            
network_output, network_output_val = lasagne.layers.get_output([l_out, l_out_val])

#network_output = lasagne.layers.get_output(l_out)

weights = lasagne.layers.get_all_params(l_out,trainable=True)
if params['regularization_type'] == 'l1':
    reg_loss = lasagne.regularization.regularize_network_params(l_out, l1) * params['lambda_regularization']
else:
    reg_loss = lasagne.regularization.regularize_network_params(l_out, l2) * params['lambda_regularization']

loss_T = categorical_crossentropy_3d(network_output, y_sym).mean() + reg_loss
loss_val_T = categorical_crossentropy_3d(network_output_val, y_sym).mean() 
loss_test = categorical_crossentropy_3d(network_output_val, y_sym).mean() 
#metric_probs = get_metric_probs(network_output, y_sym) #####             

updates = lasagne.updates.adam(loss_T, weights, learning_rate=eta)

f_train = theano.function([X_enc_sym_list, X_dec_sym, y_sym], loss_T, updates=updates, allow_input_downcast=True)
f_val = theano.function([X_enc_sym_list, X_dec_sym, Emb_mtx_sym, y_sym], loss_val_T, allow_input_downcast=True)#, on_unused_input='ignore')
forecast = theano.function([X_enc_sym_list, X_dec_sym, Emb_mtx_sym, y_sym], loss_test, allow_input_downcast=True)


1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
10.05.17 at deep night (Friday) 1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
As with Yagmur, but lists. Changing plain_et_step()
                            
import lasagne
import numpy as np
import theano
import theano.tensor as T
from theano import shared
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan
from lasagne.layers.base import Layer, MergeLayer
from lasagne.layers.input import InputLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers import helper, SliceLayer
from lasagne.layers.recurrent import Gate
import logging
import sys
import csv
from lasagne.layers import get_output
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
import os
import linecache
from lasagne.layers import InputLayer, ExpressionLayer, EmbeddingLayer
from theano.ifelse import ifelse
#            
def load_parameters2_mvrt_text(params_csvfile, line_num): 
    """
    reads the params for the new generated params
    """
    params={}
    values = linecache.getline(params_csvfile,line_num)[:-1].split(',')
    params['num_units'] = int(values[0]) #128
    params['num_att_units'] = int(values[1]) #512
    data_type = str(values[2]) 
    params['data_name'] = str(values[3])
    model = str(values[4]) 
    if params['data_name'] == 'coservit':
        params['windowise'] = 288
        params['horizon'] = 27
        params['word_dim'] = 202 # 200
        params['vocab_size'] = 22 ###### 155564
        
        #params['time_data'] = '../../../Data' + params['data_name'] + 
        #params['text_data'] = '../../../Data' + params['data_name'] + 
        #params['data_file'] =  '../../../Data/' + params['data_name'] + 'ticket2time.pkl'
        params['data_file'] =  '../../../Data/coservit/' + 'x_enc.pkl'
        params['text_file_wv'] =  '../../../Data/coservit/' + 'x_dec_wv.dat' ##params['text_file_wv'] =  '../../../Data/coservit/' + 'tickets_wv_pad_mtx.dat'
        params['text_file_w'] =  '../../../Data/coservit/' + 'x_dec_w.dat' #params['text_file_w'] =  '../../../Data/coservit/' + 'tickets_w_pad_mtx.dat'
        params['metr_dec_file'] =  '../../../Data/coservit/' + 'metr_for_dec.pkl' #list of lists with metric ids corresponding to aligned tickets. Ex.: t1,t1,t2,t3,t3,t3->[[3432, 4657], [3442], [6567, 4657, 7855]]
        params['emb_file'] =  '../../../Data/coservit/' + 'lda_emb_mtx.dat'#'emb_mtx.dat'
    else:
        pass
        """params['data_type'] = 'missing'
        #params['data_file'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/'+ params['data_name'] + '_'+ data_type +'.pkl'
        params['data_file'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/'+ params['data_name'] + '_'+ data_type +'_intr_linear.pkl'
        params['data_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/' + params['data_name'] + '_mask_' + data_type + '.pkl'
        params['missing_percent'] = data_type
        params['original_data_file'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'.pkl'
        params['original_data_mask_file'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'_mask.pkl' 
    #TODO:
    if params['data_name']  == 'pse':
        params['data2_name'] = 'weather_data'
        params['data2_file'] = '../../../Data/' + params['data2_name'] + '/missing_gaps_v2/'+ params['data2_name'] + '_'+ data_type +'.pkl'
        params['data2_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/' + params['data_name'] + '_mask_' + data_type + '.pkl'
        params['data2_variables'] = ['mintmp', 'maxtmp', 'precip'] 
    elif params['data_name']  == 'consseason':
        params['data2_name'] = 'consseason'
        params['data2_file'] = params['data_file']
        params['data2_file_missing_mask'] = params['data_file_missing_mask']
        params['data2_variables'] = ['Global_reactive_power', 'Voltage', 'Global_intensity'] 
    elif params['data_name']  == 'airq':
        params['data2_name'] = 'airq'
        params['data2_file'] = params['data_file']
        params['data2_file_missing_mask'] = params['data_file_missing_mask']
        params['data2_variables'] = ['NO2(GT)', 'CO(GT)', 'NOx(GT)'] """
            
    if data_type == 'orig':
        params['data_type'] = 'original'
    else:
        #TODO
        pass
        
    if model == 'RNN':
        params['attention'] = False
    else:
        params['attention'] = True
        if model == 'RNN-ATT':
            params['att_type'] = 'original'
        elif model == 'ATT-lambda':
            params['att_type'] = 'lambda'
        elif model == 'ATT-lambda-mu':
            params['att_type'] = 'lambda_mu' 
        elif model == 'ATT-lambda-mu-alt':
            params['att_type'] = 'lambda_mu_alt'
        elif model == 'ATT-dist':
            params['att_type'] = 'adist'
        elif model == 'RNN-text-set':
            params['att_type'] = 'set_enc'
            params['num_metrics'] = 4
            params['set_steps'] = 5 #number of iterations in set_enc part
    params['alg'] = 'adam'
    params['lambda_regularization'] = 0.0001
    params['eta_decay'] = 0.95
    params['early_stop_patience'] = 10000
    params['num_layers'] = 1
    params['regularization_type'] = 'l2'
    params['early_stop'] = True
    params['grad_clipping'] = 100
    params['train_test_ratio'] = 0.75
    params['train_val_ratio'] = 0.75
    params['patience_increase'] = 2 
    params['epsilon_improvement'] = True 
    params['lr_decay_after_n_epoch'] = 50
    params['interpolation'] = True #!!!!!!
    params['max_num_epochs'] = 25
    params['epsilon'] = 0.01
    params['learning_rate'] = 0.001
    params['batch_size'] = 64
    params['padding'] = True 
    params['learning_rate_decay'] = False
    params['interpolation_type'] = 'None'
    return params

params = load_parameters2_mvrt_text('params/fuzzy.param.text', 2)
params
#
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
results_directory = 'Results_multivariate_text_OM/'
#
att_type = params['att_type']
att_size = params['num_att_units']
word_dim = params['word_dim']
vocab_size = params['vocab_size']
grad_clip = params['grad_clipping']
hidden_size = params['num_units']
pred_len = 27
num_metrics = params['num_metrics']
set_steps = params['set_steps']
grad_clip = 100
#
#Gate
class Gate_setenc(object):
    def __init__(self, W_in=init.GlorotNormal(), W_hid=init.GlorotNormal(),
                 W_cell=init.Normal(0.1), b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        if W_in is not None:
            self.W_in = W_in
        self.W_hid = W_hid
        # Don't store a cell weight vector when cell is None
        if W_cell is not None:
            self.W_cell = W_cell
        self.b = b
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

#
#Layer
class GRULayer_setenc(lasagne.layers.Layer): #it is not mvrt lasagne.layers.Layer
    def __init__(self, incoming, num_units, 
                 resetgate=Gate_setenc(W_in=None,W_cell=None), 
                 updategate=Gate_setenc(W_in=None,W_cell=None),
                 hidden_update=Gate_setenc(W_in=None, W_cell=None),
                 nonlinearity=nonlinearities.tanh,
                 hid_init=init.Constant(0.),
                 set_steps=5,
                 att_num_units = 64,
                 W_hid_to_att = init.GlorotNormal(),
                 W_ctx_to_att = init.GlorotNormal(),
                 W_att = init.Normal(0.1), #---
                 backwards=False,
                 learn_init=True, #--???????
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=True, #-- ???????
                 **kwargs):
        #         
        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.  We will just provide the
        # layer input as incomings, unless a mask input was provided.
        #incomings = [incoming]
        #if mask_input is not None:
            #incomings.append(mask_input) #-----
        #
        # Initialize parent layer
        super(GRULayer_setenc, self).__init__(incoming, **kwargs) #super(GRULayer_setenc, self).__init__(incomings, **kwargs)
        #
        self.learn_init = learn_init
        self.num_units = num_units #128
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.set_steps = set_steps
        self.att_num_units = att_num_units #512
        self.only_return_final = only_return_final
        #
        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")
        #
        # Retrieve the dimensionality of the incoming layer
        #
        if unroll_scan and self.input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")
        #
        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(self.input_shape[2:]) #4
        print(num_inputs)
        print(self.input_shape) #(None, 256, 4)
        #print(num_inputs) #4
        #
        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            #self.add_param(gate.W_in, (num_inputs, num_units),
                                   #name="W_in_to_{}".format(gate_name))
            return (self.add_param(gate.W_hid, (3*num_units, num_units), #128
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,1),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)
        #
        # Add in all parameters from gates, nonlinearities will be sigmas, look Gate_setenc
        """(self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')
        #
        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')"""
        (self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')
        #
        (self.W_hid_to_hidden_update, self.b_hidden_update, 
         self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')
        #
        #attention Weights 
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        #
        # Initialize hidden state
        #self.hid_init = hid_init ######
        
        """if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init", #!!!!
                trainable=learn_init, regularizable=False) #???????"""
        #self.hid_init = T.dmatrix() #>>>
        self.hid_init = T.zeros((1, self.num_units*3))
        #hid_init = T.specify_shape(hid_init, (1, self.num_units))
        #hid_init = T.tile(hid_init, (input.shape[0], 1, 1))
        #T.tile(hid0, (input.shape[0], 1, 1))
        """if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError(
                    "When hid_init is provided as a TensorVariable, it should "
                    "have 2 dimensions and have shape (num_batch, num_units)")
            self.hid_init = hid_init
        else: # here
            self.hid_init = self.add_param(  #--????????#--????????#--????????#--????????#--????????#--????????#--????????
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False) #not trainable"""
        #print(self.hid_init, type(self.hid_init)) #(hid_init, <class 'theano.tensor.sharedvar.TensorSharedVariable'>
#
    def get_output_shape_for(self, input_shape):                        #(None, 256, 128)
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shp = self.input_shape[0]
        # PRINTS
        if self.only_return_final:
            return self.input_shape[0], 3*self.num_units #(None, 128)
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return self.input_shape[0], self.input_shape[1], 3*self.num_units
#
    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        #print('uiop') # PRINTS
        # Retrieve the layer input
        input = inputs
        #print(input) #<lasagne.layers.merge.ConcatLayer object at 0x7fe2e1f7fb10>
        #print(input.ndim)
        # Retrieve the mask when it is supplied
        #mask = inputs[1] if len(inputs) > 1 else None # -> mask=None
#
        #print(type(inputs)) #<class 'lasagne.layers.merge.ConcatLayer'>
        # Treat all dimensions after the second as flattened feature dimensions
        #if input.ndim > 3:
            #input = T.flatten(input, 3)
#
        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        #input = lasagne.layers.ReshapeLayer(input, ([1], [0], [2]))#
        #####input = input.dimshuffle(1, 0, 2)#
        num_batch, seq_len, _ = input.shape #256,None -- no 4 #(256, None) #seq_len, num_batch, _ = input.output_shape
        print(num_batch, seq_len) #(256, None)
        #print(input.output_shape) #(256, None, 4)
#
        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        """W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)"""
#
        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)
#
        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)
#
        #if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            #input = T.dot(input, W_in_stacked) + b_stacked
#
        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
   #     
        def plain_et_step(o_t0):#def plain_et_step(self, x_snp, o_t0)
            #reading from memory steps
            print("i'm in plain")
            bs, num_hid2, num_metr = input.shape
            m_in = input.dimshuffle(0, 2, 1)#----replace
            e_qt = T.dot(o_t0, self.W_hid_to_att)#---
            e_m = T.dot(m_in, self.W_ctx_to_att)#----
            e_q = T.tile(e_qt, (num_metr, 1, 1)) #e_q = T.tile(e_qt, (self.seq_len_m, 1, 1))
            et_p = T.tanh(e_m + e_q)
            et = T.dot(et_p, self.W_att)
            alpha = T.exp(et)
            alpha /= T.sum(alpha, axis=0)
            mt = input.dimshuffle(1, 0, 2)
            mult = T.mul(mt, alpha)
            rt = T.sum(mult, axis=2)
            #print(input.shape, rt.shape)
            print("plain finished")
            return rt.T
            """ print("i'm in plain")
            bs, seq_len_m, num_metr = input.shape#####bs, seq_len_m, _ = input.shape
            m_in = input#####.dimshuffle(1, 0, 2)#----replace
            e_qt = T.dot(o_t0, self.W_hid_to_att)#---
            e_m = T.dot(m_in, self.W_ctx_to_att)#---- ??????????????
            e_q = T.tile(e_qt, (1, 1, num_metr)) #####e_q = T.tile(e_qt, (seq_len_m, 1, 1)) #e_q = T.tile(e_qt, (self.seq_len_m, 1, 1))
            et_p = T.tanh(e_m + e_q)
            et = T.dot(et_p, self.W_att)
            #et = et.dimshuffle([0,'x',1])
            alpha = T.exp(et)
            alpha /= T.sum(alpha, axis=1)#####2)0)
            mt = input.dimshuffle(0, 2, 1)
            mult = T.mul(mt, alpha)
            rt = T.sum(mult, axis=1)#####2)
            #print(input.shape, rt.shape)
            print("plain finished")
            return rt#####.T#"""
#
        def step(hid_previous, *args): #W_hid_stacked, #W_in_stacked, b_stacked):
            #x_snp = incom
            #print(x_snp.output_shape)
            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            print("qwerty")
            one = T.alloc(np.array(1,dtype='int64'))
            #zero_vec = T.zeros((1, 3*self.num_units))
            x = ifelse(T.eq(hid_previous.shape[0], one), input.shape[0], one)
            print("after ifelse")
            hid_previous = hid_previous.repeat(x, axis=0) #hid_previous = T.tile(hid_previous, (x, 1))
            #
            #hid_previous = ifelse(T.eq(hid_previous.shape[0],one), T.tile(hid_previous, (x, 1),ndim=2),hid_previous)
            #
            hid_input = T.dot(hid_previous, W_hid_stacked) + b_stacked.T #>> ####for r, z, h tilde WHAT ABOUT THE SIZES WHEN MULT??????
#       
            print("tyui")
            if self.grad_clipping is not False:
                #input_n = theano.gradient.grad_clip(
                    #input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)
                print('d')
#
           # if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                #input_n = T.dot(input_n, W_in_stacked) + b_stacked
#
            # Reset and update gates
            resetgate = slice_w(hid_input, 0) #+ slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) #+ slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)
#
            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            #hidden_update_in = slice_w(input_n, 2)
            ####hidden_update_hid = slice_w(hid_input, 2) #h tilde
            ####hidden_update = resetgate*hidden_update_hid #hidden_update = hidden_update_in + resetgate*hidden_update_hid
            hidden_update = slice_w(hid_input, 2)
            #
            if self.grad_clipping is not False:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update) #here it is sigma, but in encoder.py this is tanh ????????????
#
            print('d')
            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid0 = (np.float32(1.0) - updategate)*hid_previous + updategate*hidden_update
            #hid0 = T.tile(hid0, (input.shape[0], 1, 1))
            rt = plain_et_step(hid0)
            print(rt.shape)
            h_t = T.concatenate([hid0,rt], axis=1)
    #        
            print("before return h_t in step")
            return h_t #------??????
        """def step(x_snp, snp_mask, m_snp_count):
            r_t = T.nnet.sigmoid(T.dot(hr_tm1, self.Wo_hh_r) + self.bo_r)
            z_t = T.nnet.sigmoid(T.dot(hr_tm1, self.W_hh_z) + self.b_z)
            h_tilde = T.tanh(T.dot(r_t * hr_tm1, self.W_hh) + self.b_hh)
            o_t = (np.float32(1.0) - z_t) * hr_tm1 + z_t * h_tilde
            #o_t: GRU hidden state
            #concatanation of hidden state and reading from memory
            rt = self.plain_et_step(x_snp, o_t)
            h_t = T.concatenate([o_t0,rt], axis=1)---------
            return h_t"""
     #       
#
        """def step_masked(input_n, mask_n, hid_previous, W_hid_stacked,
                        W_in_stacked, b_stacked):
#
            hid = step(input_n, hid_previous, W_hid_stacked, W_in_stacked,
                       b_stacked)
#
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            not_mask = 1 - mask_n
            hid = hid*mask_n + hid_previous*not_mask
#
            return hid"""
#
        """if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
            print('d')
        else:
            #sequences = input #[input]
            step_fun = step
            print("step")"""
        step_fun = step
#
        """if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)"""
#
        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked, b_stacked] #non_seqs = [W_hid_stacked] #non_seqs = [input, W_hid_stacked]
        #non_seqs += [W_ctx_stacked] #
        non_seqs += [self.W_hid_to_att, self.W_ctx_to_att, self.W_att, input] #
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        #if not self.precompute_input:
            #non_seqs += [b_stacked]#[W_in_stacked, b_stacked]
        # theano.scan only allows for positional arguments, so when
        # self.precompute_input is True, we need to supply fake placeholder
        # arguments for the input weights and biases.
        #else:
            #non_seqs += [(), ()]
#
        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else: #here
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            #hid_out, _ = theano.scan(fn=step_fun, outputs_info=o_enc_info, non_sequences=incomings[0], n_steps=self.set_steps)#self.num_set_iter) #----??????
            print('d')
            hid_out, _ = theano.scan(fn=step_fun, outputs_info=self.hid_init, non_sequences=non_seqs, n_steps=self.set_steps, strict=True)#self.num_set_iter)
        """hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]"""
#
        # dimshuffle back to (n_batch, n_time_steps, n_features))
        #######hid_out = hid_out.dimshuffle(1, 0, 2) #done below
#
        # if scan is backward reverse the output
        #if self.backwards:-------- ?????????
            #hid_out = hid_out[:, ::-1, :]-------??????????? #done below
         #
        #copied from layers.py
        """if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)
            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]     """
        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        #return theano.shared(np.random.randn(3,4))
#
        print(hid_out.shape)
        return hid_out

def categorical_crossentropy_3d(coding_dist, true_dist, lengths=None):
    #http://stackoverflow.com/questions/30225633/cross-entropy-for-batch-with-theano
    
    # Zero out the false probabilities and sum the remaining true probabilities to remove the third dimension.
    indexes = theano.tensor.arange(coding_dist.shape[2])
    mask = theano.tensor.neq(indexes, true_dist.reshape((true_dist.shape[0], true_dist.shape[1], 1)))
    pred_probs = theano.tensor.set_subtensor(coding_dist[theano.tensor.nonzero(mask)], 0.).sum(axis=2)
    pred_probs_log = T.log(pred_probs)
    pred_probs_per_sample = -pred_probs_log.sum(axis=1)
    return pred_probs_per_sample

#in network_text()
import theano.typed_list
X_enc_sym_list = theano.typed_list.TypedListType(T.dtensor3)()
name = 'x_enc_sym'
for i in range(1, num_metrics+1):
    X_enc_sym_list = theano.typed_list.append(X_enc_sym_list, T.dtensor3('x_enc_sym'+str(i)))

#theano.typed_list.basic.length(X_enc_sym_list) #Length.0
X_dec_sym = T.dtensor3('x_dec_sym') ##X_dec_sym = T.ftensor3('x_dec_sym')
y_sym = T.lmatrix('y_sym') # indexes of 1hot words, for loss 
Emb_mtx_sym = T.dmatrix('emb_mtx_sym')
eta = theano.shared(np.array(params['learning_rate'], dtype=theano.config.floatX))

#theano.typed_list.basic.length(X_enc_sym_list) #Length.0

#in model
#X = X_enc_sym[:,:,0:1]
#l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=theano.typed_list.basic.getitem(X_enc_sym_list,0)) #(None, None, 4) #l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
#l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=0, axis=2) #(None, None)
#l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #(None, None, 1)

l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                grad_clipping=grad_clip, only_return_final=True)
l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None),
                                                grad_clipping=grad_clip, only_return_final=True, backwards=True)
l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
l_enc_conc = lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1)) #(None, 256, 1)
#    
resetgate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_resetgate, W_hid=l_forward.W_hid_to_resetgate, W_cell = None, b=l_forward.b_resetgate)
updategate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_updategate, W_hid=l_forward.W_hid_to_updategate, W_cell = None, b=l_forward.b_updategate)
hidden_update_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_hidden_update, W_hid=l_forward.W_hid_to_hidden_update, W_cell=None, b=l_forward.b_hidden_update)
#
resetgate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_resetgate, W_hid=l_backward.W_hid_to_resetgate, W_cell= None, b=l_backward.b_resetgate)
updategate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_updategate, W_hid=l_backward.W_hid_to_updategate, W_cell=None, b=l_backward.b_updategate)
hidden_update_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_hidden_update, W_hid=l_backward.W_hid_to_hidden_update, W_cell=None, b=l_backward.b_hidden_update)

for i in range(1, num_metrics):
        #X = X_enc_sym[:,:,i:(i+1)]
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=theano.typed_list.basic.getitem(X_enc_sym_list,i))
        #l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=i, axis=2) #(None, None)
        #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], 1, [1]))
#        
    l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                                        resetgate=resetgate_f, 
                                                        updategate=updategate_f, 
                                                        hidden_update=hidden_update_f, 
                                                        grad_clipping=grad_clip, only_return_final=True)
    l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size,
                                                        resetgate=resetgate_b, 
                                                        updategate=updategate_b, 
                                                        hidden_update=hidden_update_b,
                                                        grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
    l_enc_conc = lasagne.layers.ConcatLayer([l_enc_conc, lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1))], axis=2) #(None, 256, 4)

#add what above to model
#l_q = theano.shared(value=np.zeros((hidden_size,1), dtype='float32'))
#l_q = lasagne.init.Constant(0.)
#l_input = lasagne.layers.InputLayer(shape=(None, 256, 4), name=l_enc_conc)
l_setenc = GRULayer_setenc(incoming=l_enc_conc, num_units=hidden_size, learn_init=False, set_steps=5, att_num_units=att_size, grad_clipping=grad_clip, 
                           nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)  #(None, 128)
#l_setenc = lasagne.layers.ConcatLayer([lasagne.layers.SliceLayer(l_enc_conc, indices=0, axis=2), \
 #           lasagne.layers.SliceLayer(lasagne.layers.SliceLayer(l_enc_conc, indices=0, axis=2), indices=slice(0,hidden_size))],axis=1) #(None, 384)
#lasagne.layers.get_output(l_setenc)

#DECODER

dec_units = 3*1 +1 #dec_units = 4*2 +1 #nr_encoder *2 + dec -> bi-directional so *2 !!important to set it right
    #decoder
l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, word_dim),input_var=X_dec_sym)#pred_len=27 #l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len,1),input_var=X_dec_sym)
    #l_mask_dec = lasagne.layers.InputLayer(shape=(None, pred_len), input_var=mask_dec)
    #    
s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #SHOULD WE RESHAPE IT???
    #s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    #s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    #h_init = lasagne.layers.ConcatLayer([T.alloc(0., (l_forward.output_shape[0], hidden_size)), l_setenc], axis=1) #
h_init = lasagne.layers.ConcatLayer([l_forward, l_setenc], axis=1) #????????????? #(None, 512)
    #
l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, hid_init=h_init,
                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)),
                                    learn_init=False, grad_clipping=grad_clip, only_return_final=True ) #l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, mask_input=s_lmask_dec1, learn_init=False, 
                                                                             #hid_init=h_init, grad_clipping=grad_clip, only_return_final=True )
r_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_resetgate, W_hid=l_dec.W_hid_to_resetgate, b=l_dec.b_resetgate)
u_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_updategate, W_hid=l_dec.W_hid_to_updategate, b=l_dec.b_updategate)
h_update = lasagne.layers.Gate(W_in=l_dec.W_in_to_hidden_update, W_hid=l_dec.W_hid_to_hidden_update, b=l_dec.b_hidden_update)                                
#
l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 128)
#TO CHANGE BACK BELOW
l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, nonlinearity=lasagne.nonlinearities.softmax)  #l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=1, nonlinearity=lasagne.nonlinearities.linear)
#
w_dense = l_out.W
b_dense = l_out.b
l_out_loop = l_out
l_out_loop_val = l_out  
l_out = lasagne.layers.ReshapeLayer(l_out, ([0], 1, [1]))
l_out_val = l_out
h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1) #--- ##(None, 512)
h_init_val = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1) #--- ##(None, 512)

i=1
s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1) #(None, 200) ##(None, 202)
s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #(None, 1, 200) ##(None, 1, 202)
l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, hid_init=h_init, learn_init=False,	 #(None, 320)
                                            grad_clipping=grad_clip, only_return_final=True, 
                                            resetgate=r_gate, updategate=u_gate, hidden_update=h_update) ##(None, 512)
l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 64) ##(None, 128)
        #
        #------ ARGMAX for validation(->forecast) parts in network via embedding matrix 18.05 ---------------------------
    	#for val and train sets
coord_max = ExpressionLayer(l_out_loop_val, lambda x: x.argmax(-1).astype('int32'), output_shape='auto') #(None,)
emb_slice = ExpressionLayer(coord_max, lambda x: Emb_mtx_sym[x,:], output_shape=(None,word_dim)) #(None, 200)
	#
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: X.argmax(-1).astype('int32'), output_shape='auto') #(None,)
    	#pred_d = ExpressionLayer(coord_max, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X], 1), output_shape='auto')
    	#pred_emb = lasagne.layers.DenseLayer(pred_d, num_units=word_dim, W=Emb_mtx_sym, nonlinearity=lasagne.nonlinearities.linear) #no biases #(None, 200)
    	#pred_emb.params[pred_emb.W].remove("trainable")
        #
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X.argmax(-1).astype('int32')], 1), output_shape='auto')
    	#hot = lasagne.layers.DenseLayer(coord_max, num_units=word_dim, nonlinearity=lasagne.nonlinearities.linear)
        #
pred = lasagne.layers.ReshapeLayer(emb_slice, ([0], 1, [1])) #######pred = lasagne.layers.ReshapeLayer(pred_emb, ([0], 1, [1])) #(None, 1, 200)
        #
l_dec_val = lasagne.layers.GRULayer(pred, num_units=hidden_size*dec_units, learn_init=False, hid_init=h_init,
                                             grad_clipping=grad_clip, only_return_final=True,#hid_init=h_init_val, grad_clipping=grad_clip, only_return_final=True,
                                             resetgate=r_gate, updategate=u_gate, hidden_update=h_update) #(None, 320)
l_dec_val_hid_state = lasagne.layers.SliceLayer(l_dec_val, indices=slice(0,hidden_size)) #(None, 64)
l_out_loop = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
l_out_loop = lasagne.layers.ReshapeLayer(l_out_loop, ([0], 1, [1])) ####### #(None, 1, 155563)
l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop], axis=1) #(None, 2, 155563)
l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
l_out_loop_val_d = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1])) #(None, 1, 155563)
l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val_d], axis=1) #(None, 2, 155563)
    	#l_out_loop_val = lasagne.layers.FlattenLayer(l_out_loop_val, 2)
h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1)  #h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1) #(None, 320)
h_init_val = lasagne.layers.ConcatLayer([l_dec_val_hid_state, l_setenc], axis=1) #(None, 320)


def model_seq2seq_GRU_setenc(X_enc_sym_list, X_dec_sym, Emb_mtx_sym, pred_len, num_metrics, set_steps, hidden_size=64, grad_clip = 100, att_size=64, vocab_size=22, word_dim=202): ##def model_seq2seq_GRU_text(X_enc_sym, X_enc_sym2, X_dec_sym, max_len, pred_len, hidden_size=64, grad_clip = 100, vocab_size=155563, word_dim=200):#(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, max_len, pred_len, hidden_size=64, grad_clip = 100) # max_len=288 - windowise, pred_len=27 - horizon
    #copy of model_seq2seq_mvrt4_GRU_text
    # model for order matters article 31.05.17
    #
    #collect last hiddens for each metric 
    #X = X_enc_sym[:,:,0:1]
    #l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=theano.typed_list.basic.getitem(X_enc_sym_list,0)) #(None, None, 4) #l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X)
    #l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=0, axis=2) #(None, None)
    #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #(None, None, 1)
    #
    l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    grad_clipping=grad_clip, only_return_final=True)
    l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128)
                                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None), 
                                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=None),
                                                    grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
    l_enc_conc = lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1)) #(None, 256, 1)
    #    
    resetgate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_resetgate, W_hid=l_forward.W_hid_to_resetgate, W_cell = None, b=l_forward.b_resetgate)
    updategate_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_updategate, W_hid=l_forward.W_hid_to_updategate, W_cell = None, b=l_forward.b_updategate)
    hidden_update_f = lasagne.layers.Gate(W_in=l_forward.W_in_to_hidden_update, W_hid=l_forward.W_hid_to_hidden_update, W_cell=None, b=l_forward.b_hidden_update)
    #
    resetgate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_resetgate, W_hid=l_backward.W_hid_to_resetgate, W_cell= None, b=l_backward.b_resetgate)
    updategate_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_updategate, W_hid=l_backward.W_hid_to_updategate, W_cell=None, b=l_backward.b_updategate)
    hidden_update_b = lasagne.layers.Gate(W_in=l_backward.W_in_to_hidden_update, W_hid=l_backward.W_hid_to_hidden_update, W_cell=None, b=l_backward.b_hidden_update)
    #                                  
    for i in range(1, num_metrics):
        #X = X_enc_sym[:,:,i:(i+1)]
        l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=theano.typed_list.basic.getitem(X_enc_sym_list,i))
        #l_in_slice = lasagne.layers.SliceLayer(l_in_enc, indices=i, axis=2) #(None, None)
        #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], [1], 1)) #l_in_slice = lasagne.layers.ReshapeLayer(l_in_slice, ([0], 1, [1]))
        #
        l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                                        resetgate=resetgate_f, 
                                                        updategate=updategate_f, 
                                                        hidden_update=hidden_update_f, 
                                                        grad_clipping=grad_clip, only_return_final=True)
        l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, #(None, 128) l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size,
                                                        resetgate=resetgate_b, 
                                                        updategate=updategate_b, 
                                                        hidden_update=hidden_update_b,
                                                        grad_clipping=grad_clip, only_return_final=True, backwards=True)
        l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1) #(None, 256)
        l_enc_conc = lasagne.layers.ConcatLayer([l_enc_conc, lasagne.layers.ReshapeLayer(l_enc, ([0], [1], 1))], axis=2) #(None, 256, 4)
    #l_setenc = lasagne.layers.ElemwiseSumLayer([l_enc_conc], axis=2) #(None, 256)
    # 
    # Set-encoder part
    #l_q = lasagne.init.Constant(0.)
    #for i in range(set_steps):
    l_setenc = GRULayer_setenc(incoming=l_enc_conc, num_units=hidden_size, learn_init=False, set_steps=set_steps, att_num_units=att_size, grad_clipping=grad_clip, 
                                nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True) #(None, 384)
    
    #l_q = lasagne.layers.SliceLayer(l_setenc, -1, 1)               ###
    #l_setenc = lasagne.layers.ConcatLayer([lasagne.layers.SliceLayer(l_enc_conc, indices=0, axis=2), \
     #           lasagne.layers.SliceLayer(lasagne.layers.SliceLayer(l_enc_conc, indices=0, axis=2), indices=slice(0,hidden_size))],axis=1) #(None, 384)
    #
    dec_units = 3*1 +1 #dec_units = 4*2 +1 #nr_encoder *2 + dec -> bi-directional so *2 !!important to set it right
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, word_dim),input_var=X_dec_sym)#pred_len=27 #l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len,1),input_var=X_dec_sym)
    #l_mask_dec = lasagne.layers.InputLayer(shape=(None, pred_len), input_var=mask_dec)
    #    
    s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
    s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #SHOULD WE RESHAPE IT???
    #s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    #s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    #h_init = lasagne.layers.ConcatLayer([T.alloc(0., (l_forward.output_shape[0], hidden_size)), l_setenc], axis=1) #
    h_init = lasagne.layers.ConcatLayer([l_forward, l_setenc], axis=1) #????????????? ##(None, 512)
    #
    l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, hid_init=h_init,
                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)),
                                    learn_init=False, grad_clipping=grad_clip, only_return_final=True ) #l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, mask_input=s_lmask_dec1, learn_init=False, 
                                         #hid_init=h_init, grad_clipping=grad_clip, only_return_final=True )
    r_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_resetgate, W_hid=l_dec.W_hid_to_resetgate, b=l_dec.b_resetgate)
    u_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_updategate, W_hid=l_dec.W_hid_to_updategate, b=l_dec.b_updategate)
    h_update = lasagne.layers.Gate(W_in=l_dec.W_in_to_hidden_update, W_hid=l_dec.W_hid_to_hidden_update, b=l_dec.b_hidden_update)                                
    #
    l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 128)
    #TO CHANGE BACK BELOW
    l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, nonlinearity=lasagne.nonlinearities.softmax)  #l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=1, nonlinearity=lasagne.nonlinearities.linear)
    #
    w_dense = l_out.W
    b_dense = l_out.b
    l_out_loop = l_out
    l_out_loop_val = l_out  
    l_out = lasagne.layers.ReshapeLayer(l_out, ([0], 1, [1]))
    l_out_val = l_out
    h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1) #--- ##(None, 512)
    h_init_val = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1) #--- ##(None, 512)
#
    for i in range(1,pred_len): #comments in this cycle are for the first iteration
        s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1) #(None, 200) ##(None, 202)
        s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #(None, 1, 200) ##(None, 1, 202)
        l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, hid_init=h_init, learn_init=False,	 #(None, 320)
                                            grad_clipping=grad_clip, only_return_final=True, 
                                            resetgate=r_gate, updategate=u_gate, hidden_update=h_update) ##(None, 512)
        l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 64) ##(None, 128)
        #
        #------ ARGMAX for validation(->forecast) parts in network via embedding matrix 18.05 ---------------------------
    	#for val and train sets
    	coord_max = ExpressionLayer(l_out_loop_val, lambda x: x.argmax(-1).astype('int32'), output_shape='auto') #(None,)
    	emb_slice = ExpressionLayer(coord_max, lambda x: Emb_mtx_sym[x,:], output_shape=(None,word_dim)) #(None, 200)
	#
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: X.argmax(-1).astype('int32'), output_shape='auto') #(None,)
    	#pred_d = ExpressionLayer(coord_max, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X], 1), output_shape='auto')
    	#pred_emb = lasagne.layers.DenseLayer(pred_d, num_units=word_dim, W=Emb_mtx_sym, nonlinearity=lasagne.nonlinearities.linear) #no biases #(None, 200)
    	#pred_emb.params[pred_emb.W].remove("trainable")
        #
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X.argmax(-1).astype('int32')], 1), output_shape='auto')
    	#hot = lasagne.layers.DenseLayer(coord_max, num_units=word_dim, nonlinearity=lasagne.nonlinearities.linear)
        #
        pred = lasagne.layers.ReshapeLayer(emb_slice, ([0], 1, [1])) #######pred = lasagne.layers.ReshapeLayer(pred_emb, ([0], 1, [1])) #(None, 1, 200)
        #
        l_dec_val = lasagne.layers.GRULayer(pred, num_units=hidden_size*dec_units, learn_init=False, hid_init=h_init,
                                             grad_clipping=grad_clip, only_return_final=True,#hid_init=h_init_val, grad_clipping=grad_clip, only_return_final=True,
                                             resetgate=r_gate, updategate=u_gate, hidden_update=h_update) #(None, 320)
    	l_dec_val_hid_state = lasagne.layers.SliceLayer(l_dec_val, indices=slice(0,hidden_size)) #(None, 64)
    	l_out_loop = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
    	l_out_loop = lasagne.layers.ReshapeLayer(l_out_loop, ([0], 1, [1])) ####### #(None, 1, 155563)
    	l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop], axis=1) #(None, 2, 155563)
    	l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #(None, 155563)
    	l_out_loop_val_d = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1])) #(None, 1, 155563)
    	l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val_d], axis=1) #(None, 2, 155563)
    	#l_out_loop_val = lasagne.layers.FlattenLayer(l_out_loop_val, 2)
    	h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_setenc], axis=1)  #h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1) #(None, 320)
    	h_init_val = lasagne.layers.ConcatLayer([l_dec_val_hid_state, l_setenc], axis=1) #(None, 320)
    #l_out = lasagne.layers.ReshapeLayer(l_out, (-1, hidden_size))
    #l_out_val = lasagne.layers.ReshapeLayer(l_out_val, (-1, hidden_size))
    return (l_out, l_out_val)

l_out, l_out_val = model_seq2seq_GRU_setenc(X_enc_sym_list, X_dec_sym, Emb_mtx_sym, params['horizon']-1, params['num_metrics'], params['set_steps'], hidden_size = params['num_units'], 
                    grad_clip = params['grad_clipping'], att_size = params['num_att_units'], vocab_size = params['vocab_size'], word_dim = params['word_dim'])
                                            
                                            
                                            
network_output, network_output_val = lasagne.layers.get_output([l_out, l_out_val])

#network_output = lasagne.layers.get_output(l_out)

weights = lasagne.layers.get_all_params(l_out,trainable=True)
if params['regularization_type'] == 'l1':
    reg_loss = lasagne.regularization.regularize_network_params(l_out, l1) * params['lambda_regularization']
else:
    reg_loss = lasagne.regularization.regularize_network_params(l_out, l2) * params['lambda_regularization']

loss_T = categorical_crossentropy_3d(network_output, y_sym).mean() + reg_loss
loss_val_T = categorical_crossentropy_3d(network_output_val, y_sym).mean() 
loss_test = categorical_crossentropy_3d(network_output_val, y_sym).mean() 
#metric_probs = get_metric_probs(network_output, y_sym) #####             

updates = lasagne.updates.adam(loss_T, weights, learning_rate=eta)

f_train = theano.function([X_enc_sym_list, X_dec_sym, y_sym], loss_T, updates=updates, allow_input_downcast=True)
f_val = theano.function([X_enc_sym_list, X_dec_sym, Emb_mtx_sym, y_sym], loss_val_T, allow_input_downcast=True)#, on_unused_input='ignore')
forecast = theano.function([X_enc_sym_list, X_dec_sym, Emb_mtx_sym, y_sym], loss_test, allow_input_downcast=True)