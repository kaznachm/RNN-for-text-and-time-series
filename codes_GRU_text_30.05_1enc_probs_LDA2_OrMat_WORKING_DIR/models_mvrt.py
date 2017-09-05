import lasagne 
from layers_mvrt import LSTMAttLayer4
from layers_mvrt import AnaLayer4
from layers import LSTMAttLayer_v2
from layers import AnaLayer_v2
from layers_mvrt import LSTMAttLayer_lambda4
from layers_mvrt import AnaLayer_lambda4
from layers_mvrt import LSTMAttLayer_lambda_mu4
from layers_mvrt import AnaLayer_lambda_mu4
from layers_mvrt import LSTMAttLayer_lambda_mu_alt4
from layers_mvrt import AnaLayer_lambda_mu_alt4
from layers_mvrt import GRULayer_setenc
import theano.tensor as T
from lasagne.layers import InputLayer, ExpressionLayer, EmbeddingLayer


def model_seq2seq_mvrt4(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, max_len, pred_len, hidden_size=64, grad_clip = 100):
    """
    Multivariate sequence to sequence without attention
    """
    #encoder
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym)
    l_mask_enc = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc)    
    #
    l_forward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True )
    l_backward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, backwards=True)
    l_enc1 = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1)
    #2nd variable encoder
    l_in_enc2 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym2)
    l_mask_enc2 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc2)    
    #
    l_forward2 = lasagne.layers.LSTMLayer(l_in_enc2, num_units=hidden_size, mask_input=l_mask_enc2, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True )
    l_backward2 = lasagne.layers.LSTMLayer(l_in_enc2, num_units=hidden_size, mask_input=l_mask_enc2, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, backwards=True)
    l_enc2 = lasagne.layers.ConcatLayer([l_forward2, l_backward2], axis=1)
    #3rd variable encoder
    l_in_enc3 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym3)
    l_mask_enc3 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc3)    
    #
    l_forward3 = lasagne.layers.LSTMLayer(l_in_enc3, num_units=hidden_size, mask_input=l_mask_enc3, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True )
    l_backward3 = lasagne.layers.LSTMLayer(l_in_enc3, num_units=hidden_size, mask_input=l_mask_enc3, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, backwards=True)
    l_enc3 = lasagne.layers.ConcatLayer([l_forward3, l_backward3], axis=1)
    ##3rd variable encoder
    l_in_enc4 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym4)
    l_mask_enc4 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc4)    
    #
    l_forward4 = lasagne.layers.LSTMLayer(l_in_enc4, num_units=hidden_size, mask_input=l_mask_enc4, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True )
    l_backward4 = lasagne.layers.LSTMLayer(l_in_enc4, num_units=hidden_size, mask_input=l_mask_enc4, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, backwards=True)
    l_enc4 = lasagne.layers.ConcatLayer([l_forward4, l_backward4], axis=1)
    #
    l_enc = lasagne.layers.ConcatLayer([l_enc1, l_enc2, l_enc3, l_enc4], axis=1) # 3*2 
    #
    dec_units = 4*2 +1 #nr_encoder *2 + dec -> bi-directional so *2 !!important to set it right
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len,1),input_var=X_dec_sym)
    l_mask_dec = lasagne.layers.InputLayer(shape=(None, pred_len), input_var=mask_dec)
    #    
    s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
    s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))
    s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    h_init = lasagne.layers.ConcatLayer([l_forward, l_enc], axis=1)
    #
    l_dec = lasagne.layers.LSTMLayer(s_lin_dec, num_units=hidden_size*dec_units, mask_input=s_lmask_dec1, learn_init=False, 
                                     hid_init=h_init, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True )
    input_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_ingate, W_hid=l_dec.W_hid_to_ingate, 
                                            W_cell= l_dec.W_cell_to_ingate, b=l_dec.b_ingate)
    output_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_outgate, W_hid=l_dec.W_hid_to_outgate, 
                                      W_cell=l_dec.W_cell_to_outgate, b=l_dec.b_outgate)
    forget_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_forgetgate, W_hid=l_dec.W_hid_to_forgetgate,
                                      W_cell=l_dec.W_cell_to_forgetgate, b=l_dec.b_forgetgate)
    cell_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_cell, W_hid=l_dec.W_hid_to_cell, W_cell=None,
                                    b=l_dec.b_cell, nonlinearity=lasagne.nonlinearities.tanh) 
    #
    l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size))
    l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=1, nonlinearity=lasagne.nonlinearities.linear)  
    w_dense = l_out.W
    b_dense = l_out.b
    l_out_loop = l_out
    l_out_val = l_out
    l_out_loop_val = l_out    
    h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1)
    h_init_val = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1)
    for i in range(1,pred_len):
        s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1)
        s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))
        s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=i, axis=1) 
        s_lmask_dec = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1)) 
        l_dec = lasagne.layers.LSTMLayer(s_lin_dec, num_units=hidden_size*dec_units, mask_input=s_lmask_dec, learn_init=False, 
                                         hid_init=h_init, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, 
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate) 
        l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size))
        #
        pred = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1]))
        l_dec_val = lasagne.layers.LSTMLayer(pred, num_units=hidden_size*dec_units, mask_input=s_lmask_dec, learn_init=False,
                                         hid_init=h_init_val, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True,
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate)
        l_dec_val_hid_state = lasagne.layers.SliceLayer(l_dec_val, indices=slice(0,hidden_size))
        #        
        l_out_loop = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop])
        #
        l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val_hid_state, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val])
        #
        h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1)
        h_init_val = lasagne.layers.ConcatLayer([l_dec_val_hid_state, l_enc], axis=1)
    return (l_out, l_out_val)
    
    

def model_seq2seq_mvrt4_GRU(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, max_len, pred_len, hidden_size=64, grad_clip = 100):
    """
    Multivariate sequence to sequence without attention. GRU
    """
    #encoder
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym)
    l_mask_enc = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc)    
    #
    l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, only_return_final=True )
    l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc1 = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1)
    #2nd variable encoder
    l_in_enc2 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym2)
    l_mask_enc2 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc2)    
    #
    l_forward2 = lasagne.layers.GRULayer(l_in_enc2, num_units=hidden_size, mask_input=l_mask_enc2, grad_clipping=grad_clip, only_return_final=True )
    l_backward2 = lasagne.layers.GRULayer(l_in_enc2, num_units=hidden_size, mask_input=l_mask_enc2, grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc2 = lasagne.layers.ConcatLayer([l_forward2, l_backward2], axis=1)
    #3rd variable encoder
    l_in_enc3 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym3)
    l_mask_enc3 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc3)    
    #
    l_forward3 = lasagne.layers.GRULayer(l_in_enc3, num_units=hidden_size, mask_input=l_mask_enc3, grad_clipping=grad_clip, only_return_final=True )
    l_backward3 = lasagne.layers.GRULayer(l_in_enc3, num_units=hidden_size, mask_input=l_mask_enc3, grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc3 = lasagne.layers.ConcatLayer([l_forward3, l_backward3], axis=1)
    ##3rd variable encoder
    l_in_enc4 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym4)
    l_mask_enc4 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc4)    
    #
    l_forward4 = lasagne.layers.GRULayer(l_in_enc4, num_units=hidden_size, mask_input=l_mask_enc4, grad_clipping=grad_clip, only_return_final=True )
    l_backward4 = lasagne.layers.GRULayer(l_in_enc4, num_units=hidden_size, mask_input=l_mask_enc4, grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc4 = lasagne.layers.ConcatLayer([l_forward4, l_backward4], axis=1)
    #
    l_enc = lasagne.layers.ConcatLayer([l_enc1, l_enc2, l_enc3, l_enc4], axis=1) # 3*2 
    #
    dec_units = 4*2 +1 #nr_encoder *2 + dec -> bi-directional so *2 !!important to set it right
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len,1),input_var=X_dec_sym)
    l_mask_dec = lasagne.layers.InputLayer(shape=(None, pred_len), input_var=mask_dec)
    #    
    s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
    s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))
    s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    h_init = lasagne.layers.ConcatLayer([l_forward, l_enc], axis=1)
    #
    """l_dec = lasagne.layers.LSTMLayer(s_lin_dec, num_units=hidden_size*dec_units, mask_input=s_lmask_dec1, learn_init=False, 
                                     hid_init=h_init, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True )
    input_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_ingate, W_hid=l_dec.W_hid_to_ingate,                                
                                            W_cell= l_dec.W_cell_to_ingate, b=l_dec.b_ingate)
    output_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_outgate, W_hid=l_dec.W_hid_to_outgate, 
                                      W_cell=l_dec.W_cell_to_outgate, b=l_dec.b_outgate)
    forget_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_forgetgate, W_hid=l_dec.W_hid_to_forgetgate,
                                      W_cell=l_dec.W_cell_to_forgetgate, b=l_dec.b_forgetgate)
    cell_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_cell, W_hid=l_dec.W_hid_to_cell, W_cell=None,
                                    b=l_dec.b_cell, nonlinearity=lasagne.nonlinearities.tanh) """
    l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, mask_input=s_lmask_dec1, learn_init=False, 
                                         hid_init=h_init, grad_clipping=grad_clip, only_return_final=True )
    r_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_resetgate, W_hid=l_dec.W_hid_to_resetgate, b=l_dec.b_resetgate)
    u_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_updategate, W_hid=l_dec.W_hid_to_updategate, b=l_dec.b_updategate)
    h_update = lasagne.layers.Gate(W_in=l_dec.W_in_to_hidden_update, W_hid=l_dec.W_hid_to_hidden_update, b=l_dec.b_hidden_update)
                                    
    #
    l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size))
    l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=1, nonlinearity=lasagne.nonlinearities.linear)
    w_dense = l_out.W
    b_dense = l_out.b
    l_out_loop = l_out
    l_out_val = l_out
    l_out_loop_val = l_out    
    h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1)
    h_init_val = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1)
    for i in range(1,pred_len):
        s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1)
        s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))
        s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=i, axis=1) 
        s_lmask_dec = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1)) 
        l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, mask_input=s_lmask_dec, learn_init=False, 
                                         hid_init=h_init, grad_clipping=grad_clip, only_return_final=True, 
                                         resetgate=r_gate, updategate=u_gate, hidden_update=h_update) 
        l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size))
        #
        pred = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1]))
        l_dec_val = lasagne.layers.GRULayer(pred, num_units=hidden_size*dec_units, mask_input=s_lmask_dec, learn_init=False,
                                         hid_init=h_init_val, grad_clipping=grad_clip, only_return_final=True,
                                         resetgate=r_gate, updategate=u_gate, hidden_update=h_update) 
        l_dec_val_hid_state = lasagne.layers.SliceLayer(l_dec_val, indices=slice(0,hidden_size))
        #        
        l_out_loop = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop])
        #
        l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val_hid_state, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val])
        #
        h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1)
        h_init_val = lasagne.layers.ConcatLayer([l_dec_val_hid_state, l_enc], axis=1)
    return (l_out, l_out_val)    


def model_seq2seq_GRU_text(X_enc_sym, X_dec_sym, Emb_mtx_sym, max_len, pred_len, hidden_size=64, grad_clip = 100, vocab_size=155564, word_dim=200): ##def model_seq2seq_GRU_text(X_enc_sym, X_enc_sym2, X_dec_sym, max_len, pred_len, hidden_size=64, grad_clip = 100, vocab_size=155563, word_dim=200):#(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, max_len, pred_len, hidden_size=64, grad_clip = 100) # max_len=288 - windowise, pred_len=27 - horizon
    #copy of model_seq2seq_mvrt4_GRU
    """
    Multivariate sequence to sequence without attention. GRU with text
    """
    """
    # MODEL 1 --------------------------- with two metrics of one host concatenated - this model was supposed to concatenate all metrics of one host ---------------------------
    #encoder
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym)
    #l_mask_enc = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc)    
    #
    l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, grad_clipping=grad_clip, only_return_final=True )#l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, only_return_final=True )
    l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, grad_clipping=grad_clip, only_return_final=True, backwards=True)#l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc1 = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1)
    #2nd variable encoder
    l_in_enc2 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym2)
    #l_mask_enc2 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc2)    
    #
    l_forward2 = lasagne.layers.GRULayer(l_in_enc2, num_units=hidden_size, grad_clipping=grad_clip, only_return_final=True )#l_forward2 = lasagne.layers.GRULayer(l_in_enc2, num_units=hidden_size, mask_input=l_mask_enc2, grad_clipping=grad_clip, only_return_final=True )
    l_backward2 = lasagne.layers.GRULayer(l_in_enc2, num_units=hidden_size, grad_clipping=grad_clip, only_return_final=True, backwards=True)#l_backward2 = lasagne.layers.GRULayer(l_in_enc2, num_units=hidden_size, mask_input=l_mask_enc2, grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc2 = lasagne.layers.ConcatLayer([l_forward2, l_backward2], axis=1)
    #
    l_enc = lasagne.layers.ConcatLayer([l_enc1, l_enc2], axis=1) # 3*2 #l_enc = lasagne.layers.ConcatLayer([l_enc1, l_enc2, l_enc3, l_enc4], axis=1)
    """
    # MODEL 2 --------------------------- learn from one metric of each host only (see notes of meeting 27.04.17 and notes 03.05.17) -------------------------------------------
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym)
    #l_mask_enc = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc)    
    #
    l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                        resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                        updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                        hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                        grad_clipping=grad_clip, only_return_final=True) #l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, grad_clipping=grad_clip, only_return_final=True )#l_forward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, only_return_final=True )
    l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, 
                                        resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                        updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                        hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)),
                                        grad_clipping=grad_clip, only_return_final=True, backwards=True)#l_backward = lasagne.layers.GRULayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, only_return_final=True, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1)
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
    h_init = lasagne.layers.ConcatLayer([l_forward, l_enc], axis=1) #?????????????
    #
    """l_dec = lasagne.layers.LSTMLayer(s_lin_dec, num_units=hidden_size*dec_units, mask_input=s_lmask_dec1, learn_init=False, 
                                     hid_init=h_init, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True )
    input_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_ingate, W_hid=l_dec.W_hid_to_ingate,                                
                                            W_cell= l_dec.W_cell_to_ingate, b=l_dec.b_ingate)
    output_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_outgate, W_hid=l_dec.W_hid_to_outgate, 
                                      W_cell=l_dec.W_cell_to_outgate, b=l_dec.b_outgate)
    forget_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_forgetgate, W_hid=l_dec.W_hid_to_forgetgate,
                                      W_cell=l_dec.W_cell_to_forgetgate, b=l_dec.b_forgetgate)
    cell_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_cell, W_hid=l_dec.W_hid_to_cell, W_cell=None,
                                    b=l_dec.b_cell, nonlinearity=lasagne.nonlinearities.tanh) """
    l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, 
                                    resetgate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    updategate=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)), 
                                    hidden_update=lasagne.layers.Gate(W_in=lasagne.init.Uniform(range=1.), W_hid=lasagne.init.Uniform(range=1.), W_cell=lasagne.init.Uniform(range=1.)),
                                    learn_init=False, hid_init=h_init, grad_clipping=grad_clip, only_return_final=True ) #l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, mask_input=s_lmask_dec1, learn_init=False, 
                                         #hid_init=h_init, grad_clipping=grad_clip, only_return_final=True )
    r_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_resetgate, W_hid=l_dec.W_hid_to_resetgate, b=l_dec.b_resetgate)
    u_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_updategate, W_hid=l_dec.W_hid_to_updategate, b=l_dec.b_updategate)
    h_update = lasagne.layers.Gate(W_in=l_dec.W_in_to_hidden_update, W_hid=l_dec.W_hid_to_hidden_update, b=l_dec.b_hidden_update)
                                    
    #
    l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size))
    #TO CHANGE BACK BELOW
    l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, nonlinearity=lasagne.nonlinearities.softmax)  #l_out = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=1, nonlinearity=lasagne.nonlinearities.linear)
    """w_dense = l_out.W
    b_dense = l_out.b
    l_out_loop = l_out
    l_out_val = l_out
    l_out_loop_val = l_out    
    h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1)
    h_init_val = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1)
    for i in range(1,pred_len): 
        s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1)
        s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))  #SHOULD WE RESHAPE IT???
        #s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=i, axis=1) 
        #s_lmask_dec = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1)) 
        l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, learn_init=False, 
                                         hid_init=h_init, grad_clipping=grad_clip, only_return_final=True, 
                                         resetgate=r_gate, updategate=u_gate, hidden_update=h_update) #l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, mask_input=s_lmask_dec, learn_init=False, 
                                         #hid_init=h_init, grad_clipping=grad_clip, only_return_final=True, 
                                         #resetgate=r_gate, updategate=u_gate, hidden_update=h_update) 
        l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size))
        #
        pred = l_out_loop_val #pred = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1]))    #SHOULD WE RESHAPE IT???
        l_dec_val = lasagne.layers.GRULayer(pred, num_units=hidden_size*dec_units, learn_init=False,
                                         hid_init=h_init_val, grad_clipping=grad_clip, only_return_final=True,
                                         resetgate=r_gate, updategate=u_gate, hidden_update=h_update) #l_dec_val = lasagne.layers.GRULayer(pred, num_units=hidden_size*dec_units, mask_input=s_lmask_dec, learn_init=False,
                                         #hid_init=h_init_val, grad_clipping=grad_clip, only_return_final=True,
                                         #resetgate=r_gate, updategate=u_gate, hidden_update=h_update) 
        l_dec_val_hid_state = lasagne.layers.SliceLayer(l_dec_val, indices=slice(0,hidden_size))
        #        
        l_out_loop = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax) #l_out_loop = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        
        l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop])
        #
        l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax)  #l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val_hid_state, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val])
        #
        h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1)
        h_init_val = lasagne.layers.ConcatLayer([l_dec_val_hid_state, l_enc], axis=1)"""
    w_dense = l_out.W
    b_dense = l_out.b
    l_out_loop = l_out
    l_out_loop_val = l_out  
    l_out = lasagne.layers.ReshapeLayer(l_out, ([0], 1, [1]))
    l_out_val = l_out
    h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1)
    h_init_val = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1)

    for i in range(1,pred_len): #comments in this cycle are for the first iteration
        s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1) #(None, 200)
        s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1])) #(None, 1, 200)
        l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, learn_init=False,	 #(None, 320)
                                            hid_init=h_init, grad_clipping=grad_clip, only_return_final=True, 
                                            resetgate=r_gate, updategate=u_gate, hidden_update=h_update)
        l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size)) #(None, 64)
        
        """
        #------ Here is no argmax for validation(->forecast) parts in network, but just mapping voc_size to word_dim for val set ---------------------------
        #pred = l_out_loop_val
        #Embedding_Matrix = T.matrix()
        #pred = T.vector()
        #pred_embedding = T.set_subtensor(Embedding_matrix, T.argmax(pred, axis=???))
        #f_pred_embed = theano.function([Embedding_Matric, pred], pred_embedding)
        #numpy_pred_embedding = f_pred_embed(numpy_embedding, numpy_pred)
        
        pred_d = lasagne.layers.DenseLayer(l_out_loop_val, num_units=word_dim, nonlinearity=lasagne.nonlinearities.linear)
    	pred = lasagne.layers.ReshapeLayer(pred_d, ([0], 1, [1])) #(None, 1, 155563)
    	#ind_word = T.argmax(pred, axis=0)
    	l_dec_val = lasagne.layers.GRULayer(pred, num_units=hidden_size*dec_units, learn_init=False,    #(None, 320)
                                             hid_init=h_init_val, grad_clipping=grad_clip, only_return_final=True,
                                             resetgate=r_gate, updategate=u_gate, hidden_update=h_update)
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
        """
        
        #------ ARGMAX for validation(->forecast) parts in network via embedding matrix 18.05 ---------------------------
    	#for val and train sets
    	coord_max = ExpressionLayer(l_out_loop_val, lambda x: x.argmax(-1).astype('int32'), output_shape='auto') #(None,)
    	emb_slice = ExpressionLayer(coord_max, lambda x: Emb_mtx_sym[x,:], output_shape=(None,word_dim)) #(None, 200)
	
    	#coord_max = ExpressionLayer(l_out_loop_val, lambda X: X.argmax(-1).astype('int32'), output_shape='auto') #(None,)
    	#pred_d = ExpressionLayer(coord_max, lambda X: T.set_subtensor(T.zeros((1,word_dim))[0,X], 1), output_shape='auto')
    	#pred_emb = lasagne.layers.DenseLayer(pred_d, num_units=word_dim, W=Emb_mtx_sym, nonlinearity=lasagne.nonlinearities.linear) #no biases #(None, 200)
    	#pred_emb.params[pred_emb.W].remove("trainable")
        
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
        
        
        """l_dec = lasagne.layers.GRULayer(s_lin_dec, num_units=hidden_size*dec_units, learn_init=False,
                                            hid_init=h_init, grad_clipping=grad_clip, only_return_final=True,
                                            resetgate=r_gate, updategate=u_gate, hidden_update=h_update)
    	l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size))
    	pred = l_out_loop_val
    	l_dec_val = lasagne.layers.GRULayer(pred, num_units=hidden_size*dec_units, learn_init=False,
                                            hid_init=h_init_val, grad_clipping=grad_clip, only_return_final=True,
                                            resetgate=r_gate, updategate=u_gate, hidden_update=h_update)
    	l_dec_val_hid_state = lasagne.layers.SliceLayer(l_dec_val, indices=slice(0,hidden_size))
    	l_out_loop = lasagne.layers.DenseLayer(l_dec_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax)
    	l_out_loop = lasagne.layers.ReshapeLayer(l_out_loop, ([0], 1, [1])) #######
    	l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop], axis=1)
    	l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val_hid_state, num_units=vocab_size, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.softmax)
    	l_out_loop_val = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1]))
    	l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val], axis=1)
    	l_out_loop_val = lasagne.layers.FlattenLayer(l_out_loop_val,  outdim=2)
    	h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1)
    	h_init_val = lasagne.layers.ConcatLayer([l_dec_val_hid_state, l_enc], axis=1)"""
    #l_out = lasagne.layers.ReshapeLayer(l_out, (-1, hidden_size))
    #l_out_val = lasagne.layers.ReshapeLayer(l_out_val, (-1, hidden_size))
    return (l_out, l_out_val) 
    
    
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
    # 
    # Set-encoder part
    #l_q = lasagne.init.Constant(0.)
    #for i in range(set_steps):
    l_setenc = GRULayer_setenc(incoming=l_enc_conc, num_units=hidden_size, learn_init=False, set_steps=set_steps, att_num_units=att_size, grad_clipping=grad_clip, 
                                nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True) 
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
    	h_init = lasagne.layers.ConcatLayer([l_dec_hid_state, l_enc], axis=1) #(None, 320)
    	h_init_val = lasagne.layers.ConcatLayer([l_dec_val_hid_state, l_enc], axis=1) #(None, 320)
    #l_out = lasagne.layers.ReshapeLayer(l_out, (-1, hidden_size))
    #l_out_val = lasagne.layers.ReshapeLayer(l_out_val, (-1, hidden_size))
    return (l_out, l_out_val) 


def model_seq2seq_att_mvrt4(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, max_len, pred_len, grad_clip = 100, hidden_size=64, att_size =64):
    """
    Multivariate sequence to sequence with attention 
    """
    #encoder
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1),input_var=X_enc_sym)
    l_mask_enc = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc)    
    #
    l_forward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc1 = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)
    #2nd variable encoder
    l_in_enc2 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym2)
    l_mask_enc2 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc2)    
    #
    l_forward2 = lasagne.layers.LSTMLayer(l_in_enc2, num_units=hidden_size, mask_input=l_mask_enc2, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward2 = lasagne.layers.LSTMLayer(l_in_enc2, num_units=hidden_size, mask_input=l_mask_enc2, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc2 = lasagne.layers.ConcatLayer([l_forward2, l_backward2], axis=2)
    #3rd variable encoder
    l_in_enc3 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym3)
    l_mask_enc3 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc3)    
    #
    l_forward3 = lasagne.layers.LSTMLayer(l_in_enc3, num_units=hidden_size, mask_input=l_mask_enc3, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward3 = lasagne.layers.LSTMLayer(l_in_enc3, num_units=hidden_size, mask_input=l_mask_enc3, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc3 = lasagne.layers.ConcatLayer([l_forward3, l_backward3], axis=2)
    ##4th variable encoder
    l_in_enc4 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym4)
    l_mask_enc4 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc4)    
    #
    l_forward4 = lasagne.layers.LSTMLayer(l_in_enc4, num_units=hidden_size, mask_input=l_mask_enc4, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward4 = lasagne.layers.LSTMLayer(l_in_enc4, num_units=hidden_size, mask_input=l_mask_enc4, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc4 = lasagne.layers.ConcatLayer([l_forward4, l_backward4], axis=2)
    #
    #l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)
    l_enc = lasagne.layers.ConcatLayer([l_enc1, l_enc2, l_enc3, l_enc4], axis=2)
    #
    l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1)   
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len,1),input_var=X_dec_sym)
    l_mask_dec = lasagne.layers.InputLayer(shape=(None, pred_len), input_var=mask_dec)
    #    
    s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
    s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))
    s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    h_init = l_forward_slice#l_enc
    c_init =  lasagne.layers.ExpressionLayer(l_enc, lambda X: X.mean(1), output_shape='auto')
    #
    l_dec = LSTMAttLayer4(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec1, learn_init=False, 
                                     hid_init=h_init, contxt_input=l_enc1, contxt_input2=l_enc2, contxt_input3=l_enc3, contxt_input4=l_enc4,
                                     ctx_init=c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True )
    input_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_ingate, W_hid=l_dec.W_hid_to_ingate, 
                                            W_cell= l_dec.W_cell_to_ingate, b=l_dec.b_ingate)
    output_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_outgate, W_hid=l_dec.W_hid_to_outgate, 
                                      W_cell=l_dec.W_cell_to_outgate, b=l_dec.b_outgate)
    forget_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_forgetgate, W_hid=l_dec.W_hid_to_forgetgate,
                                      W_cell=l_dec.W_cell_to_forgetgate, b=l_dec.b_forgetgate)
    cell_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_cell, W_hid=l_dec.W_hid_to_cell, W_cell=None,
                                    b=l_dec.b_cell, nonlinearity=lasagne.nonlinearities.tanh) 
    #
    w_ctx2in = l_dec.W_ctx_to_ingate
    w_ctx2forget = l_dec.W_ctx_to_forgetgate
    w_ctx2cell = l_dec.W_ctx_to_cell
    w_ctx2out = l_dec.W_ctx_to_outgate    
    w_att, w_att2, w_att3, w_att4 = l_dec.W_att, l_dec.W_att2, l_dec.W_att3, l_dec.W_att4
    w_hid2att, w_hid2att2, w_hid2att3, w_hid2att4 = l_dec.W_hid_to_att, l_dec.W_hid_to_att2, l_dec.W_hid_to_att3, l_dec.W_hid_to_att4
    w_ctx2att = l_dec.W_ctx_to_att
    w_ctx2att2 = l_dec.W_ctx_to_att2
    w_ctx2att3 = l_dec.W_ctx_to_att3
    w_ctx2att4 = l_dec.W_ctx_to_att4
    alphas = AnaLayer4(h_init, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc1, contxt_input2=l_enc2, 
                      contxt_input3=l_enc3, contxt_input4=l_enc4, W_att=w_att, W_att2=w_att2, W_att3=w_att3,  
                      W_att4=w_att4, W_hid_to_att=w_hid2att, W_hid_to_att2=w_hid2att2, W_hid_to_att3=w_hid2att3, 
                      W_hid_to_att4=w_hid2att4, W_ctx_to_att=w_ctx2att,
                      W_ctx_to_att2=w_ctx2att2, W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4)
    #
    l_out = lasagne.layers.DenseLayer(l_dec, num_units=1, nonlinearity=lasagne.nonlinearities.linear)  
    w_dense = l_out.W
    b_dense = l_out.b
    l_out_loop = l_out
    l_out_val = l_out
    l_out_loop_val = l_out
    h_init = l_dec
    h_init_val = l_dec
    for i in range(1,pred_len):
        s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1)
        s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))
        s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=i, axis=1) 
        s_lmask_dec = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1)) 
        l_dec = LSTMAttLayer4(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec, learn_init=False, 
                                         hid_init=h_init, contxt_input=l_enc1, contxt_input2=l_enc2, contxt_input3=l_enc3, contxt_input4=l_enc4, 
                                         ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, 
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_att2=w_att2,  W_att3=w_att3, W_att4=w_att4, 
                             W_hid_to_att=w_hid2att, W_hid_to_att2=w_hid2att2, W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4,
                             W_ctx_to_att=w_ctx2att, W_ctx_to_att2=w_ctx2att2, W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4) 
        #
        pred = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1]))
        l_dec_val = LSTMAttLayer4(pred, num_units=hidden_size, mask_input=s_lmask_dec1, learn_init=False,
                                         hid_init=h_init_val, contxt_input=l_enc1, contxt_input2=l_enc2, contxt_input3=l_enc3, contxt_input4=l_enc4,
                                         ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True,
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_att2=w_att2,  W_att3=w_att3, W_att4=w_att4, 
                             W_hid_to_att=w_hid2att, W_hid_to_att2=w_hid2att2, W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4, 
                             W_ctx_to_att=w_ctx2att, W_ctx_to_att2=w_ctx2att2, W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4)
        #
        l_out_loop = lasagne.layers.DenseLayer(l_dec, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop])
        #
        l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val])
        #
        alphas_loop = AnaLayer4(h_init_val, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc1, 
                                contxt_input2=l_enc2, contxt_input3=l_enc3, contxt_input4=l_enc4,
                               W_att=w_att, W_att2=w_att2, W_att3=w_att3, W_att4=w_att4, 
                               W_hid_to_att=w_hid2att, W_hid_to_att2=w_hid2att2, W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4, 
                               W_ctx_to_att=w_ctx2att, W_ctx_to_att2=w_ctx2att2, W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4)
        alphas = lasagne.layers.ConcatLayer([alphas, alphas_loop])
        h_init = l_dec
        h_init_val = l_dec_val
    return (l_out, l_out_val, alphas)


def model_seq2seq_att_lambda_mvrt4(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, max_len, pred_len, grad_clip = 100, hidden_size=64, att_size =64):
    """
    sequence to sequence with attention 
    """
    #encoder
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1),input_var=X_enc_sym)
    l_mask_enc = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc)    
    #
    l_forward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc1 = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)
    #2nd variable encoder
    l_in_enc2 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym2)
    l_mask_enc2 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc2)    
    #
    l_forward2 = lasagne.layers.LSTMLayer(l_in_enc2, num_units=hidden_size, mask_input=l_mask_enc2, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward2 = lasagne.layers.LSTMLayer(l_in_enc2, num_units=hidden_size, mask_input=l_mask_enc2, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc2 = lasagne.layers.ConcatLayer([l_forward2, l_backward2], axis=2)
    #3rd variable encoder
    l_in_enc3 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym3)
    l_mask_enc3 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc3)    
    #
    l_forward3 = lasagne.layers.LSTMLayer(l_in_enc3, num_units=hidden_size, mask_input=l_mask_enc3, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward3 = lasagne.layers.LSTMLayer(l_in_enc3, num_units=hidden_size, mask_input=l_mask_enc3, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc3 = lasagne.layers.ConcatLayer([l_forward3, l_backward3], axis=2)
    ##4th variable encoder
    l_in_enc4 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym4)
    l_mask_enc4 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc4)    
    #
    l_forward4 = lasagne.layers.LSTMLayer(l_in_enc4, num_units=hidden_size, mask_input=l_mask_enc4, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward4 = lasagne.layers.LSTMLayer(l_in_enc4, num_units=hidden_size, mask_input=l_mask_enc4, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc4 = lasagne.layers.ConcatLayer([l_forward4, l_backward4], axis=2)
    #
    l_enc = lasagne.layers.ConcatLayer([l_enc1, l_enc2, l_enc3, l_enc4], axis=2)
    #
    l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1)   
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len,1),input_var=X_dec_sym)
    l_mask_dec = lasagne.layers.InputLayer(shape=(None, pred_len), input_var=mask_dec)
    #    
    s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
    s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))
    s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    h_init = l_forward_slice#l_enc
    c_init =  lasagne.layers.ExpressionLayer(l_enc, lambda X: X.mean(1), output_shape='auto')
    #
    l_dec = LSTMAttLayer_lambda4(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec1, learn_init=False, pred_ind=0, ws=max_len, pred_len=pred_len,
                                     hid_init=h_init, contxt_input=l_enc1, contxt_input2=l_enc2, contxt_input3=l_enc3, contxt_input4=l_enc4, 
                                     ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True )
    input_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_ingate, W_hid=l_dec.W_hid_to_ingate, 
                                            W_cell= l_dec.W_cell_to_ingate, b=l_dec.b_ingate)
    output_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_outgate, W_hid=l_dec.W_hid_to_outgate, 
                                      W_cell=l_dec.W_cell_to_outgate, b=l_dec.b_outgate)
    forget_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_forgetgate, W_hid=l_dec.W_hid_to_forgetgate,
                                      W_cell=l_dec.W_cell_to_forgetgate, b=l_dec.b_forgetgate)
    cell_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_cell, W_hid=l_dec.W_hid_to_cell, W_cell=None,
                                    b=l_dec.b_cell, nonlinearity=lasagne.nonlinearities.tanh) 
    #
    w_ctx2in = l_dec.W_ctx_to_ingate
    w_ctx2forget = l_dec.W_ctx_to_forgetgate
    w_ctx2cell = l_dec.W_ctx_to_cell
    w_ctx2out = l_dec.W_ctx_to_outgate
    #
    w_att, w_att2, w_att3, w_att4 = l_dec.W_att, l_dec.W_att2, l_dec.W_att3, l_dec.W_att4
    w_hid2att, w_hid2att2, w_hid2att3, w_hid2att4 = l_dec.W_hid_to_att, l_dec.W_hid_to_att2, l_dec.W_hid_to_att3, l_dec.W_hid_to_att4
    #
    w_ctx2att = l_dec.W_ctx_to_att
    w_ctx2att2 = l_dec.W_ctx_to_att2
    w_ctx2att3 = l_dec.W_ctx_to_att3
    w_ctx2att4 = l_dec.W_ctx_to_att4
    #
    w_lambda = l_dec.W_lambda
    w_lambda2 = l_dec.W_lambda2
    w_lambda3 = l_dec.W_lambda3
    w_lambda4 = l_dec.W_lambda4
    alphas = AnaLayer_lambda4(h_init, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc1, contxt_input2=l_enc2, 
                              contxt_input3=l_enc3, contxt_input4=l_enc4, pred_ind=0, ws=max_len, pred_len=pred_len,
                      W_att=w_att, W_att2=w_att2, W_att3=w_att3, W_att4=w_att4, W_hid_to_att=w_hid2att,
                      W_hid_to_att2=w_hid2att2, W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4,  
                      W_ctx_to_att=w_ctx2att, W_ctx_to_att2=w_ctx2att2, W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4,
                      W_lambda=w_lambda, W_lambda2=w_lambda2, W_lambda3=w_lambda3, W_lambda4=w_lambda4)
    #
    l_out = lasagne.layers.DenseLayer(l_dec, num_units=1, nonlinearity=lasagne.nonlinearities.linear)  
    w_dense = l_out.W
    b_dense = l_out.b
    l_out_loop = l_out
    l_out_val = l_out
    l_out_loop_val = l_out
    h_init = l_dec
    h_init_val = l_dec
    for i in range(1,pred_len):
        s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1)
        s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))
        s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=i, axis=1) 
        s_lmask_dec = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1)) 
        l_dec = LSTMAttLayer_lambda4(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec, learn_init=False, pred_ind=i, ws=max_len, pred_len=pred_len,
                                         hid_init=h_init, contxt_input=l_enc1, contxt_input2=l_enc2, contxt_input3=l_enc3, contxt_input4=l_enc4, 
                                         ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, 
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_att2=w_att2, W_att3=w_att3, W_att4=w_att4, 
                             W_hid_to_att=w_hid2att, W_hid_to_att2=w_hid2att2, W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4, 
                             W_ctx_to_att=w_ctx2att, W_ctx_to_att2=w_ctx2att2, W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4,
                             W_lambda=w_lambda, W_lambda2=w_lambda2, W_lambda3=w_lambda3, W_lambda4=w_lambda4) 
        #
        pred = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1]))
        l_dec_val = LSTMAttLayer_lambda4(pred, num_units=hidden_size, mask_input=s_lmask_dec1, learn_init=False, pred_ind=i, ws=max_len, pred_len=pred_len,
                                         hid_init=h_init_val, contxt_input=l_enc1, contxt_input2=l_enc2, contxt_input3=l_enc3, contxt_input4=l_enc4, 
                                         ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True,
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_att2=w_att2, W_att3=w_att3, W_att4=w_att4, 
                             W_hid_to_att=w_hid2att, W_hid_to_att2=w_hid2att2, W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4,
                             W_ctx_to_att=w_ctx2att, W_ctx_to_att2=w_ctx2att2, W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4, W_lambda=w_lambda, 
                             W_lambda2=w_lambda2, W_lambda3=w_lambda3, W_lambda4=w_lambda4)
        #
        l_out_loop = lasagne.layers.DenseLayer(l_dec, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop])
        #
        l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val])
        #
        alphas_loop = AnaLayer_lambda4(h_init_val, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc1, 
                                       contxt_input2=l_enc2, contxt_input3=l_enc3, contxt_input4=l_enc4, pred_ind=0, 
                                       ws=max_len, pred_len=pred_len, W_att=w_att, W_att2=w_att2, W_att3=w_att3, W_att4=w_att4,
                                       W_hid_to_att=w_hid2att, W_hid_to_att2=w_hid2att2, W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4,
                                       W_ctx_to_att=w_ctx2att, W_ctx_to_att2=w_ctx2att2, W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4, 
                                       W_lambda=w_lambda, W_lambda2=w_lambda2, W_lambda3=w_lambda3, W_lambda4=w_lambda4)
        alphas = lasagne.layers.ConcatLayer([alphas, alphas_loop])
        h_init = l_dec
        h_init_val = l_dec_val
    return (l_out, l_out_val, alphas)


def model_seq2seq_att_lambda_mu_mvrt4(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4,
                                      X_dec_sym, mask_dec, delta_inds_sym, delta_inds_sym2, delta_inds_sym3, delta_inds_sym4,
                                      max_len, pred_len, grad_clip = 100, hidden_size=64, att_size =64):
    """
    sequence to sequence with attention 
    """
    #encoder
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1),input_var=X_enc_sym)
    l_mask_enc = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc)  
    l_in_delta_inds = lasagne.layers.InputLayer(shape=(None,), input_var=delta_inds_sym, dtype='int32')
    #
    l_forward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc1 = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)
    #2nd variable encoder
    l_in_enc2 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym2)
    l_mask_enc2 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc2) 
    l_in_delta_inds2 = lasagne.layers.InputLayer(shape=(None,), input_var=delta_inds_sym2, dtype='int32')
    #
    l_forward2 = lasagne.layers.LSTMLayer(l_in_enc2, num_units=hidden_size, mask_input=l_mask_enc2, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward2 = lasagne.layers.LSTMLayer(l_in_enc2, num_units=hidden_size, mask_input=l_mask_enc2, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc2 = lasagne.layers.ConcatLayer([l_forward2, l_backward2], axis=2)
    #3rd variable encoder
    l_in_enc3 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym3)
    l_mask_enc3 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc3)    
    l_in_delta_inds3 = lasagne.layers.InputLayer(shape=(None,), input_var=delta_inds_sym3, dtype='int32')
    #
    l_forward3 = lasagne.layers.LSTMLayer(l_in_enc3, num_units=hidden_size, mask_input=l_mask_enc3, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward3 = lasagne.layers.LSTMLayer(l_in_enc3, num_units=hidden_size, mask_input=l_mask_enc3, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc3 = lasagne.layers.ConcatLayer([l_forward3, l_backward3], axis=2)
    ##4th variable encoder
    l_in_enc4 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym4)
    l_mask_enc4 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc4)
    l_in_delta_inds4 = lasagne.layers.InputLayer(shape=(None,), input_var=delta_inds_sym4, dtype='int32')
    #
    l_forward4 = lasagne.layers.LSTMLayer(l_in_enc4, num_units=hidden_size, mask_input=l_mask_enc4, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward4 = lasagne.layers.LSTMLayer(l_in_enc4, num_units=hidden_size, mask_input=l_mask_enc4, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc4 = lasagne.layers.ConcatLayer([l_forward4, l_backward4], axis=2)
    #
    l_enc = lasagne.layers.ConcatLayer([l_enc1, l_enc2, l_enc3, l_enc4], axis=2)
    #
    l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1)   
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len,1),input_var=X_dec_sym)
    l_mask_dec = lasagne.layers.InputLayer(shape=(None, pred_len), input_var=mask_dec)
    #    
    s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
    s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))
    s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    h_init = l_forward_slice#l_enc
    c_init =  lasagne.layers.ExpressionLayer(l_enc, lambda X: X.mean(1), output_shape='auto')
    #
    l_dec = LSTMAttLayer_lambda_mu4(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec1, enc_mask_input=l_mask_enc, 
                                   enc_mask_input2=l_mask_enc2, enc_mask_input3=l_mask_enc3, enc_mask_input4=l_mask_enc4,
                                   delta_inds_input=l_in_delta_inds, delta_inds_input2=l_in_delta_inds2, delta_inds_input3=l_in_delta_inds3, delta_inds_input4=l_in_delta_inds4,
                                   learn_init=False, pred_ind=0, ws=max_len, pred_len=pred_len,
                                     hid_init=h_init, contxt_input=l_enc1, contxt_input2=l_enc2, contxt_input3=l_enc3, 
                                     contxt_input4=l_enc4, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True )
    input_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_ingate, W_hid=l_dec.W_hid_to_ingate, 
                                            W_cell= l_dec.W_cell_to_ingate, b=l_dec.b_ingate)
    output_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_outgate, W_hid=l_dec.W_hid_to_outgate, 
                                      W_cell=l_dec.W_cell_to_outgate, b=l_dec.b_outgate)
    forget_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_forgetgate, W_hid=l_dec.W_hid_to_forgetgate,
                                      W_cell=l_dec.W_cell_to_forgetgate, b=l_dec.b_forgetgate)
    cell_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_cell, W_hid=l_dec.W_hid_to_cell, W_cell=None,
                                    b=l_dec.b_cell, nonlinearity=lasagne.nonlinearities.tanh) 
    #
    w_ctx2in = l_dec.W_ctx_to_ingate
    w_ctx2forget = l_dec.W_ctx_to_forgetgate
    w_ctx2cell = l_dec.W_ctx_to_cell
    w_ctx2out = l_dec.W_ctx_to_outgate    
    #
    #
    w_att, w_att2, w_att3, w_att4 = l_dec.W_att, l_dec.W_att2, l_dec.W_att3, l_dec.W_att4
    w_hid2att, w_hid2att2, w_hid2att3, w_hid2att4 = l_dec.W_hid_to_att, l_dec.W_hid_to_att2, l_dec.W_hid_to_att3, l_dec.W_hid_to_att4
    #
    w_ctx2att = l_dec.W_ctx_to_att
    w_ctx2att2 = l_dec.W_ctx_to_att2
    w_ctx2att3 = l_dec.W_ctx_to_att3
    w_ctx2att4 = l_dec.W_ctx_to_att4
    #
    w_lambda = l_dec.W_lambda
    w_lambda2 = l_dec.W_lambda2
    w_lambda3 = l_dec.W_lambda3
    w_lambda4 = l_dec.W_lambda4
    #
    w_mu, w_mu2, w_mu3, w_mu4 = l_dec.W_mu, l_dec.W_mu2, l_dec.W_mu3, l_dec.W_mu4
    #
    alphas = AnaLayer_lambda_mu4(h_init, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc1, contxt_input2=l_enc2,
                      contxt_input3=l_enc3, contxt_input4=l_enc4, enc_mask_input=l_mask_enc, enc_mask_input2=l_mask_enc2, 
                      enc_mask_input3=l_mask_enc3, enc_mask_input4=l_mask_enc4, delta_inds_input=l_in_delta_inds, 
                      delta_inds_input2=l_in_delta_inds2, delta_inds_input3=l_in_delta_inds3, delta_inds_input4=l_in_delta_inds4,
                      pred_ind=0, ws=max_len, pred_len=pred_len,
                      W_att=w_att, W_att2=w_att2, W_att3=w_att3, W_att4=w_att4, W_hid_to_att=w_hid2att, W_hid_to_att2=w_hid2att2, 
                      W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4, W_ctx_to_att=w_ctx2att, W_ctx_to_att2=w_ctx2att2,
                      W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4, W_lambda=w_lambda, W_lambda2=w_lambda2, W_lambda3=w_lambda3,
                      W_lambda4=w_lambda4, W_mu=w_mu, W_mu2=w_mu2, W_mu3=w_mu3, W_mu4=w_mu4)
    #
    l_out = lasagne.layers.DenseLayer(l_dec, num_units=1, nonlinearity=lasagne.nonlinearities.linear)  
    w_dense = l_out.W
    b_dense = l_out.b
    l_out_loop = l_out
    l_out_val = l_out
    l_out_loop_val = l_out
    h_init = l_dec
    h_init_val = l_dec
    for i in range(1,pred_len):
        s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1)
        s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))
        s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=i, axis=1) 
        s_lmask_dec = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1)) 
        l_dec = LSTMAttLayer_lambda_mu4(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec, enc_mask_input=l_mask_enc,
                            enc_mask_input2=l_mask_enc2, enc_mask_input3=l_mask_enc3, enc_mask_input4=l_mask_enc4, 
                            delta_inds_input=l_in_delta_inds, delta_inds_input2=l_in_delta_inds2, delta_inds_input3=l_in_delta_inds3,
                            delta_inds_input4=l_in_delta_inds4, learn_init=False, pred_ind=i, ws=max_len, pred_len=pred_len,
                                         hid_init=h_init, contxt_input=l_enc1, contxt_input2=l_enc2, contxt_input3=l_enc3, 
                            contxt_input4=l_enc4, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                            nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, 
                            ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                            W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_att2=w_att2, W_att3=w_att3, W_att4=w_att4, W_hid_to_att=w_hid2att,
                            W_hid_to_att2=w_hid2att2, W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4, W_ctx_to_att=w_ctx2att,
                            W_ctx_to_att2=w_ctx2att2, W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4, W_lambda=w_lambda, 
                            W_lambda2=w_lambda2, W_lambda3=w_lambda3, W_lambda4=w_lambda4, W_mu=w_mu, W_mu2=w_mu2, W_mu3=w_mu3, W_mu4=w_mu4) 
        #
        pred = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1]))
        l_dec_val = LSTMAttLayer_lambda_mu4(pred, num_units=hidden_size, mask_input=s_lmask_dec1, enc_mask_input=l_mask_enc,
                            enc_mask_input2=l_mask_enc2, enc_mask_input3=l_mask_enc3, enc_mask_input4=l_mask_enc4,
                            delta_inds_input=l_in_delta_inds, delta_inds_input2=l_in_delta_inds2, delta_inds_input3=l_in_delta_inds3,
                            delta_inds_input4=l_in_delta_inds4, learn_init=False, pred_ind=i, ws=max_len, pred_len=pred_len,
                            hid_init=h_init_val, contxt_input=l_enc1, contxt_input2=l_enc2, contxt_input3=l_enc3, contxt_input4=l_enc4, 
                            ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True,
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_att2=w_att2, W_att3=w_att3, W_att4=w_att4, W_hid_to_att=w_hid2att,
                             W_hid_to_att2=w_hid2att2, W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4, W_ctx_to_att=w_ctx2att,
                             W_ctx_to_att2=w_ctx2att2, W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4, W_lambda=w_lambda,
                             W_lambda2=w_lambda2, W_lambda3=w_lambda3, W_lambda4=w_lambda4, W_mu=w_mu, W_mu2=w_mu2, W_mu3=w_mu3, W_mu4=w_mu4)
        #
        l_out_loop = lasagne.layers.DenseLayer(l_dec, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop])
        #
        l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val])
        #
        alphas_loop = AnaLayer_lambda_mu4(h_init_val, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc1, contxt_input2=l_enc2,
                               contxt_input3=l_enc3, contxt_input4=l_enc4, enc_mask_input=l_mask_enc, enc_mask_input2=l_mask_enc2,
                               enc_mask_input3=l_mask_enc3, enc_mask_input4=l_mask_enc4, delta_inds_input=l_in_delta_inds,
                               delta_inds_input2=l_in_delta_inds2, delta_inds_input3=l_in_delta_inds3, delta_inds_input4=l_in_delta_inds4, 
                               pred_ind=0, ws=max_len, pred_len=pred_len,
                               W_att=w_att, W_att2=w_att2, W_att3=w_att3, W_att4=w_att4, W_hid_to_att=w_hid2att, W_hid_to_att2=w_hid2att2,
                               W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4, W_ctx_to_att=w_ctx2att, W_ctx_to_att2=w_ctx2att2, 
                               W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4, W_lambda=w_lambda, W_lambda2=w_lambda2,
                               W_lambda3=w_lambda3, W_lambda4=w_lambda4, W_mu=w_mu, W_mu2=w_mu2, W_mu3=w_mu3, W_mu4=w_mu4)
        alphas = lasagne.layers.ConcatLayer([alphas, alphas_loop])
        h_init = l_dec
        h_init_val = l_dec_val
    return (l_out, l_out_val, alphas)


def model_seq2seq_att_lambda_mu_mvrt4_intr(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4,
                                      X_dec_sym, mask_dec, delta_mask, delta_mask2, delta_mask3, delta_mask4,
                                      delta_inds_sym, delta_inds_sym2, delta_inds_sym3, delta_inds_sym4,
                                      max_len, pred_len, grad_clip = 100, hidden_size=64, att_size =64):
    """
    sequence to sequence with attention 
    """
    #encoder
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1),input_var=X_enc_sym)
    l_mask_enc = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc)  
    l_in_delta_inds = lasagne.layers.InputLayer(shape=(None,), input_var=delta_inds_sym, dtype='int32')
    l_in_delta_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=delta_mask)
    #
    l_forward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc1 = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)
    #2nd variable encoder
    l_in_enc2 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym2)
    l_mask_enc2 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc2) 
    l_in_delta_inds2 = lasagne.layers.InputLayer(shape=(None,), input_var=delta_inds_sym2, dtype='int32')
    l_in_delta_mask2 = lasagne.layers.InputLayer(shape=(None, None), input_var=delta_mask2)
    #
    l_forward2 = lasagne.layers.LSTMLayer(l_in_enc2, num_units=hidden_size, mask_input=l_mask_enc2, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward2 = lasagne.layers.LSTMLayer(l_in_enc2, num_units=hidden_size, mask_input=l_mask_enc2, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc2 = lasagne.layers.ConcatLayer([l_forward2, l_backward2], axis=2)
    #3rd variable encoder
    l_in_enc3 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym3)
    l_mask_enc3 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc3)    
    l_in_delta_inds3 = lasagne.layers.InputLayer(shape=(None,), input_var=delta_inds_sym3, dtype='int32')
    l_in_delta_mask3 = lasagne.layers.InputLayer(shape=(None, None), input_var=delta_mask3)
    #
    l_forward3 = lasagne.layers.LSTMLayer(l_in_enc3, num_units=hidden_size, mask_input=l_mask_enc3, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward3 = lasagne.layers.LSTMLayer(l_in_enc3, num_units=hidden_size, mask_input=l_mask_enc3, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc3 = lasagne.layers.ConcatLayer([l_forward3, l_backward3], axis=2)
    ##4th variable encoder
    l_in_enc4 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym4)
    l_mask_enc4 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc4)
    l_in_delta_inds4 = lasagne.layers.InputLayer(shape=(None,), input_var=delta_inds_sym4, dtype='int32')
    l_in_delta_mask4 = lasagne.layers.InputLayer(shape=(None, None), input_var=delta_mask4) #TODO:!!!
    #
    l_forward4 = lasagne.layers.LSTMLayer(l_in_enc4, num_units=hidden_size, mask_input=l_mask_enc4, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward4 = lasagne.layers.LSTMLayer(l_in_enc4, num_units=hidden_size, mask_input=l_mask_enc4, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc4 = lasagne.layers.ConcatLayer([l_forward4, l_backward4], axis=2)
    #
    l_enc = lasagne.layers.ConcatLayer([l_enc1, l_enc2, l_enc3, l_enc4], axis=2)
    #
    l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1)   
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len,1),input_var=X_dec_sym)
    l_mask_dec = lasagne.layers.InputLayer(shape=(None, pred_len), input_var=mask_dec)
    #    
    s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
    s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))
    s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    h_init = l_forward_slice#l_enc
    c_init =  lasagne.layers.ExpressionLayer(l_enc, lambda X: X.mean(1), output_shape='auto')
    #
    l_dec = LSTMAttLayer_lambda_mu4(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec1, enc_mask_input=l_in_delta_mask, 
                                   enc_mask_input2=l_in_delta_mask2, enc_mask_input3=l_in_delta_mask3, enc_mask_input4=l_in_delta_mask4,
                                   delta_inds_input=l_in_delta_inds, delta_inds_input2=l_in_delta_inds2, delta_inds_input3=l_in_delta_inds3, delta_inds_input4=l_in_delta_inds4,
                                   learn_init=False, pred_ind=0, ws=max_len, pred_len=pred_len,
                                     hid_init=h_init, contxt_input=l_enc1, contxt_input2=l_enc2, contxt_input3=l_enc3, 
                                     contxt_input4=l_enc4, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True )
    input_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_ingate, W_hid=l_dec.W_hid_to_ingate, 
                                            W_cell= l_dec.W_cell_to_ingate, b=l_dec.b_ingate)
    output_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_outgate, W_hid=l_dec.W_hid_to_outgate, 
                                      W_cell=l_dec.W_cell_to_outgate, b=l_dec.b_outgate)
    forget_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_forgetgate, W_hid=l_dec.W_hid_to_forgetgate,
                                      W_cell=l_dec.W_cell_to_forgetgate, b=l_dec.b_forgetgate)
    cell_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_cell, W_hid=l_dec.W_hid_to_cell, W_cell=None,
                                    b=l_dec.b_cell, nonlinearity=lasagne.nonlinearities.tanh) 
    #
    w_ctx2in = l_dec.W_ctx_to_ingate
    w_ctx2forget = l_dec.W_ctx_to_forgetgate
    w_ctx2cell = l_dec.W_ctx_to_cell
    w_ctx2out = l_dec.W_ctx_to_outgate    
    #
    #
    w_att, w_att2, w_att3, w_att4 = l_dec.W_att, l_dec.W_att2, l_dec.W_att3, l_dec.W_att4
    w_hid2att, w_hid2att2, w_hid2att3, w_hid2att4 = l_dec.W_hid_to_att, l_dec.W_hid_to_att2, l_dec.W_hid_to_att3, l_dec.W_hid_to_att4
    #
    w_ctx2att = l_dec.W_ctx_to_att
    w_ctx2att2 = l_dec.W_ctx_to_att2
    w_ctx2att3 = l_dec.W_ctx_to_att3
    w_ctx2att4 = l_dec.W_ctx_to_att4
    #
    w_lambda = l_dec.W_lambda
    w_lambda2 = l_dec.W_lambda2
    w_lambda3 = l_dec.W_lambda3
    w_lambda4 = l_dec.W_lambda4
    #
    w_mu, w_mu2, w_mu3, w_mu4 = l_dec.W_mu, l_dec.W_mu2, l_dec.W_mu3, l_dec.W_mu4
    #
    alphas = AnaLayer_lambda_mu4(h_init, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc1, contxt_input2=l_enc2,
                      contxt_input3=l_enc3, contxt_input4=l_enc4, enc_mask_input=l_in_delta_mask, enc_mask_input2=l_in_delta_mask2, 
                      enc_mask_input3=l_in_delta_mask3, enc_mask_input4=l_in_delta_mask4, delta_inds_input=l_in_delta_inds, 
                      delta_inds_input2=l_in_delta_inds2, delta_inds_input3=l_in_delta_inds3, delta_inds_input4=l_in_delta_inds4,
                      pred_ind=0, ws=max_len, pred_len=pred_len,
                      W_att=w_att, W_att2=w_att2, W_att3=w_att3, W_att4=w_att4, W_hid_to_att=w_hid2att, W_hid_to_att2=w_hid2att2, 
                      W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4, W_ctx_to_att=w_ctx2att, W_ctx_to_att2=w_ctx2att2,
                      W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4, W_lambda=w_lambda, W_lambda2=w_lambda2, W_lambda3=w_lambda3,
                      W_lambda4=w_lambda4, W_mu=w_mu, W_mu2=w_mu2, W_mu3=w_mu3, W_mu4=w_mu4)
    #
    l_out = lasagne.layers.DenseLayer(l_dec, num_units=1, nonlinearity=lasagne.nonlinearities.linear)  
    w_dense = l_out.W
    b_dense = l_out.b
    l_out_loop = l_out
    l_out_val = l_out
    l_out_loop_val = l_out
    h_init = l_dec
    h_init_val = l_dec
    for i in range(1,pred_len):
        s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1)
        s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))
        s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=i, axis=1) 
        s_lmask_dec = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1)) 
        l_dec = LSTMAttLayer_lambda_mu4(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec, enc_mask_input=l_in_delta_mask,
                            enc_mask_input2=l_in_delta_mask2, enc_mask_input3=l_in_delta_mask3, enc_mask_input4=l_in_delta_mask4, 
                            delta_inds_input=l_in_delta_inds, delta_inds_input2=l_in_delta_inds2, delta_inds_input3=l_in_delta_inds3,
                            delta_inds_input4=l_in_delta_inds4, learn_init=False, pred_ind=i, ws=max_len, pred_len=pred_len,
                                         hid_init=h_init, contxt_input=l_enc1, contxt_input2=l_enc2, contxt_input3=l_enc3, 
                            contxt_input4=l_enc4, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                            nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, 
                            ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                            W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_att2=w_att2, W_att3=w_att3, W_att4=w_att4, W_hid_to_att=w_hid2att,
                            W_hid_to_att2=w_hid2att2, W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4, W_ctx_to_att=w_ctx2att,
                            W_ctx_to_att2=w_ctx2att2, W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4, W_lambda=w_lambda, 
                            W_lambda2=w_lambda2, W_lambda3=w_lambda3, W_lambda4=w_lambda4, W_mu=w_mu, W_mu2=w_mu2, W_mu3=w_mu3, W_mu4=w_mu4) 
        #
        pred = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1]))
        l_dec_val = LSTMAttLayer_lambda_mu4(pred, num_units=hidden_size, mask_input=s_lmask_dec1, enc_mask_input=l_in_delta_mask,
                            enc_mask_input2=l_in_delta_mask2, enc_mask_input3=l_in_delta_mask3, enc_mask_input4=l_in_delta_mask4,
                            delta_inds_input=l_in_delta_inds, delta_inds_input2=l_in_delta_inds2, delta_inds_input3=l_in_delta_inds3,
                            delta_inds_input4=l_in_delta_inds4, learn_init=False, pred_ind=i, ws=max_len, pred_len=pred_len,
                            hid_init=h_init_val, contxt_input=l_enc1, contxt_input2=l_enc2, contxt_input3=l_enc3, contxt_input4=l_enc4, 
                            ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True,
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_att2=w_att2, W_att3=w_att3, W_att4=w_att4, W_hid_to_att=w_hid2att,
                             W_hid_to_att2=w_hid2att2, W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4, W_ctx_to_att=w_ctx2att,
                             W_ctx_to_att2=w_ctx2att2, W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4, W_lambda=w_lambda,
                             W_lambda2=w_lambda2, W_lambda3=w_lambda3, W_lambda4=w_lambda4, W_mu=w_mu, W_mu2=w_mu2, W_mu3=w_mu3, W_mu4=w_mu4)
        #
        l_out_loop = lasagne.layers.DenseLayer(l_dec, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop])
        #
        l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val])
        #
        alphas_loop = AnaLayer_lambda_mu4(h_init_val, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc1, contxt_input2=l_enc2,
                               contxt_input3=l_enc3, contxt_input4=l_enc4, enc_mask_input=l_in_delta_mask, enc_mask_input2=l_in_delta_mask2,
                               enc_mask_input3=l_in_delta_mask3, enc_mask_input4=l_in_delta_mask4, delta_inds_input=l_in_delta_inds,
                               delta_inds_input2=l_in_delta_inds2, delta_inds_input3=l_in_delta_inds3, delta_inds_input4=l_in_delta_inds4, 
                               pred_ind=0, ws=max_len, pred_len=pred_len,
                               W_att=w_att, W_att2=w_att2, W_att3=w_att3, W_att4=w_att4, W_hid_to_att=w_hid2att, W_hid_to_att2=w_hid2att2,
                               W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4, W_ctx_to_att=w_ctx2att, W_ctx_to_att2=w_ctx2att2, 
                               W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4, W_lambda=w_lambda, W_lambda2=w_lambda2,
                               W_lambda3=w_lambda3, W_lambda4=w_lambda4, W_mu=w_mu, W_mu2=w_mu2, W_mu3=w_mu3, W_mu4=w_mu4)
        alphas = lasagne.layers.ConcatLayer([alphas, alphas_loop])
        h_init = l_dec
        h_init_val = l_dec_val
    return (l_out, l_out_val, alphas)


def model_seq2seq_att_lambda_mu_alt_mvrt4(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4,
                                      X_dec_sym, mask_dec, delta_inds_sym, delta_inds_sym2, delta_inds_sym3, delta_inds_sym4,
                                      max_len, pred_len, grad_clip = 100, hidden_size=64, att_size =64):
    """
    sequence to sequence with attention 
    """
    #encoder
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 1),input_var=X_enc_sym)
    l_mask_enc = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc)  
    l_in_delta_inds = lasagne.layers.InputLayer(shape=(None,None), input_var=delta_inds_sym)
    #
    l_forward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc1 = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)
    #2nd variable encoder
    l_in_enc2 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym2)
    l_mask_enc2 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc2) 
    l_in_delta_inds2 = lasagne.layers.InputLayer(shape=(None,None), input_var=delta_inds_sym2)
    #
    l_forward2 = lasagne.layers.LSTMLayer(l_in_enc2, num_units=hidden_size, mask_input=l_mask_enc2, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward2 = lasagne.layers.LSTMLayer(l_in_enc2, num_units=hidden_size, mask_input=l_mask_enc2, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc2 = lasagne.layers.ConcatLayer([l_forward2, l_backward2], axis=2)
    #3rd variable encoder
    l_in_enc3 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym3)
    l_mask_enc3 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc3)    
    l_in_delta_inds3 = lasagne.layers.InputLayer(shape=(None,None), input_var=delta_inds_sym3)
    #
    l_forward3 = lasagne.layers.LSTMLayer(l_in_enc3, num_units=hidden_size, mask_input=l_mask_enc3, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward3 = lasagne.layers.LSTMLayer(l_in_enc3, num_units=hidden_size, mask_input=l_mask_enc3, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc3 = lasagne.layers.ConcatLayer([l_forward3, l_backward3], axis=2)
    ##4th variable encoder
    l_in_enc4 = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=X_enc_sym4)
    l_mask_enc4 = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc4)
    l_in_delta_inds4 = lasagne.layers.InputLayer(shape=(None,None), input_var=delta_inds_sym4)
    #
    l_forward4 = lasagne.layers.LSTMLayer(l_in_enc4, num_units=hidden_size, mask_input=l_mask_enc4, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward4 = lasagne.layers.LSTMLayer(l_in_enc4, num_units=hidden_size, mask_input=l_mask_enc4, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc4 = lasagne.layers.ConcatLayer([l_forward4, l_backward4], axis=2)
    #
    l_enc = lasagne.layers.ConcatLayer([l_enc1, l_enc2, l_enc3, l_enc4], axis=2)
    #
    l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1)   
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len,1),input_var=X_dec_sym)
    l_mask_dec = lasagne.layers.InputLayer(shape=(None, pred_len), input_var=mask_dec)
    #    
    s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
    s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))
    s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    h_init = l_forward_slice#l_enc
    c_init =  lasagne.layers.ExpressionLayer(l_enc, lambda X: X.mean(1), output_shape='auto')
    #
    l_dec = LSTMAttLayer_lambda_mu_alt4(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec1, delta_inds_input=l_in_delta_inds,
                            delta_inds_input2=l_in_delta_inds2, delta_inds_input3=l_in_delta_inds3, delta_inds_input4=l_in_delta_inds4,
                                   learn_init=False, pred_ind=0, ws=max_len, pred_len=pred_len,
                                     hid_init=h_init, contxt_input=l_enc1, contxt_input2=l_enc2, contxt_input3=l_enc3, 
                                     contxt_input4=l_enc4, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True )
    input_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_ingate, W_hid=l_dec.W_hid_to_ingate, 
                                            W_cell= l_dec.W_cell_to_ingate, b=l_dec.b_ingate)
    output_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_outgate, W_hid=l_dec.W_hid_to_outgate, 
                                      W_cell=l_dec.W_cell_to_outgate, b=l_dec.b_outgate)
    forget_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_forgetgate, W_hid=l_dec.W_hid_to_forgetgate,
                                      W_cell=l_dec.W_cell_to_forgetgate, b=l_dec.b_forgetgate)
    cell_gate = lasagne.layers.Gate(W_in=l_dec.W_in_to_cell, W_hid=l_dec.W_hid_to_cell, W_cell=None,
                                    b=l_dec.b_cell, nonlinearity=lasagne.nonlinearities.tanh) 
    #
    w_ctx2in = l_dec.W_ctx_to_ingate
    w_ctx2forget = l_dec.W_ctx_to_forgetgate
    w_ctx2cell = l_dec.W_ctx_to_cell
    w_ctx2out = l_dec.W_ctx_to_outgate    
    #
    #
    w_att, w_att2, w_att3, w_att4 = l_dec.W_att, l_dec.W_att2, l_dec.W_att3, l_dec.W_att4
    w_hid2att, w_hid2att2, w_hid2att3, w_hid2att4 = l_dec.W_hid_to_att, l_dec.W_hid_to_att2, l_dec.W_hid_to_att3, l_dec.W_hid_to_att4
    #
    w_ctx2att = l_dec.W_ctx_to_att
    w_ctx2att2 = l_dec.W_ctx_to_att2
    w_ctx2att3 = l_dec.W_ctx_to_att3
    w_ctx2att4 = l_dec.W_ctx_to_att4
    #
    w_lambda = l_dec.W_lambda
    w_lambda2 = l_dec.W_lambda2
    w_lambda3 = l_dec.W_lambda3
    w_lambda4 = l_dec.W_lambda4
    #
    w_mu, w_mu2, w_mu3, w_mu4 = l_dec.W_mu, l_dec.W_mu2, l_dec.W_mu3, l_dec.W_mu4
    #
    alphas = AnaLayer_lambda_mu_alt4(h_init, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc1, contxt_input2=l_enc2,
                      contxt_input3=l_enc3, contxt_input4=l_enc4, delta_inds_input=l_in_delta_inds, 
                      delta_inds_input2=l_in_delta_inds2, delta_inds_input3=l_in_delta_inds3, delta_inds_input4=l_in_delta_inds4,
                      pred_ind=0, ws=max_len, pred_len=pred_len,
                      W_att=w_att, W_att2=w_att2, W_att3=w_att3, W_att4=w_att4, W_hid_to_att=w_hid2att, W_hid_to_att2=w_hid2att2, 
                      W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4, W_ctx_to_att=w_ctx2att, W_ctx_to_att2=w_ctx2att2,
                      W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4, W_lambda=w_lambda, W_lambda2=w_lambda2, W_lambda3=w_lambda3,
                      W_lambda4=w_lambda4, W_mu=w_mu, W_mu2=w_mu2, W_mu3=w_mu3, W_mu4=w_mu4)
    #
    l_out = lasagne.layers.DenseLayer(l_dec, num_units=1, nonlinearity=lasagne.nonlinearities.linear)  
    w_dense = l_out.W
    b_dense = l_out.b
    l_out_loop = l_out
    l_out_val = l_out
    l_out_loop_val = l_out
    h_init = l_dec
    h_init_val = l_dec
    for i in range(1,pred_len):
        s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=i, axis=1)
        s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))
        s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=i, axis=1) 
        s_lmask_dec = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1)) 
        l_dec = LSTMAttLayer_lambda_mu_alt4(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec, 
                            delta_inds_input=l_in_delta_inds, delta_inds_input2=l_in_delta_inds2, delta_inds_input3=l_in_delta_inds3,
                            delta_inds_input4=l_in_delta_inds4, learn_init=False, pred_ind=i, ws=max_len, pred_len=pred_len,
                                         hid_init=h_init, contxt_input=l_enc1, contxt_input2=l_enc2, contxt_input3=l_enc3, 
                            contxt_input4=l_enc4, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                            nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, 
                            ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                            W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_att2=w_att2, W_att3=w_att3, W_att4=w_att4, W_hid_to_att=w_hid2att,
                            W_hid_to_att2=w_hid2att2, W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4, W_ctx_to_att=w_ctx2att,
                            W_ctx_to_att2=w_ctx2att2, W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4, W_lambda=w_lambda, 
                            W_lambda2=w_lambda2, W_lambda3=w_lambda3, W_lambda4=w_lambda4, W_mu=w_mu, W_mu2=w_mu2, W_mu3=w_mu3, W_mu4=w_mu4) 
        #
        pred = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1]))
        l_dec_val = LSTMAttLayer_lambda_mu_alt4(pred, num_units=hidden_size, mask_input=s_lmask_dec1,
                            delta_inds_input=l_in_delta_inds, delta_inds_input2=l_in_delta_inds2, delta_inds_input3=l_in_delta_inds3,
                            delta_inds_input4=l_in_delta_inds4, learn_init=False, pred_ind=i, ws=max_len, pred_len=pred_len,
                            hid_init=h_init_val, contxt_input=l_enc1, contxt_input2=l_enc2, contxt_input3=l_enc3, contxt_input4=l_enc4, 
                            ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True,
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_att2=w_att2, W_att3=w_att3, W_att4=w_att4, W_hid_to_att=w_hid2att,
                             W_hid_to_att2=w_hid2att2, W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4, W_ctx_to_att=w_ctx2att,
                             W_ctx_to_att2=w_ctx2att2, W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4, W_lambda=w_lambda,
                             W_lambda2=w_lambda2, W_lambda3=w_lambda3, W_lambda4=w_lambda4, W_mu=w_mu, W_mu2=w_mu2, W_mu3=w_mu3, W_mu4=w_mu4)
        #
        l_out_loop = lasagne.layers.DenseLayer(l_dec, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop])
        #
        l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val])
        #
        alphas_loop = AnaLayer_lambda_mu_alt4(h_init_val, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc1, contxt_input2=l_enc2,
                               contxt_input3=l_enc3, contxt_input4=l_enc4, delta_inds_input=l_in_delta_inds,
                               delta_inds_input2=l_in_delta_inds2, delta_inds_input3=l_in_delta_inds3, delta_inds_input4=l_in_delta_inds4, 
                               pred_ind=0, ws=max_len, pred_len=pred_len,
                               W_att=w_att, W_att2=w_att2, W_att3=w_att3, W_att4=w_att4, W_hid_to_att=w_hid2att, W_hid_to_att2=w_hid2att2,
                               W_hid_to_att3=w_hid2att3, W_hid_to_att4=w_hid2att4, W_ctx_to_att=w_ctx2att, W_ctx_to_att2=w_ctx2att2, 
                               W_ctx_to_att3=w_ctx2att3, W_ctx_to_att4=w_ctx2att4, W_lambda=w_lambda, W_lambda2=w_lambda2,
                               W_lambda3=w_lambda3, W_lambda4=w_lambda4, W_mu=w_mu, W_mu2=w_mu2, W_mu3=w_mu3, W_mu4=w_mu4)
        alphas = lasagne.layers.ConcatLayer([alphas, alphas_loop])
        h_init = l_dec
        h_init_val = l_dec_val
    return (l_out, l_out_val, alphas)