# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:32:12 2016

@author: yagmur
"""

import lasagne 
from layers import LSTMAttLayer
from layers import AnaLayer
from layers import LSTMAttLayer_v2
from layers import AnaLayer_v2
from layers import LSTMAttLayer_lambda
from layers import AnaLayer_lambda
from layers import LSTMAttLayer_lambda_mu
from layers import AnaLayer_lambda_mu
from layers import LSTMAttLayer_lambda_mu_alt
from layers import AnaLayer_lambda_mu_alt


def model_seq2seq_att(X_enc_sym, mask_enc, X_dec_sym, mask_dec, max_len, pred_len, input_dim=1, grad_clip=100, hidden_size=64, att_size=64):
    """
    sequence to sequence with attention 
    """
    #encoder
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=X_enc_sym)
    l_mask_enc = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc)    
    #
    l_forward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)
    l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1)   #h_{T}
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, 1),input_var=X_dec_sym)
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
    l_dec = LSTMAttLayer(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec1, learn_init=False, 
                                     hid_init=h_init, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
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
    w_att = l_dec.W_att
    w_hid2att = l_dec.W_hid_to_att
    w_ctx2att = l_dec.W_ctx_to_att
    alphas = AnaLayer(h_init, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att)
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
        s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0],  1, [1]))
        s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=i, axis=1) 
        s_lmask_dec = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1)) 
        l_dec = LSTMAttLayer(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec, learn_init=False, 
                                         hid_init=h_init, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, 
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att) 
        #
        pred = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0],  1, [1]))
        l_dec_val = LSTMAttLayer(pred, num_units=hidden_size, mask_input=s_lmask_dec1, learn_init=False,
                                         hid_init=h_init_val, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True,
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att)
        #
        l_out_loop = lasagne.layers.DenseLayer(l_dec, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop])
        #
        l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val])
        #
        alphas_loop = AnaLayer(h_init_val, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att)
        alphas = lasagne.layers.ConcatLayer([alphas, alphas_loop])
        h_init = l_dec
        h_init_val = l_dec_val
    return (l_out, l_out_val, alphas)


def model_seq2seq(X_enc_sym, mask_enc, X_dec_sym, mask_dec, max_len, pred_len, input_dim=1, hidden_size=64, grad_clip=100):
    """
    sequence to sequence without attention
    """
    #encoder
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=X_enc_sym)
    l_mask_enc = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc)    
    #
    l_forward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True )
    l_backward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=1)
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, 1),input_var=X_dec_sym)
    l_mask_dec = lasagne.layers.InputLayer(shape=(None, pred_len), input_var=mask_dec)
    #    
    s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
    s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))
    s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    h_init = lasagne.layers.ConcatLayer([l_forward, l_enc], axis=1)
    #
    l_dec = lasagne.layers.LSTMLayer(s_lin_dec, num_units=hidden_size*3, mask_input=s_lmask_dec1, learn_init=False, 
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
        l_dec = lasagne.layers.LSTMLayer(s_lin_dec, num_units=hidden_size*3, mask_input=s_lmask_dec, learn_init=False, 
                                         hid_init=h_init, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, 
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate) 
        l_dec_hid_state = lasagne.layers.SliceLayer(l_dec, indices=slice(0,hidden_size))
        #
        pred = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1]))
        l_dec_val = lasagne.layers.LSTMLayer(pred, num_units=hidden_size*3, mask_input=s_lmask_dec1, learn_init=False,
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


def model_seq2seq_att_adist(X_enc_sym, mask_enc, X_dec_sym, mask_dec, max_len, pred_len, input_dim=1, grad_clip=100, hidden_size=64, att_size=64):
    """
    sequence to sequence with attention
    with corrected alpha weights considering gaps
    """
    #encoder
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=X_enc_sym)
    l_mask_enc = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc)    
    #
    l_forward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)
    l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1)   
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, input_dim),input_var=X_dec_sym)
    l_mask_dec = lasagne.layers.InputLayer(shape=(None, 1), input_var=mask_dec)
    #    
    s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
    s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))
    s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    h_init = l_forward_slice#l_enc
    c_init =  lasagne.layers.ExpressionLayer(l_enc, lambda X: X.mean(1), output_shape='auto')
    #
    l_dec = LSTMAttLayer_v2(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec1, enc_mask_input=l_mask_enc, learn_init=False, 
                                     hid_init=h_init, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
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
    w_att = l_dec.W_att
    w_hid2att = l_dec.W_hid_to_att
    w_ctx2att = l_dec.W_ctx_to_att
    alphas = AnaLayer_v2(h_init, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc,  enc_mask_input=l_mask_enc, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att)
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
        l_dec = LSTMAttLayer_v2(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec, enc_mask_input=l_mask_enc, learn_init=False, 
                                         hid_init=h_init, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, 
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att) 
        #
        pred = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1]))
        l_dec_val = LSTMAttLayer_v2(pred, num_units=hidden_size, mask_input=s_lmask_dec1, enc_mask_input=l_mask_enc, learn_init=False,
                                         hid_init=h_init_val, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True,
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att)
        #
        l_out_loop = lasagne.layers.DenseLayer(l_dec, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop])
        #
        l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val])
        #
        alphas_loop = AnaLayer_v2(h_init_val, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc, enc_mask_input=l_mask_enc, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att)
        alphas = lasagne.layers.ConcatLayer([alphas, alphas_loop])
        h_init = l_dec
        h_init_val = l_dec_val
    return (l_out, l_out_val, alphas)


def model_seq2seq_att_lambda(X_enc_sym, mask_enc, X_dec_sym, mask_dec, max_len, pred_len, input_dim=1, grad_clip=100, hidden_size=64, att_size=64):
    """
    sequence to sequence with attention 
    """
    #encoder
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=X_enc_sym)
    l_mask_enc = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc)    
    #
    l_forward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)
    l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1)   
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, 1),input_var=X_dec_sym)
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
    l_dec = LSTMAttLayer_lambda(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec1, learn_init=False, pred_ind=0, ws=max_len, pred_len=pred_len,
                                     hid_init=h_init, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
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
    w_att = l_dec.W_att
    w_hid2att = l_dec.W_hid_to_att
    w_ctx2att = l_dec.W_ctx_to_att
    w_lambda = l_dec.W_lambda
    alphas = AnaLayer_lambda(h_init, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc, pred_ind=0, ws=max_len, pred_len=pred_len,
                      W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda)
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
        l_dec = LSTMAttLayer_lambda(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec, learn_init=False, pred_ind=i, ws=max_len, pred_len=pred_len,
                                         hid_init=h_init, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, 
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda) 
        #
        pred = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1]))
        l_dec_val = LSTMAttLayer_lambda(pred, num_units=hidden_size, mask_input=s_lmask_dec1, learn_init=False, pred_ind=i, ws=max_len, pred_len=pred_len,
                                         hid_init=h_init_val, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True,
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda)
        #
        l_out_loop = lasagne.layers.DenseLayer(l_dec, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop])
        #
        l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val])
        #
        alphas_loop = AnaLayer_lambda(h_init_val, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc, pred_ind=0, ws=max_len, pred_len=pred_len,
                               W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda)
        alphas = lasagne.layers.ConcatLayer([alphas, alphas_loop])
        h_init = l_dec
        h_init_val = l_dec_val
    return (l_out, l_out_val, alphas)


def model_seq2seq_att_lambda_mu(X_enc_sym, mask_enc, X_dec_sym, mask_dec, delta_inds_sym, max_len, pred_len, input_dim=1, grad_clip=100, hidden_size=64, att_size=64):
    """
    sequence to sequence with attention 
    """
    #encoder
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=X_enc_sym)
    l_mask_enc = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc)  
    l_in_delta_inds = lasagne.layers.InputLayer(shape=(None,),input_var=delta_inds_sym, dtype='int32')
    #
    l_forward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)
    l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1)   
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, 1),input_var=X_dec_sym)
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
    l_dec = LSTMAttLayer_lambda_mu(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec1, enc_mask_input=l_mask_enc, delta_inds_input=l_in_delta_inds, learn_init=False, pred_ind=0, ws=max_len, pred_len=pred_len,
                                     hid_init=h_init, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
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
    w_att = l_dec.W_att
    w_hid2att = l_dec.W_hid_to_att
    w_ctx2att = l_dec.W_ctx_to_att
    w_lambda = l_dec.W_lambda
    w_mu = l_dec.W_mu
    alphas = AnaLayer_lambda_mu(h_init, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc, enc_mask_input=l_mask_enc, delta_inds_input=l_in_delta_inds, pred_ind=0, ws=max_len, pred_len=pred_len,
                      W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda, W_mu=w_mu)
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
        l_dec = LSTMAttLayer_lambda_mu(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec, enc_mask_input=l_mask_enc, delta_inds_input=l_in_delta_inds, learn_init=False, pred_ind=i, ws=max_len, pred_len=pred_len,
                                         hid_init=h_init, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, 
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda, W_mu=w_mu) 
        #
        pred = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1]))
        l_dec_val = LSTMAttLayer_lambda_mu(pred, num_units=hidden_size, mask_input=s_lmask_dec1, enc_mask_input=l_mask_enc, delta_inds_input=l_in_delta_inds, learn_init=False, pred_ind=i, ws=max_len, pred_len=pred_len,
                                         hid_init=h_init_val, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True,
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda, W_mu=w_mu)
        #
        l_out_loop = lasagne.layers.DenseLayer(l_dec, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop])
        #
        l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val])
        #
        alphas_loop = AnaLayer_lambda_mu(h_init_val, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc, enc_mask_input=l_mask_enc, delta_inds_input=l_in_delta_inds, pred_ind=0, ws=max_len, pred_len=pred_len,
                               W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda, W_mu=w_mu)
        alphas = lasagne.layers.ConcatLayer([alphas, alphas_loop])
        h_init = l_dec
        h_init_val = l_dec_val
    return (l_out, l_out_val, alphas)


def model_seq2seq_att_lambda_mu_alt(X_enc_sym, mask_enc, X_dec_sym, mask_dec, delta_inds_sym, max_len, pred_len, input_dim=1, grad_clip=100, hidden_size=64, att_size=64):
    """
    sequence to sequence with attention 
    """
    #encoder
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, input_dim), input_var=X_enc_sym)
    l_mask_enc = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc)  
    l_in_delta_inds = lasagne.layers.InputLayer(shape=(None,None), input_var=delta_inds_sym)
    #
    l_forward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)
    l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1)   
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, 1), input_var=X_dec_sym)
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
    l_dec = LSTMAttLayer_lambda_mu_alt(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec1, delta_inds_input=l_in_delta_inds, learn_init=False, pred_ind=0, ws=max_len, pred_len=pred_len,
                                     hid_init=h_init, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
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
    w_att = l_dec.W_att
    w_hid2att = l_dec.W_hid_to_att
    w_ctx2att = l_dec.W_ctx_to_att
    w_lambda = l_dec.W_lambda
    w_mu = l_dec.W_mu
    alphas = AnaLayer_lambda_mu_alt(h_init, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc, delta_inds_input=l_in_delta_inds, pred_ind=0, ws=max_len, pred_len=pred_len,
                      W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda, W_mu=w_mu)
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
        l_dec = LSTMAttLayer_lambda_mu_alt(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec, delta_inds_input=l_in_delta_inds, learn_init=False, pred_ind=i, ws=max_len, pred_len=pred_len,
                                         hid_init=h_init, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, 
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda, W_mu=w_mu) 
        #
        pred = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1]))
        l_dec_val = LSTMAttLayer_lambda_mu_alt(pred, num_units=hidden_size, mask_input=s_lmask_dec1, delta_inds_input=l_in_delta_inds, learn_init=False, pred_ind=i, ws=max_len, pred_len=pred_len,
                                         hid_init=h_init_val, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True,
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda, W_mu=w_mu)
        #
        l_out_loop = lasagne.layers.DenseLayer(l_dec, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop])
        #
        l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val])
        #
        alphas_loop = AnaLayer_lambda_mu_alt(h_init_val, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc, delta_inds_input=l_in_delta_inds, pred_ind=0, ws=max_len, pred_len=pred_len,
                               W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda, W_mu=w_mu)
        alphas = lasagne.layers.ConcatLayer([alphas, alphas_loop])
        h_init = l_dec
        h_init_val = l_dec_val
    return (l_out, l_out_val, alphas)


def model_seq2seq_att_lambda_mu_intr(X_enc_sym, mask_enc, X_dec_sym, mask_dec, delta_mask, delta_inds_sym, max_len, pred_len, input_dim=1, grad_clip=100, hidden_size=64, att_size=64):
    """
    sequence to sequence with attention lambda mu with delta if interpolated
    """
    #encoder
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=X_enc_sym)
    l_mask_enc = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc)  
    l_in_delta_inds = lasagne.layers.InputLayer(shape=(None,),input_var=delta_inds_sym, dtype='int32')
    #
    l_in_delta_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=delta_mask) 
    #
    l_forward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)
    l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1)   
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, 1),input_var=X_dec_sym)
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
    l_dec = LSTMAttLayer_lambda_mu(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec1, enc_mask_input=l_in_delta_mask, delta_inds_input=l_in_delta_inds, learn_init=False, pred_ind=0, ws=max_len, pred_len=pred_len,
                                     hid_init=h_init, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
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
    w_att = l_dec.W_att
    w_hid2att = l_dec.W_hid_to_att
    w_ctx2att = l_dec.W_ctx_to_att
    w_lambda = l_dec.W_lambda
    w_mu = l_dec.W_mu
    alphas = AnaLayer_lambda_mu(h_init, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc, enc_mask_input=l_in_delta_mask, delta_inds_input=l_in_delta_inds, pred_ind=0, ws=max_len, pred_len=pred_len,
                      W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda, W_mu=w_mu)
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
        l_dec = LSTMAttLayer_lambda_mu(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec, enc_mask_input=l_in_delta_mask, delta_inds_input=l_in_delta_inds, learn_init=False, pred_ind=i, ws=max_len, pred_len=pred_len,
                                         hid_init=h_init, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, 
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda, W_mu=w_mu) 
        #
        pred = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1]))
        l_dec_val = LSTMAttLayer_lambda_mu(pred, num_units=hidden_size, mask_input=s_lmask_dec1, enc_mask_input=l_in_delta_mask, delta_inds_input=l_in_delta_inds, learn_init=False, pred_ind=i, ws=max_len, pred_len=pred_len,
                                         hid_init=h_init_val, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True,
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda, W_mu=w_mu)
        #
        l_out_loop = lasagne.layers.DenseLayer(l_dec, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop])
        #
        l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val])
        #
        alphas_loop = AnaLayer_lambda_mu(h_init_val, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc, enc_mask_input=l_in_delta_mask, delta_inds_input=l_in_delta_inds, pred_ind=0, ws=max_len, pred_len=pred_len,
                               W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda, W_mu=w_mu)
        alphas = lasagne.layers.ConcatLayer([alphas, alphas_loop])
        h_init = l_dec
        h_init_val = l_dec_val
    return (l_out, l_out_val, alphas)


def model_seq2seq_att_adist_intr(X_enc_sym, mask_enc, X_dec_sym, mask_dec, delta_mask, max_len, pred_len, input_dim=1, grad_clip=100, hidden_size=64, att_size=64):
    """
    sequence to sequence with attention
    with corrected alpha weights considering gaps for interpolated data
    """
    #encoder
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=X_enc_sym)
    l_mask_enc = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc)    
    #
    l_in_delta_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=delta_mask)
    #
    l_forward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)
    l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1)   
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, input_dim),input_var=X_dec_sym)
    l_mask_dec = lasagne.layers.InputLayer(shape=(None, 1), input_var=mask_dec)
    #    
    s_lin_dec = lasagne.layers.SliceLayer(l_in_dec, indices=0, axis=1)
    s_lin_dec = lasagne.layers.ReshapeLayer(s_lin_dec, ([0], 1, [1]))
    s_lmask_dec = lasagne.layers.SliceLayer(l_mask_dec, indices=0, axis=1)
    s_lmask_dec1 = lasagne.layers.ReshapeLayer(s_lmask_dec, ([0], 1))
    #
    h_init = l_forward_slice#l_enc
    c_init =  lasagne.layers.ExpressionLayer(l_enc, lambda X: X.mean(1), output_shape='auto')
    #
    l_dec = LSTMAttLayer_v2(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec1, enc_mask_input=l_in_delta_mask, learn_init=False, 
                                     hid_init=h_init, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
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
    w_att = l_dec.W_att
    w_hid2att = l_dec.W_hid_to_att
    w_ctx2att = l_dec.W_ctx_to_att
    alphas = AnaLayer_v2(h_init, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc, enc_mask_input=l_in_delta_mask, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att)
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
        l_dec = LSTMAttLayer_v2(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec, enc_mask_input=l_in_delta_mask, learn_init=False, 
                                         hid_init=h_init, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, 
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att) 
        #
        pred = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1]))
        l_dec_val = LSTMAttLayer_v2(pred, num_units=hidden_size, mask_input=s_lmask_dec1, enc_mask_input=l_in_delta_mask, learn_init=False,
                                         hid_init=h_init_val, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True,
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att)
        #
        l_out_loop = lasagne.layers.DenseLayer(l_dec, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop])
        #
        l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val])
        #
        alphas_loop = AnaLayer_v2(h_init_val, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc, enc_mask_input=l_in_delta_mask, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att)
        alphas = lasagne.layers.ConcatLayer([alphas, alphas_loop])
        h_init = l_dec
        h_init_val = l_dec_val
    return (l_out, l_out_val, alphas)


def model_seq2seq_att_lambda_adist(X_enc_sym, mask_enc, X_dec_sym, mask_dec, max_len, pred_len, input_dim=1, grad_clip=100, hidden_size=64, att_size=64):
    """
    sequence to sequence with attention 
    """
    #encoder
    l_in_enc = lasagne.layers.InputLayer(shape=(None, None, input_dim),input_var=X_enc_sym)
    l_mask_enc = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_enc)    
    #
    l_forward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False )
    l_backward = lasagne.layers.LSTMLayer(l_in_enc, num_units=hidden_size, mask_input=l_mask_enc, grad_clipping=grad_clip, 
                                        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, backwards=True)
    l_enc = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)
    l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1)   
    #decoder
    l_in_dec = lasagne.layers.InputLayer(shape=(None, pred_len, 1),input_var=X_dec_sym)
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
    l_dec = LSTMAttLayer_lambda_adist(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec1, enc_mask_input=l_mask_enc, learn_init=False, pred_ind=0, ws=max_len, pred_len=pred_len,
                                     hid_init=h_init, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
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
    w_att = l_dec.W_att
    w_hid2att = l_dec.W_hid_to_att
    w_ctx2att = l_dec.W_ctx_to_att
    w_lambda = l_dec.W_lambda
    alphas = AnaLayer_lambda_adist(h_init, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc, enc_mask_input=l_mask_enc, pred_ind=0, ws=max_len, pred_len=pred_len,
                      W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda)
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
        l_dec = LSTMAttLayer_lambda_adist(s_lin_dec, num_units=hidden_size, mask_input=s_lmask_dec, enc_mask_input=l_mask_enc, learn_init=False, pred_ind=i, ws=max_len, pred_len=pred_len,
                                         hid_init=h_init, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, 
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda) 
        #
        pred = lasagne.layers.ReshapeLayer(l_out_loop_val, ([0], 1, [1]))
        l_dec_val = LSTMAttLayer_lambda_adist(pred, num_units=hidden_size, mask_input=s_lmask_dec1, enc_mask_input=l_mask_enc, learn_init=False, pred_ind=i, ws=max_len, pred_len=pred_len,
                                         hid_init=h_init_val, contxt_input=l_enc, ctx_init= c_init, att_num_units=att_size, grad_clipping=grad_clip, 
                                         nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True,
                                         ingate=input_gate, outgate=output_gate, forgetgate=forget_gate, cell=cell_gate,
                            W_ctx_to_ingate=w_ctx2in, W_ctx_to_forgetgate=w_ctx2forget, W_ctx_to_cell=w_ctx2cell, 
                             W_ctx_to_outgate=w_ctx2out, W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda)
        #
        l_out_loop = lasagne.layers.DenseLayer(l_dec, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out = lasagne.layers.ConcatLayer([l_out, l_out_loop])
        #
        l_out_loop_val = lasagne.layers.DenseLayer(l_dec_val, num_units=1, W=w_dense, b=b_dense, nonlinearity=lasagne.nonlinearities.linear)
        l_out_val = lasagne.layers.ConcatLayer([l_out_val, l_out_loop_val])
        #
        alphas_loop = AnaLayer_lambda_adist(h_init_val, num_units=hidden_size, att_num_units=att_size, contxt_input=l_enc, enc_mask_input=l_mask_enc, pred_ind=0, ws=max_len, pred_len=pred_len,
                               W_att=w_att, W_hid_to_att=w_hid2att, W_ctx_to_att=w_ctx2att, W_lambda=w_lambda)
        alphas = lasagne.layers.ConcatLayer([alphas, alphas_loop])
        h_init = l_dec
        h_init_val = l_dec_val
    return (l_out, l_out_val, alphas)


