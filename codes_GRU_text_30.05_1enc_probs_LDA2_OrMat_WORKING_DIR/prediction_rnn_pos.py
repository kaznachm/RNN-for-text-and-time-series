# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:27:01 2016

@author: yagmur
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:36:31 2016

@author: yagmur

prediction with(/without) position information
"""

import sys
import csv
import numpy as np
#import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import lasagne
import cPickle as pickle
import logging
from lasagne.layers import get_output
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
import os
import linecache
from models import model_seq2seq_att_adist, model_seq2seq, model_seq2seq_att, model_seq2seq_att_lambda_mu, model_seq2seq_att_lambda 


def load_pse_data(datafile):
    """
    reads polish_v2 electricity data 
    """
    with open(datafile, 'rU') as f:
        mycsv = csv.reader(f, delimiter=';')
        mycsv = list(mycsv)
    mycsv = mycsv[1:]
    data = np.zeros(len(mycsv))
    for idx in range(0,len(mycsv)):
        data[idx] = mycsv[idx][3]
    return data


def get_tr_test_index(data_len, tr_ratio):
    """
    returns the training and test indices for the given tr_ratio
    
    """
    train_index = range(0,int(data_len*tr_ratio))
    test_index = range(int(data_len*tr_ratio),data_len)
    return (train_index, test_index)


def get_train_test(data, tr_ratio):
    """
    splits into two parts according to given tr_ratio
    """
    #row, col = data.shape
    data_len = len(data)
    (train_index, test_index) = get_tr_test_index(data_len, tr_ratio)
    #train, test = (data[train_index,:], data[test_index,:])
    train, test = (data[train_index], data[test_index])
    return (train, test)


def normalize_prev(train, test):
    """
    normalizes the data by subtracting mean and dividing by std
    """
    t_mean = np.nanmean(train)
    t_std = np.nanstd(train)
    ntrain = (train - t_mean)/t_std
    ntest = (test - t_mean)/t_std
    return ntrain, ntest, t_mean, t_std


def normalize(data, mean = None, std = None):
    """
    normalizes the data by subtracting mean and dividing by std
    for validation and test data we need training mean and training standard deviation
    """
    if mean == None:
        t_mean = np.nanmean(data)
        t_std = np.nanstd(data)
    else:
        t_mean = mean
        t_std = std        
    ndata = (data - t_mean)/t_std
    if  mean == None: #so it's training we need stats as well
        return ndata, t_mean, t_std
    else:
        return ndata



def windowize(data, windowsize, f_horizon=1):
    """
    Windowize returns the data X, and ground truth of prediction Y as window according to windowsize
    """    
    X = []
    Y = []
    strt = 0
    try:
        row, col = data.shape
        while (strt + windowsize + f_horizon-1) < row:
            X.append(np.hstack(data[strt:strt + windowsize,:]))
            Y.append(np.hstack(data[strt + windowsize : strt + windowsize + f_horizon,:]))
            strt = strt +1
    except:
        row = len(data)
        while (strt + windowsize+f_horizon-1) < row:
            X.append(np.hstack(data[strt:strt + windowsize]))
            Y.append(np.hstack(data[strt + windowsize: strt + windowsize + f_horizon]))
            strt = strt +1
    X = np.array(X)
    Y = np.array(Y)
    Y = np.squeeze(Y)
    return (X, Y)


def batch_gen(X, y, X_mask, batch_size, params):
    """
    randomly samples intances for the given mini-batch size
    """
    while True:
        idx = np.random.choice(len(y), batch_size)
        if (params['data_type']=='missing' and params['padding'] == False and params['interpolation'] == False):
            x = X[idx,:]
            nd=np.where(~np.isnan(x))[1]
            x_mask = X_mask[idx,:]
            yield x[:,nd].astype('float32'), y[idx,:].astype('float32'), x_mask[:,nd].astype('float32') 
        else:
            yield X[idx,:].astype('float32'), y[idx,:].astype('float32'), X_mask[idx,:].astype('float32')


def checkpoint(l_out, directory, file_name):
    """
    saves the given check point
    """
    params = lasagne.layers.get_all_param_values(l_out)
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(params, open(directory + file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(name):
    """
    returns the saved checkpoint 
    """
    f = open(name, 'rb')
    loaded_params = pickle.load(f)
    f.close()
    return loaded_params


def non_zero_rows(y_mask):
    """
    returns non-zero row indices of the given input
    """
    r, c = np.where(y_mask==0)
    inds = range(y_mask.shape[0])
    non_zero_inds = np.array(list(set(inds)-set(r)))
    return non_zero_inds


def non_all_zero_rows(X_mask):
    """
    returns non-all-zero row indices of the given input
    --> if the all values in one row are zero return the index of that row
    """
    nr_r, nr_c = X_mask.shape
    r, c = np.where(X_mask==0)
    r_values, r_counts = np.unique(r, return_counts=True)
    all_zero_inds = r_values[np.where(r_counts==nr_c)[0]]
    non_all_zero_inds = list(set(range(nr_r))-set(all_zero_inds))
    return non_all_zero_inds
    

def polish_v2_train_test_X_y_mask(params):
    """
    """
    if params['interpolation']:
        data = pickle.load(open(params['interpolated_datafile']))
    else:
        data = load_pse_data(params['data_file'])
    train, test = get_train_test(data, params['train_test_ratio'])
    train, val = get_train_test(train, params['train_val_ratio'])
    ntrain_set, nval_set, train_mean, train_std = normalize(train, val)
    ntrain_set, ntest_set, train_mean, train_std = normalize(train, test) 
    if params['data_type'] == 'original' or params['interpolation']:        
        X_train, y_train = windowize(ntrain_set, params['windowsize'], f_horizon=params['horizon'])
        X_val, y_val = windowize(nval_set, params['windowsize'], f_horizon=params['horizon'])
        X_test, y_test = windowize(ntest_set, params['windowsize'], f_horizon=params['horizon'])
        X_train_mask, X_val_mask, X_test_mask = np.ones((X_train.shape)), np.ones((X_val.shape)), np.ones((X_test.shape))
    elif params['data_type'] == 'missing':
        X_train, y_train = windowize(ntrain_set, params['windowsize'], f_horizon=params['horizon'])
        X_val, y_val = windowize(nval_set, params['windowsize'], f_horizon=params['horizon'])
        X_test, y_test = windowize(ntest_set, params['windowsize'], f_horizon=params['horizon'])
        if params['data_file_missing'] == None:
            logging.error("Data file with missing values is None!")
        elif (not params['padding']):
            #data = np.load(params['data_file_missing'])
            data = pickle.load(open(params['data_file_missing'], 'rb'))
            train, test = get_train_test(data, params['train_test_ratio'])
            train, val = get_train_test(train, params['train_val_ratio'])
            ntrain_set, nval_set, train_mean, train_std = normalize(train, val)
            ntrain_set, ntest_set, train_mean, train_std = normalize(train, test)
            X_train, _ = windowize(ntrain_set, params['windowsize'], f_horizon=params['horizon'])
            X_val, _ = windowize(nval_set, params['windowsize'], f_horizon=params['horizon'])
            X_test, _ = windowize(ntest_set, params['windowsize'], f_horizon=params['horizon'])
        if params['data_file_missing_mask'] == None:
            logging.error("Mask file is None!")
        else:
            #data_mask = np.load(params['data_file_missing_mask'])
            data_mask = pickle.load(open(params['data_file_missing_mask'], 'rb'))
            train_mask, test_mask = get_train_test(data_mask, params['train_test_ratio'])
            train_mask, val_mask = get_train_test(train_mask, params['train_val_ratio'])
            X_train_mask, y_train_mask = windowize(train_mask, params['windowsize'], f_horizon=params['horizon'])
            X_val_mask, y_val_mask = windowize(val_mask, params['windowsize'], f_horizon=params['horizon'])
            X_test_mask, y_test_mask = windowize(test_mask, params['windowsize'], f_horizon=params['horizon'])
    if params['padding'] and params['data_type'] == 'missing':
        tr, vl, tt = non_zero_rows(y_train_mask), non_zero_rows(y_val_mask), non_zero_rows(y_test_mask)
        return (X_train[tr,:], y_train[tr,:], X_train_mask[tr,:], X_val[vl,:], y_val[vl,:], X_val_mask[vl,:], X_test[tt,:], y_test[tt,:], X_test_mask[tt,:])
    else:
        return (X_train, y_train, X_train_mask, X_val, y_val, X_val_mask, X_test, y_test, X_test_mask)


def replace_none_w_one(data):
    r = np.where(np.isnan(data))[0]
    data[r] = 1.
    return data


def elnino_train_test_X_y_mask(params):
    """
    Creates X and y matrices for train, validation and test sets and 
    corresponding mask which indicates missing values for elnino dataset
    elnino originally with missing values (gaps/randomly missing)
    """
    elnino = pickle.load(open(params['data_file'], 'rb')) #elnino data originally with gaps and missing data
    elnino_mask = pickle.load(open(params['data_file_missing_mask'], 'rb'))
    if params['elnino_variable'] == 'air_temp':
        data = np.array(elnino['air_temp'])
        data_mask = np.array(elnino_mask['air_temp'])
    elif params['elnino_variable'] == 'ss_temp':
        data = np.array(elnino['ss_temp'])
        data_mask = np.array(elnino_mask['ss_temp'])
    elif params['elnino_variable'] == 'zon_winds':
        data = np.array(elnino['zon_winds'])
        data_mask = np.array(elnino_mask['zon_winds'])
    elif params['elnino_variable'] == 'mer_winds':
        data = np.array(elnino['mer_winds'])
        data_mask = np.array(elnino_mask['mer_winds'])
    elif params['elnino_variable'] == 'humidity':
        data = np.array(elnino['humidity'])
        data_mask = np.array(elnino_mask['humidity']) 
    #data
    #data = replace_none_w_one(data)
    train, test = get_train_test(data, params['train_test_ratio'])
    train, val = get_train_test(train, params['train_val_ratio'])
    ntrain_set, nval_set, train_mean, train_std = normalize(train, val)
    ntrain_set, ntest_set, train_mean, train_std = normalize(train, test)
    ntrain_set, nval_set, ntest_set = replace_none_w_one(ntrain_set), replace_none_w_one(nval_set), replace_none_w_one(ntest_set)
    X_train, y_train = windowize(ntrain_set, params['windowsize'], f_horizon=params['horizon'])
    X_val, y_val = windowize(nval_set, params['windowsize'], f_horizon=params['horizon'])
    X_test, y_test = windowize(ntest_set, params['windowsize'], f_horizon=params['horizon'])
    #preparing mask       
    train_mask, test_mask = get_train_test(data_mask, params['train_test_ratio'])
    train_mask, val_mask = get_train_test(train_mask, params['train_val_ratio'])
    X_train_mask, y_train_mask = windowize(train_mask, params['windowsize'], f_horizon=params['horizon'])
    X_val_mask, y_val_mask = windowize(val_mask, params['windowsize'], f_horizon=params['horizon'])
    X_test_mask, y_test_mask = windowize(test_mask, params['windowsize'], f_horizon=params['horizon'])
    tr, vl, tt = non_zero_rows(y_train_mask), non_zero_rows(y_val_mask), non_zero_rows(y_test_mask)
    X_train, y_train, X_train_mask = X_train[tr,:], y_train[tr,:], X_train_mask[tr,:]
    X_val, y_val, X_val_mask = X_val[vl,:], y_val[vl,:], X_val_mask[vl,:]
    X_test, y_test, X_test_mask = X_test[tt,:], y_test[tt,:], X_test_mask[tt,:]
    tr, vl, tt = non_all_zero_rows(X_train_mask), non_all_zero_rows(X_val_mask), non_all_zero_rows(X_test_mask)
    return (X_train[tr,:], y_train[tr,:], X_train_mask[tr,:], X_val[vl,:], y_val[vl,:], X_val_mask[vl,:], X_test[tt,:], y_test[tt,:], X_test_mask[tt,:])


def compute_X_y_mask(data, data_mask, params):
    #preparing data
    train, test = get_train_test(data, params['train_test_ratio'])
    train, val = get_train_test(train, params['train_val_ratio'])
    ntrain_set, train_mean, train_std = normalize(train)
    nval_set = normalize(val, mean = train_mean, std = train_std)
    ntest_set = normalize(test, mean = train_mean, std = train_std)
    ntrain_set, nval_set, ntest_set = replace_none_w_one(ntrain_set), replace_none_w_one(nval_set), replace_none_w_one(ntest_set)
    X_train, y_train = windowize(ntrain_set, params['windowsize'], f_horizon=params['horizon'])
    X_val, y_val = windowize(nval_set, params['windowsize'], f_horizon=params['horizon'])
    X_test, y_test = windowize(ntest_set, params['windowsize'], f_horizon=params['horizon'])
    #preparing mask       
    train_mask, test_mask = get_train_test(data_mask, params['train_test_ratio'])
    train_mask, val_mask = get_train_test(train_mask, params['train_val_ratio'])
    X_train_mask, y_train_mask = windowize(train_mask, params['windowsize'], f_horizon=params['horizon'])
    X_val_mask, y_val_mask = windowize(val_mask, params['windowsize'], f_horizon=params['horizon'])
    X_test_mask, y_test_mask = windowize(test_mask, params['windowsize'], f_horizon=params['horizon'])
    tr, vl, tt = non_zero_rows(y_train_mask), non_zero_rows(y_val_mask), non_zero_rows(y_test_mask)
    X_train, y_train, X_train_mask = X_train[tr,:], y_train[tr,:], X_train_mask[tr,:]
    X_val, y_val, X_val_mask = X_val[vl,:], y_val[vl,:], X_val_mask[vl,:]
    X_test, y_test, X_test_mask = X_test[tt,:], y_test[tt,:], X_test_mask[tt,:]
    tr, vl, tt = non_all_zero_rows(X_train_mask), non_all_zero_rows(X_val_mask), non_all_zero_rows(X_test_mask)
    return (X_train[tr,:], y_train[tr,:], X_train_mask[tr,:], X_val[vl,:], y_val[vl,:], X_val_mask[vl,:], X_test[tt,:], y_test[tt,:], X_test_mask[tt,:])


def train_test_X_y_mask(params):
    """
    Creates X and y matrices for train, validation and test sets and 
    corresponding mask which indicates missing values of X
    """
    if params['data_name'] == 'pse':        
        data = pickle.load(open(params['data_file'], 'rb')) #elnino data originally with gaps and missing data
        data_mask = pickle.load(open(params['data_file_missing_mask'], 'rb'))
    else:
        data_dict = pickle.load(open(params['data_file'], 'rb')) #elnino data originally with gaps and missing data
        data_mask_dict = pickle.load(open(params['data_file_missing_mask'], 'rb'))
        data = np.array(data_dict[params['data_variable']])
        data_mask = np.array(data_mask_dict[params['data_variable']])
    X_train, y_train, X_train_mask, X_val, y_val, X_val_mask, X_test, y_test, X_test_mask = compute_X_y_mask(data, data_mask, params)
    return X_train, y_train, X_train_mask, X_val, y_val, X_val_mask, X_test, y_test, X_test_mask


def test_X_y_mask(params):
    #loading data and mask
    if params['data_name'] == 'pse':        
        data = pickle.load(open(params['data_file'], 'rb')) #elnino data originally with gaps and missing data
        data_mask = pickle.load(open(params['data_file_missing_mask'], 'rb'))
    else:
        data_dict = pickle.load(open(params['data_file'], 'rb')) #elnino data originally with gaps and missing data
        data_mask_dict = pickle.load(open(params['data_file_missing_mask'], 'rb'))
        data = np.array(data_dict[params['data_variable']])
        data_mask = np.array(data_mask_dict[params['data_variable']])
    #preparing data - X_test   
    train, test = get_train_test(data, params['train_test_ratio'])
    ntrain_set, train_mean, train_std = normalize(train)
    ntest_set = normalize(test, mean = train_mean, std = train_std)
    ntest_set = replace_none_w_one(ntest_set)
    X_test, _ = windowize(ntest_set, params['windowsize'], f_horizon=params['horizon'])
    #preparing mask - X_test_mask   
    train_mask, test_mask = get_train_test(data_mask, params['train_test_ratio'])
    train_mask, val_mask = get_train_test(train_mask, params['train_val_ratio'])
    X_test_mask, _ = windowize(test_mask, params['windowsize'], f_horizon=params['horizon'])
    #loading original data and mask
    if params['data_name'] == 'pse':        
        data_orig = pickle.load(open(params['original_data_file'], 'rb')) #elnino data originally with gaps and missing data
        data_orig_mask = pickle.load(open(params['original_data_mask_file'], 'rb'))
    else:
        data_dict = pickle.load(open(params['original_data_file'], 'rb'))#elnino data originally with gaps and missing data
        data_mask_dict = pickle.load(open(params['original_data_mask_file'], 'rb'))
        data_orig = np.array(data_dict[params['data_variable']])
        data_orig_mask = np.array(data_mask_dict[params['data_variable']])
    #y_test_orig
    train_orig, test_orig = get_train_test(data_orig, params['train_test_ratio'])
    ntest_set_orig = normalize(test_orig, mean = train_mean, std = train_std)
    _, y_test = windowize(ntest_set_orig, params['windowsize'], f_horizon=params['horizon'])
    #y_test_mask_orig
    train_orig_mask, test_orig_mask = get_train_test(data_orig_mask, params['train_test_ratio'])
    _, y_test_mask = windowize(test_orig_mask, params['windowsize'], f_horizon=params['horizon'])
    tt = non_zero_rows(y_test_mask)
    X_test, y_test, X_test_mask = X_test[tt,:], y_test[tt,:], X_test_mask[tt,:]
    tt = non_all_zero_rows(X_test_mask)
    return (X_test[tt,:], y_test[tt,:], X_test_mask[tt,:])


def network(params):
    """
    builds the network according to given optimization and model parameters and data
    """
    X_enc_sym = T.ftensor3('x_enc_sym')
    X_dec_sym = T.ftensor3('x_dec_sym')
    mask_enc = T.matrix('enc_mask')
    mask_dec = T.matrix('dec_mask')
    y_sym = T.matrix('y_sym')
    eta = theano.shared(np.array(params['learning_rate'], dtype=theano.config.floatX))
    if params['attention']:            
        if params['num_layers'] ==1:
            if params['att_type']=='original':
                l_out, l_out_val, alphas = model_seq2seq_att(X_enc_sym, mask_enc, X_dec_sym, mask_dec, params['windowsize'], params['horizon'], input_dim=params['input_ndim'], hidden_size=params['num_units'], 
                                            grad_clip=params['grad_clipping'], att_size=params['num_att_units'])                
            elif params['att_type']=='adist':
                l_out, l_out_val, alphas = model_seq2seq_att_adist(X_enc_sym, mask_enc, X_dec_sym, mask_dec, params['windowsize'], params['horizon'], input_dim=params['input_ndim'], hidden_size=params['num_units'], 
                                            grad_clip=params['grad_clipping'], att_size=params['num_att_units'])
            elif params['att_type']=='lambda':
                l_out, l_out_val, alphas = model_seq2seq_att_lambda(X_enc_sym, mask_enc, X_dec_sym, mask_dec, params['windowsize'], params['horizon'], input_dim=params['input_ndim'], hidden_size=params['num_units'], 
                                            grad_clip=params['grad_clipping'], att_size=params['num_att_units']) 
            elif params['att_type']=='lambda_adist':
                l_out, l_out_val, alphas = model_seq2seq_att_lambda_adist(X_enc_sym, mask_enc, X_dec_sym, mask_dec, params['windowsize'], params['horizon'], input_dim=params['input_ndim'], hidden_size=params['num_units'], 
                                            grad_clip=params['grad_clipping'], att_size=params['num_att_units']) 
            elif params['att_type']=='lambda_mu':
                delta_inds_sym = T.vector('d_inds', dtype='int32') 
                l_out, l_out_val, alphas = model_seq2seq_att_lambda_mu(X_enc_sym, mask_enc, X_dec_sym, mask_dec, delta_inds_sym, params['windowsize'], params['horizon'], input_dim=params['input_ndim'], hidden_size=params['num_units'], 
                                            grad_clip=params['grad_clipping'], att_size=params['num_att_units'])
        #else:
            #TODO:        
        network_output, network_output_val, alpha_weights = lasagne.layers.get_output([l_out, l_out_val, alphas])
    else:# seq2seq no attention
        if params['num_layers'] ==1:
            l_out, l_out_val= model_seq2seq(X_enc_sym, mask_enc, X_dec_sym, mask_dec, params['windowsize'], params['horizon'], input_dim=params['input_ndim'], hidden_size=params['num_units'], 
                                            grad_clip=params['grad_clipping'])
        #else:
            #TODO:
        network_output, network_output_val = lasagne.layers.get_output([l_out, l_out_val])
    weights = lasagne.layers.get_all_params(l_out,trainable=True)
    #logging.info('weights: ')  #logging.info(weights)
    if params['regularization_type'] == 'l1':
        reg_loss = lasagne.regularization.regularize_network_params(l_out, l1) * params['lambda_regularization']
    else:
        reg_loss = lasagne.regularization.regularize_network_params(l_out, l2) * params['lambda_regularization']
    loss_T = T.mean(lasagne.objectives.squared_error(network_output, y_sym)) + reg_loss
    loss_val_T = T.mean(lasagne.objectives.squared_error(network_output_val, y_sym)) 
    if params['alg'] == 'adam':
        updates = lasagne.updates.adam(loss_T, weights, learning_rate=eta)
    else:
        logging.error("Optimization algorithm needs to be specified, e.g. 'adam'")
    if params['attention'] and params['att_type']=='lambda_mu':
       f_train = theano.function([X_enc_sym, mask_enc, X_dec_sym, mask_dec, y_sym, delta_inds_sym], loss_T, updates=updates, allow_input_downcast=True)
       f_val = theano.function([X_enc_sym, mask_enc, X_dec_sym, mask_dec, y_sym, delta_inds_sym], loss_val_T, allow_input_downcast=True)
    else:
        f_train = theano.function([X_enc_sym, mask_enc, X_dec_sym, mask_dec, y_sym], loss_T, updates=updates, allow_input_downcast=True)
        f_val = theano.function([X_enc_sym, mask_enc, X_dec_sym, mask_dec, y_sym], loss_val_T, allow_input_downcast=True)
    if params['attention']:  
        if params['att_type']=='lambda_mu':
            forecast = theano.function([X_enc_sym, mask_enc, X_dec_sym, mask_dec, delta_inds_sym], [network_output_val, alpha_weights], allow_input_downcast=True)
        else:
            forecast = theano.function([X_enc_sym, mask_enc, X_dec_sym, mask_dec], [network_output_val, alpha_weights], allow_input_downcast=True)
    else:
        forecast = theano.function([X_enc_sym, mask_enc, X_dec_sym, mask_dec], network_output_val, allow_input_downcast=True)
    return l_out, f_train, f_val, forecast


def gap_lengths(zero_inds):
    gap_lengths = []
    if len(zero_inds)> 0:
        idx = 0
        cursor = zero_inds[idx]
        glen = 1
        for idx in range(1,len(zero_inds)):
            if zero_inds[idx] == cursor +1:
                cursor += 1
                glen += 1
            else:
                gap_lengths.append(glen)
                cursor = zero_inds[idx]
                glen = 1
        gap_lengths.append(glen)
    return gap_lengths


def delta(idx, gap_len):
    if (idx/gap_len)<= 0.33:
        return 0
    elif (idx/gap_len)> 0.33 and (idx/gap_len)<= 0.66:
        return 1
    elif (idx/gap_len)> 0.66:
        return 2
    else:
        logging.info('error in the delta for mu')


def delta_indices(gap_lens):
    delta_inds = []
    for glen in gap_lens:
        if glen == 1:
            delta_inds.append(0)
        else:        
            for idx in range(glen):
                delta_inds.append(delta(idx, float(glen)))
    return delta_inds


def delta_gap_mask(mask_np):
    gap_lens = []
    for idx in range(np.size(mask_np,0)):
        mask = mask_np[idx,:]
        zero_inds = np.where(mask==0)[0]
        gap_lens.extend(gap_lengths(zero_inds))
    delta_inds = delta_indices(gap_lens)
    delta_inds = np.array(delta_inds)
    return delta_inds


def train(params, results_directory):
    """
    builds the network and trains the model with according to the given parameters
    """
    logging.info('Loading data and creating regression matrix for time series')
    X_train, y_train, X_train_mask, X_val, y_val, X_val_mask, X_test, y_test, X_test_mask = train_test_X_y_mask(params)
    logging.info('Data is loaded')
    logging.info('Building network')
    l_out, f_train, f_val, forecast = network(params)
    logging.info('Network is built')
    num_batches = len(X_train) // params['batch_size']
    #num_val_batches = len(X_val) // params['batch_size']
    # Minibatch generators for the training and validation sets
    train_batches = batch_gen(X_train, y_train, X_train_mask, params['batch_size'], params)
    #val_batches = batch_gen(X_val, y_val, X_val_mask, params['batch_size'], params)
    batch_ones = np.ones((params['batch_size'],1)) # as a start signal
    X_mask_dec = np.ones((params['batch_size'], params['horizon']))
    #2:
    #position info
    if params['pos_info']:
        pos_input_val = np.tile(np.arange(1,params['windowsize']+1), (np.size(X_val,0),1))
        X_val2 = np.stack((X_val, pos_input_val), axis=2)
        pos_input = np.tile(np.arange(1,params['windowsize']+1), (params['batch_size'],1))
    else:
        X_val2 = np.expand_dims(X_val, axis = 2)
    X_val_dec = np.ones((np.size(y_val,0), np.size(y_val,1)))
    X_val_mask_dec = np.ones((np.size(y_val,0), np.size(y_val,1)))
    X_val_dec2 = np.expand_dims(X_val_dec, axis = 2)
    save_to = directory_v2(results_directory, params)
    patience = params['early_stop_patience']      
    improvement_threshold = 1 - params['epsilon'] # 0.995  
    best_validation_loss = np.inf
    best_iter = 0
    epoch = 0
    done_looping = False
    loss_train, loss_validation = [], []
    while (epoch < params['max_num_epochs']) and (not done_looping):
        epoch +=1 
        logging.info('starting epoch ' + str(epoch))
        if params['learning_rate_decay'] and epoch >= params['lr_decay_after_n_epoch']:
            eta.set_value(eta.get_value() * eta_decay)
        tr_loss = 0
        for batch_ind in range(num_batches):
            X, y, X_mask = next(train_batches)
            if params['pos_info']:
                X = np.stack((X, pos_input), axis=2)
            else:
                X = np.expand_dims(X, axis=2)
            X_dec = np.hstack((batch_ones, y[:,:-1]))
            X_dec = np.expand_dims(X_dec, axis=2)
            if params['attention'] and params['att_type']=='lambda_mu':
                d_inds = delta_gap_mask(X_mask)
                loss = f_train(X, X_mask, X_dec, X_mask_dec, y, d_inds)                
            else:
                loss = f_train(X, X_mask, X_dec, X_mask_dec, y)
            iter = (epoch - 1) * num_batches + batch_ind # iteration number
            tr_loss += loss
        tr_loss /= num_batches
        loss_train.append(tr_loss)
        #1:
        #val_loss = 0
        #for _ in range(num_val_batches):
        #    X, y, X_mask = next(val_batches)
        #    X = np.expand_dims(X, axis = 2)
        #    X_dec = np.hstack((batch_ones,y[:,:-1]))
        #    X_dec = np.expand_dims(X_dec, axis = 2)
        #    if params['attention'] and params['att_type']=='lambda_mu':
        #        d_inds = delta_gap_mask(X_mask)
        #        loss = f_val(X, X_mask, X_dec, X_mask_dec, y, d_inds) 
        #    else:
        #        loss = f_val(X, X_mask, X_dec, X_mask_dec, y) 
        #    val_loss += loss
        #val_loss /= num_val_batches
        if params['attention'] and params['att_type']=='lambda_mu':
            d_inds = delta_gap_mask(X_val_mask)
            val_loss = f_val(X_val2, X_val_mask, X_val_dec2, X_val_mask_dec, y_val, d_inds)                
        else:
            val_loss = f_val(X_val2, X_val_mask, X_val_dec2, X_val_mask_dec, y_val) 
        loss_validation.append(val_loss.mean())
        logging.info('Epoch {}, Train (val) loss {:.03f} ({:.03f}) ratio {:.03f}'.format(
                epoch, tr_loss, val_loss.mean(), val_loss/tr_loss))        
        if val_loss < best_validation_loss: # if we got the best validation score until now
            if (val_loss < best_validation_loss *improvement_threshold): #improve patience if loss improvement is good enough
                patience = max(patience, iter * params['patience_increase'])
            best_validation_loss = val_loss
            best_iter = iter
            best_epoch = epoch
            checkpoint(l_out, save_to,'best.pkl')
        checkpoint(l_out, save_to, 'last_epoch.pkl')
        pickle.dump(loss_train, open(save_to + 'train_loss.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)    
        pickle.dump(loss_validation, open(save_to + 'validation_loss.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(epoch, open(save_to +'last_epoch.pkl', 'wb'))
        if patience <= iter:
            message = 'Early stopping after seeing '+ str(iter*params['batch_size'])+ ' examples & after '+str(epoch) + ' epochs'
            pickle.dump(message, open(save_to + 'termination_message.pkl', 'wb'))
            logging.info(message)
            done_looping = True
            break
        if epoch == params['max_num_epochs']:
            message = 'Stopping after reaching max number of epoch: ' + str(epoch) + ' epochs & after seeing '+ str(iter*params['batch_size'])+' examples.'
            pickle.dump(message, open(save_to + 'termination_message.pkl', 'wb'))
            logging.info(message)
    pickle.dump(params, open(save_to + 'parameters.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    return loss_train, loss_validation


def directory_v2_prev(results_directory, params):
    save_to = results_directory         
    if params['attention']:
        save_to = save_to + 'attention_' + params['att_type'] + '/'
    else:
        save_to = save_to + 'no_attention/'
    if params['data_name'] == 'polish_electricity_v2':   
        if params['data_type']=='original':
            save_to = save_to + 'original/'
        else:
            save_to = save_to + 'missing_' + str(params['missing_percent']) + '/'
            if params['missing_type'] == 'random':
                save_to = save_to + 'random/'
            elif params['missing_type'] == 'varying_frequency':
                save_to = save_to + 'varying_frequency/'
            elif params['missing_type'] == 'gaps':
                save_to = save_to + 'gaps_'+str(params['gap_length'])+'/'        
        save_to = save_to + params['data_name'] +'_'+  params['exp_id'] + '/'
    elif params['data_name'] == 'elnino':
        save_to = save_to + params['data_name']+ '_' + params['elnino_variable'] + '/'+ params['elnino_variable'] + '_'+ params['exp_id'] +'/'    
    elif params['data_name'] == 'hhp':
        save_to = save_to + params['data_name']+ '_' + params['data_variable'] + '/'+ params['data_variable'] + '_'+ params['exp_id'] +'/'    
    elif params['data_name'] == 'airq':
        save_to = save_to + params['data_name']+ '_' + params['data_variable'] + '/'+ params['data_variable'] + '_'+ params['exp_id'] +'/'    
    print save_to
    return save_to


def directory_v2(results_directory, params):
    save_to = results_directory 
    if params['data_type'] == 'original':
        save_to = save_to + params['data_name'] + '_original/'
    else:
        save_to = save_to + params['data_name'] + params['missing_percent'] + '/'        
    if params['attention']:
        save_to = save_to + 'attention_' + params['att_type'] + '/'
    else:
        save_to = save_to + 'no_attention/' 
    if params['data_type'] == 'original':
        save_to = save_to + params['data_name'] + '_'+ str(params['num_units']) + '_'+ str(params['num_att_units']) + '/' 
    else:
        save_to = save_to + params['data_name'] + params['missing_percent'] + '_'+ str(params['num_units']) + '_'+ str(params['num_att_units']) + '/' 
    print save_to
    return save_to


def load_parameters(params_csvfile, line_num):
    keys = linecache.getline(params_csvfile,1)[:-1].split(',')
    params={}
    values = linecache.getline(params_csvfile,line_num)[:-1].split(',')
    for (k,v) in zip(keys, values):
        if "0." in v[:2]:
            v = float(v)
        else:
            try:
                v = int(v)
            except Exception as e:
                e = e
        if v == 'True':
            v = True
        if v == 'False':
            v = False
        params[k] = v
    return params


def load_parameters2(params_csvfile, line_num): 
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
    if params['data_name'] == 'airq' :
        params['data_variable'] = 'PT08.S4(NO2)'
        params['windowsize'] = 192
        params['horizon'] = 6
    if params['data_name'] == 'consseason':
        params['data_variable'] = 'Global_active_power'
        params['windowsize'] = 96
        params['horizon'] = 4
    if params['data_name'] == 'weather_data':
        params['data_variable'] = 'max_tmp' 
        params['windowsize'] = 548
        params['horizon'] = 7
    if params['data_name'] == 'pse':
        params['windowsize'] = 96
        params['horizon'] = 4
    if data_type == 'orig':
        params['data_type'] = 'original'
        params['data_file'] =  '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'.pkl'
        params['data_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'_mask.pkl'
        params['original_data_file'] = params['data_file'] #!!! for interpolation not the same file!
        params['original_data_mask_file'] = params['data_file_missing_mask']
    else:
        params['data_type'] = 'missing'
        params['data_file'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/'+ params['data_name'] + '_'+ data_type +'.pkl'
        params['data_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/missing_gaps_v2/' + params['data_name'] + '_mask_' + data_type + '.pkl'
        params['missing_percent'] = data_type
        params['original_data_file'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'.pkl'
        params['original_data_mask_file'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'_mask.pkl'
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
    params['interpolation'] = False
    params['max_num_epochs'] = 25
    params['epsilon'] = 0.01
    params['learning_rate'] = 0.001
    params['batch_size'] = 64
    params['padding'] = True 
    params['learning_rate_decay'] = False
    params['interpolation_type'] = 'None'
    return params


def mean_abs_percent_err(y_pred, y_true):
    """ the mean absolute percentage error"""
    idx = y_true != 0.0
    #mape = np.mean(np.abs((y_pred[idx]-y_true[idx])/y_true[idx])) * 100
    mape = np.abs((y_pred[idx]-y_true[idx])/y_true[idx]) #* 100
    return mape


def sym_mean_abs_percent_err(y_pred, y_true):
    """ the symmetric mean absolute percentage error"""
    #smape = np.mean(np.abs(y_pred - y_true)/(np.abs(y_pred)+np.abs(y_true))) * 200
    smape = np.abs(y_pred - y_true)/(np.abs(y_pred)+np.abs(y_true)) * 2#00
    return smape


def test(params, results_directory):
    logging.info('Loading data and creating regression matrix for time series')    
    X_test, y_test, X_test_mask = test_X_y_mask(params)
    save_to = directory_v2(results_directory, params)
    logging.info('Data is loaded')
    logging.info('Building network')
    l_out, f_train, f_val, forecast = network(params)
    logging.info('Network is built')
    logging.info('Best parameters are loaded.')
    loaded_params = load_checkpoint(save_to+ 'best.pkl')
    lasagne.layers.set_all_param_values(l_out, loaded_params) 
    #
    if params['pos_info']:
        pos_input = np.tile(np.arange(1,params['windowsize']+1), (np.size(X_test,0),1))
        X_test = np.stack((X_test, pos_input), axis=2)
    else:
        X_test = np.expand_dims(X_test, axis = 2)
    X_dec_test = np.ones((np.size(y_test,0), np.size(y_test,1)))
    X_test_mask_dec =  np.ones((np.size(y_test,0), np.size(y_test,1)))
    X_dec_test = np.expand_dims(X_dec_test, axis = 2)
    logging.info('Starting forecasting')
    if params['attention']:
        if params['att_type']=='lambda_mu':
            d_inds_test = delta_gap_mask(X_test_mask)
            y_pred, y_weights = forecast(X_test, X_test_mask, X_dec_test, X_test_mask_dec, d_inds_test)
        else:            
            y_pred, y_weights = forecast(X_test, X_test_mask, X_dec_test, X_test_mask_dec)
        pickle.dump(X_test, open(save_to + 'X_test.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_weights, open(save_to + 'y_weights.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        y_pred = forecast(X_test, X_test_mask, X_dec_test, X_test_mask_dec)
    mse = ((y_pred - y_test) ** 2).mean(axis=0)
    pickle.dump(mse, open(save_to + 'mse.pkl', 'wb'))
    logging.info('Test Prediction MSE: ' + str(mse.mean()))
    pickle.dump(y_test, open(save_to + 'y_test.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(y_pred, open(save_to + 'y_pred.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    mape = mean_abs_percent_err(y_pred, y_test)
    smape = sym_mean_abs_percent_err(y_pred, y_test)
    logging.info('Test Prediction MAPE: ' + str(mape.mean()))
    logging.info('Test Prediction SMAPE: ' + str(smape.mean()))
    pickle.dump(mape, open(save_to + 'mape.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(smape, open(save_to + 'smape.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)



def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    results_directory = 'Results_w_pos/'
    if len(sys.argv) > 4:
        strt = int(sys.argv[2]) #strt ind of line 
        end = int(sys.argv[3]) +1#end ind of line
        order = sys.argv[4] # order from top the bottom or reverse
    elif len(sys.argv) == 4:
        strt = int(sys.argv[2]) #strt ind of line 
        end = int(sys.argv[3]) +1 #end ind of line
        order = 'normal'
    elif len(sys.argv) ==2:
        strt = 1
        end = 11
        order = 'normal'
    else:
        logging.info('Problem with arguments')
    line_inds = range(strt, end)
    if order ==  'reverse':
        line_inds =  list(reversed(line_inds))
    logging.info('line_inds:')
    logging.info(line_inds)
    for i in line_inds:
        params = load_parameters2(sys.argv[1], i)
        params['pos_info'] = True #if pos_info to be used it's True
        if params['pos_info']:
            params['input_ndim'] = 2
        #params = load_parameters('params.csv', 2)
        logging.info(params)
        #
        train_loss, validation_loss = train(params, results_directory)
        #
        test(params, results_directory)
        #        
        if params['pos_info']:
            f = open(str(sys.argv[1])+"_pos_test.log","a")
        else:
            f = open(str(sys.argv[1])+".log","a")
        f.write(str(i)+':')
        #f.write(line)
        f.close()

"""    
    results_directory = 'Prediction/RNN/attention/Results/'
    #
    params = {}
    params['data_name'] = 'polish_electricity_v2'
    #params['data_file'] = 'Data/polish_v2/original/pse_8months.csv'
    params['data_file'] = '/home/yagmur/Documents/ama/yeni/lsn/data/pse_8m.csv'
    params['tr_ratio'] = 0.75
    params['val_ratio'] = 0.70
    params['windowsize'] = 100
    params['horizon'] = 4
    params['data_type'] = 'original'#'missing' # can be 'missing' or 'original'
    params['missing_type'] = None#'random' # can be 'random', 'varying_frequency','gaps', or can be None for original data
    params['missing_percent'] = 5
    params['gap_length'] = 10    
    params['data_file_missing'] = '/home/yagmur/Documents/ama/kod/hersey_cokguzel_olacak/data/pse_8m_w_gaps_5.npy'
    #'data_file_with_missing_values'
    params['data_file_missing_mask'] = '/home/yagmur/Documents/ama/kod/hersey_cokguzel_olacak/data/pse_8m_w_gaps_mask_5.npy'
    #'mask_file'
    params['padding'] = False
    params['interpolation'] = False
    params['interpolation_type'] = None #'linear' #it can be 'linear', 'quadratic','fft', 'ws' None if no interpolation applied
    #
    params = {}
    params['alg'] = 'adam'
    params['max_num_epochs'] = 25
    params['batch_size'] = 1
    params['learning_rate'] = 1e-3
    params['learning_rate_decay'] = False
    params['lr_decay_after_n_epoch'] = 50 
    params['eta_decay'] = 0.95
    params['regularization_type'] = 'l1' #can be 'l1' or 'l2'
    params['lambda_regularization'] = 1e-4
    params['early_stop'] = True
    params['early_stop_patience'] = 10000 # look as this many examples regardless
    params['patience_increase'] = 2 # wait this much longer when a new best is # found
    params['epsilon_improvement'] = True 
    params['epsilon'] = 0.005 # a relative improvement of this much is# considered significant
    #
    params = {}
    params['num_units'] = 64#  e.g. 64, 128, 256, 512
    params['num_layers'] = 1 # 2, 3, or 4
    params['attention'] = False
    params['num_att_units'] = params['num_units']# can be  64, 128, 256, 512  
    params['grad_clipping'] = 100 # e.g. 100, 50, 10
    params['adist'] = True
    #
    train_loss, validation_loss = train(params, params, params, results_directory)
    #
    #test(params, params, params, results_directory)

"""

if __name__ == '__main__':
    main()  
