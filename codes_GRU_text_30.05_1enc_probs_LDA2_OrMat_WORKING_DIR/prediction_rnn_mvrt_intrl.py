# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:27:01 2016

@author: yagmur
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:36:31 2016

@author: yagmur
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
from models_mvrt import model_seq2seq_mvrt4, model_seq2seq_att_mvrt4, model_seq2seq_att_lambda_mvrt4
from models_mvrt import model_seq2seq_att_lambda_mu_mvrt4, model_seq2seq_att_lambda_mu_alt_mvrt4, model_seq2seq_att_lambda_mu_mvrt4_intr


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
            yield x[:,nd].astype('float32'), y[idx,:].astype('float32'), x[:,nd].astype('float32') #x_mask?? #TODO:?
        else:
            yield X[idx,:].astype('float32'), y[idx,:].astype('float32'), X_mask[idx,:].astype('float32')


def batch_gen_mvrt(X, y, X_mask, X2, X_mask2, X3, X_mask3, batch_size):
    """
    randomly samples intances for the given mini-batch size
    """
    while True:
        idx = np.random.choice(len(y), batch_size)
        yield X[idx,:].astype('float32'), y[idx,:].astype('float32'), X_mask[idx,:].astype('float32'), X2[idx,:].astype('float32'), X_mask2[idx,:].astype('float32'), X3[idx,:].astype('float32'), X_mask3[idx,:].astype('float32')


def batch_gen_mvrt4(X, y, X_mask, X2, X_mask2, X3, X_mask3, X4, X_mask4, batch_size):
    """
    randomly samples intances for the given mini-batch size
    """
    while True:
        idx = np.random.choice(len(y), batch_size)
        yield X[idx,:].astype('float32'), y[idx,:].astype('float32'), X_mask[idx,:].astype('float32'), X2[idx,:].astype('float32'), X_mask2[idx,:].astype('float32'), X3[idx,:].astype('float32'), X_mask3[idx,:].astype('float32'), X4[idx,:].astype('float32'), X_mask4[idx,:].astype('float32')


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
        #TODO: reading interpolated_data
        data = pickle.load(params['interpolated_datafile'])
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


def y_train_test_orig_and_mask(params, train_mean, train_std):
    """
    returns original data y ad y_mask for train, validation and test
    """
    #loading original data and mask
    if params['data_name'] == 'pse':        
        data_orig = pickle.load(open(params['original_data_file'], 'rb')) #elnino data originally with gaps and missing data
        data_orig_mask = pickle.load(open(params['original_data_mask_file'], 'rb'))
    else:
        data_dict = pickle.load(open(params['original_data_file'], 'rb'))#elnino data originally with gaps and missing data
        data_mask_dict = pickle.load(open(params['original_data_mask_file'], 'rb'))
        data_orig = np.array(data_dict[params['data_variable']])
        data_orig_mask = np.array(data_mask_dict[params['data_variable']])
    train_orig, test_orig = get_train_test(data_orig, params['train_test_ratio'])
    train_orig, val_orig = get_train_test(train_orig, params['train_test_ratio'])
    #
    ntrain_set_orig = normalize(train_orig, mean = train_mean, std = train_std)
    nval_set_orig = normalize(val_orig, mean = train_mean, std = train_std)
    ntest_set_orig = normalize(test_orig, mean = train_mean, std = train_std)
    #
    _, y_train = windowize(ntrain_set_orig, params['windowsize'], f_horizon=params['horizon'])
    _, y_val = windowize(nval_set_orig, params['windowsize'], f_horizon=params['horizon'])
    _, y_test = windowize(ntest_set_orig, params['windowsize'], f_horizon=params['horizon'])
    # y_test_mask_orig
    train_orig_mask, test_orig_mask = get_train_test(data_orig_mask, params['train_test_ratio'])
    train_orig_mask, val_orig_mask = get_train_test(train_orig_mask, params['train_test_ratio'])
    #
    _, y_train_mask = windowize(train_orig_mask, params['windowsize'], f_horizon=params['horizon'])
    _, y_val_mask = windowize(val_orig_mask, params['windowsize'], f_horizon=params['horizon'])
    _, y_test_mask = windowize(test_orig_mask, params['windowsize'], f_horizon=params['horizon'])
    return (y_train, y_val, y_test, y_train_mask, y_val_mask, y_test_mask)


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
    if params['interpolation']:
        if (params['attention'] and (params['att_type']=='lambda_mu' or params['att_type']=='lambda_mu_alt')):
            train_mask, test_mask = get_train_test(data_mask, params['train_test_ratio'])
            train_mask, val_mask = get_train_test(train_mask, params['train_val_ratio'])
            X_train_mask, _ = windowize(train_mask, params['windowsize'], f_horizon=params['horizon'])
            X_val_mask, _ = windowize(val_mask, params['windowsize'], f_horizon=params['horizon'])
            X_test_mask, _ = windowize(test_mask, params['windowsize'], f_horizon=params['horizon'])
            y_train, y_val, y_test, y_train_mask, y_val_mask, y_test_mask = y_train_test_orig_and_mask(params, train_mean, train_std)
        else:            
            X_train_mask, X_val_mask, X_test_mask = np.ones_like(X_train), np.ones_like(X_val), np.ones_like(X_test)
            y_train, y_val, y_test, y_train_mask, y_val_mask, y_test_mask = y_train_test_orig_and_mask(params, train_mean, train_std)
    else:         
        train_mask, test_mask = get_train_test(data_mask, params['train_test_ratio'])
        train_mask, val_mask = get_train_test(train_mask, params['train_val_ratio'])
        X_train_mask, y_train_mask = windowize(train_mask, params['windowsize'], f_horizon=params['horizon'])
        X_val_mask, y_val_mask = windowize(val_mask, params['windowsize'], f_horizon=params['horizon'])
        X_test_mask, y_test_mask = windowize(test_mask, params['windowsize'], f_horizon=params['horizon'])
    tr, vl, tt = non_zero_rows(y_train_mask), non_zero_rows(y_val_mask), non_zero_rows(y_test_mask)
    params['tr_non_zero_rows'], params['vl_non_zero_rows'], params['tt_non_zero_rows'] = tr, vl, tt
    X_train, y_train, X_train_mask = X_train[tr,:], y_train[tr,:], X_train_mask[tr,:]
    X_val, y_val, X_val_mask = X_val[vl,:], y_val[vl,:], X_val_mask[vl,:]
    X_test, y_test, X_test_mask = X_test[tt,:], y_test[tt,:], X_test_mask[tt,:]
    tr, vl, tt = non_all_zero_rows(X_train_mask), non_all_zero_rows(X_val_mask), non_all_zero_rows(X_test_mask)
    params['tr_non_all_zero_rows'], params['vl_non_all_zero_rows'], params['tt_non_all_zero_rows'] = tr, vl, tt
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
    if params['interpolation']:
        if (params['attention'] and (params['att_type']=='lambda_mu' or params['att_type']=='lambda_mu_alt')):
            train_mask, test_mask = get_train_test(data_mask, params['train_test_ratio'])
            train_mask, val_mask = get_train_test(train_mask, params['train_val_ratio'])
            X_test_mask, _ = windowize(test_mask, params['windowsize'], f_horizon=params['horizon'])
        else:
            X_test_mask = np.ones_like(X_test)
    else:
        train_mask, test_mask = get_train_test(data_mask, params['train_test_ratio'])
        train_mask, val_mask = get_train_test(train_mask, params['train_val_ratio'])
        X_test_mask, _ = windowize(test_mask, params['windowsize'], f_horizon=params['horizon'])
    #y_test_orig
    _, _, y_test, _, _, y_test_mask = y_train_test_orig_and_mask(params, train_mean, train_std)
    #
    tt = non_zero_rows(y_test_mask)
    params['tt_non_zero_rows'] = tt
    X_test, y_test, X_test_mask = X_test[tt,:], y_test[tt,:], X_test_mask[tt,:]
    tt = non_all_zero_rows(X_test_mask)
    params['tt_non_all_zero_rows'] = tt
    return (X_test[tt,:], y_test[tt,:], X_test_mask[tt,:])


def compute_X_train_val_test(data, params, bool_normalize=True):
    train, test = get_train_test(data, params['train_test_ratio'])
    train, val = get_train_test(train, params['train_val_ratio'])
    if bool_normalize:        
        ntrain_set, train_mean, train_std = normalize(train)
        nval_set = normalize(val, mean = train_mean, std = train_std)
        ntest_set = normalize(test, mean = train_mean, std = train_std)
        ntrain_set, nval_set, ntest_set = replace_none_w_one(ntrain_set), replace_none_w_one(nval_set), replace_none_w_one(ntest_set)
    else:
        (ntrain_set, nval_set, ntest_set) = train, val, test #for mask
    X_train, _ = windowize(ntrain_set, params['windowsize'], f_horizon=params['horizon'])
    X_val, _ = windowize(nval_set, params['windowsize'], f_horizon=params['horizon'])
    X_test, _ = windowize(ntest_set, params['windowsize'], f_horizon=params['horizon'])
    X_train, X_val, X_test = X_train[params['tr_non_zero_rows'][:]], X_val[params['vl_non_zero_rows'][:]], X_test[params['tt_non_zero_rows'][:]]
    X_train, X_val, X_test = X_train[params['tr_non_all_zero_rows']], X_val[params['vl_non_all_zero_rows']], X_test[params['tt_non_all_zero_rows']]
    return (X_train, X_val, X_test)


def train_test_X_y_mask_mvrt(params, var=0):
    if params['data2_name'] == 'polish_weather':
        data_dict = pickle.load(open(params['data2_file'], 'rb'))
        variable_dict = data_dict[params['data2_variables'][var]]        
        mask_dict = pickle.load(open(params['data2_file_missing_mask'], 'rb'))[params['data2_variables'][var]]
        X_train, X_val, X_test = variable_dict['train'], variable_dict['val'], variable_dict['test']
        X_train_mask, X_val_mask, X_test_mask = mask_dict['train'], mask_dict['val'], mask_dict['test']
    if params['data2_name'] == 'consseason' or params['data2_name'] =='airq':
        data_dict = pickle.load(open(params['data2_file'], 'rb')) #elnino data originally with gaps and missing data
        mask_dict = pickle.load(open(params['data2_file_missing_mask'], 'rb'))
        data = np.array(data_dict[params['data2_variables'][var]])
        data_mask = np.array(mask_dict[params['data2_variables'][var]])
        X_train, X_val, X_test = compute_X_train_val_test(data, params, bool_normalize=True)
        X_train_mask, X_val_mask, X_test_mask = compute_X_train_val_test(data_mask, params, bool_normalize=False)
    return (X_train, X_train_mask, X_val, X_val_mask, X_test, X_test_mask)


def network_mvrt4(params):
    """
    builds the network according to given optimization and model parameters and data
    """
    X_enc_sym = T.ftensor3('x_enc_sym')
    X_dec_sym = T.ftensor3('x_dec_sym')
    mask_enc = T.matrix('enc_mask')
    mask_dec = T.matrix('dec_mask')
    y_sym = T.matrix('y_sym')
    X_enc_sym2 = T.ftensor3('x_enc_sym2')
    X_enc_sym3 = T.ftensor3('x_enc_sym3')
    X_enc_sym4 = T.ftensor3('x_enc_sym4')
    mask_enc2 = T.matrix('enc_mask2')
    mask_enc3 = T.matrix('enc_mask3')
    mask_enc4 = T.matrix('enc_mask4')
    eta = theano.shared(np.array(params['learning_rate'], dtype=theano.config.floatX))
    if params['attention']:
        if params['num_layers'] ==1:
            if params['att_type']=='original':
                l_out, l_out_val, alphas = model_seq2seq_att_mvrt4(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, params['windowsize'], params['horizon'], hidden_size = params['num_units'], 
                                            grad_clip = params['grad_clipping'], att_size = params['num_att_units'])                
            elif params['att_type']=='adist':
                l_out, l_out_val, alphas = model_seq2seq_att_adist_mvrt4(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, params['windowsize'], params['horizon'],  hidden_size = params['num_units'], 
                                            grad_clip = params['grad_clipping'], att_size = params['num_att_units'])
            elif params['att_type']=='lambda':
                l_out, l_out_val, alphas = model_seq2seq_att_lambda_mvrt4(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, params['windowsize'], params['horizon'],  hidden_size = params['num_units'], 
                                            grad_clip = params['grad_clipping'], att_size = params['num_att_units']) 
            elif params['att_type']=='lambda_mu':
                delta_inds_sym, delta_inds_sym2 = T.vector('d_inds', dtype='int32'), T.vector('d_inds2', dtype='int32')
                delta_inds_sym3, delta_inds_sym4 = T.vector('d_inds3', dtype='int32'), T.vector('d_inds4', dtype='int32')
                if params['interpolation']:
                    delta_mask, delta_mask2, delta_mask3, delta_mask4 = T.matrix('delta_mask'), T.matrix('delta_mask2'), T.matrix('delta_mask3'), T.matrix('delta_mask4')
                    l_out, l_out_val, alphas = model_seq2seq_att_lambda_mu_mvrt4_intr(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, delta_mask, delta_mask2, delta_mask3, delta_mask4, 
                                            delta_inds_sym, delta_inds_sym2, delta_inds_sym3, delta_inds_sym4, params['windowsize'], params['horizon'],  hidden_size = params['num_units'], 
                                            grad_clip = params['grad_clipping'], att_size = params['num_att_units'])
                else:
                    l_out, l_out_val, alphas = model_seq2seq_att_lambda_mu_mvrt4(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, 
                                            delta_inds_sym, delta_inds_sym2, delta_inds_sym3, delta_inds_sym4, params['windowsize'], params['horizon'],  hidden_size = params['num_units'], 
                                            grad_clip = params['grad_clipping'], att_size = params['num_att_units'])
            elif params['att_type']=='lambda_mu_alt':
                delta_inds_sym, delta_inds_sym2 = T.matrix('d_inds'), T.matrix('d_inds2') 
                delta_inds_sym3, delta_inds_sym4 = T.matrix('d_inds3'), T.matrix('d_inds4') 
                l_out, l_out_val, alphas = model_seq2seq_att_lambda_mu_alt_mvrt4(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, delta_inds_sym,
                                            delta_inds_sym2, delta_inds_sym3, delta_inds_sym4, params['windowsize'], params['horizon'], hidden_size=params['num_units'], 
                                            grad_clip = params['grad_clipping'], att_size = params['num_att_units'])
        network_output, network_output_val, alpha_weights = lasagne.layers.get_output([l_out, l_out_val, alphas])
    else:
        if params['num_layers'] ==1:
            l_out, l_out_val= model_seq2seq_mvrt4(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, params['windowsize'], params['horizon'],  hidden_size = params['num_units'], 
                                                grad_clip = params['grad_clipping'])
        network_output, network_output_val = lasagne.layers.get_output([l_out, l_out_val])
        #
    weights = lasagne.layers.get_all_params(l_out,trainable=True)
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
    if params['attention'] and (params['att_type']=='lambda_mu' or params['att_type']=='lambda_mu_alt'):
        if params['interpolation'] and params['att_type']=='lambda_mu':
            f_train = theano.function([X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, y_sym,
                    delta_mask, delta_mask2, delta_mask3, delta_mask4, delta_inds_sym, delta_inds_sym2, delta_inds_sym3, delta_inds_sym4], loss_T, updates=updates, allow_input_downcast=True)
            #
            f_val = theano.function([X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, y_sym,
                    delta_mask, delta_mask2, delta_mask3, delta_mask4, delta_inds_sym, delta_inds_sym2, delta_inds_sym3, delta_inds_sym4], loss_val_T, allow_input_downcast=True)
        else:
            f_train = theano.function([X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, y_sym,
                    delta_inds_sym, delta_inds_sym2, delta_inds_sym3, delta_inds_sym4], loss_T, updates=updates, allow_input_downcast=True)
            #
            f_val = theano.function([X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, y_sym,
                    delta_inds_sym, delta_inds_sym2, delta_inds_sym3, delta_inds_sym4], loss_val_T, allow_input_downcast=True)
    else:
        f_train = theano.function([X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, y_sym], loss_T, updates=updates, allow_input_downcast=True)
        f_val = theano.function([X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec, y_sym], loss_val_T, allow_input_downcast=True)
    if params['attention']:  
        if params['att_type']=='lambda_mu' or params['att_type']=='lambda_mu_alt':
            if params['interpolation'] and params['att_type']=='lambda_mu':
                forecast = theano.function([X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec,
                            delta_mask, delta_mask2, delta_mask3, delta_mask4, delta_inds_sym, delta_inds_sym2, delta_inds_sym3, delta_inds_sym4], [network_output_val, alpha_weights], allow_input_downcast=True)
            else:
                forecast = theano.function([X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec,
                            delta_inds_sym, delta_inds_sym2, delta_inds_sym3, delta_inds_sym4], [network_output_val, alpha_weights], allow_input_downcast=True)
        else:
            forecast = theano.function([X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec], [network_output_val, alpha_weights], allow_input_downcast=True)
    else:
        forecast = theano.function([X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_enc_sym4, mask_enc4, X_dec_sym, mask_dec], network_output_val, allow_input_downcast=True)
    return l_out, f_train, f_val, forecast


def network_mvrt3(params):
    """
    builds the network according to given optimization and model parameters and data
    """
    X_enc_sym = T.ftensor3('x_enc_sym')
    X_dec_sym = T.ftensor3('x_dec_sym')
    mask_enc = T.matrix('enc_mask')
    mask_dec = T.matrix('dec_mask')
    y_sym = T.matrix('y_sym')
    X_enc_sym2 = T.ftensor3('x_enc_sym2')
    X_enc_sym3 = T.ftensor3('x_enc_sym3')
    mask_enc2 = T.matrix('enc_mask2')
    mask_enc3 = T.matrix('enc_mask3')
    eta = theano.shared(np.array(params['learning_rate'], dtype=theano.config.floatX))
    if params['attention']:
        if params['num_layers'] ==1:
            if params['att_type']=='original':
                l_out, l_out_val, alphas = model_seq2seq_att_mvrt3(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_dec_sym, mask_dec, params['windowsize'], params['horizon'], hidden_size = params['num_units'], 
                                            grad_clip = params['grad_clipping'], att_size = params['num_att_units'])                
            elif params['att_type']=='adist':
                l_out, l_out_val, alphas = model_seq2seq_att_adist_mvrt3(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_dec_sym, mask_dec, params['windowsize'], params['horizon'],  hidden_size = params['num_units'], 
                                            grad_clip = params['grad_clipping'], att_size = params['num_att_units'])
            elif params['att_type']=='lambda':
                l_out, l_out_val, alphas = model_seq2seq_att_lambda_mvrt3(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_dec_sym, mask_dec, params['windowsize'], params['horizon'],  hidden_size = params['num_units'], 
                                            grad_clip = params['grad_clipping'], att_size = params['num_att_units']) 
            elif params['att_type']=='lambda_adist':
                l_out, l_out_val, alphas = model_seq2seq_att_lambda_adist_mvrt3(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_dec_sym, mask_dec, params['windowsize'], params['horizon'],  hidden_size = params['num_units'], 
                                            grad_clip = params['grad_clipping'], att_size = params['num_att_units']) 
            elif params['att_type']=='lambda_mu':
                delta_inds_sym = T.vector('d_inds', dtype='int32') 
                l_out, l_out_val, alphas = model_seq2seq_att_lambda_mu_mvrt3(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_dec_sym, mask_dec, delta_inds_sym, params['windowsize'], params['horizon'],  hidden_size = params['num_units'], 
                                            grad_clip = params['grad_clipping'], att_size = params['num_att_units'])
        network_output, network_output_val, alpha_weights = lasagne.layers.get_output([l_out, l_out_val, alphas])
    else:
        if params['num_layers'] ==1:
            l_out, l_out_val= model_seq2seq_mvrt3(X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_dec_sym, mask_dec, params['windowsize'], params['horizon'],  hidden_size = params['num_units'], 
                                                grad_clip = params['grad_clipping'])
        network_output, network_output_val = lasagne.layers.get_output([l_out, l_out_val])
        #
    weights = lasagne.layers.get_all_params(l_out,trainable=True)
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
       f_train = theano.function([X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_dec_sym, mask_dec, y_sym, delta_inds_sym], loss_T, updates=updates, allow_input_downcast=True)
       f_val = theano.function([X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_dec_sym, mask_dec, y_sym, delta_inds_sym], loss_val_T, allow_input_downcast=True)
    else:
        f_train = theano.function([X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_dec_sym, mask_dec, y_sym], loss_T, updates=updates, allow_input_downcast=True)
        f_val = theano.function([X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_dec_sym, mask_dec, y_sym], loss_val_T, allow_input_downcast=True)
    if params['attention']:  
        if params['att_type']=='lambda_mu':
            forecast = theano.function([X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_dec_sym, mask_dec, delta_inds_sym], [network_output_val, alpha_weights], allow_input_downcast=True)
        else:
            forecast = theano.function([X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_dec_sym, mask_dec], [network_output_val, alpha_weights], allow_input_downcast=True)
    else:
        forecast = theano.function([X_enc_sym, mask_enc, X_enc_sym2, mask_enc2, X_enc_sym3, mask_enc3, X_dec_sym, mask_dec], network_output_val, allow_input_downcast=True)
    return l_out, f_train, f_val, forecast    


def directory(results_directory, params):
    """
    returns the result directory and the file_name with the params
    """
    params_list = [params['data_name'], params['data_type']]
    if params['attention']:
        params_list.append('attention')
        if params['adist']:
            params_list.append('adist')
    else:
        params_list.append('no_attention')
    if params['data_type'] == 'missing':
        params_list.append(params['missing_type'])
        params_list.append('missing_percent_'+str(params['missing_percent']))
        if params['missing_type'] == 'gaps':
            params_list.append('gap_length'+ str(params['gap_length'])) 
        if params['interpolation'] == True:
            params_list.append('interpolation')
            params_list.append(params['interpolation_type'])
        else:        
            if params['padding']: 
                params_list.append('padding')
            else:
                params_list.append('no_padding')
    params_list.append('historysize_' + str(params['windowsize']))
    params_list.append('forecasthorizon_' + str(params['horizon']))
    m_list = ['num_layers', 'num_units', 'num_att_units', 'grad_clipping']
    for key in m_list:
        params_list.append(key + str(params[key]))   
    op_list = ['batch_size', 'max_num_epochs', 'learning_rate', 'regularization_type', 'lambda_regularization']
    for key in op_list:
        params_list.append(key + str(params[key]))
    if params['learning_rate_decay']:
        params_list.append('lr_decay')
        params_list.append('lr_decay_n_epoch_'+ str(params['lr_decay_after_n_epoch']))
        params_list.append('eta_decay_' + str(params['eta_decay']))
    if params['early_stop']:
        params_list.append('early_stop')
        params_list.append('patience_' + str(params['early_stop_patience']))
    else:
        params_list.append('no_early_stop')
    if params['epsilon_improvement']:
        params_list.append('epsilon_' + str(params['epsilon']))
    file_name = '_'.join(params_list) 
    save_to = os.path.join(results_directory)
    for p in params_list:
        save_to = os.path.join(save_to,p)
    save_to = save_to + '/'    
    return save_to, file_name


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


def delta_gap_depth_mask(mask_np):
    """
    returns a matrix same dimensioned as mask_np input
    the matrix is al zeros except the part corresponds to gap
    value for the gap regions corresponds to the depth of that point in the gap
    for instance: for a input mask of [1,1,0,0,0,1] returns [0,0,1,2,3,0] (for str_ind=1) or [0,0,0,1,2,0] (for str_ind=0)
    """
    str_ind = 1 #Ask about this!!!!!
    delta_depth = np.zeros_like(mask_np)
    for idx in range(np.size(mask_np,0)):
        mask = mask_np[idx,:]
        zero_inds = np.where(mask==0)[0]
        g_lengths = gap_lengths(zero_inds)
        inds_array = []
        for g_len in g_lengths:
            inds_array.extend(np.arange(str_ind,str_ind+g_len))
        delta_depth[idx, zero_inds] = np.array(inds_array)
        #delta_depth[idx, zero_inds] = inds_array
    return delta_depth


def train_mvrt4(params, results_directory):
    """
    builds the network and trains the model with according to the given parameters
    """
    logging.info('TRAINING:')
    logging.info('Loading data and creating regression matrix for time series')
    X_train, y_train, X_train_mask, X_val, y_val, X_val_mask, X_test, y_test, X_test_mask = train_test_X_y_mask(params)
    #TODO: is there any better?
    X_train2, X_train_mask2, X_val2, X_val_mask2, X_test2, X_test_mask2 = train_test_X_y_mask_mvrt(params, var = 0)
    X_train3, X_train_mask3, X_val3, X_val_mask3, X_test3, X_test_mask3 = train_test_X_y_mask_mvrt(params, var = 1)
    X_train4, X_train_mask4, X_val4, X_val_mask4, X_test4, X_test_mask4 = train_test_X_y_mask_mvrt(params, var = 2)
    #
    logging.info('Data is loaded')
    logging.info('Building network')
    l_out, f_train, f_val, forecast = network_mvrt4(params)
    logging.info('Network is built')
    num_batches = len(X_train) // params['batch_size']
    #num_val_batches = len(X_val) // params['batch_size']
    # Minibatch generators for the training and validation sets
    train_batches = batch_gen_mvrt4(X_train, y_train, X_train_mask, X_train2, X_train_mask2, X_train3, X_train_mask3, X_train4, X_train_mask4, params['batch_size'])
    #val_batches = batch_gen_mvrt(X_val, y_val, X_val_mask, params['batch_size'], params) #1
    batch_ones = np.ones((params['batch_size'],1)) # as a start signal
    X_mask_dec = np.ones((params['batch_size'], params['horizon']))
    #2:
    X_val = np.expand_dims(X_val, axis = 2)
    X_val_dec = np.ones((np.size(y_val,0), np.size(y_val,1)))
    X_val_mask_dec = np.ones((np.size(y_val,0), np.size(y_val,1)))
    X_val_dec = np.expand_dims(X_val_dec, axis = 2)
    #multi-vrt
    X_val2 = np.expand_dims(X_val2, axis = 2)
    X_val3 = np.expand_dims(X_val3, axis = 2)
    X_val4 = np.expand_dims(X_val4, axis = 2)
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
            X, y, X_mask, X2, X_mask2, X3, X_mask3, X4, X_mask4 = next(train_batches)
            X, X2, X3, X4 = np.expand_dims(X, axis = 2), np.expand_dims(X2, axis = 2), np.expand_dims(X3, axis = 2), np.expand_dims(X4, axis = 2)
            X_dec = np.hstack((batch_ones, y[:,:-1]))
            X_dec = np.expand_dims(X_dec, axis = 2)
            if params['attention'] and params['att_type']=='lambda_mu':
                d_inds = delta_gap_mask(X_mask)
                if params['interpolation']: 
                    loss = f_train(X, np.ones_like(X_mask), X2, np.ones_like(X_mask2), X3, np.ones_like(X_mask3), X4, np.ones_like(X_mask4), X_dec, X_mask_dec, y, X_mask, X_mask2, X_mask3, X_mask4, d_inds)
                else:
                    loss = f_train(X, X_mask, X2, X_mask2, X3, X_mask3, X4, X_mask4, X_dec, X_mask_dec, y, d_inds)
            elif params['attention'] and params['att_type']=='lambda_mu_alt':
                d_inds = delta_gap_depth_mask(X_mask)
                if params['interpolation']:
                    loss = f_train(X, np.ones_like(X_mask), X2, np.ones_like(X_mask2), X3, np.ones_like(X_mask3), X4, np.ones_like(X_mask4), X_dec, X_mask_dec, y, d_inds)
                else:             
                    loss = f_train(X, X_mask, X2, X_mask2, X3, X_mask3, X4, X_mask4, X_dec, X_mask_dec, y, d_inds)                
            else:
                loss = f_train(X, X_mask, X2, X_mask2, X3, X_mask3, X4, X_mask4, X_dec, X_mask_dec, y)
            iter = (epoch - 1) * num_batches + batch_ind # iteration number
            tr_loss += loss
        tr_loss /= num_batches
        loss_train.append(tr_loss)
        #2:
        if params['attention'] and params['att_type']=='lambda_mu':
            d_inds = delta_gap_mask(X_val_mask) #TODO: interpolation!!!
            if params['interpolation']: 
                val_loss = f_val(X_val, np.ones_like(X_val_mask), X_val2, np.ones_like(X_val_mask2), X_val3, np.ones_like(X_val_mask3), X_val4, np.ones_like(X_val_mask4), X_val_dec, X_val_mask_dec, y_val, X_val_mask, X_val_mask2, X_val_mask3, X_val_mask4, d_inds)
            else:
                val_loss = f_val(X_val, X_val_mask, X_val2, X_val_mask2, X_val3, X_val_mask3, X_val4, X_val_mask4, X_val_dec, X_val_mask_dec, y_val, d_inds)                
        elif params['attention'] and params['att_type']=='lambda_mu_alt':
            d_inds = delta_gap_depth_mask(X_val_mask)
            if params['interpolation']:
                val_loss = f_val(X_val, np.ones_like(X_val_mask), X_val2, np.ones_like(X_val_mask2), X_val3, np.ones_like(X_val_mask3), X_val4, np.ones_like(X_val_mask4), X_val_dec, X_val_mask_dec, y_val, d_inds)
            else:
                val_loss = f_val(X_val, X_val_mask, X_val2, X_val_mask2, X_val3, X_val_mask3, X_val4, X_val_mask4, X_val_dec, X_val_mask_dec, y_val, d_inds)
        else:
            val_loss = f_val(X_val, X_val_mask, X_val2, X_val_mask2, X_val3, X_val_mask3, X_val4, X_val_mask4, X_val_dec, X_val_mask_dec, y_val)
        #end of 2:
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


def train_mvrt(params, results_directory):
    """
    builds the network and trains the model with according to the given parameters
    """
    logging.info('Loading data and creating regression matrix for time series')
    X_train, y_train, X_train_mask, X_val, y_val, X_val_mask, X_test, y_test, X_test_mask = train_test_X_y_mask(params)
    #TODO: is there any better?
    X_train2, X_train_mask2, X_val2, X_val_mask2, X_test2, X_test_mask2 = train_test_X_y_mask_mvrt(params, var = 0)
    X_train3, X_train_mask3, X_val3, X_val_mask3, X_test3, X_test_mask3 = train_test_X_y_mask_mvrt(params, var = 1)
    #
    logging.info('Data is loaded')
    logging.info('Building network')
    l_out, f_train, f_val, forecast = network_mvrt3(params)
    logging.info('Network is built')
    num_batches = len(X_train) // params['batch_size']
    num_val_batches = len(X_val) // params['batch_size']
    # Minibatch generators for the training and validation sets
    train_batches = batch_gen_mvrt(X_train, y_train, X_train_mask, X_train2, X_train_mask2, X_train3, X_train_mask3, params['batch_size'])
    #val_batches = batch_gen_mvrt(X_val, y_val, X_val_mask, params['batch_size'], params) #1
    batch_ones = np.ones((params['batch_size'],1)) # as a start signal
    X_mask_dec = np.ones((params['batch_size'], params['horizon']))
    #2:
    X_val = np.expand_dims(X_val, axis = 2)
    X_val_dec = np.ones((np.size(y_val,0), np.size(y_val,1)))
    X_val_mask_dec = np.ones((np.size(y_val,0), np.size(y_val,1)))
    X_val_dec = np.expand_dims(X_val_dec, axis = 2)
    #multi-vrt
    X_val2 = np.expand_dims(X_val2, axis = 2)
    X_val3 = np.expand_dims(X_val3, axis = 2)
    #end of 2:
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
            X, y, X_mask, X2, X_mask2, X3, X_mask3 = next(train_batches)
            X, X2, X3 = np.expand_dims(X, axis = 2), np.expand_dims(X2, axis = 2), np.expand_dims(X3, axis = 2)
            X_dec = np.hstack((batch_ones, y[:,:-1]))
            X_dec = np.expand_dims(X_dec, axis = 2)
            if params['attention'] and params['att_type']=='lambda_mu':
                d_inds = delta_gap_mask(X_mask)
                loss = f_train(X, X_mask, X2, X_mask2, X3, X_mask3, X_dec, X_mask_dec, y, d_inds)                
            else:
                loss = f_train(X, X_mask, X2, X_mask2, X3, X_mask3, X_dec, X_mask_dec, y)
            iter = (epoch - 1) * num_batches + batch_ind # iteration number
            tr_loss += loss
        tr_loss /= num_batches
        loss_train.append(tr_loss)
        #2:
        if params['attention'] and params['att_type']=='lambda_mu':
            d_inds = delta_gap_mask(X_val_mask)
            val_loss = f_val(X_val, X_val_mask, X_val2, X_val_mask2, X_val3, X_val_mask3, X_val_dec, X_val_mask_dec, y_val, d_inds)                
        else:
            val_loss = f_val(X_val, X_val_mask, X_val2, X_val_mask2, X_val3, X_val_mask3, X_val_dec, X_val_mask_dec, y_val)
        #end of 2:
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


def load_parameters2_mvrt(params_csvfile, line_num): 
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
        params['data_variable'] = 'C6H6(GT)'#'PT08.S4(NO2)'
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
        if (params['data_name'] == 'airq') or (params['data_name'] == 'consseason'):
            params['data_file'] =  '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'_intr_linear.pkl'
        else:
            params['data_file'] =  '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'.pkl'
        params['data_file_missing_mask'] = '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'_mask.pkl'
        params['original_data_file'] = params['data_file'] =  '../../../Data/' + params['data_name'] + '/original_and_interpolation/' + params['data_name'] +'.pkl' #!!! for interpolation not the same file!
        params['original_data_mask_file'] = params['data_file_missing_mask']
    else:
        params['data_type'] = 'missing'
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
        params['data2_variables'] = ['NO2(GT)', 'CO(GT)', 'NOx(GT)'] 
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


def train_and_test_mvrt4(params, results_directory):
    """
    builds the network and trains the model with according to the given parameters
    """
    logging.info('TRAINING:')
    logging.info('Loading data and creating regression matrix for time series')
    X_train, y_train, X_train_mask, X_val, y_val, X_val_mask, X_test, y_test, X_test_mask = train_test_X_y_mask(params)
    #TODO: is there any better?
    X_train2, X_train_mask2, X_val2, X_val_mask2, X_test2, X_test_mask2 = train_test_X_y_mask_mvrt(params, var = 0)
    X_train3, X_train_mask3, X_val3, X_val_mask3, X_test3, X_test_mask3 = train_test_X_y_mask_mvrt(params, var = 1)
    X_train4, X_train_mask4, X_val4, X_val_mask4, X_test4, X_test_mask4 = train_test_X_y_mask_mvrt(params, var = 2)
    #
    logging.info('Data is loaded')
    logging.info('Building network')
    l_out, f_train, f_val, forecast = network_mvrt4(params)
    logging.info('Network is built')
    num_batches = len(X_train) // params['batch_size']
    #num_val_batches = len(X_val) // params['batch_size']
    # Minibatch generators for the training and validation sets
    train_batches = batch_gen_mvrt4(X_train, y_train, X_train_mask, X_train2, X_train_mask2, X_train3, X_train_mask3, X_train4, X_train_mask4, params['batch_size'])
    #val_batches = batch_gen_mvrt(X_val, y_val, X_val_mask, params['batch_size'], params) #1
    batch_ones = np.ones((params['batch_size'],1)) # as a start signal
    X_mask_dec = np.ones((params['batch_size'], params['horizon']))
    #2:
    X_val = np.expand_dims(X_val, axis = 2)
    X_val_dec = np.ones((np.size(y_val,0), np.size(y_val,1)))
    X_val_mask_dec = np.ones((np.size(y_val,0), np.size(y_val,1)))
    X_val_dec = np.expand_dims(X_val_dec, axis = 2)
    #multi-vrt
    X_val2 = np.expand_dims(X_val2, axis = 2)
    X_val3 = np.expand_dims(X_val3, axis = 2)
    X_val4 = np.expand_dims(X_val4, axis = 2)
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
            X, y, X_mask, X2, X_mask2, X3, X_mask3, X4, X_mask4 = next(train_batches)
            X, X2, X3, X4 = np.expand_dims(X, axis = 2), np.expand_dims(X2, axis = 2), np.expand_dims(X3, axis = 2), np.expand_dims(X4, axis = 2)
            X_dec = np.hstack((batch_ones, y[:,:-1]))
            X_dec = np.expand_dims(X_dec, axis = 2)
            if params['attention'] and params['att_type']=='lambda_mu':
                d_inds, d_inds2, d_inds3, d_inds4 = delta_gap_mask(X_mask), delta_gap_mask(X_mask2), delta_gap_mask(X_mask3), delta_gap_mask(X_mask4)
                if params['interpolation']: 
                    loss = f_train(X, np.ones_like(X_mask), X2, np.ones_like(X_mask2), X3, np.ones_like(X_mask3), X4, np.ones_like(X_mask4), X_dec, X_mask_dec, y,
                                   X_mask, X_mask2, X_mask3, X_mask4, d_inds, d_inds2, d_inds3, d_inds4)
                else:
                    loss = f_train(X, X_mask, X2, X_mask2, X3, X_mask3, X4, X_mask4, X_dec, X_mask_dec, y, d_inds, d_inds2, d_inds3, d_inds4)
            elif params['attention'] and params['att_type']=='lambda_mu_alt':
                d_inds, d_inds2, d_inds3, d_inds4 = delta_gap_depth_mask(X_mask), delta_gap_depth_mask(X_mask2), delta_gap_depth_mask(X_mask3), delta_gap_depth_mask(X_mask4)
                if params['interpolation']:
                    loss = f_train(X, np.ones_like(X_mask), X2, np.ones_like(X_mask2), X3, np.ones_like(X_mask3), X4, np.ones_like(X_mask4), X_dec, X_mask_dec, y,
                                   d_inds, d_inds2, d_inds3, d_inds4)
                else:             
                    loss = f_train(X, X_mask, X2, X_mask2, X3, X_mask3, X4, X_mask4, X_dec, X_mask_dec, y, d_inds, d_inds2, d_inds3, d_inds4)                
            else:
                loss = f_train(X, X_mask, X2, X_mask2, X3, X_mask3, X4, X_mask4, X_dec, X_mask_dec, y)
            iter = (epoch - 1) * num_batches + batch_ind # iteration number
            tr_loss += loss
        tr_loss /= num_batches
        loss_train.append(tr_loss)
        #2:
        if params['attention'] and params['att_type']=='lambda_mu':
            d_inds, d_inds2, d_inds3, d_inds4 = delta_gap_mask(X_val_mask), delta_gap_mask(X_val_mask2), delta_gap_mask(X_val_mask3), delta_gap_mask(X_val_mask4)
            if params['interpolation']: 
                val_loss = f_val(X_val, np.ones_like(X_val_mask), X_val2, np.ones_like(X_val_mask2), X_val3, np.ones_like(X_val_mask3), X_val4, np.ones_like(X_val_mask4), X_val_dec, X_val_mask_dec, y_val,
                                 X_val_mask, X_val_mask2, X_val_mask3, X_val_mask4, d_inds, d_inds2, d_inds3, d_inds4)
            else:
                val_loss = f_val(X_val, X_val_mask, X_val2, X_val_mask2, X_val3, X_val_mask3, X_val4, X_val_mask4, X_val_dec, X_val_mask_dec, y_val,
                                 d_inds, d_inds2, d_inds3, d_inds4)                
        elif params['attention'] and params['att_type']=='lambda_mu_alt':
            d_inds, d_inds2, d_inds3, d_inds4 = delta_gap_depth_mask(X_val_mask), delta_gap_depth_mask(X_val_mask2), delta_gap_depth_mask(X_val_mask3), delta_gap_depth_mask(X_val_mask4)
            if params['interpolation']:
                val_loss = f_val(X_val, np.ones_like(X_val_mask), X_val2, np.ones_like(X_val_mask2), X_val3, np.ones_like(X_val_mask3), X_val4, np.ones_like(X_val_mask4), X_val_dec, X_val_mask_dec, y_val,
                                 d_inds, d_inds2, d_inds3, d_inds4)
            else:
                val_loss = f_val(X_val, X_val_mask, X_val2, X_val_mask2, X_val3, X_val_mask3, X_val4, X_val_mask4, X_val_dec, X_val_mask_dec, y_val,
                                 d_inds, d_inds2, d_inds3, d_inds4)
        else:
            val_loss = f_val(X_val, X_val_mask, X_val2, X_val_mask2, X_val3, X_val_mask3, X_val4, X_val_mask4, X_val_dec, X_val_mask_dec, y_val)
        #end of 2:
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
    #
    logging.info('TESTING:')
    logging.info('Loading best parameters.')
    loaded_params = load_checkpoint(save_to+ 'best.pkl')
    lasagne.layers.set_all_param_values(l_out, loaded_params) 
    logging.info('Best parameters are loaded.')
    #
    X_test = np.expand_dims(X_test, axis = 2)
    X_dec_test = np.ones((np.size(y_test,0), np.size(y_test,1)))
    X_test_mask_dec =  np.ones((np.size(y_test,0), np.size(y_test,1)))
    X_dec_test = np.expand_dims(X_dec_test, axis = 2)
    X_test2 = np.expand_dims(X_test2, axis = 2)
    X_test3 = np.expand_dims(X_test3, axis = 2)
    X_test4 = np.expand_dims(X_test4, axis = 2)
    #
    logging.info('Starting forecasting')
    if params['attention']:
        if params['att_type']=='lambda_mu':
            d_inds, d_inds2, d_inds3, d_inds4 = delta_gap_mask(X_test_mask), delta_gap_mask(X_test_mask2), delta_gap_mask(X_test_mask3), delta_gap_mask(X_test_mask4)
            if params['interpolation']:
                #TODO: change model to take a separate mask for delta!
                y_pred, y_weights = forecast(X_test, np.ones_like(X_test_mask), X_test2, np.ones_like(X_test_mask2), X_test3, np.ones_like(X_test_mask3), X_test4, np.ones_like(X_test_mask4), X_dec_test, X_test_mask_dec,
                                             X_test_mask, X_test_mask2, X_test_mask3, X_test_mask4, d_inds, d_inds2, d_inds3, d_inds4)
            else:
                y_pred, y_weights = forecast(X_test, X_test_mask, X_test2, X_test_mask2, X_test3, X_test_mask3, X_test4, X_test_mask4, X_dec_test, X_test_mask_dec,
                                             d_inds, d_inds2, d_inds3, d_inds4)
        elif params['att_type']=='lambda_mu_alt':
            d_inds, d_inds2, d_inds3, d_inds4 = delta_gap_depth_mask(X_test_mask), delta_gap_depth_mask(X_test_mask2), delta_gap_depth_mask(X_test_mask3), delta_gap_depth_mask(X_test_mask4)
            if params['interpolation']:
                y_pred, y_weights = forecast(X_test, np.ones_like(X_test_mask), X_test2, np.ones_like(X_test_mask2), X_test3, np.ones_like(X_test_mask3), X_test4, np.ones_like(X_test_mask4), X_dec_test, X_test_mask_dec,
                                             d_inds, d_inds2, d_inds3, d_inds4)
            else:
                y_pred, y_weights = forecast(X_test, X_test_mask, X_test2, X_test_mask2, X_test3, X_test_mask3, X_test4, X_test_mask4, X_dec_test, X_test_mask_dec,
                                             d_inds, d_inds2, d_inds3, d_inds4)
        elif params['att_type']=='adist' and params['interpolation']:
            #TODO: change model to take separate mask for delta!
            logging.info('TODO:')
            y_pred, y_weights = forecast(X_test, X_test_mask, X_test2, X_test_mask2, X_test3, X_test_mask3, X_test4, X_test_mask4,  X_dec_test, X_test_mask_dec)
        else:            
            y_pred, y_weights = forecast(X_test, X_test_mask, X_test2, X_test_mask2, X_test3, X_test_mask3, X_test4, X_test_mask4,  X_dec_test, X_test_mask_dec)
        pickle.dump(y_weights, open(save_to + 'y_weights.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(X_test, open(save_to + 'X_test.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(X_test_mask, open(save_to + 'X_test_mask.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        y_pred = forecast(X_test, X_test_mask, X_test2, X_test_mask2, X_test3, X_test_mask3, X_test4, X_test_mask4, X_dec_test, X_test_mask_dec)
    mse = ((y_pred - y_test) ** 2).mean(axis=0)
    pickle.dump(mse, open(save_to + 'mse.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    logging.info('Test Prediction MSE: ' + str(mse.mean()))
    pickle.dump(y_test, open(save_to + 'y_test.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(y_pred, open(save_to + 'y_pred.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    mape = mean_abs_percent_err(y_pred, y_test)
    smape = sym_mean_abs_percent_err(y_pred, y_test)
    logging.info('Test Prediction MAPE: ' + str(mape.mean()))
    logging.info('Test Prediction SMAPE: ' + str(smape.mean()))
    pickle.dump(mape, open(save_to + 'mape.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(smape, open(save_to + 'smape.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)   
    return loss_train, loss_validation




def test_mvrt4(params, results_directory):
    logging.info('TESTING:')
    save_to = directory_v2(results_directory, params)
    logging.info('Loading data and creating regression matrix for time series')
    try:
        params = pickle.load(open(save_to + 'parameters.pkl', 'rb'))
        X_test, y_test, X_test_mask = test_X_y_mask(params)
    except:
        _, _, _, _, _, _, X_test, y_test, X_test_mask = train_test_X_y_mask(params)
    #TODO: is there any better way?
    _, _, _, _, X_test2, X_test_mask2 = train_test_X_y_mask_mvrt(params, var = 0)
    _, _, _, _, X_test3, X_test_mask3 = train_test_X_y_mask_mvrt(params, var = 1)
    _, _, _, _, X_test4, X_test_mask4 = train_test_X_y_mask_mvrt(params, var = 2)
    logging.info('Data is loaded')
    logging.info('Building network')
    l_out, f_train, f_val, forecast = network_mvrt4(params)
    logging.info('Network is built')
    logging.info('Best parameters are loaded.')
    loaded_params = load_checkpoint(save_to+ 'best.pkl')
    lasagne.layers.set_all_param_values(l_out, loaded_params)    
    #
    X_test = np.expand_dims(X_test, axis = 2)
    X_dec_test = np.ones((np.size(y_test,0), np.size(y_test,1)))
    X_test_mask_dec =  np.ones((np.size(y_test,0), np.size(y_test,1)))
    X_dec_test = np.expand_dims(X_dec_test, axis = 2)
    X_test2 = np.expand_dims(X_test2, axis = 2)
    X_test3 = np.expand_dims(X_test3, axis = 2)
    X_test4 = np.expand_dims(X_test4, axis = 2)
    #
    logging.info('Starting forecasting')
    if params['attention']:
        if params['att_type']=='lambda_mu':
            d_inds, d_inds2, d_inds3, d_inds4 = delta_gap_mask(X_test_mask), delta_gap_mask(X_test_mask2), delta_gap_mask(X_test_mask3), delta_gap_mask(X_test_mask4)
            if params['interpolation']:
                #TODO: change model to take a separate mask for delta!
                y_pred, y_weights = forecast(X_test, np.ones_like(X_test_mask), X_test2, np.ones_like(X_test_mask2), X_test3, np.ones_like(X_test_mask3), X_test4, np.ones_like(X_test_mask4), X_dec_test, X_test_mask_dec,
                                             X_test_mask, X_test_mask2, X_test_mask3, X_test_mask4, d_inds, d_inds2, d_inds3, d_inds4)
            else:
                y_pred, y_weights = forecast(X_test, X_test_mask, X_test2, X_test_mask2, X_test3, X_test_mask3, X_test4, X_test_mask4, X_dec_test, X_test_mask_dec,
                                             d_inds, d_inds2, d_inds3, d_inds4)
        elif params['att_type']=='lambda_mu_alt':
            d_inds, d_inds2, d_inds3, d_inds4 = delta_gap_depth_mask(X_test_mask), delta_gap_depth_mask(X_test_mask2), delta_gap_depth_mask(X_test_mask3), delta_gap_depth_mask(X_test_mask4)
            if params['interpolation']:
                y_pred, y_weights = forecast(X_test, np.ones_like(X_test_mask), X_test2, np.ones_like(X_test_mask2), X_test3, np.ones_like(X_test_mask3), X_test4, np.ones_like(X_test_mask4), X_dec_test, X_test_mask_dec,
                                             d_inds, d_inds2, d_inds3, d_inds4)
            else:
                y_pred, y_weights = forecast(X_test, X_test_mask, X_test2, X_test_mask2, X_test3, X_test_mask3, X_test4, X_test_mask4, X_dec_test, X_test_mask_dec,
                                             d_inds, d_inds2, d_inds3, d_inds4)
        elif params['att_type']=='adist' and params['interpolation']:
            #TODO: change model to take separate mask for delta!
            logging.info('TODO:')
            y_pred, y_weights = forecast(X_test, X_test_mask, X_test2, X_test_mask2, X_test3, X_test_mask3, X_test4, X_test_mask4,  X_dec_test, X_test_mask_dec)
        else:            
            y_pred, y_weights = forecast(X_test, X_test_mask, X_test2, X_test_mask2, X_test3, X_test_mask3, X_test4, X_test_mask4,  X_dec_test, X_test_mask_dec)
        pickle.dump(y_weights, open(save_to + 'y_weights.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(X_test, open(save_to + 'X_test.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(X_test_mask, open(save_to + 'X_test_mask.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        y_pred = forecast(X_test, X_test_mask, X_test2, X_test_mask2, X_test3, X_test_mask3, X_test4, X_test_mask4, X_dec_test, X_test_mask_dec)
    mse = ((y_pred - y_test) ** 2).mean(axis=0)
    pickle.dump(mse, open(save_to + 'mse.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    logging.info('Test Prediction MSE: ' + str(mse.mean()))
    pickle.dump(y_test, open(save_to + 'y_test.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(y_pred, open(save_to + 'y_pred.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    mape = mean_abs_percent_err(y_pred, y_test)
    smape = sym_mean_abs_percent_err(y_pred, y_test)
    logging.info('Test Prediction MAPE: ' + str(mape.mean()))
    logging.info('Test Prediction SMAPE: ' + str(smape.mean()))
    pickle.dump(mape, open(save_to + 'mape.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(smape, open(save_to + 'smape.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)    



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
    X_test = np.expand_dims(X_test, axis = 2)
    X_dec_test = np.ones((np.size(y_test,0), np.size(y_test,1)))
    X_test_mask_dec =  np.ones((np.size(y_test,0), np.size(y_test,1)))
    X_dec_test = np.expand_dims(X_dec_test, axis = 2)
    logging.info('Starting forecasting')
    if params['attention']:
        if params['att_type']=='lambda_mu':
            d_inds_test = delta_gap_mask(X_test_mask)
            if params['interpolation']:
                y_pred, y_weights = forecast(X_test, np.ones_like(X_test_mask), X_dec_test, X_test_mask_dec, X_test_mask, d_inds_test)
            else:
                y_pred, y_weights = forecast(X_test, X_test_mask, X_dec_test, X_test_mask_dec, d_inds_test)
        elif params['att_type']=='lambda_mu_alt':
            d_inds = delta_gap_depth_mask(X_mask)
            if params['interpolation']: 
                y_pred, y_weights = forecast(X_test, np.ones_like(X_test_mask), X_dec_test, X_test_mask_dec, d_inds_test)
            else:
                y_pred, y_weights = forecast(X_test, X_test_mask, X_dec_test, X_test_mask_dec, d_inds_test)
        else:            
            y_pred, y_weights = forecast(X_test, X_test_mask, X_dec_test, X_test_mask_dec)
        pickle.dump(X_test, open(save_to + 'X_test.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_weights, open(save_to + 'y_weights.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(X_test, open(save_to + 'X_test.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(X_test_mask, open(save_to + 'X_test_mask.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
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



def test_mvrt(params, results_directory):
    logging.info('Loading data and creating regression matrix for time series')    
    _, _, _, X_val, y_val, X_val_mask, X_test, y_test, X_test_mask = train_test_X_y_mask(params)
    #TODO: is there any better?
    _, _, X_val2, X_val_mask2, X_test2, X_test_mask2 = train_test_X_y_mask_mvrt(params, var = 0)
    _, _, X_val3, X_val_mask3, X_test3, X_test_mask3 = train_test_X_y_mask_mvrt(params, var = 1)
    save_to = directory_v2(results_directory, params)
    logging.info('Data is loaded')
    logging.info('Building network')
    l_out, f_train, f_val, forecast = network_mvrt3(params)
    logging.info('Network is built')
    logging.info('Best parameters are loaded.')
    loaded_params = load_checkpoint(save_to+ 'best.pkl')
    lasagne.layers.set_all_param_values(l_out, loaded_params)
    #
    X_val = np.expand_dims(X_val, axis = 2)
    X_dec_val = np.ones((np.size(y_val,0), np.size(y_val,1)))
    X_val_mask_dec =  np.ones((np.size(y_val,0), np.size(y_val,1)))
    X_dec_val = np.expand_dims(X_dec_val, axis = 2)
    X_val2 = np.expand_dims(X_val2, axis = 2)
    X_val3 = np.expand_dims(X_val3, axis = 2)
    #
    X_test = np.expand_dims(X_test, axis = 2)
    X_dec_test = np.ones((np.size(y_test,0), np.size(y_test,1)))
    X_test_mask_dec =  np.ones((np.size(y_test,0), np.size(y_test,1)))
    X_dec_test = np.expand_dims(X_dec_test, axis = 2)
    X_test2 = np.expand_dims(X_test2, axis = 2)
    X_test3 = np.expand_dims(X_test3, axis = 2)
    #
    logging.info('Starting forecasting')
    if params['attention']:
        if params['att_type']=='lambda_mu':
            d_inds_val = delta_gap_mask(X_val_mask)
            d_inds_test = delta_gap_mask(X_test_mask)
            y_pred_val, y_weights_val = forecast(X_val, X_val_mask, X_val2, X_val_mask2, X_val3, X_val_mask3, X_dec_val, X_val_mask_dec, d_inds_val)
            y_pred, y_weights = forecast(X_test, X_test_mask, X_test2, X_test_mask2, X_test3, X_test_mask3, X_dec_test, X_test_mask_dec, d_inds_test)
        else:            
            y_pred_val, y_weights_val = forecast(X_val, X_val_mask, X_val2, X_val_mask2, X_val3, X_val_mask3, X_dec_val, X_val_mask_dec)
            y_pred, y_weights = forecast(X_test, X_test_mask, X_test2, X_test_mask2, X_test3, X_test_mask3, X_dec_test, X_test_mask_dec)
            #np.save(save_to + 'y_alphas.npy', y_weights)
        pickle.dump(X_test, open(save_to + 'X_test.pkl', 'wb'))
        pickle.dump(y_weights, open(save_to + 'y_weights.pkl', 'wb'))
    else:
        #y_pred_train = forecast(X_train, X_train_mask, X_dec_train, X_train_mask_dec)
        y_pred_val = forecast(X_val, X_val_mask, X_val2, X_val_mask2, X_val3, X_val_mask3, X_dec_val, X_val_mask_dec)
        y_pred = forecast(X_test, X_test_mask, X_test2, X_test_mask2, X_test3, X_test_mask3, X_dec_test, X_test_mask_dec)
    mse_val = ((y_pred_val - y_val) ** 2).mean(axis=0)
    logging.info('Validation Prediction MSE: ' + str(mse_val.mean()))
    pickle.dump(mse_val, open(save_to + 'mse_val.pkl', 'wb'))
    mse = ((y_pred - y_test) ** 2).mean(axis=0)
    pickle.dump(mse, open(save_to + 'mse.pkl', 'wb'))
    logging.info('Test Prediction MSE: ' + str(mse.mean()))
    pickle.dump(y_test, open(save_to + 'y_test.pkl', 'wb'))
    pickle.dump(y_pred, open(save_to + 'y_pred.pkl', 'wb'))
    mape = mean_abs_percent_err(y_pred, y_test)
    smape = sym_mean_abs_percent_err(y_pred, y_test)
    logging.info('Test Prediction MAPE: ' + str(mape.mean()))
    logging.info('Test Prediction SMAPE: ' + str(smape.mean()))
    pickle.dump(mape, open(save_to + 'mape.pkl', 'wb'))
    pickle.dump(smape, open(save_to + 'smape.pkl', 'wb'))    
    mape_val = mean_abs_percent_err(y_pred_val, y_val)
    smape_val = sym_mean_abs_percent_err(y_pred_val, y_val)
    pickle.dump(mape_val, open(save_to + 'mape_val.pkl', 'wb'))
    pickle.dump(smape_val, open(save_to + 'smape_val.pkl', 'wb'))
    logging.info('Validation Prediction MAPE: ' + str(mape_val.mean()))
    logging.info('Validation Prediction SMAPE: ' + str(smape_val.mean()))



def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    results_directory = 'Results_multivariate_intrl/'
    #    
    train_and_test = True
    if len(sys.argv) > 5:
        strt = int(sys.argv[2]) #strt ind of line 
        end = int(sys.argv[3]) +1#end ind of line
        order = sys.argv[4] # order from top the bottom or reverse
        to_dos = sys.argv[5] # if we want only to test "only_test" otherwise "train_and_test"
        if to_dos == 'testonly':
           train_and_test = False 
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
        params = load_parameters2_mvrt(sys.argv[1], i)
        #params = load_parameters2_mvrt(params_file, line_ind)
        logging.info(params)
        #
        if train_and_test:
            #train_loss, validation_loss = train_mvrt4(params, results_directory)
            train_loss, validation_loss = train_and_test_mvrt4(params, results_directory)
        else:
            test_mvrt4(params, results_directory)
        #
        #test_mvrt4(params, results_directory)
        #        
        f = open(str(sys.argv[1])+"_mvrt_intr.log","a")
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
