import itertools
import csv
import sys

data_name = "polish_electricity_v2"
data_file = "../../../Data/polish_v2/original/pse_2years_2hrs.csv"
train_test_ratio = 0.75
train_val_ratio = 0.75
windowsize = 96
horizon = 4
data_type = "original"
missing_type = None
data_file_missing = None
data_file_missing_mask = None
padding = False
interpolation = False
interpolation_type = None
alg = "adam"
max_num_epochs = 25
batch_size = 32
learning_rate_decay = False
lr_decay_after_n_epoch = 50
eta_decay = 0.95
early_stop = True
early_stop_patience = 10000
patience_increase = 2
epsilon_improvement = True
epsilon = 0.01
num_layers = 1
attention = False
num_att_units = 128
grad_clipping = 100
gap_len = 10
missing_perc = 20
###################################################
learning_rate = [0.001, 0.01, 0.05]
regularization_type = ["l1","l2"]
lambda_regularization = [0.0001, 0.001, 0.01]
num_units = [128, 256]
###################################################
all_lists = [learning_rate,regularization_type,lambda_regularization,num_units]
all_possib = list(itertools.product(*all_lists))
###################################################
header = "data_name,data_file,train_test_ratio,train_val_ratio,windowsize,horizon,data_type,missing_type,data_file_missing,data_file_missing_mask,padding,interpolation,interpolation_type,alg,max_num_epochs,batch_size,learning_rate,learning_rate_decay,lr_decay_after_n_epoch,eta_decay,regularization_type,lambda_regularization,early_stop,early_stop_patience,patience_increase,epsilon_improvement,epsilon,num_units,num_layers,attention,num_att_units,grad_clipping,gap_len,missing_perc"
hdr = header.split(",")
wr = csv.writer(open(sys.argv[1], "w"))
wr.writerow(hdr)
ID = 1
top3 = [7,8,19]
for A in all_possib:
    if ID not in top3:
        ID = ID + 1
        continue
    print ID,',',A[0],',',A[1],',',A[2],',',A[3]
    r = []
    r.append(data_name+"_ID"+str(ID))
    r.append(data_file)
    r.append(str(train_test_ratio))
    r.append(str(train_val_ratio))
    r.append(str(windowsize))
    r.append(str(horizon))
    r.append(data_type)
    r.append(missing_type)
    r.append(data_file_missing)
    r.append(data_file_missing_mask)
    r.append(padding)
    r.append(interpolation)
    r.append(interpolation_type)
    r.append(alg)
    r.append(max_num_epochs)
    r.append(batch_size)
    r.append(str(A[0]))
    r.append(learning_rate_decay)
    r.append(lr_decay_after_n_epoch)
    r.append(eta_decay)
    r.append(A[1])
    r.append(A[2])
    r.append(early_stop)
    r.append(str(early_stop_patience))
    r.append(str(patience_increase))
    r.append(epsilon_improvement)
    r.append(str(epsilon))
    r.append(str(A[3]))
    r.append(str(num_layers))
    r.append(attention)
    r.append(str(num_att_units))
    r.append(str(grad_clipping))
    r.append(str(gap_len))
    r.append(str(missing_perc))
    wr.writerow(r)
    ID = ID + 1

