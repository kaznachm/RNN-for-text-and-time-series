******************GRU MODEL******************

This folder is almost the same as simple folder “codes”. 
What is different:

1. A file “prediction_rnn_mvrt_intrl_GRU.py” is added.
This file is a copy of a file “prediction_rnn_mvrt_intrl.py”, but in function network_mvrt4() the call of function model_seq2seq_mvrt4() is replaced with model_seq2seq_mvrt4_GRU()

2. In file “models_mvrt.py” a function model_seq2seq_mvrt4_GRU() is added.
This function is a copy of model_seq2seq_mvrt4(), but LSTM model is replaced with GRU model.


Example of compilation:
python prediction_rnn_mvrt_intrl_GRU.py params/zombie-dust.param 6 6

*********************************************

#
For univariate padded missing values:

python prediction_rnn.py params/zombie-dust.param.a 
python prediction_rnn.py params/fuzzy.param.b 1 10 #default the lines read and runned in the params file
python prediction_rnn.py params/tiger.param.c 1 10 normal #increamental order of reading lines in the params file, deafult is normal e.g. 1->10
python prediction_rnn.py params/racer.param.d 1 10 reverse #reverse order of reading lines in the params file, deafult is normal e.g. 10->1


For univariate interpolated missing values:

python prediction_rnn_linear_intr.py params/zombie-dust.param.a 
python prediction_rnn_linear_intr.py params/zombie-dust.param.a 1 10 #default the lines read and runned in the params file
python prediction_rnn_linear_intr.py params/zombie-dust.param.a 1 10 normal #increamental order of reading lines in the params file, deafult is normal e.g. 1->10
python prediction_rnn_linear_intr.py params/zombie-dust.param.a 1 10 reverse #reverse order of reading lines in the params file, deafult is normal e.g. 10->1

For multivariate


