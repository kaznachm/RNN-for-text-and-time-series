# lists of possible options
num_rnn_units = [128, 256]
num_attn_units = [256,512]
ds_types = ['orig', '5', '10', '15', '20']
ds_names = ['pse', 'consseason', 'weather_data', 'airq']
methods = ['RNN', 'RNN-ATT', 'ATT-lambda', 'ATT-lambda-mu', 'ATT-dist']

# dictionary to select server name based on the rnn_units and attn_units
server_name_dict = {(128,256): 'zombie-dust', (128,512): 'fuzzy', (256,256): 'tiger', (256,512): 'racer'}

# to generate csv files
for rnn_units in num_rnn_units:
    for attn_units in num_attn_units:
        # to create file name
        server_name = server_name_dict[(rnn_units, attn_units)]
        fname = './params/' + server_name + '.param'
        fline_list = []

        # write the params
        param_file = open(fname,'w')
        for ds_type in ds_types:
            for ds_name in ds_names:
                if (ds_name is 'airq') and (ds_type is '5'):
                    continue
                for method in methods:
                    param_file.write(str(rnn_units) + ',' + str(attn_units) + ',' + ds_type + ',' + ds_name + ',' + method + '\n')
# end
                        
                        
                
