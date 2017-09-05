# -*- coding: utf-8 -*-
"""
Created on Fri May 19 16:17:38 2017

@author: cinar
"""

class Encoder_Set(EncoderDecoderBase):
    def init_params(self):
        """ sent weights """
        #self.W_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.idim, self.rankdim), name='W_emb'))
        self.Wo_hh = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.setdim+self.edim, self.setdim), broadcastable=[False,False], name='Wo_hh'))
        self.bo_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.setdim,), dtype='float32'), name='bo_hh'))
        self.W_a = add_to_params(self.params, theano.shared(value=np.zeros((self.adim,), dtype='float32'), name='W_a'))
        self.W_am = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.adim, self.edim), name='W_am'))
        self.W_aq = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.adim, self.setdim), name='W_aq'))
        self.b_a = add_to_params(self.params, theano.shared(value=np.zeros((self.adim,), dtype='float32'), name='b_a'))
        
        #self.T = 
        if self.set_step_type == "gated":
            #self.W_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in_r'))
            #self.W_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in_z'))
            self.Wo_hh_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.setdim+self.edim, self.setdim), name='Wo_hh_r'))
            self.Wo_hh_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.setdim+self.edim, self.setdim), name='Wo_hh_z'))
            self.bo_z = add_to_params(self.params, theano.shared(value=np.zeros((self.setdim,), dtype='float32'), name='bo_z'))
            self.bo_r = add_to_params(self.params, theano.shared(value=np.zeros((self.setdim,), dtype='float32'), name='bo_r'))
    
    def plain_et_step(self, x_snp, o_t0):
        #reading from memory steps
        #bs, seq_len_m, _ = x_snp.shape
        m_in = x_snp.dimshuffle(1, 0, 2)
        e_qt = T.dot(o_t0, self.W_aq)                       
        e_m = T.dot(m_in, self.W_am)
        e_q = T.tile(e_qt, (self.seq_len_m, 1, 1))
        et_p = T.tanh(e_m + e_q + self.b_a)
        et = T.dot(et_p, self.W_a)
        alpha = T.exp(et)
        alpha /= T.sum(alpha, axis=0)
        mt = x_snp.dimshuffle(2, 1, 0)
        mult = T.mul(mt, alpha)
        rt = T.sum(mult, axis=1)
        return rt.T
    
    def seq_enc_step(self, h_tm1, x_snp):
        #xr, xc, xz = x_snp.shape
        #simple rnn
        #o_t = T.tanh(T.dot(hr_tm1, self.Wo_hh) + self.bo_hh)
        #gru:
        r_t = T.nnet.sigmoid(T.dot(hr_tm1, self.Wo_hh_r) + self.bo_r)
        z_t = T.nnet.sigmoid(T.dot(hr_tm1, self.W_hh_z) + self.b_z)
        h_tilde = T.tanh(T.dot(r_t * hr_tm1, self.W_hh) + self.b_hh)
        o_t = (np.float32(1.0) - z_t) * hr_tm1 + z_t * h_tilde
        #o_t: GRU hidden state
        #concatanation of hidden state and reading from memory
        rt = self.plain_et_step(x_snp, o_t)
        h_t = T.concatenate([o_t0,rt], axis=1)
        return h_t
    
    _res, _ = theano.scan(seq_enc_step, outputs_info=o_enc_info, sequences=[xmask] non_sequences=[x_snp, snp_mask, m_snp_count], n_steps=self.num_set_iter)