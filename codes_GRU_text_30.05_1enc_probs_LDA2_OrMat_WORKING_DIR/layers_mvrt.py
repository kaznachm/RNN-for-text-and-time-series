# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:00:21 2017

@author: yagmur
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 15:45:58 2017

@author: yagmur
"""

import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan
from lasagne.layers.base import Layer, MergeLayer
from lasagne.layers.input import InputLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers import helper, SliceLayer
from lasagne.layers.recurrent import Gate


class myGate(object):
    """
    lasagne.layers.recurrent.Gate(W_in=lasagne.init.Normal(0.1),
    W_hid=lasagne.init.Normal(0.1), W_cell=lasagne.init.Normal(0.1),
    b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.sigmoid)
    Simple class to hold the parameters for a gate connection.  We define
    a gate loosely as something which computes the linear mix of two inputs,
    optionally computes an element-wise product with a third, adds a bias, and
    applies a nonlinearity.
    Parameters
    ----------
    W_in : Theano shared variable, numpy array or callable
        Initializer for input-to-gate weight matrix.
    W_hid : Theano shared variable, numpy array or callable
        Initializer for hidden-to-gate weight matrix.
    W_cell : Theano shared variable, numpy array, callable, or None
        Initializer for cell-to-gate weight vector.  If None, no cell-to-gate
        weight vector will be stored.
    b : Theano shared variable, numpy array or callable
        Initializer for input gate bias vector.
    nonlinearity : callable or None
        The nonlinearity that is applied to the input gate activation. If None
        is provided, no nonlinearity will be applied.
    Examples
    --------
    For :class:`LSTMLayer` the bias of the forget gate is often initialized to
    a large positive value to encourage the layer initially remember the cell
    value, see e.g. [1]_ page 15.
    >>> import lasagne
    >>> forget_gate = Gate(b=lasagne.init.Constant(5.0))
    >>> l_lstm = LSTMLayer((10, 20, 30), num_units=10,
    ...                    forgetgate=forget_gate)
    References
    ----------
    .. [1] Gers, Felix A., Jürgen Schmidhuber, and Fred Cummins. "Learning to
           forget: Continual prediction with LSTM." Neural computation 12.10
           (2000): 2451-2471.
    """
    def __init__(self, W_in=init.GlorotNormal(), W_hid=init.GlorotNormal(),
                 W_cell=init.Normal(0.1), b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
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
            

class Gate_setenc(object):
    """
    lasagne.layers.recurrent.Gate(W_in=lasagne.init.Normal(0.1),
    W_hid=lasagne.init.Normal(0.1), W_cell=lasagne.init.Normal(0.1),
    b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.sigmoid)
    Simple class to hold the parameters for a gate connection.  We define
    a gate loosely as something which computes the linear mix of two inputs,
    optionally computes an element-wise product with a third, adds a bias, and
    applies a nonlinearity.
    Parameters
    ----------
    W_in : Theano shared variable, numpy array or callable
        Initializer for input-to-gate weight matrix.
    W_hid : Theano shared variable, numpy array or callable
        Initializer for hidden-to-gate weight matrix.
    W_cell : Theano shared variable, numpy array, callable, or None
        Initializer for cell-to-gate weight vector.  If None, no cell-to-gate
        weight vector will be stored.
    b : Theano shared variable, numpy array or callable
        Initializer for input gate bias vector.
    nonlinearity : callable or None
        The nonlinearity that is applied to the input gate activation. If None
        is provided, no nonlinearity will be applied.
    Examples
    --------
    For :class:`LSTMLayer` the bias of the forget gate is often initialized to
    a large positive value to encourage the layer initially remember the cell
    value, see e.g. [1]_ page 15.
    >>> import lasagne
    >>> forget_gate = Gate(b=lasagne.init.Constant(5.0))
    >>> l_lstm = LSTMLayer((10, 20, 30), num_units=10,
    ...                    forgetgate=forget_gate)
    References
    ----------
    .. [1] Gers, Felix A., Jürgen Schmidhuber, and Fred Cummins. "Learning to
           forget: Continual prediction with LSTM." Neural computation 12.10
           (2000): 2451-2471.
    """
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



class AnaLayer4(MergeLayer):
    def __init__(self, incoming, num_units, att_num_units = 64, contxt_input=init.Constant(0.), 
                 contxt_input2=init.Constant(0.), contxt_input3=init.Constant(0.), 
                 contxt_input4=init.Constant(0.), W_att=init.Normal(0.1), 
                 W_att2=init.Normal(0.1), W_att3=init.Normal(0.1), W_att4=init.Normal(0.1), 
                 W_hid_to_att=init.GlorotNormal(), W_hid_to_att2=init.GlorotNormal(), W_hid_to_att3=init.GlorotNormal(),
                 W_hid_to_att4=init.GlorotNormal(), W_ctx_to_att=init.GlorotNormal(),
                 W_ctx_to_att2=init.GlorotNormal(), W_ctx_to_att3=init.GlorotNormal(), W_ctx_to_att4=init.GlorotNormal(), 
                 **kwargs):        
        incomings = [incoming]        
        self.contxt_input_incoming_index = -1
        self.contxt_input2_incoming_index = -1
        self.contxt_input3_incoming_index = -1
        self.contxt_input4_incoming_index = -1
        if isinstance(contxt_input, Layer):
            incomings.append(contxt_input)
            self.contxt_input_incoming_index = len(incomings)-1 
        if isinstance(contxt_input2, Layer):
            incomings.append(contxt_input2)
            self.contxt_input2_incoming_index = len(incomings)-1 
        if isinstance(contxt_input3, Layer):
            incomings.append(contxt_input3)
            self.contxt_input3_incoming_index = len(incomings)-1
        if isinstance(contxt_input4, Layer):
            incomings.append(contxt_input4)
            self.contxt_input4_incoming_index = len(incomings)-1 
        super(AnaLayer4, self).__init__(incomings, **kwargs)
        self.num_units = num_units  #number of attention units
        #num_inputs = np.prod(input_shape[2:])
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_hid_to_att2 = self.add_param(W_hid_to_att2, (num_units, att_num_units), name='W_hid_to_att2')
        self.W_hid_to_att3 = self.add_param(W_hid_to_att3, (num_units, att_num_units), name='W_hid_to_att3')
        self.W_hid_to_att4 = self.add_param(W_hid_to_att4, (num_units, att_num_units), name='W_hid_to_att4')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_ctx_to_att2 = self.add_param(W_ctx_to_att2, (2*num_units, att_num_units), name='W_ctx_to_att2')
        self.W_ctx_to_att3 = self.add_param(W_ctx_to_att3, (2*num_units, att_num_units), name='W_ctx_to_att3')
        self.W_ctx_to_att4 = self.add_param(W_ctx_to_att4, (2*num_units, att_num_units), name='W_ctx_to_att4')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        self.W_att2 = self.add_param(W_att2, (att_num_units,), name='W_att2')
        self.W_att3 = self.add_param(W_att3, (att_num_units,), name='W_att3')
        self.W_att4 = self.add_param(W_att4, (att_num_units,), name='W_att4')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        self.att_num_units = att_num_units
        if isinstance(contxt_input, Layer):
            self.contxt_input = contxt_input
            #_, self.seq_len_enc, ctx_fea_len = contxt_input.shape
    def get_output_shape_for(self, input_shape):
        #_, seq_len_enc, _ = self.contxt_input.shape
        return (input_shape[0][0], None)
    def get_output_for(self, inputs, **kwargs):
        hid_previous = inputs[0]      
        #
        def calculate_alpha(ctx_input, ctx2att_mtx, hid2att_mtx, v_a):
            _, seq_len_enc, _ = ctx_input.shape
            contxt_sh = ctx_input.dimshuffle(1, 0, 2)
            pre_comp_ctx = T.dot(contxt_sh, ctx2att_mtx)# self.W_ctx_to_att)
            e_dec = T.dot(hid_previous, hid2att_mtx)#self.W_hid_to_att)
            e_conct = T.tile(e_dec, (seq_len_enc,1,1))
            ener_i = self.nonlinearity_att(e_conct +pre_comp_ctx)
            e_i = T.dot(ener_i, v_a)#self.W_att)
            alpha = T.exp(e_i)
            alpha /= T.sum(alpha, axis=0) 
            return alpha.T
        contxt_input = None 
        contxt_input2 = None
        contxt_input3 = None
        contxt_input4 = None
        if self.contxt_input_incoming_index > 0:
            contxt_input = inputs[self.contxt_input_incoming_index]
        if self.contxt_input2_incoming_index > 0:
            contxt_input2 = inputs[self.contxt_input2_incoming_index]
        if self.contxt_input3_incoming_index > 0:
            contxt_input3 = inputs[self.contxt_input3_incoming_index]
        if self.contxt_input4_incoming_index > 0:
            contxt_input4 = inputs[self.contxt_input4_incoming_index]
        #
        alpha = calculate_alpha(contxt_input, self.W_ctx_to_att, self.W_hid_to_att, self.W_att)   
        alpha2 = calculate_alpha(contxt_input2, self.W_ctx_to_att2, self.W_hid_to_att2, self.W_att2)
        alpha3 = calculate_alpha(contxt_input3, self.W_ctx_to_att3, self.W_hid_to_att3, self.W_att3)   
        alpha4 = calculate_alpha(contxt_input4, self.W_ctx_to_att4, self.W_hid_to_att4, self.W_att4)
        alpha_conct = T.concatenate([alpha, alpha2, alpha3, alpha4], axis=0)
        return alpha_conct


class GRULayer_setenc(MergeLayer): #it is not mvrt
    #https://github.com/Lasagne/Lasagne/blob/v0.1/lasagne/layers/recurrent.py#L1064-L1416
    r"""
    lasagne.layers.recurrent.GRULayer(incoming, num_units,
    resetgate=lasagne.layers.Gate(W_cell=None),
    updategate=lasagne.layers.Gate(W_cell=None),
    hidden_update=lasagne.layers.Gate(
    W_cell=None, lasagne.nonlinearities.tanh),
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=True,
    gradient_steps=-1, grad_clipping=False, unroll_scan=False,
    precompute_input=True, mask_input=None, **kwargs)
    Gated Recurrent Unit (GRU) Layer
    Implements the recurrent step proposed in [1]_, which computes the output
    by
    .. math ::
        r_t &= \sigma_r(x_t W_{xr} + h_{t - 1} W_{hr} + b_r)\\
        u_t &= \sigma_u(x_t W_{xu} + h_{t - 1} W_{hu} + b_u)\\
        c_t &= \sigma_c(x_t W_{xc} + r_t \odot (h_{t - 1} W_{hc}) + b_c)\\
        h_t &= (1 - u_t) \odot h_{t - 1} + u_t \odot c_t
    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden units in the layer.
    resetgate : Gate
        Parameters for the reset gate (:math:`r_t`): :math:`W_{xr}`,
        :math:`W_{hr}`, :math:`b_r`, and :math:`\sigma_r`.
    updategate : Gate
        Parameters for the update gate (:math:`u_t`): :math:`W_{xu}`,
        :math:`W_{hu}`, :math:`b_u`, and :math:`\sigma_u`.
    hidden_update : Gate
        Parameters for the hidden update (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    hid_init : callable, np.ndarray, theano.shared or TensorVariable
        Initializer for initial hidden state (:math:`h_0`).  If a
        TensorVariable (Theano expression) is supplied, it will not be learned
        regardless of the value of `learn_init`.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned. If `hid_init` is a
        TensorVariable then the TensorVariable is used and
        `learn_init` is ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : False or float
        If a float is provided, the gradient messages are clipped during the
        backward pass.  If False, the gradients will not be clipped.  See [1]_
        (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    References
    ----------
    .. [1] Cho, Kyunghyun, et al: On the properties of neural
       machine translation: Encoder-decoder approaches.
       arXiv preprint arXiv:1409.1259 (2014).
    .. [2] Chung, Junyoung, et al.: Empirical Evaluation of Gated
       Recurrent Neural Networks on Sequence Modeling.
       arXiv preprint arXiv:1412.3555 (2014).
    .. [3] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    Notes
    -----
    An alternate update for the candidate hidden state is proposed in [2]_:
    .. math::
        c_t &= \sigma_c(x_t W_{ic} + (r_t \odot h_{t - 1})W_{hc} + b_c)\\
    We use the formulation from [1]_ because it allows us to do all matrix
    operations in a single dot product.
    """
    def __init__(self, incoming, num_units, 
                 resetgate=Gate_setenc(W_in=None,W_cell=None), 
                 updategate=Gate_setenc(W_in=None,W_cell=None),
                 hidden_update=Gate_setenc(W_in=None, W_cell=None),
                 nonlinearity=nonlinearities.tanh,
                 hid_init=init.Constant(0.),
                 set_steps = 5,
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
                 #only_return_final=True, #-- ???????
                 **kwargs):
                 
        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.  We will just provide the
        # layer input as incomings, unless a mask input was provided.
        incomings = [incoming]
        if mask_input is not None:
            incomings.append(mask_input) #-----

        # Initialize parent layer
        super(GRULayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.set_steps = set_steps
        

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates, nonlinearities will be sigmas, look Gate_setenc
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')
        
        #attention Weights 
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        self.att_num_units = att_num_units
        
        # Initialize hidden state
        if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError(
                    "When hid_init is provided as a TensorVariable, it should "
                    "have 2 dimensions and have shape (num_batch, num_units)")
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1], self.num_units

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
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = inputs[1] if len(inputs) > 1 else None

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        """W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)"""

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)

        #if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            #input = T.dot(input, W_in_stacked) + b_stacked

        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
        
        def plain_et_step(self, o_t0):
            #reading from memory steps
            #bs, seq_len_m, _ = x_snp.shape
            m_in = inco.dimshuffle(1, 0, 2)----replace
            e_qt = T.dot(o_t0, self.W_hid_to_att)---
            e_m = T.dot(m_in, self.W_ctx_to_att)----
            e_q = T.tile(e_qt, (self.seq_len_m, 1, 1))
            et_p = T.tanh(e_m + e_q)
            et = T.dot(et_p, self.W_att)
            alpha = T.exp(et)
            alpha /= T.sum(alpha, axis=0)
            mt = x_snp.dimshuffle(2, 1, 0)
            mult = T.mul(mt, alpha)
            rt = T.sum(mult, axis=1)
            return rt.T
            
        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(incomings, hid_previous, W_hid_stacked, #W_in_stacked,
                 b_stacked):
            x_snp = incomings[0]
            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping is not False:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

           # if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                #input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) #+ slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) #+ slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            #hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = resetgate*hidden_update_hid #hidden_update = hidden_update_in + resetgate*hidden_update_hid
            if self.grad_clipping is not False:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid0 = (1 - updategate)*hid_previous + updategate*hidden_update
            rt = self.plain_et_step(x_snp, hid0)
            h_t = T.concatenate([hid0,rt], axis=1) 
            #setenc part
            
            return hid
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
            

        def step_masked(input_n, mask_n, hid_previous, W_hid_stacked,
                        W_in_stacked, b_stacked):

            hid = step(input_n, hid_previous, W_hid_stacked, W_in_stacked,
                       b_stacked)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            not_mask = 1 - mask_n
            hid = hid*mask_n + hid_previous*not_mask

            return hid

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = [input]
            step_fun = step

        if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        #if not self.precompute_input:
         #   non_seqs += [W_in_stacked, b_stacked]
        # theano.scan only allows for positional arguments, so when
        # self.precompute_input is True, we need to supply fake placeholder
        # arguments for the input weights and biases.
        else:
            non_seqs += [(), ()]

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
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out, _ = theano.scan(fn=step_fun, outputs_info=o_enc_info, sequences=[xmask], non_sequences=[x_snp, snp_mask, set_steps], n_steps=self.set_steps)
            """hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]"""

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        hid_out = hid_out.dimshuffle(1, 0, 2)

        # if scan is backward reverse the output
        if self.backwards:
            hid_out = hid_out[:, ::-1, :]

        return hid_out


class LSTMAttLayer4(MergeLayer):
    r"""
    lasagne.layers.recurrent.LSTMLayer(incoming, num_units,
    ingate=lasagne.layers.Gate(), forgetgate=lasagne.layers.Gate(),
    cell=lasagne.layers.Gate(
    W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
    outgate=lasagne.layers.Gate(),
    nonlinearity=lasagne.nonlinearities.tanh,
    cell_init=lasagne.init.Constant(0.),
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    peepholes=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)
    A long short-term memory (LSTM) layer.
    Includes optional "peephole connections" and a forget gate.  Based on the
    definition in [1]_, which is the current common definition.  The output is
    computed by
    .. math ::
        i_t &= \sigma_i(x_t W_{xi} + h_{t-1} W_{hi}
               + w_{ci} \odot c_{t-1} + b_i)\\
        f_t &= \sigma_f(x_t W_{xf} + h_{t-1} W_{hf}
               + w_{cf} \odot c_{t-1} + b_f)\\
        c_t &= f_t \odot c_{t - 1}
               + i_t \odot \sigma_c(x_t W_{xc} + h_{t-1} W_{hc} + b_c)\\
        o_t &= \sigma_o(x_t W_{xo} + h_{t-1} W_{ho} + w_{co} \odot c_t + b_o)\\
        h_t &= o_t \odot \sigma_h(c_t)
    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden/cell units in the layer.
    ingate : Gate
        Parameters for the input gate (:math:`i_t`): :math:`W_{xi}`,
        :math:`W_{hi}`, :math:`w_{ci}`, :math:`b_i`, and :math:`\sigma_i`.
    forgetgate : Gate
        Parameters for the forget gate (:math:`f_t`): :math:`W_{xf}`,
        :math:`W_{hf}`, :math:`w_{cf}`, :math:`b_f`, and :math:`\sigma_f`.
    cell : Gate
        Parameters for the cell computation (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    outgate : Gate
        Parameters for the output gate (:math:`o_t`): :math:`W_{xo}`,
        :math:`W_{ho}`, :math:`w_{co}`, :math:`b_o`, and :math:`\sigma_o`.
    nonlinearity : callable or None
        The nonlinearity that is applied to the output (:math:`\sigma_h`). If
        None is provided, no nonlinearity will be applied.
    cell_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial cell state (:math:`c_0`).
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    peepholes : bool
        If True, the LSTM uses peephole connections.
        When False, `ingate.W_cell`, `forgetgate.W_cell` and
        `outgate.W_cell` are ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.
    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """
    def __init__(self, incoming, num_units,
                 ingate=myGate(),
                 forgetgate=myGate(),
                 cell=myGate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=myGate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.), 
                 contxt_input= init.Constant(0.), contxt_input2= init.Constant(0.), contxt_input3= init.Constant(0.), contxt_input4= init.Constant(0.),
                 ctx_init = init.Constant(0.),
                 att_num_units = 64,
                 W_hid_to_att = init.GlorotNormal(), W_hid_to_att2=init.GlorotNormal(), W_hid_to_att3=init.GlorotNormal(),
                 W_hid_to_att4=init.GlorotNormal(),
                 W_ctx_to_att=init.GlorotNormal(), W_ctx_to_att2=init.GlorotNormal(), W_ctx_to_att3=init.GlorotNormal(), W_ctx_to_att4=init.GlorotNormal(),
                 W_att = init.Normal(0.1), W_att2=init.Normal(0.1), W_att3=init.Normal(0.1), W_att4=init.Normal(0.1), 
                 W_ctx_to_ingate = init.GlorotNormal(),
                 W_ctx_to_forgetgate = init.GlorotNormal(),
                 W_ctx_to_cell = init.GlorotNormal(),
                 W_ctx_to_outgate = init.GlorotNormal(),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=True,
                 **kwargs):
        # This layer inherits from a MergeLayer, because it can have four
        # inputs - the layer input, the mask, the initial hidden state and the
        # inital cell state. We will just provide the layer input as incomings,
        # unless a mask input, inital hidden state or initial cell state was
        # provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.cell_init_incoming_index = -1
        self.contxt_input_incoming_index = -1
        self.contxt_input2_incoming_index = -1
        self.contxt_input3_incoming_index = -1
        self.contxt_input4_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1
        if isinstance(contxt_input, Layer):
            incomings.append(contxt_input)
            self.contxt_input_incoming_index = len(incomings)-1
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1
        if isinstance(ctx_init, Layer):
            incomings.append(ctx_init)
            self.ctx_init_incoming_index = len(incomings)-1
        if isinstance(contxt_input2, Layer):
            incomings.append(contxt_input2)
            self.contxt_input2_incoming_index = len(incomings)-1 
        if isinstance(contxt_input3, Layer):
            incomings.append(contxt_input3)
            self.contxt_input3_incoming_index = len(incomings)-1
        if isinstance(contxt_input4, Layer):
            incomings.append(contxt_input4)
            self.contxt_input4_incoming_index = len(incomings)-1 
        # Initialize parent layer
        super(LSTMAttLayer4, self).__init__(incomings, **kwargs)
        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity
        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")
        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]
        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")
        num_inputs = np.prod(input_shape[2:])
        #def cal_attent():       
        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)
        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate, 
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')
        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate, 'forgetgate')
        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')
        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')
        #  
        #IMP:
        num_ctx_input = 4
        self.W_ctx_to_ingate = self.add_param(W_ctx_to_ingate, (2*num_units*num_ctx_input, num_units), name='W_ctx_ingate')
        self.W_ctx_to_forgetgate = self.add_param(W_ctx_to_forgetgate, (2*num_units*num_ctx_input, num_units), name='W_ctx_forgetgate')
        self.W_ctx_to_cell = self.add_param(W_ctx_to_cell, (2*num_units*num_ctx_input, num_units), name='W_ctx_cell')
        self.W_ctx_to_outgate = self.add_param(W_ctx_to_outgate, (2*num_units*num_ctx_input, num_units), name='W_ctx_outgate')
        #
        #attention Weights        
        #
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_hid_to_att2 = self.add_param(W_hid_to_att2, (num_units, att_num_units), name='W_hid_to_att2')
        self.W_hid_to_att3 = self.add_param(W_hid_to_att3, (num_units, att_num_units), name='W_hid_to_att3')
        self.W_hid_to_att4 = self.add_param(W_hid_to_att4, (num_units, att_num_units), name='W_hid_to_att4')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_ctx_to_att2 = self.add_param(W_ctx_to_att2, (2*num_units, att_num_units), name='W_ctx_to_att2')
        self.W_ctx_to_att3 = self.add_param(W_ctx_to_att3, (2*num_units, att_num_units), name='W_ctx_to_att3')
        self.W_ctx_to_att4 = self.add_param(W_ctx_to_att4, (2*num_units, att_num_units), name='W_ctx_to_att4')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        self.W_att2 = self.add_param(W_att2, (att_num_units,), name='W_att2')
        self.W_att3 = self.add_param(W_att3, (att_num_units,), name='W_att3')
        self.W_att4 = self.add_param(W_att4, (att_num_units,), name='W_att4')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        self.att_num_units = att_num_units
        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_units, ), name="W_cell_to_ingate")
            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (num_units, ), name="W_cell_to_forgetgate")
            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (num_units, ), name="W_cell_to_outgate")
        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)        
        if isinstance(contxt_input, Layer):
            self.contxt_input = contxt_input        
        if isinstance(ctx_init, Layer):
            self.ctx_init = ctx_init
        else:
            self.ctx_init = self.add_param(
            ctx_init, (1, self.num_units*2*num_ctx_input), name='ctx_init',
            trainable=True, regularizable=False)
    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units
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
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        contxt_input = None
        contxt_input2 = None
        contxt_input3 = None
        contxt_input4 = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]
        if self.contxt_input_incoming_index > 0:
            contxt_input = inputs[self.contxt_input_incoming_index]
        if self.ctx_init_incoming_index > 0:
            ctx_init = inputs[self.ctx_init_incoming_index] 
        if self.contxt_input2_incoming_index > 0:
            contxt_input2 = inputs[self.contxt_input2_incoming_index]
        if self.contxt_input3_incoming_index > 0:
            contxt_input3 = inputs[self.contxt_input3_incoming_index]
        if self.contxt_input4_incoming_index > 0:
            contxt_input4 = inputs[self.contxt_input4_incoming_index]
        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)
        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape
        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)
        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)
        #DONE: buraya stack att'yi ekle
        W_ctx_stacked = T.concatenate(
            [self.W_ctx_to_ingate, self.W_ctx_to_forgetgate,
             self.W_ctx_to_cell, self.W_ctx_to_outgate], axis=1)
        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)
        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked
        #
        #_, seq_len_enc, ctx_fea_len = contxt_input.shape
        #contxt_input.output_shape        
        #pre_ctx: (seq_len_enc, n_batch, num_att_units), ctx_shuffle: (seq_len_enc, n_batch, n_feature)
        #contxt_sh = contxt_input.dimshuffle(1, 0, 2)
        #pre_comp_ctx = T.dot(contxt_sh, self.W_ctx_to_att)
        #contxt_sht = contxt_input.dimshuffle(2, 1, 0)
        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
        def cal_contx(ctx_in, hid_previous, ctx2att_mtx, hid2att_mtx, v_a):
            _, seq_len_enc, _ = ctx_in.shape
            contxt_sh = ctx_in.dimshuffle(1, 0, 2)
            pre_comp_ctx = T.dot(contxt_sh, ctx2att_mtx)
            contxt_sht = ctx_in.dimshuffle(2, 1, 0)
            e_dec = T.dot(hid_previous, hid2att_mtx)#self.W_hid_to_att)
            e_conct = T.tile(e_dec, (seq_len_enc,1,1))
            ener_i = self.nonlinearity_att(e_conct +pre_comp_ctx)
            e_i = T.dot(ener_i, v_a)#self.W_att)
            alpha = T.exp(e_i)
            alpha /= T.sum(alpha, axis=0)          
            mult = T.mul(contxt_sht, alpha)
            ctx = T.sum(mult, axis=1)
            return ctx.T
        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, contxt_previous, *args):
            contxt1 = cal_contx(contxt_input, hid_previous, self.W_ctx_to_att, self.W_hid_to_att, self.W_att)
            contxt2 = cal_contx(contxt_input2, hid_previous, self.W_ctx_to_att2, self.W_hid_to_att2, self.W_att2)
            contxt3 = cal_contx(contxt_input3, hid_previous, self.W_ctx_to_att3, self.W_hid_to_att3, self.W_att3)
            contxt4 = cal_contx(contxt_input4, hid_previous, self.W_ctx_to_att4, self.W_hid_to_att4, self.W_att4)
            contxt = T.concatenate([contxt1, contxt2, contxt3, contxt4], axis=1)
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked
            # Calculate gates pre-activations and slice
            gates1 = input_n + T.dot(hid_previous, W_hid_stacked) 
            #att etkisini ekle
            gates = gates1 + T.dot(contxt, W_ctx_stacked)
            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)
            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)
            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate
            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)
            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input
            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)
            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            return [cell, hid, contxt]
        def step_masked(input_n, mask_n, cell_previous, hid_previous, contxt_previous, *args):
            cell, hid, contxt = step(input_n, cell_previous, hid_previous, contxt_previous, *args)
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            contxt = T.switch(mask_n, contxt, contxt_previous)
            if not mask_n:
                print('Mask 0 here something wrong going on!')
            return [cell, hid, contxt]
        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step
        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)
        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)        
        if not isinstance(self.ctx_init, Layer):
            ctx_init = T.dot(ones, self.ctx_init)
        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        non_seqs += [W_ctx_stacked]
        non_seqs += [self.W_hid_to_att, self.W_ctx_to_att, self.W_att]
        non_seqs += [self.W_hid_to_att2, self.W_ctx_to_att2, self.W_att2]
        non_seqs += [self.W_hid_to_att3, self.W_ctx_to_att3, self.W_att3]
        non_seqs += [self.W_hid_to_att4, self.W_ctx_to_att4, self.W_att4]
        non_seqs += [contxt_input, contxt_input2, contxt_input3, contxt_input4]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]
        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out, ctx_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init, ctx_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out, ctx_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init, ctx_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=False)[0]
        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)
            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]
        return hid_out 



class AnaLayer_lambda4(MergeLayer):
    def __init__(self, incoming, num_units, att_num_units=64, ws=96, pred_len=4, pred_ind=0, contxt_input=init.Constant(0.), 
                 contxt_input2=init.Constant(0.), contxt_input3=init.Constant(0.), 
                 contxt_input4=init.Constant(0.), W_att=init.Normal(0.1),
                 W_att2=init.Normal(0.1), W_att3=init.Normal(0.1), W_att4=init.Normal(0.1),
                 W_lambda=init.Normal(0.1), 
                 W_lambda2=init.Normal(0.1), W_lambda3=init.Normal(0.1), W_lambda4=init.Normal(0.1),
                 W_hid_to_att=init.GlorotNormal(), 
                 W_hid_to_att2=init.GlorotNormal(), W_hid_to_att3=init.GlorotNormal(), W_hid_to_att4=init.GlorotNormal(), 
                 W_ctx_to_att=init.GlorotNormal(),
                 W_ctx_to_att2=init.GlorotNormal(), W_ctx_to_att3=init.GlorotNormal(), W_ctx_to_att4=init.GlorotNormal(),
                 **kwargs):        
        incomings = [incoming]        
        self.contxt_input_incoming_index = -1
        self.contxt_input2_incoming_index = -1
        self.contxt_input3_incoming_index = -1
        self.contxt_input4_incoming_index = -1        
        if isinstance(contxt_input, Layer):
            incomings.append(contxt_input)
            self.contxt_input_incoming_index = len(incomings)-1
        if isinstance(contxt_input2, Layer):
            incomings.append(contxt_input2)
            self.contxt_input2_incoming_index = len(incomings)-1 
        if isinstance(contxt_input3, Layer):
            incomings.append(contxt_input3)
            self.contxt_input3_incoming_index = len(incomings)-1
        if isinstance(contxt_input4, Layer):
            incomings.append(contxt_input4)
            self.contxt_input4_incoming_index = len(incomings)-1 
        super(AnaLayer_lambda4, self).__init__(incomings, **kwargs)
        self.num_units = num_units  #number of attention units
        #num_inputs = np.prod(input_shape[2:])
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_hid_to_att2 = self.add_param(W_hid_to_att2, (num_units, att_num_units), name='W_hid_to_att2')
        self.W_hid_to_att3 = self.add_param(W_hid_to_att3, (num_units, att_num_units), name='W_hid_to_att3')
        self.W_hid_to_att4 = self.add_param(W_hid_to_att4, (num_units, att_num_units), name='W_hid_to_att4')
        #
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_ctx_to_att2 = self.add_param(W_ctx_to_att2, (2*num_units, att_num_units), name='W_ctx_to_att2')
        self.W_ctx_to_att3 = self.add_param(W_ctx_to_att3, (2*num_units, att_num_units), name='W_ctx_to_att3')
        self.W_ctx_to_att4 = self.add_param(W_ctx_to_att4, (2*num_units, att_num_units), name='W_ctx_to_att4')
        #
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        self.W_att2 = self.add_param(W_att2, (att_num_units,), name='W_att2')
        self.W_att3 = self.add_param(W_att3, (att_num_units,), name='W_att3')
        self.W_att4 = self.add_param(W_att4, (att_num_units,), name='W_att4')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        self.att_num_units = att_num_units
        if isinstance(contxt_input, Layer):
            self.contxt_input = contxt_input
            #_, self.seq_len_enc, ctx_fea_len = contxt_input.shape
        self.W_lambda = self.add_param(W_lambda, (ws + pred_len -1,), name='W_lambda')
        self.W_lambda2 = self.add_param(W_lambda2, (ws + pred_len -1,), name='W_lambda2')
        self.W_lambda3 = self.add_param(W_lambda3, (ws + pred_len -1,), name='W_lambda3')
        self.W_lambda4 = self.add_param(W_lambda4, (ws + pred_len -1,), name='W_lambda4')
        self.pred_len = pred_len
        self.pred_ind = pred_ind
    def get_output_shape_for(self, input_shape):
        #_, seq_len_enc, _ = self.contxt_input.shape
        return (input_shape[0][0], None)
    def get_output_for(self, inputs, **kwargs):
        hid_previous = inputs[0]      
        #
        contxt_input = None  
        contxt_input2 = None
        contxt_input3 = None
        contxt_input4 = None
        if self.contxt_input_incoming_index > 0:
            contxt_input = inputs[self.contxt_input_incoming_index]
        if self.contxt_input2_incoming_index > 0:
            contxt_input2 = inputs[self.contxt_input2_incoming_index]
        if self.contxt_input3_incoming_index > 0:
            contxt_input3 = inputs[self.contxt_input3_incoming_index]
        if self.contxt_input4_incoming_index > 0:
            contxt_input4 = inputs[self.contxt_input4_incoming_index]
        #here we assume all input sequences are same lenght by using one self.ws!
        bs, seq_len_enc, _ = contxt_input.shape
        def delta_mtx():
            dia = T.ones((seq_len_enc,seq_len_enc))
            diag = T.identity_like(dia)
            anti_diag = diag[::-1]
            delt = T.zeros((seq_len_enc+self.pred_len-1,seq_len_enc))
            delta =  T.set_subtensor(delt[self.pred_ind:self.pred_ind+seq_len_enc, :], anti_diag)
            return delta
        def calculate_alpha(ctx_input, ctx2att_mtx, hid2att_mtx, v_a, lambda_mtx):
            _, seq_len_enc, _ = ctx_input.shape
            contxt_sh = ctx_input.dimshuffle(1, 0, 2)
            pre_comp_ctx = T.dot(contxt_sh, ctx2att_mtx)#self.W_ctx_to_att)
            e_dec = T.dot(hid_previous, hid2att_mtx)#self.W_hid_to_att)
            e_conct = T.tile(e_dec, (seq_len_enc,1,1))
            ener_i = self.nonlinearity_att(e_conct +pre_comp_ctx)
            e_i = T.dot(ener_i, v_a)#self.W_att)
            delta = delta_mtx()
            lambda_delta = T.dot(lambda_mtx.T, delta)
            lambda_delta_tile = T.tile(lambda_delta, (bs,1), ndim=2).T
            e_i_new = e_i * lambda_delta_tile
            alpha = T.exp(e_i_new)
            alpha /= T.sum(alpha, axis=0) 
            return alpha.T
        alpha = calculate_alpha(contxt_input, self.W_ctx_to_att, self.W_hid_to_att, self.W_att, self.W_lambda)
        alpha2 = calculate_alpha(contxt_input2, self.W_ctx_to_att2, self.W_hid_to_att2, self.W_att2, self.W_lambda2)
        alpha3 = calculate_alpha(contxt_input3, self.W_ctx_to_att3, self.W_hid_to_att3, self.W_att3, self.W_lambda3)
        alpha4 = calculate_alpha(contxt_input4, self.W_ctx_to_att4, self.W_hid_to_att4, self.W_att4, self.W_lambda4)
        alpha_conct = T.concatenate([alpha, alpha2, alpha3, alpha4], axis=0)
        return alpha_conct


class LSTMAttLayer_lambda4(MergeLayer):
    r"""
    lasagne.layers.recurrent.LSTMLayer(incoming, num_units,
    ingate=lasagne.layers.Gate(), forgetgate=lasagne.layers.Gate(),
    cell=lasagne.layers.Gate(
    W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
    outgate=lasagne.layers.Gate(),
    nonlinearity=lasagne.nonlinearities.tanh,
    cell_init=lasagne.init.Constant(0.),
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    peepholes=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)
    A long short-term memory (LSTM) layer.
    Includes optional "peephole connections" and a forget gate.  Based on the
    definition in [1]_, which is the current common definition.  The output is
    computed by
    .. math ::
        i_t &= \sigma_i(x_t W_{xi} + h_{t-1} W_{hi}
               + w_{ci} \odot c_{t-1} + b_i)\\
        f_t &= \sigma_f(x_t W_{xf} + h_{t-1} W_{hf}
               + w_{cf} \odot c_{t-1} + b_f)\\
        c_t &= f_t \odot c_{t - 1}
               + i_t \odot \sigma_c(x_t W_{xc} + h_{t-1} W_{hc} + b_c)\\
        o_t &= \sigma_o(x_t W_{xo} + h_{t-1} W_{ho} + w_{co} \odot c_t + b_o)\\
        h_t &= o_t \odot \sigma_h(c_t)
    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden/cell units in the layer.
    ingate : Gate
        Parameters for the input gate (:math:`i_t`): :math:`W_{xi}`,
        :math:`W_{hi}`, :math:`w_{ci}`, :math:`b_i`, and :math:`\sigma_i`.
    forgetgate : Gate
        Parameters for the forget gate (:math:`f_t`): :math:`W_{xf}`,
        :math:`W_{hf}`, :math:`w_{cf}`, :math:`b_f`, and :math:`\sigma_f`.
    cell : Gate
        Parameters for the cell computation (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    outgate : Gate
        Parameters for the output gate (:math:`o_t`): :math:`W_{xo}`,
        :math:`W_{ho}`, :math:`w_{co}`, :math:`b_o`, and :math:`\sigma_o`.
    nonlinearity : callable or None
        The nonlinearity that is applied to the output (:math:`\sigma_h`). If
        None is provided, no nonlinearity will be applied.
    cell_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial cell state (:math:`c_0`).
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    peepholes : bool
        If True, the LSTM uses peephole connections.
        When False, `ingate.W_cell`, `forgetgate.W_cell` and
        `outgate.W_cell` are ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.
    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """
    def __init__(self, incoming, num_units,
                 ingate=myGate(),
                 forgetgate=myGate(),
                 cell=myGate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=myGate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.), 
                 contxt_input= init.Constant(0.), contxt_input2=init.Constant(0.), contxt_input3=init.Constant(0.), 
                 contxt_input4=init.Constant(0.),
                 ctx_init = init.Constant(0.),
                 att_num_units = 64,
                 ws=96, pred_len=4, pred_ind=0,
                 W_hid_to_att = init.GlorotNormal(),
                 W_hid_to_att2=init.GlorotNormal(), W_hid_to_att3=init.GlorotNormal(), W_hid_to_att4=init.GlorotNormal(), 
                 W_ctx_to_att = init.GlorotNormal(), W_ctx_to_att2=init.GlorotNormal(), W_ctx_to_att3=init.GlorotNormal(), 
                 W_ctx_to_att4=init.GlorotNormal(),
                 W_att = init.Normal(0.1), W_lambda= init.Normal(0.1),
                 W_att2=init.Normal(0.1), W_att3=init.Normal(0.1), W_att4=init.Normal(0.1),
                 W_lambda2=init.Normal(0.1), W_lambda3=init.Normal(0.1), W_lambda4=init.Normal(0.1),
                 W_ctx_to_ingate = init.GlorotNormal(),
                 W_ctx_to_forgetgate = init.GlorotNormal(),
                 W_ctx_to_cell = init.GlorotNormal(),
                 W_ctx_to_outgate = init.GlorotNormal(), 
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=True,
                 **kwargs):
        # This layer inherits from a MergeLayer, because it can have four
        # inputs - the layer input, the mask, the initial hidden state and the
        # inital cell state. We will just provide the layer input as incomings,
        # unless a mask input, inital hidden state or initial cell state was
        # provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.cell_init_incoming_index = -1
        self.contxt_input_incoming_index = -1
        self.contxt_input2_incoming_index = -1
        self.contxt_input3_incoming_index = -1
        self.contxt_input4_incoming_index = -1        
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1
        if isinstance(contxt_input, Layer):
            incomings.append(contxt_input)
            self.contxt_input_incoming_index = len(incomings)-1
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1
        if isinstance(contxt_input2, Layer):
            incomings.append(contxt_input2)
            self.contxt_input2_incoming_index = len(incomings)-1 
        if isinstance(contxt_input3, Layer):
            incomings.append(contxt_input3)
            self.contxt_input3_incoming_index = len(incomings)-1
        if isinstance(contxt_input4, Layer):
            incomings.append(contxt_input4)
            self.contxt_input4_incoming_index = len(incomings)-1 
        if isinstance(ctx_init, Layer):
            incomings.append(ctx_init)
            self.ctx_init_incoming_index = len(incomings)-1      
        # Initialize parent layer
        super(LSTMAttLayer_lambda4, self).__init__(incomings, **kwargs)
        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity
        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")
        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]
        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")
        num_inputs = np.prod(input_shape[2:])
        #def cal_attent():       
        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)
        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate, 
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')
        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                                 'forgetgate')
        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')
        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')
        #
        #IMP:
        num_ctx_input = 4         
        self.W_ctx_to_ingate = self.add_param(W_ctx_to_ingate, (2*num_units*num_ctx_input, num_units), name='W_ctx_ingate')
        self.W_ctx_to_forgetgate = self.add_param(W_ctx_to_forgetgate, (2*num_units*num_ctx_input, num_units), name='W_ctx_forgetgate')
        self.W_ctx_to_cell = self.add_param(W_ctx_to_cell, (2*num_units*num_ctx_input, num_units), name='W_ctx_cell')
        self.W_ctx_to_outgate = self.add_param(W_ctx_to_outgate, (2*num_units*num_ctx_input, num_units), name='W_ctx_outgate')
        #
        #attention Weights        
        #b_att = init.Constant(0.)
        #
        b_s, seq_len_enc, ctx_fea_len  = contxt_input.output_shape
        #b_s, seq_len_enc, ctx_fea_len = contxt_input.shape
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_hid_to_att2 = self.add_param(W_hid_to_att2, (num_units, att_num_units), name='W_hid_to_att2')
        self.W_hid_to_att3 = self.add_param(W_hid_to_att3, (num_units, att_num_units), name='W_hid_to_att3')
        self.W_hid_to_att4 = self.add_param(W_hid_to_att4, (num_units, att_num_units), name='W_hid_to_att4')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_ctx_to_att2 = self.add_param(W_ctx_to_att2, (2*num_units, att_num_units), name='W_ctx_to_att2')
        self.W_ctx_to_att3 = self.add_param(W_ctx_to_att3, (2*num_units, att_num_units), name='W_ctx_to_att3')
        self.W_ctx_to_att4 = self.add_param(W_ctx_to_att4, (2*num_units, att_num_units), name='W_ctx_to_att4')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        self.W_att2 = self.add_param(W_att2, (att_num_units,), name='W_att2')
        self.W_att3 = self.add_param(W_att3, (att_num_units,), name='W_att3')
        self.W_att4 = self.add_param(W_att4, (att_num_units,), name='W_att4')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        self.att_num_units = att_num_units
        self.W_lambda = self.add_param(W_lambda, (ws + pred_len -1,), name='W_lambda')
        self.W_lambda2 = self.add_param(W_lambda2, (ws + pred_len -1,), name='W_lambda2')
        self.W_lambda3 = self.add_param(W_lambda3, (ws + pred_len -1,), name='W_lambda3')
        self.W_lambda4 = self.add_param(W_lambda4, (ws + pred_len -1,), name='W_lambda4')
        self.pred_len = pred_len
        self.pred_ind = pred_ind
        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_units, ), name="W_cell_to_ingate")
            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (num_units, ), name="W_cell_to_forgetgate")
            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (num_units, ), name="W_cell_to_outgate")
        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)        
        if isinstance(contxt_input, Layer):
            self.contxt_input = contxt_input        
        if isinstance(ctx_init, Layer):
            self.ctx_init = ctx_init
        else:
            self.ctx_init = self.add_param(
            ctx_init, (1, self.num_units*2*num_ctx_input), name='ctx_init',
            trainable=True, regularizable=False)
    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units
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
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        contxt_input = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]
        if self.contxt_input_incoming_index > 0:
            contxt_input = inputs[self.contxt_input_incoming_index]
        if self.contxt_input2_incoming_index > 0:
            contxt_input2 = inputs[self.contxt_input2_incoming_index]
        if self.contxt_input3_incoming_index > 0:
            contxt_input3 = inputs[self.contxt_input3_incoming_index]
        if self.contxt_input4_incoming_index > 0:
            contxt_input4 = inputs[self.contxt_input4_incoming_index]
        if self.ctx_init_incoming_index > 0:
            ctx_init = inputs[self.ctx_init_incoming_index]        
        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)
        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape
        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)
        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)
        #DONE: buraya stack att'yi ekle
        W_ctx_stacked = T.concatenate(
            [self.W_ctx_to_ingate, self.W_ctx_to_forgetgate,
             self.W_ctx_to_cell, self.W_ctx_to_outgate], axis=1)
        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)
        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked
        #
        bs, seq_len_enc, _ = contxt_input.shape
        def delta_mtx():
            dia = T.ones((seq_len_enc,seq_len_enc))
            diag = T.identity_like(dia)
            anti_diag = diag[::-1]
            delt = T.zeros((seq_len_enc+self.pred_len-1,seq_len_enc))
            delta =  T.set_subtensor(delt[self.pred_ind:self.pred_ind+seq_len_enc, :], anti_diag)
            return delta
        #contxt_input.output_shape        
        #pre_ctx: (seq_len_enc, n_batch, num_att_units), ctx_shuffle: (seq_len_enc, n_batch, n_feature)
        
        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
        def cal_contx(ctx_input, hid_previous, ctx2att_mtx, hid2att_mtx, v_a, lambda_mtx):
            _, seq_len_enc, _ = ctx_input.shape
            contxt_sh = ctx_input.dimshuffle(1, 0, 2)
            pre_comp_ctx = T.dot(contxt_sh, ctx2att_mtx)#self.W_ctx_to_att)
            contxt_sht = ctx_input.dimshuffle(2, 1, 0)
            e_dec = T.dot(hid_previous, hid2att_mtx)#self.W_hid_to_att)
            e_conct = T.tile(e_dec, (seq_len_enc,1,1))
            ener_i = self.nonlinearity_att(e_conct + pre_comp_ctx)
            e_i = T.dot(ener_i, v_a)#self.W_att)
            delta = delta_mtx()
            lambda_delta = T.dot(lambda_mtx.T, delta)#T.dot(self.W_lambda.T, delta)
            lambda_delta_tile = T.tile(lambda_delta, (bs,1), ndim=2).T
            e_i_new = e_i * lambda_delta_tile
            alpha = T.exp(e_i_new)
            alpha /= T.sum(alpha, axis=0)          
            mult = T.mul(contxt_sht, alpha)
            ctx = T.sum(mult, axis=1)
            return ctx.T
        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, contxt_previous, *args):
            #contxt = cal_contx(hid_previous)
            contxt1 = cal_contx(contxt_input, hid_previous, self.W_ctx_to_att, self.W_hid_to_att, self.W_att, self.W_lambda)
            contxt2 = cal_contx(contxt_input2, hid_previous, self.W_ctx_to_att2, self.W_hid_to_att2, self.W_att2, self.W_lambda2)
            contxt3 = cal_contx(contxt_input3, hid_previous, self.W_ctx_to_att3, self.W_hid_to_att3, self.W_att3, self.W_lambda3)
            contxt4 = cal_contx(contxt_input4, hid_previous, self.W_ctx_to_att4, self.W_hid_to_att4, self.W_att4, self.W_lambda4)
            contxt = T.concatenate([contxt1, contxt2, contxt3, contxt4], axis=1)
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked
            # Calculate gates pre-activations and slice
            gates1 = input_n + T.dot(hid_previous, W_hid_stacked) 
            #att etkisini ekle
            gates = gates1 + T.dot(contxt, W_ctx_stacked)
            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)
            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)
            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate
            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)
            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input
            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)
            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            return [cell, hid, contxt]
        def step_masked(input_n, mask_n, cell_previous, hid_previous, contxt_previous, *args):
            cell, hid, contxt = step(input_n, cell_previous, hid_previous, contxt_previous, *args)
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            contxt = T.switch(mask_n, contxt, contxt_previous)
            if not mask_n:
                print('Mask 0 here something wrong going on!')
            return [cell, hid, contxt]
        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step
        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)
        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)        
        if not isinstance(self.ctx_init, Layer):
            ctx_init = T.dot(ones, self.ctx_init)
        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        non_seqs += [W_ctx_stacked]
        non_seqs += [self.W_hid_to_att, self.W_ctx_to_att, self.W_att]
        non_seqs += [self.W_hid_to_att2, self.W_ctx_to_att2, self.W_att2]
        non_seqs += [self.W_hid_to_att3, self.W_ctx_to_att3, self.W_att3]
        non_seqs += [self.W_hid_to_att4, self.W_ctx_to_att4, self.W_att4]
        non_seqs += [self.W_lambda, self.W_lambda2, self.W_lambda3, self.W_lambda4] 
        non_seqs += [contxt_input, contxt_input2, contxt_input3, contxt_input4]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]
        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out, ctx_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init, ctx_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out, ctx_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init, ctx_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=False)[0]
        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)
            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]
        return hid_out 




class AnaLayer_lambda_mu4(MergeLayer):
    def __init__(self, incoming, num_units, att_num_units = 64, ws=96, pred_len=4, pred_ind=0, contxt_input=init.Constant(0.), 
                 contxt_input2=init.Constant(0.), contxt_input3=init.Constant(0.), contxt_input4=init.Constant(0.),                 
                 W_att=init.Normal(0.1), W_att2=init.Normal(0.1), W_att3=init.Normal(0.1), W_att4=init.Normal(0.1),
                 W_lambda=init.Normal(0.1), W_lambda2=init.Normal(0.1), W_lambda3=init.Normal(0.1),
                 W_lambda4=init.Normal(0.1), W_mu=init.Normal(0.1), W_mu2=init.Normal(0.1), W_mu3=init.Normal(0.1), 
                 W_mu4=init.Normal(0.1), enc_mask_input=None, enc_mask_input2=None, enc_mask_input3=None, enc_mask_input4=None,
                 W_hid_to_att=init.GlorotNormal(), W_hid_to_att2=init.GlorotNormal(), W_hid_to_att3=init.GlorotNormal(),
                 W_hid_to_att4=init.GlorotNormal(),  W_ctx_to_att=init.GlorotNormal(), 
                 W_ctx_to_att2=init.GlorotNormal(), W_ctx_to_att3=init.GlorotNormal(), W_ctx_to_att4=init.GlorotNormal(),
                 delta_inds_input=None, delta_inds_input2=None, delta_inds_input3=None, delta_inds_input4=None,
                 **kwargs):        
        incomings = [incoming]        
        self.contxt_input_incoming_index = -1        
        if isinstance(contxt_input, Layer):
            incomings.append(contxt_input)
            self.contxt_input_incoming_index = len(incomings)-1
        if isinstance(contxt_input2, Layer):
            incomings.append(contxt_input2)
            self.contxt_input2_incoming_index = len(incomings)-1 
        if isinstance(contxt_input3, Layer):
            incomings.append(contxt_input3)
            self.contxt_input3_incoming_index = len(incomings)-1
        if isinstance(contxt_input4, Layer):
            incomings.append(contxt_input4)
            self.contxt_input4_incoming_index = len(incomings)-1 
        if enc_mask_input is not None:
            incomings.append(enc_mask_input)
            self.enc_mask_incoming_index = len(incomings)-1
        if enc_mask_input2 is not None:
            incomings.append(enc_mask_input2)
            self.enc_mask_incoming_index2 = len(incomings)-1
        if enc_mask_input3 is not None:
            incomings.append(enc_mask_input3)
            self.enc_mask_incoming_index3 = len(incomings)-1
        if enc_mask_input4 is not None:
            incomings.append(enc_mask_input4)
            self.enc_mask_incoming_index4 = len(incomings)-1
        if delta_inds_input is not None:
            incomings.append(delta_inds_input)
            self.delta_inds_incoming_index = len(incomings)-1
        if delta_inds_input2 is not None:
            incomings.append(delta_inds_input2)
            self.delta_inds_incoming_index2 = len(incomings)-1
        if delta_inds_input3 is not None:
            incomings.append(delta_inds_input3)
            self.delta_inds_incoming_index3 = len(incomings)-1
        if delta_inds_input4 is not None:
            incomings.append(delta_inds_input4)
            self.delta_inds_incoming_index4 = len(incomings)-1
        super(AnaLayer_lambda_mu4, self).__init__(incomings, **kwargs)
        self.num_units = num_units  #number of attention units
        #num_inputs = np.prod(input_shape[2:])
        #
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')        
        self.W_hid_to_att2 = self.add_param(W_hid_to_att2, (num_units, att_num_units), name='W_hid_to_att2')
        self.W_hid_to_att3 = self.add_param(W_hid_to_att3, (num_units, att_num_units), name='W_hid_to_att3')
        self.W_hid_to_att4 = self.add_param(W_hid_to_att4, (num_units, att_num_units), name='W_hid_to_att4')
        #
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_ctx_to_att2 = self.add_param(W_ctx_to_att2, (2*num_units, att_num_units), name='W_ctx_to_att2')
        self.W_ctx_to_att3 = self.add_param(W_ctx_to_att3, (2*num_units, att_num_units), name='W_ctx_to_att3')
        self.W_ctx_to_att4 = self.add_param(W_ctx_to_att4, (2*num_units, att_num_units), name='W_ctx_to_att4')
        #
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        self.W_att2 = self.add_param(W_att2, (att_num_units,), name='W_att2')
        self.W_att3 = self.add_param(W_att3, (att_num_units,), name='W_att3')
        self.W_att4 = self.add_param(W_att4, (att_num_units,), name='W_att4')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        self.att_num_units = att_num_units
        if isinstance(contxt_input, Layer):
            self.contxt_input = contxt_input
            #_, self.seq_len_enc, ctx_fea_len = contxt_input.shape
        #using same ws -> assumption of the sequence length is the same
        self.W_lambda = self.add_param(W_lambda, (ws + pred_len -1,), name='W_lambda')
        self.W_lambda2 = self.add_param(W_lambda2, (ws + pred_len -1,), name='W_lambda2')
        self.W_lambda3 = self.add_param(W_lambda2, (ws + pred_len -1,), name='W_lambda3')
        self.W_lambda4 = self.add_param(W_lambda2, (ws + pred_len -1,), name='W_lambda4')
        self.pred_len = pred_len
        self.pred_ind = pred_ind
        self.W_mu = self.add_param(W_mu, (3,), name='W_mu')
        self.W_mu2 = self.add_param(W_mu2, (3,), name='W_mu2')
        self.W_mu3 = self.add_param(W_mu3, (3,), name='W_mu3')
        self.W_mu4 = self.add_param(W_mu4, (3,), name='W_mu4')
    def get_output_shape_for(self, input_shape):
        #_, seq_len_enc, _ = self.contxt_input.shape
        return (input_shape[0][0], None)
    def get_output_for(self, inputs, **kwargs):
        hid_previous = inputs[0]      
        #
        contxt_input = None  
        contxt_input2 = None
        contxt_input3 = None
        contxt_input4 = None       
        if self.contxt_input_incoming_index > 0:
            contxt_input = inputs[self.contxt_input_incoming_index]
        if self.contxt_input2_incoming_index > 0:
            contxt_input2 = inputs[self.contxt_input2_incoming_index]
        if self.contxt_input3_incoming_index > 0:
            contxt_input3 = inputs[self.contxt_input3_incoming_index]
        if self.contxt_input4_incoming_index > 0:
            contxt_input4 = inputs[self.contxt_input4_incoming_index]
        if self.enc_mask_incoming_index > 0:
            enc_mask = inputs[self.enc_mask_incoming_index]
        if self.enc_mask_incoming_index2 > 0:
            enc_mask2 = inputs[self.enc_mask_incoming_index2]
        if self.enc_mask_incoming_index3 > 0:
            enc_mask3 = inputs[self.enc_mask_incoming_index3]
        if self.enc_mask_incoming_index4 > 0:
            enc_mask4 = inputs[self.enc_mask_incoming_index4]
        if self.delta_inds_incoming_index > 0:
            delta_inds = inputs[self.delta_inds_incoming_index]
        if self.delta_inds_incoming_index2 > 0:
            delta_inds2 = inputs[self.delta_inds_incoming_index2]
        if self.delta_inds_incoming_index3 > 0:
            delta_inds3 = inputs[self.delta_inds_incoming_index3]
        if self.delta_inds_incoming_index4 > 0:
            delta_inds4 = inputs[self.delta_inds_incoming_index4]
        #
        #here we assume all input sequences are same lenght by using one seq_len_enc
        #bs, seq_len_enc, _ = contxt_input.shape
        def delta_mtx(seq_len_enc):
            dia = T.ones((seq_len_enc,seq_len_enc))
            diag = T.identity_like(dia)
            anti_diag = diag[::-1]
            delt = T.zeros((seq_len_enc+self.pred_len-1,seq_len_enc))
            delta =  T.set_subtensor(delt[self.pred_ind:self.pred_ind+seq_len_enc, :], anti_diag)
            return delta        
        def calculate_alpha(ctx_input, ctx2att_mtx, hid2att_mtx, v_a, lambda_mtx, enc_mask_mtx, delta_inds_vec, mu_vec):
            bs, seq_len_enc, _ = ctx_input.shape
            contxt_sh = ctx_input.dimshuffle(1, 0, 2)
            pre_comp_ctx = T.dot(contxt_sh, ctx2att_mtx)#self.W_ctx_to_att)
            e_dec = T.dot(hid_previous, hid2att_mtx)#self.W_hid_to_att)
            e_conct = T.tile(e_dec, (seq_len_enc,1,1))
            ener_i = self.nonlinearity_att(e_conct +pre_comp_ctx)
            e_i = T.dot(ener_i, v_a)#self.W_att)
            delta = delta_mtx(seq_len_enc)
            lambda_delta = T.dot(lambda_mtx.T, delta)
            lambda_delta_tile = T.tile(lambda_delta, (bs,1), ndim=2).T
            e_i_n = e_i * lambda_delta_tile
            delta_gap = T.zeros((bs* seq_len_enc, 3))
            mask_flat = T.flatten(enc_mask_mtx)
            zeros_inds = T.eq(mask_flat, 0).nonzero()
            delta_gap2 = theano.tensor.set_subtensor(delta_gap[zeros_inds, delta_inds_vec],1)
            delta_gap3 = delta_gap2.reshape((bs, seq_len_enc, 3))
            delta_gap4 = delta_gap3.dimshuffle(1,0,2)
            mudelta = T.dot(delta_gap4, mu_vec)#self.W_mu) ###HERE we are!
            e_i_new = e_i_n * (1 + mudelta)
            alpha = T.exp(e_i_new)
            alpha /= T.sum(alpha, axis=0) 
            return alpha.T
        #
        alpha1 = calculate_alpha(contxt_input, self.W_ctx_to_att, self.W_hid_to_att, self.W_att, self.W_lambda, enc_mask, delta_inds, self.W_mu)
        alpha2 = calculate_alpha(contxt_input2, self.W_ctx_to_att2, self.W_hid_to_att2, self.W_att2, self.W_lambda2, enc_mask2, delta_inds2, self.W_mu2)
        alpha3 = calculate_alpha(contxt_input3, self.W_ctx_to_att3, self.W_hid_to_att3, self.W_att3, self.W_lambda3, enc_mask3, delta_inds3, self.W_mu3)
        alpha4 = calculate_alpha(contxt_input4, self.W_ctx_to_att4, self.W_hid_to_att4, self.W_att4, self.W_lambda4, enc_mask4, delta_inds4, self.W_mu4)
        alpha_conct = T.concatenate([alpha1, alpha2, alpha3, alpha4], axis=0)
        return alpha_conct


class LSTMAttLayer_lambda_mu4(MergeLayer):
    r"""
    lasagne.layers.recurrent.LSTMLayer(incoming, num_units,
    ingate=lasagne.layers.Gate(), forgetgate=lasagne.layers.Gate(),
    cell=lasagne.layers.Gate(
    W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
    outgate=lasagne.layers.Gate(),
    nonlinearity=lasagne.nonlinearities.tanh,
    cell_init=lasagne.init.Constant(0.),
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    peepholes=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)
    A long short-term memory (LSTM) layer.
    Includes optional "peephole connections" and a forget gate.  Based on the
    definition in [1]_, which is the current common definition.  The output is
    computed by
    .. math ::
        i_t &= \sigma_i(x_t W_{xi} + h_{t-1} W_{hi}
               + w_{ci} \odot c_{t-1} + b_i)\\
        f_t &= \sigma_f(x_t W_{xf} + h_{t-1} W_{hf}
               + w_{cf} \odot c_{t-1} + b_f)\\
        c_t &= f_t \odot c_{t - 1}
               + i_t \odot \sigma_c(x_t W_{xc} + h_{t-1} W_{hc} + b_c)\\
        o_t &= \sigma_o(x_t W_{xo} + h_{t-1} W_{ho} + w_{co} \odot c_t + b_o)\\
        h_t &= o_t \odot \sigma_h(c_t)
    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden/cell units in the layer.
    ingate : Gate
        Parameters for the input gate (:math:`i_t`): :math:`W_{xi}`,
        :math:`W_{hi}`, :math:`w_{ci}`, :math:`b_i`, and :math:`\sigma_i`.
    forgetgate : Gate
        Parameters for the forget gate (:math:`f_t`): :math:`W_{xf}`,
        :math:`W_{hf}`, :math:`w_{cf}`, :math:`b_f`, and :math:`\sigma_f`.
    cell : Gate
        Parameters for the cell computation (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    outgate : Gate
        Parameters for the output gate (:math:`o_t`): :math:`W_{xo}`,
        :math:`W_{ho}`, :math:`w_{co}`, :math:`b_o`, and :math:`\sigma_o`.
    nonlinearity : callable or None
        The nonlinearity that is applied to the output (:math:`\sigma_h`). If
        None is provided, no nonlinearity will be applied.
    cell_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial cell state (:math:`c_0`).
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    peepholes : bool
        If True, the LSTM uses peephole connections.
        When False, `ingate.W_cell`, `forgetgate.W_cell` and
        `outgate.W_cell` are ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.
    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """
    def __init__(self, incoming, num_units,
                 ingate=myGate(),
                 forgetgate=myGate(),
                 cell=myGate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=myGate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.), 
                 contxt_input= init.Constant(0.), contxt_input2=init.Constant(0.), contxt_input3=init.Constant(0.), contxt_input4=init.Constant(0.), 
                 ctx_init = init.Constant(0.),
                 att_num_units = 64, W_mu=init.Normal(0.1), W_mu2=init.Normal(0.1), W_mu3=init.Normal(0.1), 
                 W_mu4=init.Normal(0.1), enc_mask_input = None,
                 enc_mask_input2=None, enc_mask_input3=None, enc_mask_input4=None,
                 ws=96, pred_len=4, pred_ind=0, delta_inds_input=None, 
                 delta_inds_input2=None, delta_inds_input3=None, delta_inds_input4=None,
                 W_hid_to_att = init.GlorotNormal(), W_hid_to_att2=init.GlorotNormal(), W_hid_to_att3=init.GlorotNormal(),
                 W_hid_to_att4=init.GlorotNormal(),
                 W_ctx_to_att = init.GlorotNormal(), W_ctx_to_att2=init.GlorotNormal(), W_ctx_to_att3=init.GlorotNormal(), 
                 W_ctx_to_att4=init.GlorotNormal(),
                 W_att = init.Normal(0.1), W_att2=init.Normal(0.1), W_att3=init.Normal(0.1), W_att4=init.Normal(0.1),
                 W_lambda= init.Normal(0.1), W_lambda2=init.Normal(0.1), W_lambda3=init.Normal(0.1),
                 W_lambda4=init.Normal(0.1), 
                 W_ctx_to_ingate = init.GlorotNormal(),
                 W_ctx_to_forgetgate = init.GlorotNormal(),
                 W_ctx_to_cell = init.GlorotNormal(),
                 W_ctx_to_outgate = init.GlorotNormal(), 
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=True,
                 **kwargs):
        # This layer inherits from a MergeLayer, because it can have four
        # inputs - the layer input, the mask, the initial hidden state and the
        # inital cell state. We will just provide the layer input as incomings,
        # unless a mask input, inital hidden state or initial cell state was
        # provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.cell_init_incoming_index = -1
        self.contxt_input_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1
        if isinstance(contxt_input, Layer):
            incomings.append(contxt_input)
            self.contxt_input_incoming_index = len(incomings)-1
        if isinstance(contxt_input2, Layer):
            incomings.append(contxt_input2)
            self.contxt_input2_incoming_index = len(incomings)-1 
        if isinstance(contxt_input3, Layer):
            incomings.append(contxt_input3)
            self.contxt_input3_incoming_index = len(incomings)-1
        if isinstance(contxt_input4, Layer):
            incomings.append(contxt_input4)
            self.contxt_input4_incoming_index = len(incomings)-1 
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1
        if isinstance(ctx_init, Layer):
            incomings.append(ctx_init)
            self.ctx_init_incoming_index = len(incomings)-1     
        if enc_mask_input is not None:
            incomings.append(enc_mask_input)
            self.enc_mask_incoming_index = len(incomings)-1
        if enc_mask_input2 is not None:
            incomings.append(enc_mask_input2)
            self.enc_mask_incoming_index2 = len(incomings)-1
        if enc_mask_input3 is not None:
            incomings.append(enc_mask_input3)
            self.enc_mask_incoming_index3 = len(incomings)-1
        if enc_mask_input4 is not None:
            incomings.append(enc_mask_input4)
            self.enc_mask_incoming_index4 = len(incomings)-1
        if delta_inds_input is not None:
            incomings.append(delta_inds_input)
            self.delta_inds_incoming_index = len(incomings)-1
        if delta_inds_input2 is not None:
            incomings.append(delta_inds_input2)
            self.delta_inds_incoming_index2 = len(incomings)-1
        if delta_inds_input3 is not None:
            incomings.append(delta_inds_input3)
            self.delta_inds_incoming_index3 = len(incomings)-1
        if delta_inds_input4 is not None:
            incomings.append(delta_inds_input4)
            self.delta_inds_incoming_index4 = len(incomings)-1
        # Initialize parent layer
        super(LSTMAttLayer_lambda_mu4, self).__init__(incomings, **kwargs)
        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity
        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")
        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]
        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")
        num_inputs = np.prod(input_shape[2:])
        #def cal_attent():       
        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)
        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate, 
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')
        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                                 'forgetgate')
        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')
        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')
        #    
        #IMP:
        num_ctx_input = 4
        self.W_ctx_to_ingate = self.add_param(W_ctx_to_ingate, (2*num_units*num_ctx_input, num_units), name='W_ctx_ingate')
        self.W_ctx_to_forgetgate = self.add_param(W_ctx_to_forgetgate, (2*num_units*num_ctx_input, num_units), name='W_ctx_forgetgate')
        self.W_ctx_to_cell = self.add_param(W_ctx_to_cell, (2*num_units*num_ctx_input, num_units), name='W_ctx_cell')
        self.W_ctx_to_outgate = self.add_param(W_ctx_to_outgate, (2*num_units*num_ctx_input, num_units), name='W_ctx_outgate')
        #
        #attention Weights        
        #b_att = init.Constant(0.)
        #
        b_s, seq_len_enc, ctx_fea_len  = contxt_input.output_shape
        #b_s, seq_len_enc, ctx_fea_len = contxt_input.shape
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_hid_to_att2 = self.add_param(W_hid_to_att2, (num_units, att_num_units), name='W_hid_to_att2')
        self.W_hid_to_att3 = self.add_param(W_hid_to_att3, (num_units, att_num_units), name='W_hid_to_att3')
        self.W_hid_to_att4 = self.add_param(W_hid_to_att4, (num_units, att_num_units), name='W_hid_to_att4')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_ctx_to_att2 = self.add_param(W_ctx_to_att2, (2*num_units, att_num_units), name='W_ctx_to_att2')
        self.W_ctx_to_att3 = self.add_param(W_ctx_to_att3, (2*num_units, att_num_units), name='W_ctx_to_att3')
        self.W_ctx_to_att4 = self.add_param(W_ctx_to_att4, (2*num_units, att_num_units), name='W_ctx_to_att4')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        self.W_att2 = self.add_param(W_att2, (att_num_units,), name='W_att2')
        self.W_att3 = self.add_param(W_att3, (att_num_units,), name='W_att3')
        self.W_att4 = self.add_param(W_att4, (att_num_units,), name='W_att4')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        self.att_num_units = att_num_units
        self.W_lambda = self.add_param(W_lambda, (ws + pred_len -1,), name='W_lambda')
        self.W_lambda2 = self.add_param(W_lambda2, (ws + pred_len -1,), name='W_lambda2')
        self.W_lambda3 = self.add_param(W_lambda3, (ws + pred_len -1,), name='W_lambda3')
        self.W_lambda4 = self.add_param(W_lambda4, (ws + pred_len -1,), name='W_lambda4')
        self.pred_len = pred_len
        self.pred_ind = pred_ind
        self.W_mu = self.add_param(W_mu, (3,), name='W_mu')
        self.W_mu2 = self.add_param(W_mu2, (3,), name='W_mu2')
        self.W_mu3 = self.add_param(W_mu3, (3,), name='W_mu3')
        self.W_mu4 = self.add_param(W_mu4, (3,), name='W_mu4')
        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_units, ), name="W_cell_to_ingate")
            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (num_units, ), name="W_cell_to_forgetgate")
            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (num_units, ), name="W_cell_to_outgate")
        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)        
        if isinstance(contxt_input, Layer):
            self.contxt_input = contxt_input        
        if isinstance(ctx_init, Layer):
            self.ctx_init = ctx_init
        else:
            self.ctx_init = self.add_param(
            ctx_init, (1, self.num_units*2*num_ctx_input), name='ctx_init',
            trainable=True, regularizable=False)
    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units
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
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        contxt_input = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]
        if self.contxt_input_incoming_index > 0:
            contxt_input = inputs[self.contxt_input_incoming_index]
        if self.contxt_input2_incoming_index > 0:
            contxt_input2 = inputs[self.contxt_input2_incoming_index]
        if self.contxt_input3_incoming_index > 0:
            contxt_input3 = inputs[self.contxt_input3_incoming_index]
        if self.contxt_input4_incoming_index > 0:
            contxt_input4 = inputs[self.contxt_input4_incoming_index]
        if self.ctx_init_incoming_index > 0:
            ctx_init = inputs[self.ctx_init_incoming_index]        
        if self.enc_mask_incoming_index > 0:
            enc_mask = inputs[self.enc_mask_incoming_index]
        if self.enc_mask_incoming_index2 > 0:
            enc_mask2 = inputs[self.enc_mask_incoming_index2]
        if self.enc_mask_incoming_index3 > 0:
            enc_mask3 = inputs[self.enc_mask_incoming_index3]
        if self.enc_mask_incoming_index4 > 0:
            enc_mask4 = inputs[self.enc_mask_incoming_index4]
        if self.delta_inds_incoming_index > 0:
            delta_inds = inputs[self.delta_inds_incoming_index]
        if self.delta_inds_incoming_index2 > 0:
            delta_inds2 = inputs[self.delta_inds_incoming_index2]
        if self.delta_inds_incoming_index3 > 0:
            delta_inds3 = inputs[self.delta_inds_incoming_index3]
        if self.delta_inds_incoming_index4 > 0:
            delta_inds4 = inputs[self.delta_inds_incoming_index4]
        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)
        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape
        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)
        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)
        #DONE: buraya stack att'yi ekle
        W_ctx_stacked = T.concatenate(
            [self.W_ctx_to_ingate, self.W_ctx_to_forgetgate,
             self.W_ctx_to_cell, self.W_ctx_to_outgate], axis=1)
        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)
        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked
        #
        
        def delta_mtx(seq_len_enc):
            dia = T.ones((seq_len_enc,seq_len_enc))
            diag = T.identity_like(dia)
            anti_diag = diag[::-1]
            delt = T.zeros((seq_len_enc+self.pred_len-1,seq_len_enc))
            delta =  T.set_subtensor(delt[self.pred_ind:self.pred_ind+seq_len_enc, :], anti_diag)
            return delta
        #contxt_input.output_shape        
        #pre_ctx: (seq_len_enc, n_batch, num_att_units), ctx_shuffle: (seq_len_enc, n_batch, n_feature)
        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
        def cal_contx(hid_previous, ctx_input, ctx2att_mtx, hid2att_mtx, v_a, lambda_mtx, enc_mask_mtx, delta_inds_vec, mu_vec):
            bs, seq_len_enc, _ = ctx_input.shape
            contxt_sh = ctx_input.dimshuffle(1, 0, 2)
            pre_comp_ctx = T.dot(contxt_sh, ctx2att_mtx)#self.W_ctx_to_att)
            contxt_sht = ctx_input.dimshuffle(2, 1, 0)
            e_dec = T.dot(hid_previous, hid2att_mtx)#self.W_hid_to_att)
            e_conct = T.tile(e_dec, (seq_len_enc,1,1))
            ener_i = self.nonlinearity_att(e_conct + pre_comp_ctx)
            e_i = T.dot(ener_i, v_a)#self.W_att)
            delta = delta_mtx(seq_len_enc)
            lambda_delta = T.dot(lambda_mtx.T, delta)#T.dot(self.W_lambda.T, delta)
            lambda_delta_tile = T.tile(lambda_delta, (bs,1), ndim=2).T
            len_mask = bs* seq_len_enc
            delta_gap = T.zeros((len_mask, 3),dtype='float32')
            mask_flat = T.flatten(enc_mask_mtx)#enc_mask)
            zeros_inds = T.eq(mask_flat, 0).nonzero()
            delta_gap2 = theano.tensor.set_subtensor(delta_gap[zeros_inds, delta_inds_vec],1) #theano.tensor.set_subtensor(delta_gap[zeros_inds, delta_inds],1)
            delta_gap3 = delta_gap2.reshape((bs, seq_len_enc, 3), ndim=3)
            delta_gap4 = delta_gap3.dimshuffle(1,0,2)
            mudelta = T.dot(delta_gap4, mu_vec)#self.W_mu)
            e_i_n = e_i * lambda_delta_tile
            e_i_new = e_i_n * (1 + mudelta)
            alpha = T.exp(e_i_new)
            alpha /= T.sum(alpha, axis=0)          
            mult = T.mul(contxt_sht, alpha)
            ctx = T.sum(mult, axis=1)
            return ctx.T
        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, contxt_previous, *args):
            contxt1 = cal_contx(hid_previous, contxt_input, self.W_ctx_to_att, self.W_hid_to_att, self.W_att, self.W_lambda, enc_mask, delta_inds, self.W_mu)
            contxt2 = cal_contx(hid_previous, contxt_input2, self.W_ctx_to_att2, self.W_hid_to_att2, self.W_att2, self.W_lambda2, enc_mask2, delta_inds2, self.W_mu2)
            contxt3 = cal_contx(hid_previous, contxt_input3, self.W_ctx_to_att3, self.W_hid_to_att3, self.W_att3, self.W_lambda3, enc_mask3, delta_inds3, self.W_mu3)
            contxt4 = cal_contx(hid_previous, contxt_input4, self.W_ctx_to_att4, self.W_hid_to_att4, self.W_att4, self.W_lambda4, enc_mask4, delta_inds4, self.W_mu4)
            contxt = T.concatenate([contxt1, contxt2, contxt3, contxt4], axis=1)                                
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked
            # Calculate gates pre-activations and slice
            gates1 = input_n + T.dot(hid_previous, W_hid_stacked) 
            #att etkisini ekle
            gates = gates1 + T.dot(contxt, W_ctx_stacked)
            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)
            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)
            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate
            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)
            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input
            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)
            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            return [cell, hid, contxt]
        def step_masked(input_n, mask_n, cell_previous, hid_previous, contxt_previous, *args):
            cell, hid, contxt = step(input_n, cell_previous, hid_previous, contxt_previous, *args)
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            contxt = T.switch(mask_n, contxt, contxt_previous)
            if not mask_n:
                print('Mask 0 here something wrong going on!')
            return [cell, hid, contxt]
        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step
        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)
        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)        
        if not isinstance(self.ctx_init, Layer):
            ctx_init = T.dot(ones, self.ctx_init)
        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        non_seqs += [W_ctx_stacked]
        non_seqs += [self.W_hid_to_att, self.W_ctx_to_att, self.W_att]
        non_seqs += [self.W_hid_to_att2, self.W_ctx_to_att2, self.W_att2]
        non_seqs += [self.W_hid_to_att3, self.W_ctx_to_att3, self.W_att3]
        non_seqs += [self.W_hid_to_att4, self.W_ctx_to_att4, self.W_att4]
        non_seqs += [self.W_lambda, self.W_lambda2, self.W_lambda3, self.W_lambda4]
        non_seqs += [self.W_mu, self.W_mu2, self.W_mu3, self.W_mu4]
        non_seqs += [contxt_input, contxt_input2, contxt_input3, contxt_input4]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]
        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out, ctx_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init, ctx_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out, ctx_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init, ctx_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=False)[0]
        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)
            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]
        return hid_out 



class AnaLayer_lambda_mu_alt4(MergeLayer):
    def __init__(self, incoming, num_units, att_num_units = 64, ws=96, pred_len=4, pred_ind=0, contxt_input=init.Constant(0.),
                 contxt_input2=init.Constant(0.), contxt_input3=init.Constant(0.), contxt_input4=init.Constant(0.),
                 W_att=init.Normal(0.1), W_att2=init.Normal(0.1), W_att3=init.Normal(0.1), W_att4=init.Normal(0.1),
                 W_lambda= init.Normal(0.1), W_lambda2=init.Normal(0.1), W_lambda3=init.Normal(0.1),
                 W_lambda4=init.Normal(0.1), W_mu=init.Normal(0.1), W_mu2=init.Normal(0.1), W_mu3=init.Normal(0.1), 
                 W_mu4=init.Normal(0.1), delta_inds_input=None, delta_inds_input2=None, delta_inds_input3=None, delta_inds_input4=None,
                 W_hid_to_att=init.GlorotNormal(), W_hid_to_att2=init.GlorotNormal(), W_hid_to_att3=init.GlorotNormal(),
                 W_hid_to_att4=init.GlorotNormal(), W_ctx_to_att=init.GlorotNormal(), W_ctx_to_att2=init.GlorotNormal(),
                 W_ctx_to_att3=init.GlorotNormal(), W_ctx_to_att4=init.GlorotNormal(),
                 **kwargs):        
        incomings = [incoming]        
        self.contxt_input_incoming_index = -1
        self.delta_inds_incoming_index = -1
        if isinstance(contxt_input, Layer):
            incomings.append(contxt_input)
            self.contxt_input_incoming_index = len(incomings)-1 
        if isinstance(contxt_input2, Layer):
            incomings.append(contxt_input2)
            self.contxt_input2_incoming_index = len(incomings)-1 
        if isinstance(contxt_input3, Layer):
            incomings.append(contxt_input3)
            self.contxt_input3_incoming_index = len(incomings)-1
        if isinstance(contxt_input4, Layer):
            incomings.append(contxt_input4)
            self.contxt_input4_incoming_index = len(incomings)-1
        if delta_inds_input is not None:
            incomings.append(delta_inds_input)
            self.delta_inds_incoming_index = len(incomings)-1
        if delta_inds_input2 is not None:
            incomings.append(delta_inds_input2)
            self.delta_inds_incoming_index2 = len(incomings)-1
        if delta_inds_input3 is not None:
            incomings.append(delta_inds_input3)
            self.delta_inds_incoming_index3 = len(incomings)-1
        if delta_inds_input4 is not None:
            incomings.append(delta_inds_input4)
            self.delta_inds_incoming_index4 = len(incomings)-1
        super(AnaLayer_lambda_mu_alt4, self).__init__(incomings, **kwargs)
        self.num_units = num_units  #number of attention units
        #num_inputs = np.prod(input_shape[2:])
        #
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')        
        self.W_hid_to_att2 = self.add_param(W_hid_to_att2, (num_units, att_num_units), name='W_hid_to_att2')
        self.W_hid_to_att3 = self.add_param(W_hid_to_att3, (num_units, att_num_units), name='W_hid_to_att3')
        self.W_hid_to_att4 = self.add_param(W_hid_to_att4, (num_units, att_num_units), name='W_hid_to_att4')
        #
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_ctx_to_att2 = self.add_param(W_ctx_to_att2, (2*num_units, att_num_units), name='W_ctx_to_att2')
        self.W_ctx_to_att3 = self.add_param(W_ctx_to_att3, (2*num_units, att_num_units), name='W_ctx_to_att3')
        self.W_ctx_to_att4 = self.add_param(W_ctx_to_att4, (2*num_units, att_num_units), name='W_ctx_to_att4')
        #
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        self.W_att2 = self.add_param(W_att2, (att_num_units,), name='W_att2')
        self.W_att3 = self.add_param(W_att3, (att_num_units,), name='W_att3')
        self.W_att4 = self.add_param(W_att4, (att_num_units,), name='W_att4')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        self.att_num_units = att_num_units
        if isinstance(contxt_input, Layer):
            self.contxt_input = contxt_input
            #_, self.seq_len_enc, ctx_fea_len = contxt_input.shape
        self.W_lambda = self.add_param(W_lambda, (ws + pred_len -1,), name='W_lambda')
        self.W_lambda2 = self.add_param(W_lambda2, (ws + pred_len -1,), name='W_lambda2')
        self.W_lambda3 = self.add_param(W_lambda3, (ws + pred_len -1,), name='W_lambda3')
        self.W_lambda4 = self.add_param(W_lambda4, (ws + pred_len -1,), name='W_lambda4')
        self.pred_len = pred_len
        self.pred_ind = pred_ind
        self.W_mu = self.add_param(W_mu, (1,), name='W_mu')
        self.W_mu2 = self.add_param(W_mu2, (1,), name='W_mu2')
        self.W_mu3 = self.add_param(W_mu3, (1,), name='W_mu3')
        self.W_mu4 = self.add_param(W_mu4, (1,), name='W_mu4')
    def get_output_shape_for(self, input_shape):
        #_, seq_len_enc, _ = self.contxt_input.shape
        return (input_shape[0][0], None)
    def get_output_for(self, inputs, **kwargs):
        hid_previous = inputs[0]      
        #
        contxt_input = None
        contxt_input2 = None
        contxt_input3 = None
        contxt_input4 = None
        if self.contxt_input_incoming_index > 0:
            contxt_input = inputs[self.contxt_input_incoming_index]
        if self.contxt_input2_incoming_index > 0:
            contxt_input2 = inputs[self.contxt_input2_incoming_index]
        if self.contxt_input3_incoming_index > 0:
            contxt_input3 = inputs[self.contxt_input3_incoming_index]
        if self.contxt_input4_incoming_index > 0:
            contxt_input4 = inputs[self.contxt_input4_incoming_index]
        if self.delta_inds_incoming_index > 0:
            delta_inds = inputs[self.delta_inds_incoming_index]
        if self.delta_inds_incoming_index2 > 0:
            delta_inds2 = inputs[self.delta_inds_incoming_index2]
        if self.delta_inds_incoming_index3 > 0:
            delta_inds3 = inputs[self.delta_inds_incoming_index3]
        if self.delta_inds_incoming_index4 > 0:
            delta_inds4 = inputs[self.delta_inds_incoming_index4]
        #
        #bs, seq_len_enc, _ = contxt_input.shape 
        def delta_mtx(seq_len_enc):
            dia = T.ones((seq_len_enc,seq_len_enc))
            diag = T.identity_like(dia)
            anti_diag = diag[::-1]
            delt = T.zeros((seq_len_enc+self.pred_len-1,seq_len_enc))
            delta =  T.set_subtensor(delt[self.pred_ind:self.pred_ind+seq_len_enc, :], anti_diag)
            return delta
        #
        def calculate_alpha(ctx_input, ctx2att_mtx, hid2att_mtx, v_a, lambda_mtx, delta_inds_vec, mu_scalar):
            bs, seq_len_enc, _ = ctx_input.shape
            contxt_sh = ctx_input.dimshuffle(1, 0, 2)
            pre_comp_ctx = T.dot(contxt_sh, ctx2att_mtx)#self.W_ctx_to_att)
            e_dec = T.dot(hid_previous, hid2att_mtx)#self.W_hid_to_att)
            e_conct = T.tile(e_dec, (seq_len_enc,1,1))
            ener_i = self.nonlinearity_att(e_conct +pre_comp_ctx)
            e_i = T.dot(ener_i, v_a)#self.W_att)
            delta = delta_mtx(seq_len_enc)
            lambda_delta = T.dot(lambda_mtx.T, delta)
            lambda_delta_tile = T.tile(lambda_delta, (bs,1), ndim=2).T
            e_i_l = e_i * lambda_delta_tile        
            e_i_l = e_i * lambda_delta_tile
            mu_exp = T.exp(-mu_scalar*delta_inds_vec) #T.exp(-self.W_mu*delta_inds)
            e_i_lm = e_i_l * mu_exp.T
            alpha = T.exp(e_i_lm)
            alpha /= T.sum(alpha, axis=0)          
            return alpha.T
        #
        alpha1 = calculate_alpha(contxt_input, self.W_ctx_to_att, self.W_hid_to_att, self.W_att, self.W_lambda, delta_inds, self.W_mu)
        alpha2 = calculate_alpha(contxt_input2, self.W_ctx_to_att2, self.W_hid_to_att2, self.W_att2, self.W_lambda2, delta_inds2, self.W_mu2)
        alpha3 = calculate_alpha(contxt_input3, self.W_ctx_to_att3, self.W_hid_to_att3, self.W_att3, self.W_lambda3, delta_inds3, self.W_mu3)
        alpha4 = calculate_alpha(contxt_input4, self.W_ctx_to_att4, self.W_hid_to_att4, self.W_att4, self.W_lambda4, delta_inds4, self.W_mu4)
        alpha_conct = T.concatenate([alpha1, alpha2, alpha3, alpha4], axis=0)
        return alpha_conct


class LSTMAttLayer_lambda_mu_alt4(MergeLayer):
    r"""
    lasagne.layers.recurrent.LSTMLayer(incoming, num_units,
    ingate=lasagne.layers.Gate(), forgetgate=lasagne.layers.Gate(),
    cell=lasagne.layers.Gate(
    W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
    outgate=lasagne.layers.Gate(),
    nonlinearity=lasagne.nonlinearities.tanh,
    cell_init=lasagne.init.Constant(0.),
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    peepholes=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)
    A long short-term memory (LSTM) layer.
    Includes optional "peephole connections" and a forget gate.  Based on the
    definition in [1]_, which is the current common definition.  The output is
    computed by
    .. math ::
        i_t &= \sigma_i(x_t W_{xi} + h_{t-1} W_{hi}
               + w_{ci} \odot c_{t-1} + b_i)\\
        f_t &= \sigma_f(x_t W_{xf} + h_{t-1} W_{hf}
               + w_{cf} \odot c_{t-1} + b_f)\\
        c_t &= f_t \odot c_{t - 1}
               + i_t \odot \sigma_c(x_t W_{xc} + h_{t-1} W_{hc} + b_c)\\
        o_t &= \sigma_o(x_t W_{xo} + h_{t-1} W_{ho} + w_{co} \odot c_t + b_o)\\
        h_t &= o_t \odot \sigma_h(c_t)
    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden/cell units in the layer.
    ingate : Gate
        Parameters for the input gate (:math:`i_t`): :math:`W_{xi}`,
        :math:`W_{hi}`, :math:`w_{ci}`, :math:`b_i`, and :math:`\sigma_i`.
    forgetgate : Gate
        Parameters for the forget gate (:math:`f_t`): :math:`W_{xf}`,
        :math:`W_{hf}`, :math:`w_{cf}`, :math:`b_f`, and :math:`\sigma_f`.
    cell : Gate
        Parameters for the cell computation (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    outgate : Gate
        Parameters for the output gate (:math:`o_t`): :math:`W_{xo}`,
        :math:`W_{ho}`, :math:`w_{co}`, :math:`b_o`, and :math:`\sigma_o`.
    nonlinearity : callable or None
        The nonlinearity that is applied to the output (:math:`\sigma_h`). If
        None is provided, no nonlinearity will be applied.
    cell_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial cell state (:math:`c_0`).
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    peepholes : bool
        If True, the LSTM uses peephole connections.
        When False, `ingate.W_cell`, `forgetgate.W_cell` and
        `outgate.W_cell` are ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.
    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """
    def __init__(self, incoming, num_units,
                 ingate=myGate(),
                 forgetgate=myGate(),
                 cell=myGate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=myGate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.), 
                 contxt_input= init.Constant(0.), contxt_input2=init.Constant(0.), contxt_input3=init.Constant(0.), contxt_input4=init.Constant(0.),
                 ctx_init = init.Constant(0.),
                 att_num_units = 64, W_mu=init.Normal(0.1), W_mu2=init.Normal(0.1), W_mu3=init.Normal(0.1), 
                 W_mu4=init.Normal(0.1), delta_inds_input=None, delta_inds_input2=None, delta_inds_input3=None, delta_inds_input4=None,
                 ws=96, pred_len=4, pred_ind=0,
                 W_hid_to_att = init.GlorotNormal(), W_hid_to_att2=init.GlorotNormal(), W_hid_to_att3=init.GlorotNormal(),
                 W_hid_to_att4=init.GlorotNormal(),
                 W_ctx_to_att = init.GlorotNormal(), W_ctx_to_att2=init.GlorotNormal(), W_ctx_to_att3=init.GlorotNormal(), 
                 W_ctx_to_att4=init.GlorotNormal(),
                 W_att = init.Normal(0.1), W_att2=init.Normal(0.1), W_att3=init.Normal(0.1), W_att4=init.Normal(0.1),
                 W_lambda= init.Normal(0.1), W_lambda2=init.Normal(0.1), W_lambda3=init.Normal(0.1),
                 W_lambda4=init.Normal(0.1),
                 W_ctx_to_ingate = init.GlorotNormal(),
                 W_ctx_to_forgetgate = init.GlorotNormal(),
                 W_ctx_to_cell = init.GlorotNormal(),
                 W_ctx_to_outgate = init.GlorotNormal(), 
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=True,
                 **kwargs):
        # This layer inherits from a MergeLayer, because it can have four
        # inputs - the layer input, the mask, the initial hidden state and the
        # inital cell state. We will just provide the layer input as incomings,
        # unless a mask input, inital hidden state or initial cell state was
        # provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.cell_init_incoming_index = -1
        self.contxt_input_incoming_index = -1
        self.delta_inds_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1
        if isinstance(contxt_input, Layer):
            incomings.append(contxt_input)
            self.contxt_input_incoming_index = len(incomings)-1
        if isinstance(contxt_input2, Layer):
            incomings.append(contxt_input2)
            self.contxt_input2_incoming_index = len(incomings)-1 
        if isinstance(contxt_input3, Layer):
            incomings.append(contxt_input3)
            self.contxt_input3_incoming_index = len(incomings)-1
        if isinstance(contxt_input4, Layer):
            incomings.append(contxt_input4)
            self.contxt_input4_incoming_index = len(incomings)-1 
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1
        if isinstance(ctx_init, Layer):
            incomings.append(ctx_init)
            self.ctx_init_incoming_index = len(incomings)-1 
        if delta_inds_input is not None:
            incomings.append(delta_inds_input)
            self.delta_inds_incoming_index = len(incomings)-1
        if delta_inds_input2 is not None:
            incomings.append(delta_inds_input2)
            self.delta_inds_incoming_index2 = len(incomings)-1
        if delta_inds_input3 is not None:
            incomings.append(delta_inds_input3)
            self.delta_inds_incoming_index3 = len(incomings)-1
        if delta_inds_input4 is not None:
            incomings.append(delta_inds_input4)
            self.delta_inds_incoming_index4 = len(incomings)-1
        # Initialize parent layer
        super(LSTMAttLayer_lambda_mu_alt4, self).__init__(incomings, **kwargs)
        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity
        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")
        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]
        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")
        num_inputs = np.prod(input_shape[2:])
        #def cal_attent():       
        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)
        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate, 
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')
        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                                 'forgetgate')
        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')
        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')
        #     
        #IMP:
        num_ctx_input = 4
        self.W_ctx_to_ingate = self.add_param(W_ctx_to_ingate, (2*num_units*num_ctx_input, num_units), name='W_ctx_ingate')
        self.W_ctx_to_forgetgate = self.add_param(W_ctx_to_forgetgate, (2*num_units*num_ctx_input, num_units), name='W_ctx_forgetgate')
        self.W_ctx_to_cell = self.add_param(W_ctx_to_cell, (2*num_units*num_ctx_input, num_units), name='W_ctx_cell')
        self.W_ctx_to_outgate = self.add_param(W_ctx_to_outgate, (2*num_units*num_ctx_input, num_units), name='W_ctx_outgate')
        #
        #attention Weights        
        #b_att = init.Constant(0.)
        #
        b_s, seq_len_enc, ctx_fea_len  = contxt_input.output_shape
        #b_s, seq_len_enc, ctx_fea_len = contxt_input.shape
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_hid_to_att2 = self.add_param(W_hid_to_att2, (num_units, att_num_units), name='W_hid_to_att2')
        self.W_hid_to_att3 = self.add_param(W_hid_to_att3, (num_units, att_num_units), name='W_hid_to_att3')
        self.W_hid_to_att4 = self.add_param(W_hid_to_att4, (num_units, att_num_units), name='W_hid_to_att4')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_ctx_to_att2 = self.add_param(W_ctx_to_att2, (2*num_units, att_num_units), name='W_ctx_to_att2')
        self.W_ctx_to_att3 = self.add_param(W_ctx_to_att3, (2*num_units, att_num_units), name='W_ctx_to_att3')
        self.W_ctx_to_att4 = self.add_param(W_ctx_to_att4, (2*num_units, att_num_units), name='W_ctx_to_att4')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        self.W_att2 = self.add_param(W_att2, (att_num_units,), name='W_att2')
        self.W_att3 = self.add_param(W_att3, (att_num_units,), name='W_att3')
        self.W_att4 = self.add_param(W_att4, (att_num_units,), name='W_att4')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        self.att_num_units = att_num_units
        self.W_lambda = self.add_param(W_lambda, (ws + pred_len -1,), name='W_lambda')
        self.W_lambda2 = self.add_param(W_lambda2, (ws + pred_len -1,), name='W_lambda2')
        self.W_lambda3 = self.add_param(W_lambda3, (ws + pred_len -1,), name='W_lambda3')
        self.W_lambda4 = self.add_param(W_lambda4, (ws + pred_len -1,), name='W_lambda4')
        self.pred_len = pred_len
        self.pred_ind = pred_ind
        self.W_mu = self.add_param(W_mu, (1,), name='W_mu')
        self.W_mu2 = self.add_param(W_mu2, (1,), name='W_mu2')
        self.W_mu3 = self.add_param(W_mu3, (1,), name='W_mu3')
        self.W_mu4 = self.add_param(W_mu4, (1,), name='W_mu4')
        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_units, ), name="W_cell_to_ingate")
            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (num_units, ), name="W_cell_to_forgetgate")
            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (num_units, ), name="W_cell_to_outgate")
        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)        
        if isinstance(contxt_input, Layer):
            self.contxt_input = contxt_input        
        if isinstance(ctx_init, Layer):
            self.ctx_init = ctx_init
        else:
            self.ctx_init = self.add_param(
            ctx_init, (1, self.num_units*2*num_ctx_input), name='ctx_init',
            trainable=True, regularizable=False)
    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units
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
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        contxt_input = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]
        if self.contxt_input_incoming_index > 0:
            contxt_input = inputs[self.contxt_input_incoming_index]
        if self.contxt_input2_incoming_index > 0:
            contxt_input2 = inputs[self.contxt_input2_incoming_index]
        if self.contxt_input3_incoming_index > 0:
            contxt_input3 = inputs[self.contxt_input3_incoming_index]
        if self.contxt_input4_incoming_index > 0:
            contxt_input4 = inputs[self.contxt_input4_incoming_index]
        if self.ctx_init_incoming_index > 0:
            ctx_init = inputs[self.ctx_init_incoming_index]  
        if self.delta_inds_incoming_index > 0:
            delta_inds = inputs[self.delta_inds_incoming_index]
        if self.delta_inds_incoming_index2 > 0:
            delta_inds2 = inputs[self.delta_inds_incoming_index2]
        if self.delta_inds_incoming_index3 > 0:
            delta_inds3 = inputs[self.delta_inds_incoming_index3]
        if self.delta_inds_incoming_index4 > 0:
            delta_inds4 = inputs[self.delta_inds_incoming_index4]
        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)
        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape
        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)
        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)
        #DONE: buraya stack att'yi ekle
        W_ctx_stacked = T.concatenate(
            [self.W_ctx_to_ingate, self.W_ctx_to_forgetgate,
             self.W_ctx_to_cell, self.W_ctx_to_outgate], axis=1)
        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)
        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked
        #
        bs, seq_len_enc, _ = contxt_input.shape
        def delta_mtx():
            dia = T.ones((seq_len_enc,seq_len_enc))
            diag = T.identity_like(dia)
            anti_diag = diag[::-1]
            delt = T.zeros((seq_len_enc+self.pred_len-1,seq_len_enc))
            delta =  T.set_subtensor(delt[self.pred_ind:self.pred_ind+seq_len_enc, :], anti_diag)
            return delta
        #contxt_input.output_shape        
        #pre_ctx: (seq_len_enc, n_batch, num_att_units), ctx_shuffle: (seq_len_enc, n_batch, n_feature)
        
        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
        def cal_contx(hid_previous, ctx_input, ctx2att_mtx, hid2att_mtx, v_a, lambda_mtx, delta_inds_vec, mu_scalar):
            contxt_sh = ctx_input.dimshuffle(1, 0, 2)
            pre_comp_ctx = T.dot(contxt_sh, ctx2att_mtx)#self.W_ctx_to_att)
            contxt_sht = ctx_input.dimshuffle(2, 1, 0)
            e_dec = T.dot(hid_previous, hid2att_mtx)#self.W_hid_to_att)
            e_conct = T.tile(e_dec, (seq_len_enc,1,1))
            ener_i = self.nonlinearity_att(e_conct + pre_comp_ctx)
            e_i = T.dot(ener_i, v_a)#self.W_att)
            delta = delta_mtx()
            lambda_delta = T.dot(lambda_mtx.T, delta)#T.dot(self.W_lambda.T, delta)
            lambda_delta_tile = T.tile(lambda_delta, (bs,1), ndim=2).T
            e_i_l = e_i * lambda_delta_tile
            mu_exp = T.exp(-mu_scalar*delta_inds_vec)#T.exp(-self.W_mu*delta_inds)
            e_i_lm = e_i_l * mu_exp.T
            alpha = T.exp(e_i_lm)
            alpha /= T.sum(alpha, axis=0)          
            mult = T.mul(contxt_sht, alpha)
            ctx = T.sum(mult, axis=1)
            return ctx.T
        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, contxt_previous, *args):
            contxt1 = cal_contx(hid_previous, contxt_input, self.W_ctx_to_att, self.W_hid_to_att, self.W_att, self.W_lambda, delta_inds, self.W_mu)
            contxt2 = cal_contx(hid_previous, contxt_input2, self.W_ctx_to_att2, self.W_hid_to_att2, self.W_att2, self.W_lambda2, delta_inds2, self.W_mu2)
            contxt3 = cal_contx(hid_previous, contxt_input3, self.W_ctx_to_att3, self.W_hid_to_att3, self.W_att3, self.W_lambda3, delta_inds3, self.W_mu3)
            contxt4 = cal_contx(hid_previous, contxt_input4, self.W_ctx_to_att4, self.W_hid_to_att4, self.W_att4, self.W_lambda4, delta_inds4, self.W_mu4)
            contxt = T.concatenate([contxt1, contxt2, contxt3, contxt4], axis=1)   
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked
            # Calculate gates pre-activations and slice
            gates1 = input_n + T.dot(hid_previous, W_hid_stacked) 
            #att etkisini ekle
            gates = gates1 + T.dot(contxt, W_ctx_stacked)
            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)
            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)
            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate
            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)
            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input
            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)
            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            return [cell, hid, contxt]
        def step_masked(input_n, mask_n, cell_previous, hid_previous, contxt_previous, *args):
            cell, hid, contxt = step(input_n, cell_previous, hid_previous, contxt_previous, *args)
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            contxt = T.switch(mask_n, contxt, contxt_previous)
            if not mask_n:
                print('Mask 0 here something wrong going on!')
            return [cell, hid, contxt]
        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step
        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)
        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)        
        if not isinstance(self.ctx_init, Layer):
            ctx_init = T.dot(ones, self.ctx_init)
        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        non_seqs += [W_ctx_stacked]
        non_seqs += [self.W_hid_to_att, self.W_ctx_to_att, self.W_att]
        non_seqs += [self.W_hid_to_att2, self.W_ctx_to_att2, self.W_att2]
        non_seqs += [self.W_hid_to_att3, self.W_ctx_to_att3, self.W_att3]
        non_seqs += [self.W_hid_to_att4, self.W_ctx_to_att4, self.W_att4]
        non_seqs += [self.W_lambda, self.W_lambda2, self.W_lambda3, self.W_lambda4]
        non_seqs += [self.W_mu, self.W_mu2, self.W_mu3, self.W_mu4]
        non_seqs += [contxt_input, contxt_input2, contxt_input3, contxt_input4]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]
        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out, ctx_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init, ctx_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out, ctx_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init, ctx_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=False)[0]
        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)
            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]
        return hid_out 

