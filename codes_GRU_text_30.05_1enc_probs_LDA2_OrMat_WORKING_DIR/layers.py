# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:14:26 2016

@author: yagmur

Modified Lasagne layers and custom layers

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



class AnaLayer(MergeLayer):
    def __init__(self, incoming, num_units, att_num_units = 64, contxt_input=init.Constant(0.), 
                 W_att=init.Normal(0.1), 
                 W_hid_to_att=init.GlorotNormal(), W_ctx_to_att=init.GlorotNormal(),
                 **kwargs):        
        incomings = [incoming]        
        self.contxt_input_incoming_index = -1        
        if isinstance(contxt_input, Layer):
            incomings.append(contxt_input)
            self.contxt_input_incoming_index = len(incomings)-1        
        super(AnaLayer, self).__init__(incomings, **kwargs)
        self.num_units = num_units  #number of attention units
        #num_inputs = np.prod(input_shape[2:])
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
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
        contxt_input = None       
        if self.contxt_input_incoming_index > 0:
            contxt_input = inputs[self.contxt_input_incoming_index]
        #
        _, seq_len_enc, _ = contxt_input.shape   
        contxt_sh = contxt_input.dimshuffle(1, 0, 2)
        pre_comp_ctx = T.dot(contxt_sh, self.W_ctx_to_att)
        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        e_dec = T.dot(hid_previous, self.W_hid_to_att)
        e_conct = T.tile(e_dec, (seq_len_enc,1,1))
        ener_i = self.nonlinearity_att(e_conct +pre_comp_ctx)
        e_i = T.dot(ener_i, self.W_att)
        alpha = T.exp(e_i)
        alpha /= T.sum(alpha, axis=0)          
        alpha = alpha.T
        return alpha


class LSTMAttLayer(MergeLayer):
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
                 contxt_input= init.Constant(0.),
                 ctx_init = init.Constant(0.),
                 att_num_units = 64,
                 W_hid_to_att = init.GlorotNormal(),
                 W_ctx_to_att = init.GlorotNormal(),
                 W_att = init.Normal(0.1),
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
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1
        if isinstance(ctx_init, Layer):
            incomings.append(ctx_init)
            self.ctx_init_incoming_index = len(incomings)-1      
        # Initialize parent layer
        super(LSTMAttLayer, self).__init__(incomings, **kwargs)
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
        self.W_ctx_to_ingate = self.add_param(W_ctx_to_ingate, (2*num_units, num_units), name='W_ctx_ingate')
        self.W_ctx_to_forgetgate = self.add_param(W_ctx_to_forgetgate, (2*num_units, num_units), name='W_ctx_forgetgate')
        self.W_ctx_to_cell = self.add_param(W_ctx_to_cell, (2*num_units, num_units), name='W_ctx_cell')
        self.W_ctx_to_outgate = self.add_param(W_ctx_to_outgate, (2*num_units, num_units), name='W_ctx_outgate')
        #
        #attention Weights        
        #b_att = init.Constant(0.)
        #
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
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
            ctx_init, (1, self.num_units), name='ctx_init',
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
        _, seq_len_enc, ctx_fea_len = contxt_input.shape
        #contxt_input.output_shape        
        #pre_ctx: (seq_len_enc, n_batch, num_att_units), ctx_shuffle: (seq_len_enc, n_batch, n_feature)
        contxt_sh = contxt_input.dimshuffle(1, 0, 2)
        pre_comp_ctx = T.dot(contxt_sh, self.W_ctx_to_att)
        contxt_sht = contxt_input.dimshuffle(2, 1, 0)
        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
        def cal_contx(hid_previous):
            e_dec = T.dot(hid_previous, self.W_hid_to_att)
            e_conct = T.tile(e_dec, (seq_len_enc,1,1))
            ener_i = self.nonlinearity_att(e_conct +pre_comp_ctx)
            e_i = T.dot(ener_i, self.W_att)
            alpha = T.exp(e_i)
            alpha /= T.sum(alpha, axis=0)          
            mult = T.mul(contxt_sht, alpha)
            ctx = T.sum(mult, axis=1)
            return ctx.T
        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, contxt_previous, *args):
            contxt = cal_contx(hid_previous)
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
        non_seqs += [contxt_input]
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



class AnaLayer_lambda(MergeLayer):
    def __init__(self, incoming, num_units, att_num_units = 64, ws=96, pred_len=4, pred_ind=0, contxt_input=init.Constant(0.), 
                 W_att=init.Normal(0.1), W_lambda= init.Normal(0.1),
                 W_hid_to_att=init.GlorotNormal(), W_ctx_to_att=init.GlorotNormal(),
                 **kwargs):        
        incomings = [incoming]        
        self.contxt_input_incoming_index = -1        
        if isinstance(contxt_input, Layer):
            incomings.append(contxt_input)
            self.contxt_input_incoming_index = len(incomings)-1        
        super(AnaLayer_lambda, self).__init__(incomings, **kwargs)
        self.num_units = num_units  #number of attention units
        #num_inputs = np.prod(input_shape[2:])
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        self.att_num_units = att_num_units
        if isinstance(contxt_input, Layer):
            self.contxt_input = contxt_input
            #_, self.seq_len_enc, ctx_fea_len = contxt_input.shape
        self.W_lambda = self.add_param(W_lambda, (ws + pred_len -1,), name='W_lambda')
        self.pred_len = pred_len
        self.pred_ind = pred_ind
    def get_output_shape_for(self, input_shape):
        #_, seq_len_enc, _ = self.contxt_input.shape
        return (input_shape[0][0], None)
    def get_output_for(self, inputs, **kwargs):
        hid_previous = inputs[0]      
        #
        contxt_input = None       
        if self.contxt_input_incoming_index > 0:
            contxt_input = inputs[self.contxt_input_incoming_index]
        #
        bs, seq_len_enc, _ = contxt_input.shape
        def delta_mtx():
            dia = T.ones((seq_len_enc,seq_len_enc))
            diag = T.identity_like(dia)
            anti_diag = diag[::-1]
            delt = T.zeros((seq_len_enc+self.pred_len-1,seq_len_enc))
            delta =  T.set_subtensor(delt[self.pred_ind:self.pred_ind+seq_len_enc, :], anti_diag)
            return delta
        contxt_sh = contxt_input.dimshuffle(1, 0, 2)
        pre_comp_ctx = T.dot(contxt_sh, self.W_ctx_to_att)
        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        e_dec = T.dot(hid_previous, self.W_hid_to_att)
        e_conct = T.tile(e_dec, (seq_len_enc,1,1))
        ener_i = self.nonlinearity_att(e_conct +pre_comp_ctx)
        e_i = T.dot(ener_i, self.W_att)
        delta = delta_mtx()
        lambda_delta = T.dot(self.W_lambda.T, delta)
        lambda_delta_tile = T.tile(lambda_delta, (bs,1), ndim=2).T
        e_i_new = e_i * lambda_delta_tile
        alpha = T.exp(e_i_new)
        alpha /= T.sum(alpha, axis=0)          
        alpha = alpha.T
        return alpha


class LSTMAttLayer_lambda(MergeLayer):
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
                 contxt_input= init.Constant(0.),
                 ctx_init = init.Constant(0.),
                 att_num_units = 64,
                 ws=96, pred_len=4, pred_ind=0,
                 W_hid_to_att = init.GlorotNormal(),
                 W_ctx_to_att = init.GlorotNormal(),
                 W_att = init.Normal(0.1), W_lambda= init.Normal(0.1),
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
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1
        if isinstance(ctx_init, Layer):
            incomings.append(ctx_init)
            self.ctx_init_incoming_index = len(incomings)-1      
        # Initialize parent layer
        super(LSTMAttLayer_lambda, self).__init__(incomings, **kwargs)
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
        self.W_ctx_to_ingate = self.add_param(W_ctx_to_ingate, (2*num_units, num_units), name='W_ctx_ingate')
        self.W_ctx_to_forgetgate = self.add_param(W_ctx_to_forgetgate, (2*num_units, num_units), name='W_ctx_forgetgate')
        self.W_ctx_to_cell = self.add_param(W_ctx_to_cell, (2*num_units, num_units), name='W_ctx_cell')
        self.W_ctx_to_outgate = self.add_param(W_ctx_to_outgate, (2*num_units, num_units), name='W_ctx_outgate')
        #
        #attention Weights        
        #b_att = init.Constant(0.)
        #
        b_s, seq_len_enc, ctx_fea_len  = contxt_input.output_shape
        #b_s, seq_len_enc, ctx_fea_len = contxt_input.shape
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        self.att_num_units = att_num_units
        self.W_lambda = self.add_param(W_lambda, (ws + pred_len -1,), name='W_lambda')
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
            ctx_init, (1, self.num_units), name='ctx_init',
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
        contxt_sh = contxt_input.dimshuffle(1, 0, 2)
        pre_comp_ctx = T.dot(contxt_sh, self.W_ctx_to_att)
        contxt_sht = contxt_input.dimshuffle(2, 1, 0)
        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
        def cal_contx(hid_previous):
            e_dec = T.dot(hid_previous, self.W_hid_to_att)
            e_conct = T.tile(e_dec, (seq_len_enc,1,1))
            ener_i = self.nonlinearity_att(e_conct + pre_comp_ctx)
            e_i = T.dot(ener_i, self.W_att)
            delta = delta_mtx()
            lambda_delta = T.dot(self.W_lambda.T, delta)
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
            contxt = cal_contx(hid_previous)
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
        non_seqs += [contxt_input]
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


class AnaLayer_lambda_mu(MergeLayer):
    def __init__(self, incoming, num_units, att_num_units = 64, ws=96, pred_len=4, pred_ind=0, contxt_input=init.Constant(0.), 
                 W_att=init.Normal(0.1), W_lambda=init.Normal(0.1), W_mu=init.Normal(0.1), enc_mask_input = None,
                 W_hid_to_att=init.GlorotNormal(), W_ctx_to_att=init.GlorotNormal(), delta_inds_input=None,
                 **kwargs):        
        incomings = [incoming]        
        self.contxt_input_incoming_index = -1        
        if isinstance(contxt_input, Layer):
            incomings.append(contxt_input)
            self.contxt_input_incoming_index = len(incomings)-1
        if enc_mask_input is not None:
            incomings.append(enc_mask_input)
            self.enc_mask_incoming_index = len(incomings)-1
        if delta_inds_input is not None:
            incomings.append(delta_inds_input)
            self.delta_inds_incoming_index = len(incomings)-1
        super(AnaLayer_lambda_mu, self).__init__(incomings, **kwargs)
        self.num_units = num_units  #number of attention units
        #num_inputs = np.prod(input_shape[2:])
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        self.att_num_units = att_num_units
        if isinstance(contxt_input, Layer):
            self.contxt_input = contxt_input
            #_, self.seq_len_enc, ctx_fea_len = contxt_input.shape
        self.W_lambda = self.add_param(W_lambda, (ws + pred_len -1,), name='W_lambda')
        self.pred_len = pred_len
        self.pred_ind = pred_ind
        self.W_mu = self.add_param(W_mu, (3,), name='W_mu')
    def get_output_shape_for(self, input_shape):
        #_, seq_len_enc, _ = self.contxt_input.shape
        return (input_shape[0][0], None)
    def get_output_for(self, inputs, **kwargs):
        hid_previous = inputs[0]      
        #
        contxt_input = None       
        if self.contxt_input_incoming_index > 0:
            contxt_input = inputs[self.contxt_input_incoming_index]
        if self.enc_mask_incoming_index > 0:
            enc_mask = inputs[self.enc_mask_incoming_index]
        if self.delta_inds_incoming_index > 0:
            delta_inds = inputs[self.delta_inds_incoming_index]
        #
        bs, seq_len_enc, _ = contxt_input.shape
        def delta_mtx():
            dia = T.ones((seq_len_enc,seq_len_enc))
            diag = T.identity_like(dia)
            anti_diag = diag[::-1]
            delt = T.zeros((seq_len_enc+self.pred_len-1,seq_len_enc))
            delta =  T.set_subtensor(delt[self.pred_ind:self.pred_ind+seq_len_enc, :], anti_diag)
            return delta
        contxt_sh = contxt_input.dimshuffle(1, 0, 2)
        pre_comp_ctx = T.dot(contxt_sh, self.W_ctx_to_att)
        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        e_dec = T.dot(hid_previous, self.W_hid_to_att)
        e_conct = T.tile(e_dec, (seq_len_enc,1,1))
        ener_i = self.nonlinearity_att(e_conct +pre_comp_ctx)
        e_i = T.dot(ener_i, self.W_att)
        delta = delta_mtx()
        lambda_delta = T.dot(self.W_lambda.T, delta)
        lambda_delta_tile = T.tile(lambda_delta, (bs,1), ndim=2).T
        delta_gap = T.zeros((bs* seq_len_enc, 3))
        mask_flat = T.flatten(enc_mask)
        zeros_inds = T.eq(mask_flat, 0).nonzero()
        delta_gap2 = theano.tensor.set_subtensor(delta_gap[zeros_inds, delta_inds],1)
        delta_gap3 = delta_gap2.reshape((bs, seq_len_enc, 3))
        delta_gap4 = delta_gap3.dimshuffle(1,0,2)
        mudelta = T.dot(delta_gap4, self.W_mu)
        e_i_n = e_i * lambda_delta_tile
        e_i_new = e_i_n * (1 + mudelta)
        alpha = T.exp(e_i_new)
        alpha /= T.sum(alpha, axis=0)          
        alpha = alpha.T
        return alpha


class LSTMAttLayer_lambda_mu(MergeLayer):
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
                 contxt_input= init.Constant(0.),
                 ctx_init = init.Constant(0.),
                 att_num_units = 64, W_mu=init.Normal(0.1), enc_mask_input = None,
                 ws=96, pred_len=4, pred_ind=0, delta_inds_input=None,
                 W_hid_to_att = init.GlorotNormal(),
                 W_ctx_to_att = init.GlorotNormal(),
                 W_att = init.Normal(0.1), W_lambda= init.Normal(0.1),
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
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1
        if isinstance(ctx_init, Layer):
            incomings.append(ctx_init)
            self.ctx_init_incoming_index = len(incomings)-1     
        if enc_mask_input is not None:
            incomings.append(enc_mask_input)
            self.enc_mask_incoming_index = len(incomings)-1
        if delta_inds_input is not None:
            incomings.append(delta_inds_input)
            self.delta_inds_incoming_index = len(incomings)-1
        # Initialize parent layer
        super(LSTMAttLayer_lambda_mu, self).__init__(incomings, **kwargs)
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
        self.W_ctx_to_ingate = self.add_param(W_ctx_to_ingate, (2*num_units, num_units), name='W_ctx_ingate')
        self.W_ctx_to_forgetgate = self.add_param(W_ctx_to_forgetgate, (2*num_units, num_units), name='W_ctx_forgetgate')
        self.W_ctx_to_cell = self.add_param(W_ctx_to_cell, (2*num_units, num_units), name='W_ctx_cell')
        self.W_ctx_to_outgate = self.add_param(W_ctx_to_outgate, (2*num_units, num_units), name='W_ctx_outgate')
        #
        #attention Weights        
        #b_att = init.Constant(0.)
        #
        b_s, seq_len_enc, ctx_fea_len  = contxt_input.output_shape
        #b_s, seq_len_enc, ctx_fea_len = contxt_input.shape
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        self.att_num_units = att_num_units
        self.W_lambda = self.add_param(W_lambda, (ws + pred_len -1,), name='W_lambda')
        self.pred_len = pred_len
        self.pred_ind = pred_ind
        self.W_mu = self.add_param(W_mu, (3,), name='W_mu')
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
            ctx_init, (1, self.num_units), name='ctx_init',
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
        if self.ctx_init_incoming_index > 0:
            ctx_init = inputs[self.ctx_init_incoming_index]
        if self.enc_mask_incoming_index > 0:
            enc_mask = inputs[self.enc_mask_incoming_index]
        if self.delta_inds_incoming_index > 0:
            delta_inds = inputs[self.delta_inds_incoming_index]
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
        contxt_sh = contxt_input.dimshuffle(1, 0, 2)
        pre_comp_ctx = T.dot(contxt_sh, self.W_ctx_to_att)
        contxt_sht = contxt_input.dimshuffle(2, 1, 0)
        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
        def cal_contx(hid_previous):
            e_dec = T.dot(hid_previous, self.W_hid_to_att)
            e_conct = T.tile(e_dec, (seq_len_enc,1,1))
            ener_i = self.nonlinearity_att(e_conct + pre_comp_ctx)
            e_i = T.dot(ener_i, self.W_att)
            delta = delta_mtx()
            lambda_delta = T.dot(self.W_lambda.T, delta)
            lambda_delta_tile = T.tile(lambda_delta, (bs,1), ndim=2).T
            len_mask = bs* seq_len_enc
            delta_gap = T.zeros((len_mask, 3),dtype='float32')
            mask_flat = T.flatten(enc_mask)
            zeros_inds = T.eq(mask_flat, 0).nonzero()
            delta_gap2 = theano.tensor.set_subtensor(delta_gap[zeros_inds, delta_inds],1)
            delta_gap3 = delta_gap2.reshape((bs, seq_len_enc, 3), ndim=3)
            delta_gap4 = delta_gap3.dimshuffle(1,0,2)
            mudelta = T.dot(delta_gap4, self.W_mu)
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
            contxt = cal_contx(hid_previous)
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
        non_seqs += [self.W_hid_to_att, self.W_ctx_to_att, self.W_att, self.W_mu, self.W_lambda]
        non_seqs += [contxt_input]
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



class AnaLayer_v2(MergeLayer):
    def __init__(self, incoming, num_units, att_num_units = 64, contxt_input=init.Constant(0.), 
                 W_att=init.Normal(0.1), enc_mask_input = None,
                 W_hid_to_att=init.GlorotNormal(), W_ctx_to_att=init.GlorotNormal(),
                 **kwargs):        
        incomings = [incoming]        
        self.contxt_input_incoming_index = -1        
        if isinstance(contxt_input, Layer):
            incomings.append(contxt_input)
            self.contxt_input_incoming_index = len(incomings)-1 
        if enc_mask_input is not None:
            incomings.append(enc_mask_input)
            self.enc_mask_incoming_index = len(incomings)-1
        super(AnaLayer_v2, self).__init__(incomings, **kwargs)
        self.num_units = num_units  #number of attention units
        #num_inputs = np.prod(input_shape[2:])
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
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
        contxt_input = None       
        if self.contxt_input_incoming_index > 0:
            contxt_input = inputs[self.contxt_input_incoming_index]
        if self.enc_mask_incoming_index > 0:
            enc_mask = inputs[self.enc_mask_incoming_index]
        #
        _, seq_len_enc, _ = contxt_input.shape   
        contxt_sh = contxt_input.dimshuffle(1, 0, 2)
        pre_comp_ctx = T.dot(contxt_sh, self.W_ctx_to_att)
        r_zero, c_zero = T.eq(enc_mask, 0).nonzero()
        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        e_dec = T.dot(hid_previous, self.W_hid_to_att)
        e_conct = T.tile(e_dec, (seq_len_enc,1,1))
        ener_i = self.nonlinearity_att(e_conct +pre_comp_ctx)
        e_i = T.dot(ener_i, self.W_att)
        alpha = T.exp(e_i)
        alpha /= T.sum(alpha, axis=0)          
        alpha = alpha.T
        n_b, n_seq = alpha.shape
        zeros = theano.tensor.zeros_like(alpha)
        divider = theano.tensor.sum(enc_mask, axis=1)
        w_betas = theano.tensor.zeros_like(alpha)
        w_betas= theano.tensor.set_subtensor(w_betas[r_zero,c_zero], alpha[r_zero,c_zero])
        sum_w_betas = theano.tensor.sum(w_betas,axis=1)
        add_value = sum_w_betas/ divider
        add_value = theano.tensor.reshape(add_value, (n_b,1))
        add_mtx = theano.tensor.tile(add_value, n_seq) #N is the nr of columns
        new_alpha = alpha + add_mtx
        new_alpha = theano.tensor.set_subtensor(new_alpha[r_zero,c_zero],zeros[r_zero,c_zero])
        return new_alpha


class LSTMAttLayer_v2(MergeLayer):
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
                 contxt_input= init.Constant(0.),
                 ctx_init = init.Constant(0.), enc_mask_input = None,
                 att_num_units = 64,
                 W_hid_to_att = init.GlorotNormal(),
                 W_ctx_to_att = init.GlorotNormal(),
                 W_att = init.Normal(0.1),
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
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1
        if isinstance(ctx_init, Layer):
            incomings.append(ctx_init)
            self.ctx_init_incoming_index = len(incomings)-1 
        if enc_mask_input is not None:
            incomings.append(enc_mask_input)
            self.enc_mask_incoming_index = len(incomings)-1
        # Initialize parent layer
        super(LSTMAttLayer_v2, self).__init__(incomings, **kwargs)
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
        self.W_ctx_to_ingate = self.add_param(W_ctx_to_ingate, (2*num_units, num_units), name='W_ctx_ingate')
        self.W_ctx_to_forgetgate = self.add_param(W_ctx_to_forgetgate, (2*num_units, num_units), name='W_ctx_forgetgate')
        self.W_ctx_to_cell = self.add_param(W_ctx_to_cell, (2*num_units, num_units), name='W_ctx_cell')
        self.W_ctx_to_outgate = self.add_param(W_ctx_to_outgate, (2*num_units, num_units), name='W_ctx_outgate')
        #
        #attention Weights        
        #b_att = init.Constant(0.)
        #
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
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
            ctx_init, (1, self.num_units), name='ctx_init',
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
        if self.ctx_init_incoming_index > 0:
            ctx_init = inputs[self.ctx_init_incoming_index]     
        if self.enc_mask_incoming_index > 0:
            enc_mask = inputs[self.enc_mask_incoming_index]
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
        _, seq_len_enc, ctx_fea_len = contxt_input.shape
        #contxt_input.output_shape        
        #pre_ctx: (seq_len_enc, n_batch, num_att_units), ctx_shuffle: (seq_len_enc, n_batch, n_feature)
        contxt_sh = contxt_input.dimshuffle(1, 0, 2)
        pre_comp_ctx = T.dot(contxt_sh, self.W_ctx_to_att)
        contxt_sht = contxt_input.dimshuffle(2, 1, 0)
        divider = theano.tensor.sum(enc_mask, axis=1)
        r_zero, c_zero = T.eq(enc_mask, 0).nonzero()
        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
        def cal_contx(hid_previous):
            e_dec = T.dot(hid_previous, self.W_hid_to_att)
            e_conct = T.tile(e_dec, (seq_len_enc,1,1))
            ener_i = self.nonlinearity_att(e_conct +pre_comp_ctx)
            e_i = T.dot(ener_i, self.W_att)
            alpha = T.exp(e_i)
            alpha /= T.sum(alpha, axis=0)          
            n_seq, n_b = alpha.shape
            zeros = theano.tensor.zeros_like(alpha)            
            w_betas = theano.tensor.zeros_like(alpha)
            w_betas= theano.tensor.set_subtensor(w_betas[c_zero,r_zero], alpha[c_zero,r_zero])
            sum_w_betas = theano.tensor.sum(w_betas,axis=0)
            add_value = sum_w_betas/ divider
            add_value = theano.tensor.reshape(add_value, (n_b,1))
            add_mtx = theano.tensor.tile(add_value, n_seq) #N is the nr of columns
            new_alphas = alpha + add_mtx.T
            new_alphas = theano.tensor.set_subtensor(new_alphas[c_zero,r_zero],zeros[c_zero,r_zero])
            mult = T.mul(contxt_sht, new_alphas)
            ctx = T.sum(mult, axis=1)
            return ctx.T
        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, contxt_previous, *args):
            contxt = cal_contx(hid_previous)
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
        non_seqs += [contxt_input]
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




class AnaLayer_lambda_mu_alt(MergeLayer):
    def __init__(self, incoming, num_units, att_num_units = 64, ws=96, pred_len=4, pred_ind=0, contxt_input=init.Constant(0.), 
                 W_att=init.Normal(0.1), W_lambda= init.Normal(0.1), W_mu=init.Normal(0.1), delta_inds_input=None,
                 W_hid_to_att=init.GlorotNormal(), W_ctx_to_att=init.GlorotNormal(),
                 **kwargs):        
        incomings = [incoming]        
        self.contxt_input_incoming_index = -1
        self.delta_inds_incoming_index = -1
        if isinstance(contxt_input, Layer):
            incomings.append(contxt_input)
            self.contxt_input_incoming_index = len(incomings)-1  
        if delta_inds_input is not None:
            incomings.append(delta_inds_input)
            self.delta_inds_incoming_index = len(incomings)-1
        super(AnaLayer_lambda_mu_alt, self).__init__(incomings, **kwargs)
        self.num_units = num_units  #number of attention units
        #num_inputs = np.prod(input_shape[2:])
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        self.att_num_units = att_num_units
        if isinstance(contxt_input, Layer):
            self.contxt_input = contxt_input
            #_, self.seq_len_enc, ctx_fea_len = contxt_input.shape
        self.W_lambda = self.add_param(W_lambda, (ws + pred_len -1,), name='W_lambda')
        self.pred_len = pred_len
        self.pred_ind = pred_ind
        self.W_mu = self.add_param(W_mu, (1,), name='W_mu')
    def get_output_shape_for(self, input_shape):
        #_, seq_len_enc, _ = self.contxt_input.shape
        return (input_shape[0][0], None)
    def get_output_for(self, inputs, **kwargs):
        hid_previous = inputs[0]      
        #
        contxt_input = None       
        if self.contxt_input_incoming_index > 0:
            contxt_input = inputs[self.contxt_input_incoming_index]
        if self.delta_inds_incoming_index > 0:
            delta_inds = inputs[self.delta_inds_incoming_index]
        #
        bs, seq_len_enc, _ = contxt_input.shape
        def delta_mtx():
            dia = T.ones((seq_len_enc,seq_len_enc))
            diag = T.identity_like(dia)
            anti_diag = diag[::-1]
            delt = T.zeros((seq_len_enc+self.pred_len-1,seq_len_enc))
            delta =  T.set_subtensor(delt[self.pred_ind:self.pred_ind+seq_len_enc, :], anti_diag)
            return delta
        contxt_sh = contxt_input.dimshuffle(1, 0, 2)
        pre_comp_ctx = T.dot(contxt_sh, self.W_ctx_to_att)
        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        e_dec = T.dot(hid_previous, self.W_hid_to_att)
        e_conct = T.tile(e_dec, (seq_len_enc,1,1))
        ener_i = self.nonlinearity_att(e_conct +pre_comp_ctx)
        e_i = T.dot(ener_i, self.W_att)
        delta = delta_mtx()
        lambda_delta = T.dot(self.W_lambda.T, delta)
        lambda_delta_tile = T.tile(lambda_delta, (bs,1), ndim=2).T
        e_i_l = e_i * lambda_delta_tile
        mu_exp = T.exp(-self.W_mu*delta_inds)
        e_i_lm = e_i_l * mu_exp.T
        alpha = T.exp(e_i_lm)
        alpha /= T.sum(alpha, axis=0)          
        alpha = alpha.T
        return alpha


class LSTMAttLayer_lambda_mu_alt(MergeLayer):
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
                 contxt_input= init.Constant(0.),
                 ctx_init = init.Constant(0.),
                 att_num_units = 64, W_mu=init.Normal(0.1), delta_inds_input=None,
                 ws=96, pred_len=4, pred_ind=0,
                 W_hid_to_att = init.GlorotNormal(),
                 W_ctx_to_att = init.GlorotNormal(),
                 W_att = init.Normal(0.1), W_lambda= init.Normal(0.1),
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
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1
        if isinstance(ctx_init, Layer):
            incomings.append(ctx_init)
            self.ctx_init_incoming_index = len(incomings)-1 
        if delta_inds_input is not None:
            incomings.append(delta_inds_input)
            self.delta_inds_incoming_index = len(incomings)-1
        # Initialize parent layer
        super(LSTMAttLayer_lambda_mu_alt, self).__init__(incomings, **kwargs)
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
        self.W_ctx_to_ingate = self.add_param(W_ctx_to_ingate, (2*num_units, num_units), name='W_ctx_ingate')
        self.W_ctx_to_forgetgate = self.add_param(W_ctx_to_forgetgate, (2*num_units, num_units), name='W_ctx_forgetgate')
        self.W_ctx_to_cell = self.add_param(W_ctx_to_cell, (2*num_units, num_units), name='W_ctx_cell')
        self.W_ctx_to_outgate = self.add_param(W_ctx_to_outgate, (2*num_units, num_units), name='W_ctx_outgate')
        #
        #attention Weights        
        #b_att = init.Constant(0.)
        #
        b_s, seq_len_enc, ctx_fea_len  = contxt_input.output_shape
        #b_s, seq_len_enc, ctx_fea_len = contxt_input.shape
        self.W_hid_to_att = self.add_param(W_hid_to_att, (num_units, att_num_units), name='W_hid_to_att')
        self.W_ctx_to_att = self.add_param(W_ctx_to_att, (2*num_units, att_num_units), name='W_ctx_to_att')
        self.W_att = self.add_param(W_att, (att_num_units,), name='W_att')
        #self.b_att = self.add_param(b_att, (att_num_units,), name='b_att', regularizable=False)
        self.nonlinearity_att = nonlinearities.tanh
        self.att_num_units = att_num_units
        self.W_lambda = self.add_param(W_lambda, (ws + pred_len -1,), name='W_lambda')
        self.pred_len = pred_len
        self.pred_ind = pred_ind
        self.W_mu = self.add_param(W_mu, (1,), name='W_mu')
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
            ctx_init, (1, self.num_units), name='ctx_init',
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
        if self.ctx_init_incoming_index > 0:
            ctx_init = inputs[self.ctx_init_incoming_index]  
        if self.delta_inds_incoming_index > 0:
            delta_inds = inputs[self.delta_inds_incoming_index]
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
        contxt_sh = contxt_input.dimshuffle(1, 0, 2)
        pre_comp_ctx = T.dot(contxt_sh, self.W_ctx_to_att)
        contxt_sht = contxt_input.dimshuffle(2, 1, 0)
        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]
        def cal_contx(hid_previous):
            e_dec = T.dot(hid_previous, self.W_hid_to_att)
            e_conct = T.tile(e_dec, (seq_len_enc,1,1))
            ener_i = self.nonlinearity_att(e_conct + pre_comp_ctx)
            e_i = T.dot(ener_i, self.W_att)
            delta = delta_mtx()
            lambda_delta = T.dot(self.W_lambda.T, delta)
            lambda_delta_tile = T.tile(lambda_delta, (bs,1), ndim=2).T
            e_i_l = e_i * lambda_delta_tile
            mu_exp = T.exp(-self.W_mu*delta_inds)
            e_i_lm = e_i_l * mu_exp.T
            alpha = T.exp(e_i_lm)
            alpha /= T.sum(alpha, axis=0)          
            mult = T.mul(contxt_sht, alpha)
            ctx = T.sum(mult, axis=1)
            return ctx.T
        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, contxt_previous, *args):
            contxt = cal_contx(hid_previous)
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
        non_seqs += [contxt_input]
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

