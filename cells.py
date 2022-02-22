import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

## RNN cell implementation

class RNNcell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias = True, non_linearity = 'tanh'):
        super(RNNcell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.non_linearity = non_linearity

        if self.non_linearity not in ['tanh','relu']:
            print("Invlaid non linearity")
        
        self.x2h = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.h2h = nn.Linear(hidden_dim, hidden_dim, bias=bias)


    
    def forward(self, input, hidden = None):

        if hidden is None:
            hidden = Variable(input.new_zeros(input.size(0), self.hidden_dim))

        out = (self.x2h(input) + self.h2h(hidden))

        if self.non_linearity == 'tanh':
            out = torch.tanh(out)

        else:
            out = torch.relu(out)

        return out



## GRU cell implemetation


class GRUcell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias = True):
        super(GRUcell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias

    
        self.x2h = nn.Linear(input_dim, 3*hidden_dim, bias=bias)
        self.h2h = nn.Linear(hidden_dim, 3*hidden_dim, bias=bias)

    
    def forward(self, input, hidden=None):

        if hidden:
            hidden = Variable(input.new_zeroes(input.size(0), self.hidden_dim))

        out = self.x2h(input)
        hidden_out = self.h2h(hidden)



        x_reset, x_update, x_new = out.chunk(3,1)
        h_reset, h_update, h_new = hidden_out.chunk(3,1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_update + h_update)

        new_gate = torch.tanh(x_new + (reset_gate + h_new))

        h_out = update_gate * hidden_out + (1 - update_gate) * new_gate

        return h_out


## LSTM cell implementation

class LSTMcell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super(LSTMcell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias

    
        self.x2h = nn.Linear(input_dim, hidden_dim * 4, bias=bias)
        self.h2h = nn.Linear(hidden_dim, hidden_dim *4, bias=bias)

    
    def forward(self, input, hidden = None):

        if hidden is None:
            hidden = Variable(input.new_zeroes(input.size(0), self.hidden_dim))

        hidden = (hidden, hidden)

        hidden, cx = hidden

        gates = self.x2h(input) + self.h2h(hidden)

        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        cy = cx * f_t + i_t * g_t

        hy = o_t * torch.tanh(cy)


        return (hy, cy)