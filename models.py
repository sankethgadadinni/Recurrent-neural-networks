from turtle import forward
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from cells import RNNcell, GRUcell, LSTMcell




class RNNnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bias, activation="tanh"):
        super(RNNnet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bias = bias
        self.activation = activation

        self.rnn_cell_list = nn.ModuleList()

        if self.activation == 'tanh':
            self.rnn_cell_list.append(RNNcell(self.input_dim, self.hidden_dim, self.bias, "tanh"))

            for l in range(num_layers):
                self.rnn_cell_list.append(RNNcell(self.hidden_dim, self.hidden_dim, self.bias, "tanh"))

        elif self.activation == 'relu':
            self.rnn_cell_list.append(RNNcell(self.input_dim, self.hidden_dim, self.bias, "relu"))

            for l in range(num_layers):
                self.rnn_cell_list.append(RNNcell(self.hidden_dim, self.hidden_dim, self.bias, "relu"))

        
        else:
            print("Invalid activation function")

        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    

    def forward(self, input, hidden=None):

        # Input of shape (batch_size, seqence length , input_size)
        # Output of shape (batch_size, output_size)

        if hidden is None:
            hid = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_dim))

        else:
            hid = hidden


        outs = []
        hx = []

        for layer in range(self.num_layers):
            hx.append(hid[layer, :, :])

        
        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_1 = self.rnn_cell_list[layer](input[:,t,:], hx[layer])
                
                else:
                    hidden_1 = self.rnn_cell_list[layer](hx[layer - 1], hx[layer])

                hx[layer] = hidden_1

            outs.append(hidden_1)

        out = outs[-1].squeeze()
        out = self.fc(out)

        return out


        


class GRUnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bias):
        super(GRUnet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.bias = bias


        self.gru_cell_list = nn.ModuleList()

        self.gru_cell_list.append(GRUcell(self.input_dim, self.hidden_dim, self.bias))

        for l in range(self.num_layers):
            self.gru_cell_list.append(GRUcell(self.hidden_dim, self.hidden_dim, self.bias))

        


        self.fc = nn.Linear(self.hidden_dim, self.output_dim)


    
    def forward(self, input, hidden = None):

        # Input of shape (batch_size, seqence length , input_size)
        # Output of shape (batch_size, output_size)

        if hidden is None:
            hid = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_dim))

        else:
            hid = hidden

        

        outs = []
        hx = []

        for layer in range(self.num_layers):
            hx.append(hid[layer, :, :])

        
        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_1 = self.gru_cell_list[layer](input[:, t, :], hx[layer])
                
                else:
                    hidden_1 = self.gru_cell_list[layer](hx[layer-1], hx[layer])

                hx[layer] = hidden_1

            outs.append(hidden_1)

        out = outs[-1].squeeze()

        out = self.fc(out)
        return out





class LSTMnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bias):
        super(LSTMnet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.bias = bias


        self.lstm_cell_list = nn.ModuleList()

        self.lstm_cell_list.append(GRUcell(self.input_dim, self.hidden_dim, self.bias))

        for l in range(self.num_layers):
            self.lstm_cell_list.append(GRUcell(self.hidden_dim, self.hidden_dim, self.bias))

        
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)


    
    def forward(self, input, hidden = None):

        # Input of shape (batch_size, seqence length , input_size)
        # Output of shape (batch_size, output_size)

        if hidden is None:
            hid = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_dim))

        else:
            hid = hidden

        

        outs = []
        hx = []

        for layer in range(self.num_layers):
            hx.append(hid[layer, :, :])

        
        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_1 = self.lstm_cell_list[layer](input[:, t, :], hx[layer])
                
                else:
                    hidden_1 = self.lstm_cell_list[layer](hx[layer-1], hx[layer])

                hx[layer] = hidden_1

            outs.append(hidden_1)

        out = outs[-1].squeeze()

        out = self.fc(out)
        return out