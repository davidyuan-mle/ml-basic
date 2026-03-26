import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLSTM(nn.ModuleList):
  def __init__(self, input_size, hidden_size, cell_type = nn.LSTMCell):
    super(MyLSTM, self).__init__()

    # init the parameters
    self.hidden_dim = hidden_size
    self.input_size = input_size

    self.lstm1 = cell_type(input_size, hidden_size)
    self.lstm2 = cell_type(hidden_size, hidden_size)
    self.linear = nn.Linear(hidden_size, input_size)
    self.log_softmax = nn.LogSoftmax(dim=1)

  def forward(self, x, hc):
    # Return values
    output = None
    hidden = None
    cell = None

    h0, c0 = hc
    h1, c1 = self.lstm1(x, (h0, c0))
    h2, c2 = self.lstm2(h1, (h0, c0))
    logits = self.linear(h2)
    output = self.log_softmax(logits)

    hidden = h2
    cell = c2

    return output, (hidden.detach(), cell.detach())

  def init_hidden(self):
    # initialize the hidden state and the cell state to zeros
    return (torch.zeros(1, self.hidden_dim), # 1 is the batch size
            torch.zeros(1, self.hidden_dim)) # 1 is the batch size
