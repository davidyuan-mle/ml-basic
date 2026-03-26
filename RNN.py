import torch
import torch.nn as nn
import torch.nn.functional as F

class MyRNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size=None):
    super(MyRNN, self).__init__()

    if output_size is None:
      output_size = input_size
    
    self.input_size = input_size      # the size of the input vocabulary
    self.hidden_size = hidden_size    # the size of the hidden state
    self.output_size = output_size    # the size of the output vocabulary (if different)

    self.linear1 = nn.Linear(self.input_size + self.hidden_size, self.hidden_size) # input_size + hidden_size --> hidden_size
    self.linear2 = nn.Linear(self.hidden_size, self.output_size) # hidden_size --> output_size

  def forward(self, x, hidden_state):
    output = None
    hidden = None

    x_concat = torch.cat((x, hidden_state), dim=1) # concatenate
    hidden = torch.sigmoid(self.linear1(x_concat)) # new hidden state via linear mapping and sigmoid activation
    score = self.linear2(hidden)  # map to the output size
    output = F.log_softmax(score, dim=1) # get the log softmax

    return output, hidden

  # Make an initial hidden state with some randomness to the values
  def init_hidden(self):
    return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))

