import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLSTMCell(torch.nn.Module):

  def __init__(self, input_size=10, hidden_size=64):
    super(MyLSTMCell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size

    # forget gate
    self.linear_fx = nn.Linear(input_size, hidden_size)
    self.linear_fh = nn.Linear(hidden_size, hidden_size)

    # input gate
    self.linear_ix = nn.Linear(input_size, hidden_size)
    self.linear_ih = nn.Linear(hidden_size, hidden_size)

    # candidate cell memory
    self.linear_gx = nn.Linear(input_size, hidden_size)
    self.linear_gh = nn.Linear(hidden_size, hidden_size)

    # output gate
    self.linear_ox = nn.Linear(input_size, hidden_size)
    self.linear_oh = nn.Linear(hidden_size, hidden_size)

    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()

  ### The Forget Gate takes in the input (x) and hidden state (h)
  ### The input and hidden state pass through their own linear compression layers,
  ### then are concatenated and passed through a sigmoid
  def forget_gate(self, x, h):
    f = self.sigmoid(self.linear_fx(x) + self.linear_fh(h))
    return f

  ### The Input Gate takes the input (x) and hidden state (h)
  ### The input and hidden state pass through their own linear compression layers,
  ### then are concatenated and passed through a sigmoid
  def input_gate(self, x, h):
    i = self.sigmoid(self.linear_ix(x) + self.linear_ih(h))
    return i

  ### The Cell memory gate takes the results from the input gate (i), the results from the forget gate (f)
  ### the original input (x), the hidden state(h) and the previous cell state (c_prev).
  ### 1. The Cell memory gate compresses the input and hidden and concatenates them and passes it through a Tanh.
  ### 2. The resultant intermediate tensor is multiplied by the results from the input gate to determine
  ###    what new information is allowed to carry on
  ### 3. The results from the forget state are multiplied against the previous cell state (c_prev) to determine
  ###    what should be removed from the cell state.
  ### 4. The new cell state (c_next) is the new information that survived the input gate and the previous
  ###    cell state that survived the forget gate.
  ### The new cell state c_next is returned
  def cell_memory(self, i, f, x, h, c_prev):
    g = self.tanh(self.linear_gx(x) + self.linear_gh(h))
    c_next = f * c_prev + i * g
    return c_next

  ### The Out gate takes the original input (x) and the hidden state (h)
  ### The gate passes the input and hidden through their own compression layers and
  ### then concatenates to send through a sigmoid
  def out_gate(self, x, h):
    o = self.sigmoid(self.linear_ox(x) + self.linear_oh(h))
    return o

  ### This function assembles the new hidden state, give the results of the output gate (o)
  ### and the new cells sate (c_next).
  ### This function runs c_next through a tanh to get a 1 or -1 which will flip some of the
  ### elements of the output.
  def hidden_out(self, o, c_next):
    h_next = o * self.tanh(c_next)
    return h_next

  def forward(self, x, hc):
    (h, c_prev) = hc
    # Equation 1. input gate
    i = self.input_gate(x, h)

    # Equation 2. forget gate
    f = self.forget_gate(x, h)

    # Equation 3. updating the cell memory
    c_next = self.cell_memory(i, f, x, h, c_prev)

    # Equation 4. calculate the main output gate
    o = self.out_gate(x, h)

    # Equation 5. produce next hidden output
    h_next = self.hidden_out(o, c_next)

    return h_next, c_next

  def init_hidden(self):
    return (torch.zeros(1, self.hidden_size),
            torch.zeros(1, self.hidden_size))