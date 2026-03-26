# ML Basic Algorithms

### Logistic Regression
Binary classifier implemented from scratch in NumPy. Uses a sigmoid activation on a linear combination of inputs (`z = Xw + b`) to produce probabilities. Training follows the standard `forward()` → `backward()` → `loss()` loop:
- **Forward**: computes `sigmoid(Xw + b)` to get predicted probabilities
- **Loss**: binary cross-entropy `−mean(y·log(ŷ) + (1−y)·log(1−ŷ))`
- **Backward**: computes gradients `dw = Xᵀ(ŷ−y)/n` and `db = mean(ŷ−y)`, then updates weights via gradient descent

### RNN
Vanilla (Elman) RNN implemented in PyTorch. At each time step, the input and previous hidden state are concatenated and passed through a linear layer with sigmoid activation to produce the new hidden state. A second linear layer maps the hidden state to output logits, followed by log-softmax for next-token prediction.

Architecture diagram from Georgia Tech CS7650:
<img width="684" height="300" alt="image" src="https://github.com/user-attachments/assets/43c12853-05a6-4bd5-9038-4fed26632b93" />

### LSTM
Two-layer LSTM implemented in PyTorch using `nn.LSTMCell`. Addresses the vanishing gradient problem by introducing a cell state alongside the hidden state, controlled by three gates (forget, input, output). The architecture stacks two LSTM cells — the first processes the input and produces a hidden state, which the second LSTM cell takes as input — followed by a linear layer and log-softmax for next-token prediction. Hidden and cell states are detached between steps to enable truncated backpropagation through time (TBPTT).

Architecture diagram from Georgia Tech CS7650:
<img width="704" height="397" alt="image" src="https://github.com/user-attachments/assets/28a3e895-bc33-418e-863b-a662a38f6d94" />

