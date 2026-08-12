# ML Basic Algorithms

### Logistic Regression
Binary classifier implemented from scratch in NumPy. Uses a sigmoid activation on a linear combination of inputs (`z = Xw + b`) to produce probabilities. Training follows the standard `forward()` → `backward()` → `loss()` loop:
- **Forward**: computes `sigmoid(Xw + b)` to get predicted probabilities
- **Loss**: binary cross-entropy `−mean(y·log(ŷ) + (1−y)·log(1−ŷ))`
- **Backward**: computes gradients `dw = Xᵀ(ŷ−y)/n` and `db = mean(ŷ−y)`, then updates weights via gradient descent

### RNN
Vanilla RNN implemented in PyTorch. At each time step, the input and previous hidden state are concatenated and passed through a linear layer with sigmoid activation to produce the new hidden state. A second linear layer maps the hidden state to output logits, followed by log-softmax for next-token prediction.

Architecture diagram from Georgia Tech CS7650:
<img width="684" height="300" alt="image" src="https://github.com/user-attachments/assets/43c12853-05a6-4bd5-9038-4fed26632b93" />

### LSTM
LSTM cell implemented from scratch in PyTorch. Each gate is built with explicit linear layers rather than using `nn.LSTMCell`, making the internal mechanics fully transparent:
- **Forget gate**: `f = sigmoid(W_fx · x + W_fh · h)` — decides what to discard from the cell state
- **Input gate**: `i = sigmoid(W_ix · x + W_ih · h)` — decides what new information to store
- **Cell memory**: `c_next = f * c_prev + i * tanh(W_gx · x + W_gh · h)` — updates the cell state
- **Output gate**: `o = sigmoid(W_ox · x + W_oh · h)` — determines what to expose as hidden state
- **Hidden output**: `h_next = o * tanh(c_next)`

Architecture diagram from Georgia Tech CS7650:
<img width="704" height="397" alt="image" src="https://github.com/user-attachments/assets/28a3e895-bc33-418e-863b-a662a38f6d94" />

### Multi-Head Attention
Multi-head attention implemented from scratch in PyTorch. Projects input into multiple parallel attention heads, each operating on a lower-dimensional subspace, then concatenates and projects back:
- **Linear projections**: separate `W_q`, `W_k`, `W_v` project input `x` into queries, keys, and values, then reshape into `(B, n_heads, T, d_k)` where `d_k = d_model / n_heads`
- **Scaled dot-product attention**: `scores = QKᵀ / √d_k`, with optional mask applied before softmax to produce attention weights
- **Output projection**: weighted values from all heads are concatenated and passed through a final linear layer `W_o`

### ROC / AUC
ROC curve and AUC implemented from scratch in pure Python, validated against `sklearn.metrics.roc_auc_score`. A classifier outputs a score, not a label — every threshold gives a different `(FPR, TPR)` trade-off, and the ROC curve is the set of all of them:
- **Threshold sweep**: sorting by score descending turns the naive O(n²) rescan into a single pass — each step admits one more point into the predicted-positive set, so `TP` and `FP` only ever increment (O(n log n), dominated by the sort)
- **Rates**: `TPR = TP / P` (of all real positives, how many were caught) and `FPR = FP / N` (of all real negatives, how many were falsely flagged)
- **Tie handling**: points with equal scores can't be separated by any threshold, so a point is only emitted when the score changes, plus one final point for the all-positive prediction
- **AUC**: trapezoid integration over the curve, each step contributing a rectangle plus a triangle — equal to `P(random positive scores above random negative)`, where 0.5 is a coin flip and 1.0 is perfect

