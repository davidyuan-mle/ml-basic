import numpy as np 

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, x):
        p = x.shape[1]
        logits = x @ self.weights + self.bias
        yhat = self.sigmoid(logits) 
        return yhat 
    
    def backward(self, x, y, yhat):
        z = yhat - y  # nx1
        dw = x.T @ z / len(y) # p x 1, p is the number of features
        db = np.sum(z) / len(y)
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def loss(self, y, yhat):
        return -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
    
    def train(self, x, y):
        p = x.shape[1]
        self.weights = np.random.randn(p, 1) * 0.01
        self.bias = np.random.randn(1) * 0.01

        for epoch in range(self.epochs):
            yhat = self.forward(x)
            self.backward(x, y, yhat)
            loss = self.loss(y, yhat)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
    
    def predict_proba(self, x):
        return self.forward(x)

    def predict(self, x):
        yhat = self.forward(x)
        return (yhat >= 0.5).astype(int)


# simulate data with 2 features and 100 samples 
x = np.random.randn(5000, 2)
weights = np.array([[2], [1]])
bias = 0.5
y = (1 / (1 + np.exp(-(x @ weights + bias))) > 0.5).astype(int)

# train the model 
model = LogisticRegression(learning_rate=2e-2, epochs=1000)
model.train(x, y)

# predict the labels 
print(model.weights, model.bias)
