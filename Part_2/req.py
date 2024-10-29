import numpy as np

class LinearLayer:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(out_features, in_features) * 0.01
        self.bias = np.zeros((out_features, 1))
        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)
    
    def forward(self, x):
        self.input = x
        return np.dot(self.weights, x) + self.bias

    def backward(self, grad_output):
        # Calculate gradient with respect to weights and biases
        self.dweights = np.dot(grad_output, self.input.T)
        self.dbias = np.sum(grad_output, axis=1, keepdims=True)
        return np.dot(self.weights.T, grad_output)


class BatchNormalization:
    def __init__(self, size, epsilon=1e-5):
        self.gamma = np.ones((size, 1))
        self.beta = np.zeros((size, 1))
        self.epsilon = epsilon
    
    def forward(self, x):
        self.input = x
        self.mean = np.mean(x, axis=1, keepdims=True)
        self.variance = np.var(x, axis=1, keepdims=True)
        self.x_normalized = (x - self.mean) / np.sqrt(self.variance + self.epsilon)
        return self.gamma * self.x_normalized + self.beta

    def backward(self, grad_output):
        # Calculate gradients for gamma and beta
        self.dgamma = np.sum(grad_output * self.x_normalized, axis=1, keepdims=True)
        self.dbeta = np.sum(grad_output, axis=1, keepdims=True)
        
        # Gradient with respect to input
        N = grad_output.shape[1]  # Batch size
        dx_norm = grad_output * self.gamma
        dvar = np.sum(dx_norm * (self.input - self.mean) * -0.5 * (self.variance + self.epsilon)**(-1.5), axis=1, keepdims=True)
        dmean = np.sum(dx_norm * -1.0 / np.sqrt(self.variance + self.epsilon), axis=1, keepdims=True) + dvar * np.mean(-2.0 * (self.input - self.mean), axis=1, keepdims=True)
        
        return (dx_norm / np.sqrt(self.variance + self.epsilon)) + (dvar * 2 * (self.input - self.mean) / N) + (dmean / N)


class TanhActivation:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)

class CrossEntropyLoss:
    def forward(self, logits, targets):
        self.logits = logits
        self.targets = targets
        exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
        loss = -np.sum(targets * np.log(self.probs + 1e-8)) / logits.shape[1]
        return loss

    def backward(self):
        return (self.probs - self.targets) / self.targets.shape[1]


class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.layer1 = LinearLayer(input_size, hidden_size)
        self.bn1 = BatchNormalization(hidden_size)
        self.activation = TanhActivation()
        self.layer2 = LinearLayer(hidden_size, output_size)
        self.bn2 = BatchNormalization(output_size)
        self.loss_fn = CrossEntropyLoss()
    
    def forward(self, x, y):
        # Forward pass
        out = self.layer1.forward(x)
        out = self.bn1.forward(out)
        out = self.activation.forward(out)
        out = self.layer2.forward(out)
        out = self.bn2.forward(out)
        loss = self.loss_fn.forward(out, y)
        return loss

    def backward(self):
        # Backward pass
        grad = self.loss_fn.backward()
        grad = self.bn2.backward(grad)
        grad = self.layer2.backward(grad)
        grad = self.activation.backward(grad)
        grad = self.bn1.backward(grad)
        grad = self.layer1.backward(grad)
