import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def nguyen_widrow_init(nn_shape):
    weights, biases = [], []
    for i in range(len(nn_shape) - 1):
        n_in, n_out = nn_shape[i], nn_shape[i + 1]
        w = np.random.uniform(-0.5, 0.5, (n_out, n_in))
        beta = 0.7 * (n_out ** (1.0 / n_in))
        for j in range(n_out):
            norm = np.linalg.norm(w[j])
            if norm != 0:
                w[j] = (beta * w[j]) / norm
        b = np.random.uniform(-beta, beta, (n_out, 1))
        weights.append(w)
        biases.append(b)
    genome = np.concatenate([w.flatten() for w in weights] + [b.flatten() for b in biases])
    return genome

def decode(nn, genome):
    idx = 0
    for i in range(len(nn.layer_sizes) - 1):
        w_size = nn.layer_sizes[i + 1] * nn.layer_sizes[i]
        b_size = nn.layer_sizes[i + 1]
        w = genome[idx : idx + w_size].reshape(nn.layer_sizes[i + 1], nn.layer_sizes[i])
        idx += w_size
        b = genome[idx : idx + b_size].reshape(nn.layer_sizes[i + 1], 1)
        idx += b_size
        nn.weights[i], nn.biases[i] = w, b
