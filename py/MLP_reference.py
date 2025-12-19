import os

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Config
np.random.seed(0)  # deterministic for testing
HIDDEN = 8
EPOCHS = 30
BATCH = 128
LR = 0.05


def one_hot(label: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Transforms class labels into one-hot vectors.

    Parameters
    ----------
    label : np.ndarray, size (1, batch_size)
        Class labels vector.
    n_classes : int
        Number of classes.

    Returns
    -------
    out : np.ndarray, size (batch_size, n_classes)
        One-hot vector transformation of class labels
    """
    out = np.zeros((label.size, n_classes), dtype=np.float32)
    out[np.arange(label.size), label] = 1.0
    return out


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Softmax formula implementation for array of logits.

    Parameters
    ----------
    logits : np.ndarray
        Array of logits.

    Returns
    -------
    sf : np.ndarray
        Array of distributed probabilities after softmax activation.
    """
    logits = logits - np.max(logits, axis=1, keepdims=True)  # prevent overflow
    e = np.exp(logits)
    sf = e / (np.sum(e, axis=1, keepdims=True) + 1e-12)
    return sf


def ce_loss(pred_probs: np.ndarray, true_classes: np.ndarray) -> float:
    """
    Calculate cross-entropy loss of prediction probabilities.

    Parameters
    ----------
    pred_probs : np.narray, shape (samples, classes)
        Output probability distribution.
    true_classes: np.ndarray, shape (samples, classes)
        Correct one-hot classifications for samples.

    Returns
    -------
    loss : float
        Average cross-entropy loss of samples.
    """
    n_samples = pred_probs.shape[0]
    true_class_indices = np.argmax(true_classes, axis=1)
    pred_confidence = pred_probs[np.arange(n_samples), true_class_indices]
    loss = -np.mean(np.log(pred_confidence) + 1e-12)
    return loss


def ce_grad(pred_probs: np.ndarray, true_classes: np.ndarray) -> np.ndarray:
    """
    Calculate cross-entropy loss gradient of an array of predictions.

    Parameters
    ----------
    pred_probs : np.ndarray, shape (samples, classes)
        Output probability distribution.
    true_classes : np.ndarray, shape (samples, classes)
        Correct one-hot classifications for samples.

    Returns
    -------
    grad : np.ndarray, shape (pred_probs)
        Cross-entropy gradient
    """
    grad = (pred_probs - true_classes) / pred_probs.shape[0]
    return grad


# Layers
class Linear:
    def __init__(self, i_feats: int, o_feats: int):
        """
        Create dense neural network layer and initialize learnable parameters.

        Attributes
        ----------
        i_feats : int
            Number of input features.
        o_feats : int
            Number of output features.
        """
        lim = np.sqrt(6.0 / (i_feats + o_feats))  # Xavier initialization
        self.weights = np.random.uniform(-lim, lim, size=(i_feats, o_feats))
        self.weights = self.weights.astype(np.float32)
        self.biases = np.zeros((1, o_feats), dtype=np.float32)

    def forward(self, x: np.ndarray):
        """
        Set input array and calculate MAC operation (X @ W + b).

        Parameters
        ----------
        x : np.ndarray, shape (batch_size, i_feats)

        Returns
        -------
        mac : np.ndarray, shape (batch_size, o_feats)
        """
        self.x = x
        mac = x @ self.weights + self.biases
        return mac

    def backward(self, gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        """
        Handle parameter learning using loss gradient as feedback.

        Parameters
        ----------
        loss_gradient : np.ndarray, shape (batch, o_feats)
            Cross-entropy loss gradient.
        learn_rate : float
            Learning rate for backpropagation.

        Returns
        -------
        dx : np.ndarray
        """
        dW = self.x.T @ gradient
        db = np.sum(gradient, axis=0, keepdims=True)
        dx = gradient @ self.weights.T
        self.weights -= learn_rate * dW
        self.biases -= learn_rate * db
        return dx


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.positive_mask = (x > 0).astype(np.float32)
        return x * self.positive_mask

    def backward(self, gradient: np.ndarray, _: any) -> np.ndarray:
        return gradient * self.positive_mask


class MLP:
    def __init__(self):
        """Create an MLP network with no layers."""
        self.layers = []

    def add(self, layer_type: Linear | ReLU):
        """Create new layer in network of defined type."""
        self.layers.append(layer_type)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass input into network. Returns network output."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        """Backward pass gradient through the network."""
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learn_rate)

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        learn_rate: float = 0.05,
        verbose: bool = True,
    ) -> list:
        """
        Implement mini-batch gradient descent training

        Passes through the network forwards and backwards {epochs} times.
        Each backward pass trains weights and biases.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, i_feats)
            Input data array.
        Y_train : np.ndarray, shape (n_samples, n_classes)
            True labels arrays.
        epochs : int, optional
            Number of passes over dataset.
        batch_size : int, optional
            Mini-batch size.
        learn_rate : float, optional
            Learning rate.
        verbose : bool, optional
            Print training progress.

        Returns
        -------
        hist : np.ndarray
            History of losses through each epoch to visualize training improvement
        """
        n_samples = X_train.shape[0]
        hist = []
        for e in range(epochs):
            perm = np.random.permutation(n_samples)  # avoid order-dependent patterns
            X_train, Y_train = X_train[perm], Y_train[perm]
            loss_sum = 0.0

            for i in range(0, n_samples, batch_size):
                x_batch = X_train[i: i + batch_size]
                y_batch = Y_train[i: i + batch_size]

                logits = self.forward(x_batch)
                prob_dist = softmax(logits)
                loss = ce_loss(prob_dist, y_batch)
                gradient = ce_grad(prob_dist, y_batch)
                self.backward(gradient, learn_rate)

                loss_sum += loss * x_batch.shape[0]

            loss_epoch = loss_sum / n_samples
            hist.append(loss_epoch)

            if verbose:
                print(f"Epoch {e+1:02d} | loss {loss_epoch:.4f}")

        return hist

    def predict(self, X: np.ndarray):
        return np.argmax(softmax(self.forward(X)), axis=1)


"""
=========================
Loading and Running model
=========================
"""
# Load 8x8 bit images
digits = load_digits()
X = digits.data.astype(np.float32)  # (1797, 64)
y = digits.target.astype(np.int64)  # (1797,)
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
Y_train = one_hot(y_train, 10)
Y_test = one_hot(y_test, 10)

# Build & train small model: 64 -> 8 -> 10
model = MLP()
lin1 = Linear(64, HIDDEN)
relu = ReLU()
lin2 = Linear(HIDDEN, 10)
model.add(lin1)
model.add(relu)
model.add(lin2)

model.train(
    X_train, Y_train, epochs=EPOCHS, batch_size=BATCH, learn_rate=LR, verbose=True
)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nMLP (64->{HIDDEN}->10) Test Accuracy: {acc*100:.2f}%\n")

"""
====================
PRINTING AND LOGGING
====================
"""


def print_c_array(name: str, arr: np.ndarray, ctype: str = "float"):
    """
    Prints values from array

    Takes an array with name and data type details. Iterates through to print
    each value in hex form.

    Parameters
    ----------
    name : str
    arr : np.ndarray
    ctype : str
    """
    flat = arr.reshape(-1)
    print(f"// {name} shape: {list(arr.shape)}")
    print(f"const {ctype} {name}[{flat.size}] = {{")
    line = []
    for idx, value in enumerate(flat):
        val_str = f"{float(value):.7g}"
        if "." not in val_str:
            val_str += ".0"
        if ctype == "float":
            val_str += "f"
        line.append(val_str)
        if (idx + 1) % 8 == 0:
            print("  " + ", ".join(line) + ",")
            line = []
    if line:
        print("  " + ", ".join(line) + ",")
    print("};\n")


"""
========================
Print weights and biases
========================
"""
print("// ================= FLOAT32 PARAMETERS =================")
print_c_array("W1", lin1.weights, "float")
print_c_array("b1", lin1.biases.squeeze(0), "float")
print_c_array("W2", lin2.weights, "float")
print_c_array("b2", lin2.biases.squeeze(0), "float")


def int8_to_hex(x) -> str:
    """Convert signed int8 (-128..127) to 2's complement hex (00..FF)."""
    return f"{int(x) & 0xFF:02X}"


def int32_to_hex(x) -> str:
    """Convert signed int32 to 8-digit 2's complement hex."""
    return f"{int(x) & 0xFFFFFFFF:08X}"


# ================================================
# INT 8 conversion for printing and export to .mem
# ================================================
def quantize_symmetric_int8(arr: np.ndarray) -> np.ndarray:
    """
    Quantize weight tensor into symettric 8-bit integers.

    Takes the maximum value in input matrix and divides by 127 to calculate
    scale factor. Scale factor is used to quantize input matrix and bind to
    -128 - 127 range.

    Parameters
    ----------
    arr : np.ndarray

    Returns
    -------
    Q : np.ndarray
        Quanitized matrix.
    scale_factor : float
        Factor for scaling used in quantization.
    """
    qmax = 127
    scale_factor = float(np.max(np.abs(arr))) / max(qmax, 1e-12)
    scale_factor = 1.0 if scale_factor == 0 else scale_factor
    Q = np.clip(np.round(arr / scale_factor), -128, 127).astype(np.int8)
    return Q, scale_factor


W1_q, scale_W1 = quantize_symmetric_int8(lin1.weights)
b1_q = np.round(lin1.biases / scale_W1).astype(np.int32)
W2_q, scale_W2 = quantize_symmetric_int8(lin2.weights)
b2_q = np.round(lin2.biases / scale_W2).astype(np.int32)

# ==========================
# Path definition for export
# ==========================

folder = "mem"
weight_mem = "W1_q_orig.mem"
bias_mem = "b1_q.mem"
save_w = os.path.join(".", folder, weight_mem)
save_b = os.path.join(".", folder, bias_mem)

print("// ================= INT8 PARAMETERS (per-tensor symmetric) =================")
print("// Scales used during dequant: real_W = int8_W * scale")
print(f"const float SCALE_W1 = {scale_W1:.9g}f;")
print(f"const float SCALE_W2 = {scale_W2:.9g}f;\n")
print_c_array("W1_q", W1_q.astype(np.int8), "int8_t")
np.savetxt(save_w, [int8_to_hex(v) for v in W1_q.reshape(-1)], fmt="%s")
print_c_array("b1_q", b1_q.squeeze(0).astype(np.int32), "int32_t")
np.savetxt(save_b, [int32_to_hex(v) for v in b1_q.reshape(-1)], fmt="%s")
print_c_array("W2_q", W2_q.astype(np.int8), "int8_t")
print_c_array("b2_q", b2_q.squeeze(0).astype(np.int32), "int32_t")


def quantized_inference(
    X_f32: float,
    W1_q: np.ndarray,
    b1_q: np.ndarray,
    scale_W1: float,
    W2_q: np.ndarray,
    b2_q: np.ndarray,
    scale_W2: float,
) -> np.ndarray:
    """
    Perform quantized inference for 2-layer MLP.

    Simulates running on INT8 hardware (e.g. FPGA) while still producing
    a float32 softmax output for each sample.

    Parameters
    ----------
    Xf32 : float
        Input in float32 (normalized data)
    W1_q, W2_q : np.ndarray
        Quantized weights
    b1_q, b2_q : np.ndarray
        Quantized biases
    sW1, sW2 : float
        Per-tensor scales

    Returns
    ----------
    preds : np.ndarray
        Vector of predicted classes for each sample
    """

    X_q, x_scale = quantize_symmetric_int8(X_f32)
    Z1_int32 = X_q.astype(np.int32) @ W1_q.astype(np.int32) + b1_q.astype(np.int32)
    Z1_f32 = Z1_int32 * (x_scale * scale_W1)

    # Apply ReLU
    A1_f32 = np.maximum(Z1_f32, 0.0)
    A1_q, a1_scale = quantize_symmetric_int8(A1_f32)

    # Second layer
    Z2_int32 = A1_q.astype(np.int32) @ W2_q.astype(np.int32) + b2_q.astype(np.int32)
    Z2_f32 = Z2_int32 * (a1_scale * scale_W2)

    # Prediction
    probs = np.exp(Z2_f32 - np.max(Z2_f32, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)

    return preds


# -----------------------
# Run quantized inference
# -----------------------
y_pred_q = quantized_inference(X_test, W1_q, b1_q, scale_W1, W2_q, b2_q, scale_W2)
acc_q = accuracy_score(y_test, y_pred_q)

print(f"Quantized Test Accuracy: {acc_q*100:.2f}%")


"""
=======================================
Printing for comparison setup with CUDA
=======================================
"""
sample = X_test[0:1]

print("\n// ===== Input sample exported for CUDA =====")
print_c_array("X_sample", sample.squeeze(), "float")

ref_hidden = np.maximum(sample @ lin1.weights + lin1.biases, 0)
ref_logits = ref_hidden @ lin2.weights + lin2.biases

print("\nPython Hidden:", ref_hidden)
print("Python Logits:", ref_logits)
print("Python Argmax:", np.argmax(ref_logits))
