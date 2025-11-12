import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -----------------------------
# Config
# -----------------------------
np.random.seed(0)  # reproducibility
HIDDEN = 8         # tiny hidden layer: 64 -> 8 -> 10
EPOCHS = 30
BATCH = 128
LR = 0.05

# -----------------------------
# Utils
# -----------------------------
def one_hot(y, C):
    out = np.zeros((y.size, C), dtype=np.float32)
    out[np.arange(y.size), y] = 1.0
    return out

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)

def ce_loss(p, yoh):
    n = p.shape[0]
    return -np.mean(np.log(p[np.arange(n), np.argmax(yoh, axis=1)] + 1e-12))

def ce_grad(p, yoh):
    return (p - yoh) / p.shape[0]

# -----------------------------
# Layers
# -----------------------------
class Linear:
    def __init__(self, in_f, out_f):
        # Xavier/Glorot init
        lim = np.sqrt(6.0 / (in_f + out_f))
        self.W = np.random.uniform(-lim, lim, size=(in_f, out_f)).astype(np.float32)
        self.b = np.zeros((1, out_f), dtype=np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, go, lr):
        dW = self.x.T @ go
        db = np.sum(go, axis=0, keepdims=True)
        dx = go @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dx

class ReLU:
    def forward(self, x):
        self.m = (x > 0).astype(np.float32)
        return x * self.m
    def backward(self, go, lr):
        return go * self.m

# -----------------------------
# Model
# -----------------------------
class MLP:
    def __init__(self):
        self.layers = []
    def add(self, l): self.layers.append(l)
    def forward(self, x):
        for l in self.layers: x = l.forward(x)
        return x
    def backward(self, g, lr):
        for l in reversed(self.layers): g = l.backward(g, lr)
    def train(self, X, Y, epochs=20, bs=64, lr=0.05, verbose=True):
        n = X.shape[0]; hist=[]
        for e in range(epochs):
            perm = np.random.permutation(n)
            X, Y = X[perm], Y[perm]
            loss_sum = 0.0
            for s in range(0, n, bs):
                xb = X[s:s+bs]; yb = Y[s:s+bs]
                logits = self.forward(xb)
                p = softmax(logits)
                loss = ce_loss(p, yb)
                g = ce_grad(p, yb)
                self.backward(g, lr)
                loss_sum += loss * xb.shape[0]
            loss_epoch = loss_sum / n
            hist.append(loss_epoch)
            if verbose: print(f"Epoch {e+1:02d} | loss {loss_epoch:.4f}")
        return hist
    def predict(self, X):
        return np.argmax(softmax(self.forward(X)), axis=1)

# -----------------------------
# Data: sklearn Digits (1797 samples, 64 features)
# -----------------------------
digits = load_digits()
X = digits.data.astype(np.float32)     # (1797, 64)
y = digits.target.astype(np.int64)     # (1797,)
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
Y_train = one_hot(y_train, 10)
Y_test  = one_hot(y_test, 10)

# -----------------------------
# Build & train small model: 64 -> 8 -> 10
# -----------------------------
model = MLP()
lin1 = Linear(64, HIDDEN)
relu = ReLU()
lin2 = Linear(HIDDEN, 10)
model.add(lin1); model.add(relu); model.add(lin2)

model.train(X_train, Y_train, epochs=EPOCHS, bs=BATCH, lr=LR, verbose=True)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTiny MLP (64->{HIDDEN}->10) Test Accuracy: {acc*100:.2f}%\n")

# -----------------------------
# Printing helpers
# -----------------------------
def print_c_array(name, arr, ctype="float"):
    flat = arr.reshape(-1)
    print(f"// {name} shape: {list(arr.shape)}")
    print(f"const {ctype} {name}[{flat.size}] = {{")
    # pretty formatting: 8 numbers per line
    line = []
    for i, v in enumerate(flat):
        if ctype in ("float", "double"):
            s = f"{float(v):.7g}"
            if ctype == "float":
                s += "f"
        else:  # integer types
            s = str(int(v))
        line.append(s)
        if (i+1) % 8 == 0:
            print("  " + ", ".join(line) + ",")
            line = []
    if line:
        print("  " + ", ".join(line) + ",")
    print("};\n")

# -----------------------------
# 1) Print FLOAT32 weights/biases (direct use if FPGA supports float)
# -----------------------------
print("// ================= FLOAT32 PARAMETERS =================")
print_c_array("W1", lin1.W, "float")
print_c_array("b1", lin1.b.squeeze(0), "float")
print_c_array("W2", lin2.W, "float")
print_c_array("b2", lin2.b.squeeze(0), "float")

# -----------------------------
# 2) Int8 per-tensor symmetric quantization + scales
#    (int8 weights, int32 biases in weight scale domain)
# -----------------------------
def quantize_weight_symmetric_int8(W):
    qmax = 127
    s = float(np.max(np.abs(W))) / max(qmax, 1e-12)
    s = 1.0 if s == 0 else s
    Q = np.clip(np.round(W / s), -128, 127).astype(np.int8)
    return Q, s

W1_q, sW1 = quantize_weight_symmetric_int8(lin1.W)
b1_q = np.round(lin1.b / sW1).astype(np.int32)
W2_q, sW2 = quantize_weight_symmetric_int8(lin2.W)
b2_q = np.round(lin2.b / sW2).astype(np.int32)

print("// ================= INT8 PARAMETERS (per-tensor symmetric) =================")
print(f"// Scales used during dequant: real_W = int8_W * scale")
print(f"const float SCALE_W1 = {sW1:.9g}f;")
print(f"const float SCALE_W2 = {sW2:.9g}f;\n")
print_c_array("W1_q", W1_q.astype(np.int8), "int8_t")
print_c_array("b1_q", b1_q.squeeze(0).astype(np.int32), "int32_t")
print_c_array("W2_q", W2_q.astype(np.int8), "int8_t")
print_c_array("b2_q", b2_q.squeeze(0).astype(np.int32), "int32_t")

def quantized_inference(Xf32, W1_q, b1_q, sW1, W2_q, b2_q, sW2):
    """
    Xf32 : input in float32 (normalized data)
    W*_q, b*_q : quantized weights/biases
    sW1, sW2 : per-tensor scales
    Returns predicted labels
    """

    # 1️⃣ Quantize input to int8 (simulating FPGA input)
    # We can use a simple symmetric scheme for demo
    x_scale = np.max(np.abs(Xf32)) / 127
    X_q = np.clip(np.round(Xf32 / x_scale), -128, 127).astype(np.int8)

    # 2️⃣ First layer: int8 * int8 → int32 accum
    Z1_int32 = X_q.astype(np.int32) @ W1_q.astype(np.int32) + b1_q.astype(np.int32)

    # Convert back to float (dequantize)
    Z1_f32 = Z1_int32 * (x_scale * sW1)

    # 3️⃣ Apply ReLU
    A1_f32 = np.maximum(Z1_f32, 0.0)

    a1_scale = np.max(np.abs(A1_f32)) / 127
    A1_q = np.clip(np.round(A1_f32 / a1_scale), -128, 127).astype(np.int8)

    # 4️⃣ Second layer: int8 * int8 → int32 + bias
    Z2_int32 = A1_q.astype(np.int32) @ W2_q.astype(np.int32) + b2_q.astype(np.int32)

    # Dequantize back to float
    Z2_f32 = Z2_int32 * (a1_scale * sW2)

    # 5️⃣ Apply softmax and pick max class
    probs = np.exp(Z2_f32 - np.max(Z2_f32, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    return preds


# ---- Run quantized inference ----
y_pred_q = quantized_inference(X_test, W1_q, b1_q, sW1, W2_q, b2_q, sW2)
acc_q = accuracy_score(y_test, y_pred_q)

print(f"Quantized Test Accuracy: {acc_q*100:.2f}%")

