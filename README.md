import numpy as np
from scipy.signal import convolve2d
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.reshape(-1, 28, 28).astype(np.float32)
    y = mnist.target.astype(np.int32)
    return X, y


def preprocess(X, y):
    # 归一化
    X = X / 255.0
    # 添加通道维度
    X = np.expand_dims(X, axis=1)
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Xavier初始化
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros(out_channels)

        self.cache = None

    def forward(self, x):
        batch_size, in_channels, in_h, in_w = x.shape
        out_h = (in_h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (in_w - self.kernel_size + 2 * self.padding) // self.stride + 1

        # 添加padding
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding),
                                  (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x

        # 初始化输出
        output = np.zeros((batch_size, self.out_channels, out_h, out_w))

        # 执行卷积
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for c_in in range(self.in_channels):
                    output[b, c_out] += convolve2d(x_padded[b, c_in], self.weights[c_out, c_in], mode='valid')[
                                        ::self.stride, ::self.stride]
                output[b, c_out] += self.bias[c_out]

        self.cache = x
        return output

    def backward(self, dout):
        x = self.cache
        batch_size, in_channels, in_h, in_w = x.shape
        _, _, out_h, out_w = dout.shape

        # 初始化梯度
        dw = np.zeros_like(self.weights)
        db = np.zeros_like(self.bias)
        dx = np.zeros_like(x)

        # 添加padding
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding),
                                  (self.padding, self.padding)), mode='constant')
            dx_padded = np.pad(dx, ((0, 0), (0, 0), (self.padding, self.padding),
                                    (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x
            dx_padded = dx

        # 计算梯度
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for c_in in range(self.in_channels):
                    # 计算dw
                    for i in range(self.kernel_size):
                        for j in range(self.kernel_size):
                            x_slice = x_padded[b, c_in, i:i + out_h * self.stride:self.stride,
                                      j:j + out_w * self.stride:self.stride]
                            dw[c_out, c_in, i, j] += np.sum(x_slice * dout[b, c_out])

                    # 计算dx
                    grad_padded = np.zeros((in_h + 2 * self.padding, in_w + 2 * self.padding))
                    for h in range(out_h):
                        for w in range(out_w):
                            h_start = h * self.stride
                            w_start = w * self.stride
                            grad_padded[h_start:h_start + self.kernel_size,
                            w_start:w_start + self.kernel_size] += \
                                self.weights[c_out, c_in] * dout[b, c_out, h, w]

                    dx_padded[b, c_in] += grad_padded

                # 计算db
                db[c_out] += np.sum(dout[b, c_out])

        # 去除padding
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded

        return dx, dw, db


class AvgPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None

    def forward(self, x):
        batch_size, channels, in_h, in_w = x.shape
        out_h = (in_h - self.pool_size) // self.stride + 1
        out_w = (in_w - self.pool_size) // self.stride + 1

        output = np.zeros((batch_size, channels, out_h, out_w))
        self.cache = x.shape

        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.pool_size
                        w_end = w_start + self.pool_size

                        window = x[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, h, w] = np.mean(window)

        return output

    def backward(self, dout):
        batch_size, channels, out_h, out_w = dout.shape
        in_h, in_w = self.cache[2], self.cache[3]
        dx = np.zeros(self.cache)

        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.pool_size
                        w_end = w_start + self.pool_size

                        # 平均池化的梯度是均匀分布的
                        grad = dout[b, c, h, w] / (self.pool_size * self.pool_size)
                        dx[b, c, h_start:h_end, w_start:w_end] += grad

        return dx


class ReLU:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dout):
        x = self.cache
        dx = dout * (x > 0)
        return dx


class Flatten:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.cache)


class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # Xavier初始化
        scale = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(in_features, out_features) * scale
        self.bias = np.zeros(out_features)

        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, dout):
        x = self.cache
        dx = np.dot(dout, self.weights.T)
        dw = np.dot(x.T, dout)
        db = np.sum(dout, axis=0)
        return dx, dw, db


class LeNet5:
    def __init__(self):
        self.conv1 = Conv2D(1, 6, 5, padding=0)
        self.relu1 = ReLU()
        self.pool1 = AvgPool2D(2, 2)

        self.conv2 = Conv2D(6, 16, 5, padding=0)
        self.relu2 = ReLU()
        self.pool2 = AvgPool2D(2, 2)

        self.flatten = Flatten()
        self.fc1 = Linear(16 * 4 * 4, 120)
        self.relu3 = ReLU()
        self.fc2 = Linear(120, 84)
        self.relu4 = ReLU()
        self.fc3 = Linear(84, 10)

        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.flatten,
            self.fc1, self.relu3,
            self.fc2, self.relu4,
            self.fc3
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            if isinstance(layer, ReLU) or isinstance(layer, Flatten) or isinstance(layer, AvgPool2D):
                dout = layer.backward(dout)
            else:
                dout, dw, db = layer.backward(dout)
                layer.dw = dw
                layer.db = db
        return dout


# 4. 损失函数与评估
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss


def cross_entropy_grad(y_pred, y_true):
    m = y_true.shape[0]
    grad = y_pred.copy()
    grad[range(m), y_true] -= 1
    grad /= m
    return grad


def accuracy(y_pred, y_true):
    preds = np.argmax(y_pred, axis=1)
    return np.mean(preds == y_true)


# 5. 训练过程
class SGD:
    def __init__(self, model, lr=0.01, momentum=0.0):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

        # 初始化速度
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'weights'):
                self.velocities[f'dw_{i}'] = np.zeros_like(layer.weights)
                self.velocities[f'db_{i}'] = np.zeros_like(layer.bias)

    def step(self):
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'weights'):
                # 更新速度
                self.velocities[f'dw_{i}'] = self.momentum * self.velocities[f'dw_{i}'] + layer.dw
                self.velocities[f'db_{i}'] = self.momentum * self.velocities[f'db_{i}'] + layer.db

                # 更新参数
                layer.weights -= self.lr * self.velocities[f'dw_{i}']
                layer.bias -= self.lr * self.velocities[f'db_{i}']


def train(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=128, lr=0.01, momentum=0.9):
    n_train = X_train.shape[0]
    n_batches = n_train // batch_size

    optimizer = SGD(model, lr=lr, momentum=momentum)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(epochs):
        # 打乱数据
        indices = np.random.permutation(n_train)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        epoch_loss = 0.0
        correct = 0

        for batch in range(n_batches):
            # 获取当前batch
            start = batch * batch_size
            end = start + batch_size
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            # 前向传播
            logits = model.forward(X_batch)
            probs = softmax(logits)
            loss = cross_entropy_loss(probs, y_batch)

            # 计算准确率
            correct += np.sum(np.argmax(probs, axis=1) == y_batch)
            epoch_loss += loss

            # 反向传播
            grad = cross_entropy_grad(probs, y_batch)
            model.backward(grad)

            # 参数更新
            optimizer.step()

        # 计算训练集指标
        train_loss = epoch_loss / n_batches
        train_acc = correct / n_train

        # 验证集评估
        val_loss, val_acc = evaluate(model, X_val, y_val, batch_size)

        # 记录历史
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

    return train_loss_history, train_acc_history, val_loss_history, val_acc_history


def evaluate(model, X, y, batch_size=128):
    n_samples = X.shape[0]
    n_batches = n_samples // batch_size

    total_loss = 0.0
    correct = 0

    for batch in range(n_batches):
        start = batch * batch_size
        end = start + batch_size
        X_batch = X[start:end]
        y_batch = y[start:end]

        logits = model.forward(X_batch)
        probs = softmax(logits)
        loss = cross_entropy_loss(probs, y_batch)

        correct += np.sum(np.argmax(probs, axis=1) == y_batch)
        total_loss += loss

    # 处理剩余样本
    if n_samples % batch_size != 0:
        start = n_batches * batch_size
        X_batch = X[start:]
        y_batch = y[start:]

        logits = model.forward(X_batch)
        probs = softmax(logits)
        loss = cross_entropy_loss(probs, y_batch)

        correct += np.sum(np.argmax(probs, axis=1) == y_batch)
        total_loss += loss

    avg_loss = total_loss / (n_batches + (1 if n_samples % batch_size != 0 else 0))
    acc = correct / n_samples

    return avg_loss, acc


def plot_history(train_loss, train_acc, val_loss, val_acc):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.tight_layout()
    plt.show()


def predict(model, X):
    return model.forward(X)


def save_params(model, filename):
    params = {}
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'weights'):
            params[f'weights_{i}'] = layer.weights
        if hasattr(layer, 'bias'):
            params[f'bias_{i}'] = layer.bias
    with open(filename, 'wb') as f:
        pickle.dump(params, f)


def load_params(model, filename):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'weights'):
            layer.weights = params[f'weights_{i}']
        if hasattr(layer, 'bias'):
            layer.bias = params[f'bias_{i}']


# 6. 主程序
def main():
    # 加载并预处理数据
    X, y = load_mnist()
    X_train, X_test, y_train, y_test = preprocess(X, y)

    # 初始化LeNet-5模型
    model = LeNet5()

    # 训练模型
    print("Training LeNet-5...")
    train_loss, train_acc, val_loss, val_acc = train(
        model, X_train, y_train, X_test, y_test,
        epochs=2, lr=0.01, momentum=0.9)

    # 绘制训练曲线
    plot_history(train_loss, train_acc, val_loss, val_acc)

    # 最终评估
    test_loss, test_acc = evaluate(model, X_test, y_test)
    print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
