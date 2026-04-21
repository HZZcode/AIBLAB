import numpy as np
import sys

exp_type = sys.argv[1] # 'linear' or 'nonlinear'
if exp_type not in ['linear', 'nonlinear']:
    raise Exception("`exp_type` parameter should be 'linear' or 'nonlinear'.")

np.random.seed(0)

def generate_linear_data(n_samples_per_class=100, n_features=10):
    """
    生成线性可分的高维多类数据
    参数：
        n_samples_per_class : 每个类别的样本数
        n_features : 总特征维度
    返回：
        X : 数据矩阵 (n_samples, n_features)
        y : 标签 (n_samples,)
    """
    X = []
    y = []
    for class_idx in range(4):
        mean = np.zeros(n_features)
        mean[class_idx] = 5
        data = np.random.normal(loc=mean, scale=1, 
                               size=(n_samples_per_class, n_features))
        labels = np.full(n_samples_per_class, class_idx, dtype=int)
        X.append(data)
        y.append(labels)

    X = np.vstack(X)
    y = np.concatenate(y)
    shuffle_idx = np.random.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]

def generate_nonlinear_data(n_samples_per_class=100, n_features=10):
    """
    生成非线性可分的高维多类数据
    参数：
        n_samples_per_class : 每个类别的样本数
        n_features : 总特征维度(必须 >= 4)
    返回：
        X : 数据矩阵 (n_samples, n_features)
        y : 标签 (n_samples,)
    """
    noise = 0.3
    np.random.seed(42)
    assert n_features >= 4, "特征维度需至少为4"

    X = []
    y = []
    
    # 生成四类数据的核心非线性结构
    for class_idx in range(4):
        # 生成基础结构（前4个维度定义非线性关系）
        base_dims = np.zeros((n_samples_per_class, 4))
        
        # 不同类别的非线性模式
        if class_idx == 0:
            # 环形数据（内环）
            theta = np.linspace(0, 2*np.pi, n_samples_per_class)
            r = 2 + noise * np.random.randn(n_samples_per_class)
            base_dims[:, 0] = r * np.cos(theta)
            base_dims[:, 1] = r * np.sin(theta)
        elif class_idx == 1:
            # 环形数据（外环）
            theta = np.linspace(0, 2*np.pi, n_samples_per_class)
            r = 4 + noise * np.random.randn(n_samples_per_class)
            base_dims[:, 0] = r * np.cos(theta)
            base_dims[:, 1] = r * np.sin(theta)
        elif class_idx == 2:
            # 月牙形数据（类似sklearn的make_moons）
            theta = np.linspace(0, np.pi, n_samples_per_class)
            r = 3 + noise * np.random.randn(n_samples_per_class)
            base_dims[:, 2] = r * np.cos(theta)
            base_dims[:, 3] = r * np.sin(theta) + 2.5
        else:
            # 螺旋形数据
            theta = np.linspace(0, 4*np.pi, n_samples_per_class)
            r = np.linspace(0.5, 3, n_samples_per_class)
            base_dims[:, 2] = r * np.cos(theta)
            base_dims[:, 3] = r * np.sin(theta)
        
        # 添加噪声到其他维度
        noise_dims = noise * np.random.randn(n_samples_per_class, n_features-4)
        full_data = np.hstack([base_dims, noise_dims])
        
        X.append(full_data)
        y.extend([class_idx] * n_samples_per_class)
    
    X = np.vstack(X)
    y = np.array(y)
    return X, y

def relu(x):
    '''
    ReLU 函数
    '''
    return np.maximum(0, x)

def relu_derivative(x):
    '''
    ReLU 函数的导数
    '''
    return (x > 0).astype(float)

def softmax(x):
    '''
    Softmax 函数
    '''
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


class ReluNeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        # Xavier初始化
        stddev = np.sqrt(2 / (input_size + hidden1_size))
        self.weights_1 = np.random.normal(0,stddev, (input_size, hidden1_size))
        self.bias_1 = np.zeros((1, hidden1_size))
        

        stddev = np.sqrt(2 / (hidden1_size + hidden2_size))
        self.weights_2 = np.random.normal(0,stddev, (hidden1_size, hidden2_size))
        self.bias_2 = np.zeros((1, hidden2_size))

        stddev = np.sqrt(2 / (hidden2_size + output_size))
        self.weights_3 = np.random.normal(0,stddev, (hidden2_size, output_size))
        self.bias_3 = np.zeros((1, output_size))

    def forward(self, X):

        '''
        前向传播:第一隐藏层激活 a_1 的计算 [代码已经给出]
        '''
        self.z_1 = X @ self.weights_1 + self.bias_1
        self.a_1 = relu(self.z_1)
        
        '''
        前向传播:第二隐藏层激活 a_2 的计算 [需要补全 None 代表部分的代码] 
        '''  
        self.z_2 = self.a_1 @ self.weights_2 + self.bias_2
        self.a_2 = relu(self.z_2)
        
        '''
        前向传播:输出层 y_pred 的计算 [需要补全 None 代表部分的代码] 
        '''  
        self.z_3 = self.a_2 @ self.weights_3 + self.bias_3
        self.y_pred = softmax(self.z_3)

        return self.y_pred
    

    def backward(self, X, y, batch_size, y_pred, learning_rate):
        '''
        反向传播: self.weights_3, self.bias_3 参数梯度的计算和更新
        '''
        # 计算误差项 e_3
        e_3 = y_pred - y
        # self.weights_3, self.bias_3 参数梯度的计算
        grad_weights_3 = self.a_2.T @ e_3 / batch_size
        bias_weights_3 = np.sum(e_3, axis=0, keepdims=True) / batch_size


        '''
        反向传播: self.weights_1, self.bias_1 参数梯度的计算和更新
         [需要补全 None 代表部分的代码] 
        '''
        e_2 = (e_3 @ self.weights_3.T) * relu_derivative(self.z_2)
        grad_weights_2 = self.a_1.T @ e_2 / batch_size
        grad_bias_2 = np.sum(e_2, axis=0, keepdims=True) / batch_size
        
        '''
        反向传播: self.weights_2, self.bias_2 参数梯度的计算和更新
         [需要补全 None 代表部分的代码] 
        '''
        e_1 = (e_2 @ self.weights_2.T) * relu_derivative(self.z_1)
        grad_weights_1 = X.T @ e_1 / batch_size
        grad_bias_1 = np.sum(e_1, axis=0, keepdims=True) / batch_size

        # 梯度下降，更新网络参数
        self.weights_3 -= grad_weights_3 * learning_rate
        self.bias_3 -= bias_weights_3 * learning_rate
        self.weights_2 -= grad_weights_2 * learning_rate
        self.bias_2 -= grad_bias_2 * learning_rate
        self.weights_1 -= grad_weights_1 * learning_rate
        self.bias_1 -= grad_bias_1 * learning_rate


    def train(self, X, y, epochs, learning_rate):
        
        for epoch in range(epochs):
            batch_size = X.shape[0]

            # 前向传播过程
            y_pred = self.forward(X)

            # 反向传播过程
            self.backward(X, y, batch_size, y_pred, learning_rate)

            # 训练过程中每隔10个epoch，打印损失函数值
            if epoch % 10 == 0: 
                loss = -np.mean(y * np.log(y_pred + 1e-8))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")


class LinearNeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        # Xavier初始化
        stddev = np.sqrt(2 / (input_size + hidden1_size))
        self.weights_1 = np.random.normal(0,stddev, (input_size, hidden1_size))
        self.bias_1 = np.zeros((1, hidden1_size))
        

        stddev = np.sqrt(2 / (hidden1_size + hidden2_size))
        self.weights_2 = np.random.normal(0,stddev, (hidden1_size, hidden2_size))
        self.bias_2 = np.zeros((1, hidden2_size))

        stddev = np.sqrt(2 / (hidden2_size + output_size))
        self.weights_3 = np.random.normal(0,stddev, (hidden2_size, output_size))
        self.bias_3 = np.zeros((1, output_size))

    def forward(self, X):
        self.z_1 = X @ self.weights_1 + self.bias_1
        self.a_1 = self.z_1 
        self.z_2 = self.a_1 @ self.weights_2 + self.bias_2
        self.a_2 = self.z_2
        self.z_3 = self.a_2 @ self.weights_3 + self.bias_3
        self.y_pred = softmax(self.z_3)
        return self.y_pred
    

    def backward(self, X, y, batch_size, y_pred, learning_rate):
        e_3 = y_pred - y
        grad_weights_3 = self.a_2.T @ e_3 / batch_size
        grad_bias_3 = np.sum(e_3, axis=0, keepdims=True) / batch_size

        e_2 = e_3 @ self.weights_3.T
        grad_weights_2 = self.a_1.T @ e_2 / batch_size
        grad_bias_2 = np.sum(e_2, axis=0, keepdims=True) / batch_size

        e_1 = e_2 @ self.weights_2.T
        grad_weights_1 = X.T @ e_1 / batch_size
        grad_bias_1 = np.sum(e_1, axis=0, keepdims=True) / batch_size

        self.weights_3 -= grad_weights_3 * learning_rate
        self.bias_3 -= grad_bias_3 * learning_rate
        self.weights_2 -= grad_weights_2 * learning_rate
        self.bias_2 -= grad_bias_2 * learning_rate
        self.weights_1 -= grad_weights_1 * learning_rate
        self.bias_1 -= grad_bias_1 * learning_rate


    def train(self, X, y, epochs, learning_rate):

        for epoch in range(epochs):
            batch_size = X.shape[0]

            # 前向传播过程
            y_pred = self.forward(X)

            # 反向传播过程
            self.backward(X, y, batch_size, y_pred, learning_rate)

            # 训练过程中每隔10个epoch，打印损失函数值
            if epoch % 10 == 0: 
                loss = -np.mean(y * np.log(y_pred + 1e-8))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")


# 生成数据供训练、测试使用
if exp_type == 'nonlinear':
    # 生成非线性可分的数据（训练集，测试集）
    X_train, y_train = generate_nonlinear_data(n_samples_per_class=100, n_features=10)
    X_test, y_test = generate_nonlinear_data(n_samples_per_class=100, n_features=10)
    
else:
    # 生成线性可分的数据（训练集，测试集）
    X_train, y_train = generate_linear_data(n_samples_per_class=100, n_features=10)
    X_test, y_test = generate_linear_data(n_samples_per_class=100, n_features=10)

'''
生成数据信息:
X : 数据矩阵 shape = (n_samples, n_features) = (400, 10)
y : 标签 shape = (n_samples,) = (400,)
'''
print(X_train.shape, y_train.shape)
print('具体例子 X_train[0]:', X_train[0])
print('具体例子 y_train[0]:', y_train[0])

'''
将标签y转换one-hot向量
'''
y_train_onehot = np.eye(4)[y_train]
y_test_onehot = np.eye(4)[y_test]

print('具体例子 y_train_onehot[0]:', y_train_onehot[0])


# 配置神经网络中间层的尺寸
input_size = X_train.shape[1] # input_size = 10
hidden1_size = 16  
hidden2_size = 16
output_size = 4

# 初始化神经网络
nn_non_linear = ReluNeuralNetwork(input_size, hidden1_size, hidden2_size, output_size)
nn_linear = LinearNeuralNetwork(input_size, hidden1_size, hidden2_size, output_size)

# 训练神经网络
learning_rate = 0.1
nn_non_linear.train(X_train, y_train_onehot, epochs=300, learning_rate=learning_rate)
nn_linear.train(X_train, y_train_onehot, epochs=300, learning_rate=learning_rate)

# 进行预测并评估准确率
predictions = nn_non_linear.forward(X_test)
predicted_classes = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_classes == y_test)
print(f"非线性模型测试准确率: {accuracy:.2f}")

predictions = nn_linear.forward(X_test)
predicted_classes = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_classes == y_test)
print(f"线性模型测试准确率: {accuracy:.2f}")

'''
结果：
python experiment_bp.py linear
非线性模型测试准确率: 1.00
线性模型测试准确率: 1.00
python experiment_bp.py nonlinear
非线性模型测试准确率: 0.99
线性模型测试准确率: 0.57
'''