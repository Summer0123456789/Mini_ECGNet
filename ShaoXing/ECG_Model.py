import json

import tensorflow as tf
from keras.initializers.initializers_v2 import RandomUniform
from keras.layers import MultiHeadAttention, GlobalAveragePooling1D, Reshape
from tensorflow.keras.layers import Layer, Dense, Flatten, Conv1D, BatchNormalization, ReLU, Add, MaxPooling1D, \
    LayerNormalization, Dropout


def load_model_folders(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)


def select_folder(config_file):
    model_folders = load_model_folders(config_file)
    print("Available Models:")
    for i, (model_name, folder_path) in enumerate(model_folders.items(), start=1):
        print(f"{i}. {model_name} ({folder_path})")

    try:
        choice = int(input("Select a model by its number: ")) - 1
        if 0 <= choice < len(model_folders):
            model_name = list(model_folders.keys())[choice]
            folder_path = model_folders[model_name]
            print(f"Selected model: {model_name}, folder: {folder_path}")
            return model_name, folder_path
        else:
            print("Invalid selection.")
            return None, None
    except ValueError:
        print("Invalid input.")
        return None, None


def Covariance_pooling(x, eps=1e-7):
    """
    输入形状: [Batch, Time, Channels]
    输出形状: [Batch, Channels, Channels]

    Args:
        x: 输入张量，形状为 [B, T, C]
        eps: 正则化项，防止协方差矩阵不可逆
    Returns:
        cov_matrix: 协方差矩阵，形状为 [B, C, C]
    """
    # 确保输入维度正确
    if len(x.shape) != 3:
        raise ValueError("输入形状应为 [Batch, Time, Channels]")

    # 计算每个通道的均值（沿时间维度 axis=1）
    mean = tf.reduce_mean(x, axis=1, keepdims=True)  # [B, 1, C]

    # 中心化数据：减去均值
    x_centered = x - mean  # [B, T, C]

    # 调整维度顺序为 [B, C, T]，以便矩阵乘法
    x_centered = tf.transpose(x_centered, perm=[0, 2, 1])  # [B, C, T]

    # 计算协方差矩阵
    batch_size, C, T = tf.shape(x_centered)[0], tf.shape(x_centered)[1], tf.shape(x_centered)[2]
    cov_matrix = tf.matmul(x_centered, x_centered, transpose_b=True)  # [B, C, C]
    cov_matrix = cov_matrix / (tf.cast(T, tf.float32) - 1.0)  # 无偏估计（除以 T-1）

    # 添加正则化项（防止矩阵奇异）
    identity = tf.eye(C) * eps  # [C, C]
    cov_matrix = cov_matrix + identity  # [B, C, C]

    return cov_matrix


class SEBlock(Layer):
    def __init__(self, reduction=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.dense2 = None
        self.dense1 = None
        self.global_pool = None
        self.reduction = reduction

    def build(self, input_shape):
        # 全局平均池化适配一维数据
        self.global_pool = GlobalAveragePooling1D()
        # 两个全连接层
        self.dense1 = Dense(input_shape[-1] // self.reduction, activation='relu')
        self.dense2 = Dense(input_shape[-1], activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        # 全局平均池化，生成通道描述
        x = self.global_pool(inputs)  # 输出形状为 (batch_size, channels)
        x = Reshape((1, inputs.shape[-1]))(x)  # 调整为 (batch_size, 1, channels)
        # 全连接层生成通道权重
        x = self.dense1(x)
        x = self.dense2(x)
        # 通道加权，保持形状不变
        return inputs * x  # 逐通道加权

    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({
            'reduction': self.reduction,  # 返回 reduction 参数
        })
        return config


class SimplifiedAttention(Layer):
    def __init__(self, output_dim=None, **kwargs):
        """
        output_dim: 用来指定权重矩阵的输出维度。默认为 None，如果没有提供，将使用 input_shape[-1] 作为维度。
        """
        self.W = None
        self.output_dim = output_dim
        super(SimplifiedAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 选择output_dim，如果没有提供则默认为 input_shape[-1]
        if self.output_dim is None:
            self.output_dim = input_shape[-1]

        # 这里的input_shape[-1] 是输入向量 X 的维度
        self.W = self.add_weight(
            name='W',
            shape=(input_shape[-1], self.output_dim),  # W 的形状根据 output_dim 设置
            initializer=tf.keras.initializers.GlorotUniform(),
            regularizer=tf.keras.regularizers.L2(1e-4),  # L2 正则化
            trainable=True
        )
        super(SimplifiedAttention, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):

        K = tf.matmul(inputs, self.W)
        output = tf.matmul(tf.transpose(K, perm=[0, 2, 1]), inputs)  # 转置 K 并与 X 做点积

        return output

    def get_config(self):
        """
        返回自定义层的配置字典，包括所有需要保存的参数。
        """
        config = super(SimplifiedAttention, self).get_config()
        config.update({
            'output_dim': self.output_dim,  # 返回 output_dim 参数
        })
        return config


class ConvBlock(Layer):
    def __init__(self, filters, kernel_size, pool_size, dilation_rate=1, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dilation_rate = dilation_rate

        # Conv1D layers with 'same' padding and no strides (output same length as input)
        self.conv1 = Conv1D(filters, kernel_size, strides=1, dilation_rate=self.dilation_rate, padding="same", activation=None)
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.conv2 = Conv1D(filters, kernel_size, strides=1, dilation_rate=self.dilation_rate, padding="same", activation=None)  # 'same' padding
        self.bn2 = BatchNormalization()
        self.add = Add()

        # Pooling layer to reduce sequence length
        self.pool = MaxPooling1D(pool_size=self.pool_size)

        # Adjust shortcut channel dimensions (if needed)
        self.adjust_shortcut = Conv1D(filters, 1, padding="same", strides=1, activation=None)

    def call(self, inputs, training=None, mask=None):
        # Conv Path
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = self.relu(x)
        return x

    def get_config(self):
        config = super(ConvBlock, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "pool_size": self.pool_size,
            "dilation_rate": self.dilation_rate,
        })
        return config


def Mini_ECGNet(input_shape=(5000, 6), num_classes=4, stand_power=8, dilation_rate=12):
    inputs = tf.keras.Input(shape=input_shape)

    # Initial Conv Layer with 'same' padding
    x = Conv1D(8, kernel_size=3, dilation_rate=dilation_rate, padding="same", activation="relu")(inputs)

    # Add Residual Blocks
    x = ConvBlock(8, kernel_size=25, dilation_rate=dilation_rate // 6, pool_size=2)(x)
    x = ConvBlock(16, kernel_size=13, dilation_rate=dilation_rate // 3, pool_size=2)(x)
    x = ConvBlock(8, kernel_size=7, dilation_rate=dilation_rate, pool_size=2)(x)

    # Attention layer: Use AttentionWithLeftMultiply
    x = LayerNormalization()(x)
    attention_layer = SimplifiedAttention(output_dim=stand_power)
    combined_output = attention_layer(x)  # Call the custom attention layer

    # Flatten the output
    x = Flatten()(combined_output)
    x = LayerNormalization()(x)
    x = Dropout(0.8)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.8)(x)
    x = Dense(num_classes, activation="softmax")(x)

    # Create Model
    a_model = tf.keras.Model(inputs, x)

    return a_model


if __name__ == '__main__':
    model_names, folder_paths = select_folder("model_folders.json")

    # 根据模型名称加载对应模型
    input_shapes = (5000, 6)
    nums_classes = 4

    model = Mini_ECGNet(input_shapes, nums_classes)
    model.summary()
