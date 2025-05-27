import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import BatchNormalization, Add, ReLU

from tensorflow.keras.layers import Layer, Dense, Flatten, Conv1D, LayerNormalization, Dropout, BatchNormalization, \
    MaxPooling1D


class ConvBlock(Layer):
    def __init__(self, filters, kernel_size, pool_size, dilation_rate=1, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dilation_rate = dilation_rate

        # Conv1D layers with 'same' padding and no strides (output same length as input)
        self.conv1 = Conv1D(filters, kernel_size, strides=1, dilation_rate=self.dilation_rate, padding="same",
                            activation=None)
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.conv2 = Conv1D(filters, kernel_size, strides=1, dilation_rate=self.dilation_rate, padding="same",
                            activation=None)  # 'same' padding
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
        # Step 1: 计算 K = XW
        K = tf.matmul(inputs, self.W)

        # Step 2: 计算 K^T * X
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


class RenameCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, temp_path, model_name, exist_type=True, noise_type="Clean"):
        super().__init__()
        self.temp_path = temp_path
        self.model_name = str(model_name)
        self.best_epoch = 0
        self.best_val_accuracy = 0
        self.best_val_loss = float('inf')
        self.exist_type = exist_type
        self.noise_type = str(noise_type)
        self.root_folder = os.path.join(self.model_name, self.noise_type)
        os.makedirs(self.root_folder, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            val_accuracy = logs.get("val_binary_accuracy")
            val_loss = logs.get("val_loss")
            if self.exist_type:
                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1
            else:
                if val_loss < self.best_val_loss:
                    self.best_val_accuracy = val_accuracy
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1

    def on_train_end(self, logs=None):
        if os.path.exists(self.temp_path):
            final_checkpoint_path = os.path.join(
                self.root_folder,
                f"{self.model_name}_epoch_{self.best_epoch:03d}-"
                f"{self.best_val_accuracy:.4f}-{self.best_val_loss:.4f}.h5"
            )
            os.rename(self.temp_path, final_checkpoint_path)
            print(f"模型保存为最终文件：{final_checkpoint_path}")


class RenameCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, temp_path, model_name, sp_0=8, sp_1=12, exist_type=True, noise_type='Clean'):
        """
        初始化回调，动态设置保存路径为模型名称。

        :param temp_path: 临时文件路径
        :param model_name: 模型名称，用于替换根目录
        :param sp_0: Attention 层的第一个参数
        :param sp_1: Attention 层的第二个参数
        """
        super().__init__()
        self.temp_path = temp_path
        self.model_name = str(model_name)  # 确保模型名称为字符串
        self.sp_0 = sp_0
        self.sp_1 = sp_1
        self.best_epoch = 0
        self.best_val_accuracy = 0
        self.best_val_loss = float('inf')
        self.exist_type = exist_type
        self.noise_type = str(noise_type)

        # 设置根目录为模型名称
        self.root_folder = self.model_name + '/' + self.noise_type
        os.makedirs(self.root_folder, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            val_accuracy = logs.get("val_binary_accuracy")
            val_loss = logs.get("val_loss")
            if self.exist_type:
                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1  # 记录 epoch，+1 是因为 epoch 从 0 开始
            else:
                if val_loss < self.best_val_loss:
                    self.best_val_accuracy = val_accuracy
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1  # 记录 epoch，+1 是因为 epoch 从 0 开始

    def on_train_end(self, logs=None):
        if os.path.exists(self.temp_path):
            # 构造最终文件名，直接在模型名目录下保存文件
            final_checkpoint_path = os.path.join(
                self.root_folder,
                f"sp0_{self.sp_0}_sp1_{self.sp_1}_epoch_{self.best_epoch:03d}-"
                f"{self.best_val_accuracy:.4f}-{self.best_val_loss:.4f}.h5"
            )
            os.rename(self.temp_path, final_checkpoint_path)
            print(f"模型保存为最终文件：{final_checkpoint_path}")


def data_generator(train_h5_file, val_h5_file, batch_size, shuffle=True):
    """
    从训练集和验证集的 HDF5 文件中逐批次加载数据。

    参数:
        train_h5_file (str): 训练集 HDF5 文件路径。
        val_h5_file (str): 验证集 HDF5 文件路径。
        batch_size (int): 每批次的样本数量。
        shuffle (bool): 是否打乱数据顺序。

    返回:
        train_gen: 训练集生成器，每次返回一个批次的 (features, labels)。
        val_gen: 验证集生成器，每次返回一个批次的 (features, labels)。
        steps_per_epoch: 每个 epoch 的训练步数。
        validation_steps: 每个 epoch 的验证步数。
    """
    # 打开 HDF5 文件
    train_hf = h5py.File(train_h5_file, 'r')
    val_hf = h5py.File(val_h5_file, 'r')

    train_features = train_hf['features']
    train_labels = train_hf['labels']
    val_features = val_hf['features']
    val_labels = val_hf['labels']

    # 获取数据的总样本数
    num_train_samples = train_features.shape[0]
    num_val_samples = val_features.shape[0]

    # 训练集生成器
    def train_generator():
        while True:
            indices = np.arange(num_train_samples)
            if shuffle:
                np.random.shuffle(indices)  # 打乱训练集索引
            for start_idx in range(0, len(indices), batch_size):
                end_idx = min(start_idx + batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]

                # 确保索引是递增的
                batch_indices_sorted = np.sort(batch_indices)

                features_batch = train_features[batch_indices_sorted]
                labels_batch = train_labels[batch_indices_sorted]

                yield np.array(features_batch), np.array(labels_batch)

    # 验证集生成器
    def val_generator():
        while True:
            indices = np.arange(num_val_samples)
            if shuffle:
                np.random.shuffle(indices)  # 打乱验证集索引
            for start_idx in range(0, len(indices), batch_size):
                end_idx = min(start_idx + batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]

                # 确保索引是递增的
                batch_indices_sorted = np.sort(batch_indices)

                features_batch = val_features[batch_indices_sorted]
                labels_batch = val_labels[batch_indices_sorted]

                yield np.array(features_batch), np.array(labels_batch)

    # 计算每个 epoch 的训练步数和验证步数
    steps_per_epoch = num_train_samples // batch_size
    validation_steps = num_val_samples // batch_size

    return train_generator(), val_generator(), steps_per_epoch, validation_steps, train_hf, val_hf


def Mini_ECGNet(input_shape=(5000, 12), num_classes=4, stand_power=8, dilation_rate=12):
    inputs = tf.keras.Input(shape=input_shape)

    # Initial Conv Layer with 'same' padding
    x = Conv1D(16, kernel_size=3, dilation_rate=dilation_rate, padding="same", activation="relu")(inputs)

    # Add Residual Blocks
    x = ConvBlock(16, kernel_size=25, dilation_rate=dilation_rate // 6, pool_size=2)(x)
    x = ConvBlock(32, kernel_size=13, dilation_rate=dilation_rate // 3, pool_size=2)(x)
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
    x = Dense(num_classes, activation="sigmoid")(x)

    # Create Model
    a_model = tf.keras.Model(inputs, x)

    return a_model


if __name__ == '__main__':
    train_dataset = "D:/A-ECGdata/PTBXL/new_train_data.h5"
    val_dataset = "D:/A-ECGdata/PTBXL/new_val_data.h5"

    model = Mini_ECGNet()
    model.summary()

    # 使用生成器加载数据
    batch_size = 128  # 可以根据需要调整
    train_gen, val_gen, steps_per_epoch, validation_steps, train_hf, val_hf = data_generator(
        train_dataset, val_dataset, batch_size, shuffle=True
    )

    # 编译模型
    loss_fn = 'binary_crossentropy'
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[tf.keras.metrics.BinaryAccuracy()])

    # 临时保存路径
    temp_checkpoint_path = "temp_model_weights.h5"

    rename_callback = RenameCheckpointCallback(temp_checkpoint_path, 'Mini_ECGNet')

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=temp_checkpoint_path, save_best_only=True, monitor="val_binary_accuracy", mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.85, patience=10, min_lr=1e-5, verbose=1
        ),
        rename_callback  # 添加自定义重命名回调
    ]

    # 训练模型
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,  # 每个 epoch 的训练步数
        epochs=100,
        validation_data=val_gen,  # 验证集生成器
        validation_steps=validation_steps,  # 每个 epoch 的验证步数
        callbacks=callbacks,
        verbose=1
    )

    # 训练完成后关闭文件句柄
    train_hf.close()
    val_hf.close()
