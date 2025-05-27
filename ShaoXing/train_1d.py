import gc
import os
import h5py
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from ECG_Model import Mini_ECGNet


class RenameCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, temp_path, model_name, sp_0, sp_1=0, exist_type=True, noise_type=None):
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
            val_accuracy = logs.get("val_accuracy")
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


def train_models(
        model_type="ECGNet",
        input_shape=(5000, 6),
        num_classes=4,
        sp_0=8,
        sp_1=12,
        summary=False,
        learning_rate=0.001,
        factor=0.85,
        epochs=100,
        batch_size=128,
        train_dataset="D:/A-ECGdata/Clean/dataset_train.h5",
        random_num=33,

):
    # 1. 数据加载
    with h5py.File(train_dataset, 'r') as hf:
        features = np.array(hf['features'])
        labels = np.array(hf['labels'])

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=1 / 9, stratify=labels, random_state=random_num
    )

    # 2. model加载
    model = Mini_ECGNet(input_shape=input_shape, num_classes=num_classes)

    if summary:
        model.summary()

    # 3. 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 4. 临时保存路径
    temp_checkpoint_path = "temp_model_weights.h5"

    # 5. 使用模型名称替换根目录
    rename_callback = RenameCheckpointCallback(temp_checkpoint_path, model_type, sp_0, sp_1, noise_type="Clean")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=temp_checkpoint_path, save_best_only=True, monitor="val_accuracy", mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=factor, patience=15, min_lr=1e-5, verbose=1
        ),
        rename_callback  # 添加自定义重命名回调
    ]

    # 6. 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # 7. 绘制结果
    if summary:
        # 创建一个包含2行1列子图的图形窗口
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # 绘制第一个图（Loss Curve）
        axes[0].plot(history.history['loss'], label='Training Loss')
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Loss Curve')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid()

        # 绘制第二个图（Accuracy Curve）
        axes[1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Accuracy Curve')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid()

        # 显示图形
        plt.tight_layout()  # 自动调整子图之间的间距
        plt.show()

    return rename_callback.best_val_accuracy, summary


if __name__ == "__main__":
    hdf5_file_path = "D:/A-ECGdata/Clean/dataset_train.h5"
    train_models(model_type="ECGNet", train_dataset=hdf5_file_path, random_num=33, summary=True)
