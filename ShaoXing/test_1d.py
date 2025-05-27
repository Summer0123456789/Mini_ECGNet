import numpy as np
import h5py
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from ECG_Model import Mini_ECGNet


if "__main__" == __name__:
    # 加载数据
    hdf5_file_path1 = "D:/A-ECGdata/Clean/dataset_test.h5"
    with h5py.File(hdf5_file_path1, 'r') as hf:
        features = np.array(hf['features'])
        labels = np.array(hf['labels'])

    X_val, y_val = features, labels

    weight_file_path = "Mini_ECGNet/Clean/sp0_8_sp1_12_epoch_055-0.9756-0.1023.h5"

    # 根据模型名称加载对应模型
    input_shape = (5000, 6)
    num_classes = 4

    model = Mini_ECGNet(input_shape, num_classes)
    model.load_weights(weight_file_path)
    model.summary()
    print(f"Loaded weights from: {weight_file_path}")

    # 验证模型
    y_pred_probs = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred_probs, axis=-1)

    accuracy = accuracy_score(y_val, y_pred_classes)
    precision = precision_score(y_val, y_pred_classes, average='macro')
    recall = recall_score(y_val, y_pred_classes, average='macro')
    f1 = f1_score(y_val, y_pred_classes, average='macro')

    print(f"Validation Accuracy: {accuracy}")
    print(f"Validation Precision: {precision}")
    print(f"Validation Recall: {recall}")
    print(f"Validation F1 Score: {f1}")

    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(y_val, y_pred_classes)
    labels = ['SB', 'SR', 'AFIB', 'GSVT']

    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True) * 100
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix_percent, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix (Percentage)')
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f'{conf_matrix_percent[i, j]:.2f}%', horizontalalignment="center",
                     color="white" if conf_matrix_percent[i, j] > conf_matrix_percent.max() / 2. else "black")

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix_percent, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f'{conf_matrix[i, j]}', horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > conf_matrix.max() // 2. else "black")

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.show()
