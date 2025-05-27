import gc
import json
import os

import pandas as pd
import tensorflow as tf


import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    multilabel_confusion_matrix
)
from train_model import Mini_ECGNet


if __name__ == "__main__":
    # 选择一个模型
    main_folder_path = "Mini_ECGNet/"

    # 遍历主文件夹内的所有子文件夹
    for subfolder in os.listdir(main_folder_path):
        subfolder_path = os.path.join(main_folder_path, subfolder)

        if not os.path.isdir(subfolder_path):  # 只处理文件夹
            continue

        print(f"Processing folder: {subfolder_path}")
        print()

        # 获取所有 `.h5` 文件
        h5_files = [f for f in os.listdir(subfolder_path) if f.endswith('.h5')]
        if not h5_files:
            print(f"No .h5 files found in {subfolder_path}. Skipping...")
            continue

        # 结果存储
        results = []

        for weight_file in h5_files:
            weight_file_path = os.path.join(subfolder_path, weight_file)

            # 选择并加载模型
            input_shape = (5000, 12)
            num_classes = 4

            model = Mini_ECGNet(input_shape, num_classes)
            model.summary()

            # 加载权重
            model.load_weights(weight_file_path)

            # 加载数据
            hdf5_file_path1 = "D:/A-ECGdata/PTBXL/new_test_data.h5"
            with h5py.File(hdf5_file_path1, 'r') as hf:
                features = np.array(hf['features'])
                labels = np.array(hf['labels'])

            X_test, y_test = features, labels

            # 预测并二值化
            y_pred_probs = model.predict(X_test)
            y_pred_binary = (y_pred_probs > 0.5).astype(int)

            # 初始化指标存储
            label_names = ['CD', 'HYP', 'MI', 'STTC']
            metrics = {
                'Accuracy': [],
                'Precision': [],
                'Recall': [],
                'F1': []
            }

            # 1. 逐标签计算指标（Label-wise）
            # 输出当前权重文件名
            print(f"\n>>> Evaluating weight file: {weight_file}")
            print("\n逐标签性能指标（Label-wise）:")
            print(f"{'Class':<6} {'Accuracy':<8} {'Precision':<8} {'Recall':<8} {'F1':<6}")
            for i in range(num_classes):
                y_true = y_test[:, i]
                y_pred = y_pred_binary[:, i]

                metrics['Accuracy'].append(accuracy_score(y_true, y_pred))
                metrics['Precision'].append(precision_score(y_true, y_pred, zero_division=0))
                metrics['Recall'].append(recall_score(y_true, y_pred, zero_division=0))
                metrics['F1'].append(f1_score(y_true, y_pred, zero_division=0))

                # 打印当前类别指标
                print(f"{label_names[i]:<6} {metrics['Accuracy'][i]:<8.4f} {metrics['Precision'][i]:<8.4f} "
                      f"{metrics['Recall'][i]:<8.4f} {metrics['F1'][i]:<6.4f}")

            # 2. 全局Micro/Macro指标（包括准确率）
            micro_accuracy = accuracy_score(y_test.ravel(), y_pred_binary.ravel())

            # Macro准确率 = 所有标签准确率的平均值
            macro_accuracy = np.mean(metrics['Accuracy'])

            micro_precision = precision_score(y_test, y_pred_binary, average='micro', zero_division=0)
            micro_recall = recall_score(y_test, y_pred_binary, average='micro', zero_division=0)
            micro_f1 = f1_score(y_test, y_pred_binary, average='micro', zero_division=0)

            macro_precision = precision_score(y_test, y_pred_binary, average='macro', zero_division=0)
            macro_recall = recall_score(y_test, y_pred_binary, average='macro', zero_division=0)
            macro_f1 = f1_score(y_test, y_pred_binary, average='macro', zero_division=0)

            print("全局平均指标:")
            print(f"{'Metric':<10} {'Micro':<8} {'Macro':<8}")
            print(f"{'Accuracy':<10} {micro_accuracy:<8.4f} {macro_accuracy:<8.4f}")
            print(f"{'Precision':<10} {micro_precision:<8.4f} {macro_precision:<8.4f}")
            print(f"{'Recall':<10} {micro_recall:<8.4f} {macro_recall:<8.4f}")
            print(f"{'F1':<10} {micro_f1:<8.4f} {macro_f1:<8.4f}")
            # 存储结果
            results.append({
                "model_name": "Mini_ECGNet",
                "subfolder": subfolder,
                "weight_file": weight_file,
                "accuracy": macro_accuracy,
                "precision": macro_precision,
                "recall": macro_recall,
                "f1_score": macro_f1
            })

            # 3. 绘制每个类别的混淆矩阵
            conf_matrices = multilabel_confusion_matrix(y_test, y_pred_binary)

            # 设置子图：每个类别一个子图
            num_labels = len(label_names)
            fig, axes = plt.subplots(1, num_labels, figsize=(6 * num_labels, 4))

            if num_labels == 1:
                axes = [axes]  # 确保axes是列表

            for i, (ax, label) in enumerate(zip(axes, label_names)):
                matrix = conf_matrices[i]

                # 计算每行的百分比
                conf_matrix_percent = matrix.astype('float') / matrix.sum(axis=1, keepdims=True) * 100

                ax.imshow(conf_matrix_percent, cmap='Blues', vmin=0, vmax=conf_matrix_percent.max() * 1.2)
                ax.set_title(f'Confusion Matrix - {label}', pad=20)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Negative', 'Positive'])
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['Negative', 'Positive'])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')

                # 在每个单元格上标注原始数值和百分比
                for (j, k), val in np.ndenumerate(matrix):
                    row_sum = matrix[j, :].sum()  # 计算每行的总数
                    perc = val / row_sum * 100 if row_sum > 0 else 0
                    # 根据百分比的值决定文本颜色
                    text_color = 'white' if perc > conf_matrix_percent.max() / 2 else 'black'
                    ax.text(k, j, f'{val}\n({perc:.1f}%)', ha='center', va='center', color=text_color)

            plt.tight_layout()
            # 清理 GPU 和内存
            tf.keras.backend.clear_session()
            gc.collect()

        plt.show()

    # 保存 CSV 文件到对应的子文件夹
    if results:
        df_results = pd.DataFrame(results)
        results_file = os.path.join(subfolder_path, f"Mini_ECGNet_evaluation_results.csv")
        df_results.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}\n")
