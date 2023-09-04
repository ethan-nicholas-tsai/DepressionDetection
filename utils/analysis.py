#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/12 12:10
# @Author  : cendeavor
# @Site    :
# @File    : analysis_utils.py
# @Software: PyCharm

import tensorflow as tf
import keras
import numpy as np


def cal_basic_metrics(y_true, y_pred):
    from sklearn.metrics import (
        confusion_matrix,
        precision_score,
        accuracy_score,
        recall_score,
        f1_score,
        roc_auc_score,
        precision_recall_fscore_support,
        roc_curve,
        classification_report,
    )

    # print('accuracy:{}'.format(accuracy_score(y_true, y_pred))) # 不存在average
    # print('precision:{}'.format(precision_score(y_true, y_pred,average='micro')))
    # print('recall:{}'.format(recall_score(y_true, y_pred,average='micro')))
    # print('f1-score:{}'.format(f1_score(y_true, y_pred,average='micro')))
    # print('f1-score-for-each-class:{}'.format(precision_recall_fscore_support(y_true, y_pred))) # for macro
    ans = classification_report(y_true, y_pred, digits=5)  # 小数点后保留5位有效数字
    print(ans)
    return ans


def cal_auc(y_true_one_hot, y_pred_prob):
    from sklearn.metrics import roc_auc_score

    # AUC值
    # 使用micro，会计算n_classes个roc曲线，再取平均
    auc = roc_auc_score(y_true_one_hot, y_pred_prob, average="micro")
    print("AUC y_pred = proba:", auc)
    return auc


def plot_roc(y_true, y_pred_prob):
    # The magic happens here
    import matplotlib.pyplot as plt
    import scikitplot as skplt

    fig, ax = plt.subplots()  # 可以用figsize=(16,12)指定画布大小
    skplt.metrics.plot_roc(y_true, y_pred_prob, ax=ax)
    return fig
    # skplt.metrics.plot_roc(y_true, y_pred_prob)
    # plt.show()  #没必要
    # return plt


def plot_prc(y_true, y_pred_prob):
    import matplotlib.pyplot as plt
    import scikitplot as skplt

    fig, ax = plt.subplots()  # 可以用figsize=(16,12)指定画布大小
    skplt.metrics.plot_precision_recall_curve(y_true, y_pred_prob, ax=ax)
    return fig


def plot_confusion_matrix(y_true, y_pred):
    import matplotlib.pyplot as plt
    import scikitplot as skplt

    fig, ax = plt.subplots()
    skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=True, ax=ax)
    return fig
    # plot = skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=True)
    # plt.show()  #没必要
    # return plt


def report_model_performance(y_true_one_hot, y_pred_prob):
    import numpy as np

    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_true_one_hot, axis=1)

    cal_basic_metrics(y_true, y_pred)
    cal_auc(y_true_one_hot, y_pred_prob)
    plt1 = plot_roc(y_true, y_pred_prob)
    plt2 = plot_confusion_matrix(y_true, y_pred)
    return plt1, plt2


# 精确率评价指标
def metric_precision(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    precision = TP / (TP + FP)
    return precision


# 召回率评价指标
def metric_recall(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    recall = TP / (TP + FN)
    return recall


# F1-score评价指标
def metric_F1score(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1score = 2 * precision * recall / (precision + recall)
    return F1score


# 编译阶段引用自定义评价指标示例
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy',
#                        metric_precision,
#                        metric_recall,
#                        metric_F1score])


def get_hardest_k_examples(test_dataset, model, k=32):
    class_probs = model(test_dataset.x)
    predictions = np.argmax(class_probs, axis=1)
    losses = keras.losses.categorical_crossentropy(test_dataset.y, class_probs)
    argsort_loss = np.argsort(losses)

    highest_k_losses = np.array(losses)[argsort_loss[-k:]]
    hardest_k_examples = test_dataset.x[argsort_loss[-k:]]
    true_labels = np.argmax(test_dataset.y[argsort_loss[-k:]], axis=1)

    return highest_k_losses, hardest_k_examples, true_labels, predictions


def calc_metrics_binary(model, X_test, y_test):
    from sklearn.metrics import (
        classification_report,
        accuracy_score,
        f1_score,
        roc_auc_score,
        recall_score,
        precision_score,
    )

    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=["Normal", "Depressed"], digits=4
    )
    # # 计算准确率
    # acc = accuracy_score(y_test, y_pred)
    # # 计算F1-Score
    # f1 = f1_score(y_test, y_pred, pos_label='1')
    # # 计算精确率
    # prec = precision_score(y_test, y_pred)
    # # 计算召回率
    # rec = recall_score(y_test, y_pred)
    # # 计算AUC面积(2分类)
    # AUC = roc_auc_score(y_test, y_pred)
    return report  # , acc, f1, prec, rec, AUC
