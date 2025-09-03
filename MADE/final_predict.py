"""
MADE 最终预测与标签纠错模块
=========================

本文件使用多种传统机器学习模型的集成来对样本进行校正，
将模型一致判定的样本划分为纠正后的 benign/malicious 集合，
并输出纠正结果统计信息。

作者: RAPIER 开发团队
版本: 1.0
"""

import os
import numpy as np
import xgboost
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 使用多模型集成来纠正剩余样本标签
def main(feat_dir):
    """
    使用多个传统机器学习模型对样本进行投票纠错
    
    步骤：
    1) 读取 groundtruth 与 unknown 集合
    2) 训练多种分类器并对所有样本给出概率/预测
    3) 进行多数投票，将高置信度样本划入纠正后的集合
    4) 保存纠正后的 benign/malicious 特征，并输出统计
    
    参数:
        feat_dir (str): 特征目录
    """

    be_g = np.load(os.path.join(feat_dir, 'be_groundtruth.npy'))
    ma_g = np.load(os.path.join(feat_dir, 'ma_groundtruth.npy'))
    be_u = np.load(os.path.join(feat_dir, 'be_unknown.npy'))
    ma_u = np.load(os.path.join(feat_dir, 'ma_unknown.npy'))

    X_train = np.concatenate([be_g, ma_g], axis=0)
    Y_train = np.concatenate([np.zeros(be_g.shape[0]), np.ones(ma_g.shape[0])], axis=0)

    X_test = np.concatenate([be_g, ma_g, be_u, ma_u], axis=0)
    Y_test = np.zeros(X_test.shape[0])

    dtrain = xgboost.DMatrix(X_train, label=Y_train)
    dtest = xgboost.DMatrix(X_test, label=Y_test)
    params = {}

    # GaussianNB
    Gaussiannb = GaussianNB()
    Gaussiannb.fit(X_train, Y_train)
    possibility = Gaussiannb.predict(X_test)
    y_pred = possibility > 0.5

    ensemble = y_pred.astype(int)
    ensemble_pos = possibility

    # xgboost
    bst = xgboost.train(params, dtrain)
    possibility = bst.predict(dtest)
    y_pred = possibility > 0.5

    ensemble = ensemble + y_pred.astype(int)
    ensemble_pos = ensemble_pos + possibility

    # AdaBoost
    AdaBoost = AdaBoostClassifier()
    AdaBoost.fit(X_train, Y_train)
    possibility = AdaBoost.predict(X_test)
    y_pred = possibility > 0.5

    ensemble = ensemble + y_pred.astype(int)
    ensemble_pos = ensemble_pos + possibility

    # Linear Discriminant Analysis
    LDA = LinearDiscriminantAnalysis()
    LDA.fit(X_train, Y_train)
    possibility = LDA.predict(X_test)
    y_pred = possibility > 0.5

    ensemble = ensemble + y_pred.astype(int)
    ensemble_pos = ensemble_pos + possibility

    # SVM
    svm = SVC(kernel = 'rbf', probability=True)
    svm.fit(X_train, Y_train)
    possibility = svm.predict(X_test)
    y_pred = possibility > 0.5

    ensemble = ensemble + y_pred.astype(int)
    ensemble_pos = ensemble_pos + possibility

    # random forest
    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    possibility = rf.predict(X_test)
    y_pred = possibility > 0.5

    ensemble = ensemble + y_pred.astype(int)
    ensemble_pos = ensemble_pos + possibility

    # logistic regression
    logistic = LogisticRegression(penalty='l2')
    logistic.fit(X_train, Y_train)
    possibility = logistic.predict(X_test)
    y_pred = possibility > 0.5

    ensemble = ensemble + y_pred.astype(int)
    ensemble_pos = ensemble_pos + possibility

    # 多数投票：>=4 判为恶意，其余为良性
    ensemble_pred = []
    ensemble_test = []
    be_num = 0
    ma_num = 0
    be_all_final = []
    ma_all_final = []
    for i in range(len(ensemble)):
        if ensemble[i] >= 4:
            ensemble_pred.append(True)
            ensemble_test.append(Y_test[i])
            ma_num = ma_num + 1
            ma_all_final.append(X_test[i])

        else:
            ensemble_pred.append(False)
            ensemble_test.append(Y_test[i])
            be_num = be_num + 1
            be_all_final.append(X_test[i])

    be_all_final = np.array(be_all_final)
    ma_all_final = np.array(ma_all_final)
    np.random.shuffle(be_all_final)
    np.random.shuffle(ma_all_final)
    np.save(os.path.join(feat_dir, 'be_corrected.npy'), be_all_final)
    np.save(os.path.join(feat_dir, 'ma_corrected.npy'), ma_all_final)
    
    # 统计与日志
    wrong_be = be_all_final[:, -1].sum()
    wrong_ma = ma_all_final.shape[0] - ma_all_final[:, -1].sum()
    print('malicious in benign set: %d/%d'%(be_all_final.shape[0], wrong_be))
    print('benign in malicious set: %d/%d'%(ma_all_final.shape[0], wrong_ma))
    
    with open('../data/result/label_correction.txt', 'w') as fp:
        fp.write('malicious in benign set: %d(%d)\n'%(wrong_be, be_all_final.shape[0]))
        fp.write('benign in malicious set: %d(%d)\n'%(wrong_ma, ma_all_final.shape[0]))
        fp.write('Remaining noise ratio: %.2f%%\n'%(100 * (wrong_be + wrong_ma) / (be_all_final.shape[0] + ma_all_final.shape[0])))
