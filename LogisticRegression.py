import os

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def add_one(X):
    m, n = X.shape
    X = np.column_stack((X, np.ones((m, 1))))
    return X


class LogisticRegression:
    def __init__(self, n=None, K=None):
        self.n = n
        self.K = K
        if self.n is not None and self.K is not None:
            self.W = np.zeros((K, n + 1))
        else:
            self.W = None
        self.file_W = 'lr_weight.npy'

    # X:(m, n) return: (m, K)

    def predict_added(self, X):
        m, n = X.shape
        numerator = np.exp(np.dot(X, self.W.T))  # (m, K)
        numerator = np.clip(numerator, -2 ** 31, 2 ** 31)
        denominator = np.sum(numerator, axis=1).reshape(m, 1)  # (m, 1)
        return numerator / denominator

    def predict(self, X):
        X = add_one(X)
        return self.predict_added(X)

    def loss_added(self, X, y):
        l1 = np.sum(np.log(np.sum(np.exp(np.dot(self.W, X.T)), axis=0)))
        l2 = np.sum(y.dot(self.W) * X)
        return l1 - l2

    def loss(self, X, y):
        X = add_one(X)
        return self.loss_added(X, y)

    def gradient_added(self, X, y, regular='l2', regular_weight=0.1):  # output:(K, n + 1)
        y_predict = self.predict_added(X)  # (m, K)
        g = (y_predict - y).T.dot(X)
        if regular == 'l2':
            reg = regular_weight * self.W
        else:
            reg = 0
        return g + reg

    def gradient(self, X, y, regular='l2', regular_weight=0.1):  # output:(K, n + 1)
        X = add_one(X)
        return self.gradient_added(X, y, regular, regular_weight)

    def fit_added(self, X_train, y_train, X_val=None, y_val=None, *,
                  max_iters=100, learn_rate=0.0001, regular='l2', regular_weight=0.1, add_noise=False):

        if add_noise:
            cov = np.eye(self.n + 1) * (learn_rate ** 2)
            mean = [0 for i in range(self.n + 1)]

        for i in range(1, max_iters + 1):
            grad = self.gradient_added(X_train, y_train, regular, regular_weight)
            if add_noise:
                noise = np.random.multivariate_normal(mean, cov, self.K)
            else:
                noise = 0
            self.W = self.W - learn_rate * grad + noise

    def fit(self, X_train, y_train, X_val=None, y_val=None, *,
            max_iters=100, learn_rate=0.0001, regular='l2', regular_weight=0.1, add_noise=False):

        X_train = add_one(X_train)
        self.fit_added(X_train, y_train, X_val, y_val, max_iters=max_iters, learn_rate=learn_rate, regular=regular,
                       regular_weight=regular_weight, add_noise=add_noise)

    def save_weight(self, file='lr_weight.npy'):
        np.save(file, self.W)

    def load_weight(self, file='lr_weight.npy'):
        self.W = np.load(file)


def show_img(x, y, y_predict):
    plt.imshow(x)
    plt.show()
    print(y, y_predict)


if __name__ == '__main__':
    # 准备数据
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784) / 255
    X_test = X_test.reshape(-1, 784) / 255

    X_train = add_one(X_train)
    X_test = add_one(X_test)

    y_train = label_binarize(y_train, classes=np.arange(10))
    y_test = label_binarize(y_test, classes=np.arange(10))

    # 设置参数
    max_iters = (10, 20, 30, 50, 100, 150, 200)
    delta_iters = (10, 10, 10, 20, 50, 50, 50)
    learn_rates = (0.01, 0.0001, 0.00001, 0.000001, 0.0000001)
    weight_update_methods = ('none', 'l2', 'noise')
    regular_weights = (10, 1, 0.1, 0.01, 0.001, 0.0001)
    # plt.figure(dpi=200)

    if not os.path.isdir('imgs'):
        os.makedirs('imgs')
    if not os.path.isdir('models'):
        os.makedirs('models')

    max_iter = 0
    for delta_iter in delta_iters:
        max_iter += delta_iter
        # legend = []
        draws = []
        for learn_rate in learn_rates:
            for method in weight_update_methods:
                for regular_weight in regular_weights:

                    lr = LogisticRegression(784, 10)
                    # 载入已有模型
                    if max_iter != 10:
                        if method == 'none':
                            file_str = f'lr_weight-iter{max_iter - delta_iter}-lr{learn_rate}'
                        elif method == 'noise':
                            file_str = f'lr_weight-iter{max_iter - delta_iter}-lr{learn_rate}-{method}'
                        else:
                            file_str = f'lr_weight-iter{max_iter - delta_iter}-lr{learn_rate}-{method}-rw{regular_weight}'
                        lr.load_weight('models/' + file_str + '.npy')

                    # 训练模型
                    lr.fit_added(X_train, y_train, X_test, y_test,
                                 max_iters=delta_iter,
                                 learn_rate=learn_rate,
                                 regular='l2' if method == 'l2' else None,
                                 regular_weight=regular_weight,
                                 add_noise=True if method == 'noise' else False,
                                 )

                    # 保存模型
                    if method == 'none':
                        file_str = f'lr_weight-iter{max_iter}-lr{learn_rate}'
                    elif method == 'noise':
                        file_str = f'lr_weight-iter{max_iter}-lr{learn_rate}-{method}'
                    else:
                        file_str = f'lr_weight-iter{max_iter}-lr{learn_rate}-{method}-rw{regular_weight}'
                    lr.save_weight('models/' + file_str + '.npy')

                    # 绘制roc曲线
                    # y_test_onehot = label_binarize(y_test, classes=np.arange(10))
                    y_predict = lr.predict_added(X_test)
                    fpr, tpr, thresholds = metrics.roc_curve(y_test.ravel(), y_predict.ravel())
                    auc = metrics.auc(fpr, tpr)
                    # plt.plot(fpr, tpr)

                    if method == 'none':
                        legend_str = f'{auc:.3f}-iter{max_iter}-lr{learn_rate}'
                    elif method == 'noise':
                        legend_str = f'{auc:.3f}-iter{max_iter}-lr{learn_rate}-{method}'
                    else:
                        legend_str = f'{auc:.3f}-iter{max_iter}-lr{learn_rate}-{method}-rw{regular_weight}'
                    draws.append((auc, fpr, tpr, legend_str))
                    # legend.append(legend_str)
                    # 控制台输出
                    if method == 'none':
                        print_str = f'max_iter: {max_iter}, learn_rate: {learn_rate}, auc: {auc}'
                    elif method == 'noise':
                        print_str = f'max_iter: {max_iter}, learn_rate: {learn_rate}, method: {method}, auc: {auc}'
                    else:
                        print_str = f'max_iter: {max_iter}, learn_rate: {learn_rate}, method: {method}, regular_weight: {regular_weight}, auc: {auc}'
                    print(print_str)

                    if method != 'l2':
                        break
        plt.clf()
        plt.figure(figsize=(19.2, 10.8))
        plt.title(f'多元逻辑回归在MNIST上的ROC曲线. 迭代次数: {max_iter}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        draws.sort(key=lambda x: x[0], reverse=True)
        for x in draws:
            plt.plot(x[1], x[2])
        legend = [x[3] for x in draws]
        plt.legend(legend, loc='lower right')
        plt.savefig(f'imgs/lr_weight-iter{max_iter}.jpg')

# plt.show()
# lr.update_weight(X_train, y_train, X_test, y_test, max_times=100)
# lr.save_weight()
# rs = np.random.randint(0, y_test.shape[0], 100)
# for i in rs:
#     show_img(X_test[i].reshape(28, 28), y_test[i],
#              np.argmax(lr.predict(X_test[i].reshape(1, -1))[0]))

# y_test_onehot = np.zeros((y_test.shape[0], 10))
# # y_test_onehot = label_binarize(y_test, np.arange(10))
# print(y_test_onehot.shape)
# for i in range(y_test.shape[0]):
#     y_test_onehot[i][y_test[i]] = 1
# for i in range(y_test_onehot.shape[0]):
#     for j in range(y_test_onehot.shape[1]):
#         x1 = y_test_onehot[i][j]
#         x2 = y_test_onehot2[i][j]
#         if x1 != x2:
#             print(f'pos:({i}, {j}),x1: {x1},x2: {x2}')
# y_test_onehot = y_test_onehot.reshape(-1)
# y_predict_onehot = y_predict_onehot.reshape(-1)


'''
    loss:
        # l2 = np.sum(self.W[y] * X)
        # l1 = 0
        # W:(K, n + 1)
        # X:(m, n + 1)
        # for i in range(m):
        #     s = 0
        #     for k in range(self.K):
        #         s += np.exp(np.dot(self.W[i], X[i].T))
        #     l1 += np.log(s)

        # W[y]: m * (n+1)
        # X:(m, n + 1)

        # l2 = 0
        # for i in range(m):
        #     self.W[y].dot(X.T)
        #     l2 += np.dot(self.W[y[i]], X[i].T)  # 1*(n+1) (n+1)*1
'''

'''
gradient:
        # X = np.array()
        # denominator = np.zeros((m, 1))
        # numerator = np.zeros((m, K))
        # W:(K, n + 1)
        # X:(m, n + 1)
        # for i in range(K):
        #     t = np.dot(self.W[i].reshape(1, -1), X.T)
        #     t = np.clip(t, -30, 30)
        #     numerator[:, i] = np.exp(t).T.reshape(m)
        #     denominator += numerator[:, i].reshape(m, 1)
'''

'''
            # if X_val is not None and i % 10 == 0:
            #     y_predict_train = self.predict(X_train)
            #     y_predict_train = np.argmax(y_predict_train, axis=1)
            #     true_num_train = np.sum(y_predict_train == y_train)
            #
            #     y_predict_val = self.predict(X_val)
            #     y_predict_val = np.argmax(y_predict_val, axis=1)
            #     true_num_val = np.sum(y_predict_val == y_val)
            #     print(
            #         f'max_iters: {max_iters}, learn_rate: {learn_rate}, regular: {regular}, regular_weight: {regular_weight}, add_noise: {add_noise}. '
            #         f'{i}th train, '
            #         f'train precise: {true_num_train / y_train.shape[0]:.2f}, '
            #         f'val precise: {true_num_val / y_val.shape[0]:.2f}')
'''
