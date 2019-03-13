import random
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from scipy.signal import resample


np.random.seed(43)


"""
Data augmentation
"""
def augment(x):
    
    result = np.zeros(shape=(4, 187))
    for i in range(3):
        if random.random() < 0.33:
            new_y = stretch(x)
        elif random.random() < 0.66:
            new_y = amplify(x)
        else:
            new_y = stretch(x)
            new_y = amplify(new_y)
        result[i, :] = new_y
        
    return result

# stretch
def stretch(x):
    # 将 187 的长度随机重新采样
    l = int(187 * (1 + (random.random()-0.5)/3))
    y = resample(x, l)
    # 长度小于 187 的填充 0
    if l < 187:
        y_ = np.zeros(shape=(187,))
        y_[:l] = y
    # 长度大于 187 的裁剪到 187 的长度
    else:
        y_ = y[:187]
    return y_

# amplify
def amplify(x):
    alpha = (random.random()-0.5)
    factor = -alpha*x + (1+alpha) # 放大的幅度几乎很小
    return x*factor


def get_data_set():
    
    """
    Read data
    """
    df1 = pd.read_csv("D:/cao/kaggle/ECG/mitbih_train.csv", header=None)
    df2 = pd.read_csv("D:/cao/kaggle/ECG/mitbih_test.csv", header=None)

    # concate
    df = pd.concat([df1, df2], axis=0)

    M = df.values
    X = M[:, :-1]
    y = M[:, -1].astype(int)


    # apply augment 

    C0 = np.argwhere(y == 0).flatten()
    C1 = np.argwhere(y == 1).flatten()
    C2 = np.argwhere(y == 2).flatten()
    C3 = np.argwhere(y == 3).flatten()
    C4 = np.argwhere(y == 4).flatten()

    # 将 augment 这个函数按照所设定的 axis 应用
    result = np.apply_along_axis(augment, axis=1, arr=X[C3]).reshape(-1, 187)

    # 将 C3 的数据增强到与其他一样的级别
    classes = np.ones(shape=(result.shape[0],),dtype=int)*3

    # 按垂直方向堆叠扩增的数据
    X = np.vstack([X, result]) 
    y = np.hstack([y, classes])


    """
    Split dataset
    """

    subC0 = np.random.choice(C0, 800)
    subC1 = np.random.choice(C1, 800)
    subC2 = np.random.choice(C2, 800)
    subC3 = np.random.choice(C3, 800)
    subC4 = np.random.choice(C4, 800)

    # test set
    X_test = np.vstack([X[subC0], X[subC1], X[subC2], X[subC3], X[subC4]])
    y_test = np.hstack([y[subC0], y[subC1], y[subC2], y[subC3], y[subC4]])
    print("test set shape: ", X_test.shape)
    print("test examples: ", X_test.shape[0])

    # training set
    X_train = np.delete(X, [subC0, subC1, subC2, subC3, subC4], axis=0)
    y_train = np.delete(y, [subC0, subC1, subC2, subC3, subC4], axis=0)
    print("train examples: ", y_train.shape[0])

    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    X_test, y_test = shuffle(X_test, y_test, random_state=0)

    X_train = np.expand_dims(X_train, 2)
    X_test = np.expand_dims(X_test, 2)

    # one-hot encoding

    ohe = OneHotEncoder()
    y_train = ohe.fit_transform(y_train.reshape(-1, 1)).toarray()
    y_test = ohe.transform(y_test.reshape(-1, 1)).toarray()

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":

    X_train, y_train, X_test, y_test = get_data_set()

    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)