import numpy as np

RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


def get_weight(array):
    row = array.shape[0]  # 计算出阶数
    a_axis_0_sum = array.sum(axis=0)
    b = array / a_axis_0_sum  # 新的矩阵b
    b_axis_1_sum = b.sum(axis=1)  # 每一行的特征向量
    w = b_axis_1_sum / row  # 归一化处理(特征向量)
    nw = w * row
    AW = (w * array).sum(axis=1)
    max_max = sum(AW / w)/row
    CI = (max_max - row) / (row - 1)
    CR = CI / RI_dict[row]
    if CR < 0.1:
        return w
    else:
        print('不满足一致性，请进行修改')


def AHP(array):
    return get_weight(np.array(array))


if __name__ == '__main__':
    e = np.array([[1, 2, 7, 5, 5], [1 / 2, 1, 4, 3, 3], [1 / 7, 1 / 4, 1, 1 / 2, 1 / 3], [1 / 5, 1 / 3, 2, 1, 1], [1 / 5, 1 / 3, 3, 1, 1]])
    a = np.array([[1, 1 / 3, 1 / 8], [3, 1, 1 / 3], [8, 3, 1]])
    b = np.array([[1, 2, 5], [1 / 2, 1, 2], [1 / 5, 1 / 2, 1]])
    c = np.array([[1, 1, 3], [1, 1, 3], [1 / 3, 1 / 3, 1]])
    d = np.array([[1, 3, 4], [1 / 3, 1, 1], [1 / 4, 1, 1]])
    f = np.array([[1, 4, 1 / 2], [1 / 4, 1, 1 / 4], [2, 4, 1]])
    e = AHP(e)
    a = AHP(a)
    b = AHP(b)
    c = AHP(c)
    d = AHP(d)
    f = AHP(f)

    try:
        res = np.array([a, b, c, d, f])
        ret = (np.transpose(res) * e).sum(axis=1)
        print(ret)
    except TypeError:
        print('数据有误，可能不满足一致性，请进行修改')