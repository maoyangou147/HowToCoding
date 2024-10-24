'''
import numpy as np

def kmeans(data, k=3, normalize=False, limit=500):
    # normalize 数据
    if normalize:
        stats = (data.mean(axis=0), data.std(axis=0))
        data = (data - stats[0]) / stats[1]
    
    # 直接将前K个数据当成簇中心
    centers = data[:k]

    for i in range(limit):
        # 首先利用广播机制计算每个样本到簇中心的距离，之后根据最小距离重新归类
        classifications = np.argmin(((data[:, :, None] - centers.T[None, :, :])**2).sum(axis=1), axis=1)
        # 对每个新的簇计算簇中心
        new_centers = np.array([data[classifications == j, :].mean(axis=0) for j in range(k)])

        # 簇中心不再移动的话，结束循环
        if (new_centers == centers).all():
            break
        else:
            centers = new_centers
    else:
        # 如果在for循环里正常结束，下面不会执行
        raise RuntimeError(f"Clustering algorithm did not complete within {limit} iterations")
            
    # 如果normalize了数据，簇中心会反向 scaled 到原来大小
    if normalize:
        centers = centers * stats[1] + stats[0]

    return classifications, centers

data = np.random.rand(200, 2)
classifications, centers = kmeans(data, k=5)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


plt.figure(figsize=(12, 8))
plt.scatter(x=data[:, 0], y=data[:, 1], s=100, c=classifications)
plt.scatter(x=centers[:, 0], y=centers[:, 1], s=500, c='k', marker='^')
plt.show()
'''

import numpy as np

def kmeans(nums, k=3, normalize=False, limit=500):
    if normalize:
        mean = nums.mean(axis=0)
        std = nums.std(axis=0)
        nums = (nums - mean) / std
    
    centers = nums[:k]

    for _ in range(limit):
        classifications = np.argmin(((nums[:, :, None] - centers.T[None, :, :]) ** 2).sum(axis=1), axis=1)
        new_centers = np.array([nums[classifications == i, :].mean(axis=0) for i in range(k)])

        if (centers == new_centers).all():
            break
        else:
            centers = new_centers

    if normalize:
        centers = centers * std + mean
    
    return classifications, centers

nums = np.random.rand(200, 2)
classifications, centers = kmeans(nums, 3, False, 500)
