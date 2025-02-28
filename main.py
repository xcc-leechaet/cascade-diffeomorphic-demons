import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter as gaus_flter
import cv2
import os
import matplotlib.pyplot as plt
import time
from cupyx.scipy.ndimage import zoom
import numpy as np
"""
验证新的微分同胚转化方式
这份对lena配准效果奇好无比
"""

class diff_demons:
    def __init__(self, fixed, moving):
        self.fixed = fixed
        self.moving = moving
        self.rows, self.cols = fixed.shape[0], moving.shape[1]
        # 坐标索引
        grid_loc = []
        for i in range(2):
            grid_loc.append(cp.arange(0, fixed.shape[i], 1))
        meshgrid = cp.meshgrid(grid_loc[0],
                               grid_loc[1])  # 对于Meshgrid的使用还是不太清楚,具体看https://zhuanlan.zhihu.com/p/105148901
        self.x_ord, self.y_ord = meshgrid[0].transpose([1, 0]).reshape(-1, 1), \
                                 meshgrid[1].transpose([1, 0]).reshape(-1, 1)

        self.field = cp.zeros((2, self.rows, self.cols))
        self.sigma = 0.1
        self.energies = []

    def reverse_field(self, u, iteration=100):
        v = -u.copy()
        v_new = -u.copy()
        iterate = 0
        while (iterate < iteration):
            iterate += 1
            v_new[0, :, :] = self.binli_iterpolator(-u[0, :, :], v)
            v_new[1, :, :] = self.binli_iterpolator(-u[1, :, :], v)
            v = v_new.copy()
        return v

    def field2diffeomorphic(self, deformation_field):
        N=10
        vector=deformation_field.copy()
        vector/=cp.power(2,N)
        for i in range(int(N)):
            t1=self.binli_iterpolator(vector[0],vector)
            t2=self.binli_iterpolator(vector[1],vector)
            vector[0]+=t1
            vector[1]+=t2
        return vector

    def field2diffeomorphic_mask(self,field,mask):
        field_copy=field.copy()
        field_copy[0]=field_copy[0]*(1-mask)
        field_copy[1]=field_copy[1]*(1-mask)
        field_diff=self.field2diffeomorphic(field)
        field_diff[0]=field_diff[0]*mask
        field_diff[1]=field_diff[1]*mask
        return field_copy+field_diff

    def generate_mask(self,img1,img2,threshold=1):
        diff=cp.abs(img1-img2)
        mask=cp.zeros(img1.shape)
        location=cp.where(diff>threshold)
        mask[location]=1
        return mask



    # 对形变场求逆
    # 根据形变场对图像进行双线性插值
    def binli_iterpolator(self, img, deformation_field):
        fnx = self.x_ord + deformation_field[0, :, :].reshape(-1, 1)
        fny = self.y_ord + deformation_field[1, :, :].reshape(-1, 1)
        dx = fnx - cp.floor(fnx)
        dy = fny - cp.floor(fny)
        fx1 = cp.clip((cp.floor(fnx)).astype(cp.int32), 0, self.rows - 2)
        fy1 = cp.clip((cp.floor(fny)).astype(cp.int32), 0, self.cols - 2)
        fx2 = fx1 + 1
        fy2 = fy1 + 1

        dstimg = (img[fx1, fy1] * (1 - dx) + img[fx2, fy1] * dx) * (1 - dy) \
                 + (img[fx1, fy2] * (1 - dx) + img[fx2, fy2] * dx) * dy

        return dstimg.reshape(self.rows, self.cols)

    def binli_iterpolator_field(self, img, vx, vy):

        fnx = self.x_ord + vx.reshape(-1, 1)
        fny = self.y_ord + vy.reshape(-1, 1)
        dx = fnx - cp.floor(fnx)
        dy = fny - cp.floor(fny)
        fx1 = cp.clip((cp.floor(fnx)).astype(cp.int32), 0, self.rows - 2)
        fy1 = cp.clip((cp.floor(fny)).astype(cp.int32), 0, self.cols - 2)
        fx2 = fx1 + 1
        fy2 = fy1 + 1

        dstimg = (img[fx1, fy1] * (1 - dx) + img[fx2, fy1] * dx) * (1 - dy) \
                 + (img[fx1, fy2] * (1 - dx) + img[fx2, fy2] * dx) * dy

        return dstimg.reshape(self.rows, self.cols)

    # 计算互信息
    def cal_NCC(self, fixed, moving):
        # fm=fixed*moving
        # ff=fixed*fixed
        # mm=moving*moving
        # ncc=cp.sum(fm)/cp.sqrt(cp.sum(ff)*cp.sum(mm))
        di = fixed - moving
        ncc = cp.sum(di ** 2) / (fixed.shape[0] * moving.shape[1])
        return ncc

    # 计算图像梯度
    def grd_img(self, img, flag):
        if flag == True:
            x, y = cp.gradient(img)
        else:
            x = cp.zeros(self.fixed.shape)
            y = cp.zeros(self.moving.shape)
            x[1:-2, :] = img[:-3, :] + img[2:-1, :] - 2 * img[1:-2, :]
            y[:, 1:-2] = img[:, :-3] + img[:, 2:-1] - 2 * img[:, 1:-2]

        return x, y

    # 计算速度场
    def cal_vector(self, fixed, moving, alpha, fx, fy, flag):
        diff = fixed - moving

        mx, my = self.grd_img(moving, flag)
        vx = diff * (mx / (mx ** 2 + my ** 2 + alpha * diff**2) + fx / (fx ** 2 + fy ** 2 + alpha * diff**2))
        vy = diff * (my / (mx ** 2 + my ** 2 + alpha * diff**2) + fy / (fx ** 2 + fy ** 2 + alpha * diff**2))
        vx[cp.isnan(vx)] = 0
        vx[cp.isinf(vx)] = 0
        vy[cp.isnan(vy)] = 0
        vy[cp.isinf(vy)] = 0
        vx = gaus_flter(vx, self.sigma)
        vy = gaus_flter(vy, self.sigma)
        return vx, vy

    def optimize(self, iteration, flag):

        iterate = iteration  # 迭代计数器
        alpha = 1 # 初始化alpha
        flag_best = 1e10  # 初始化互相关系数最小值
        v = cp.zeros(self.field.shape)  # 初始化速度场
        fx, fy = self.grd_img(self.fixed, flag)
        self.field_best = cp.zeros(self.field.shape)
        while (iterate > 0):
            iterate -= 1
            tempy = self.binli_iterpolator(self.moving, self.field)
            Ncc_iter = self.cal_NCC(self.fixed, tempy)
            self.energies.append(cp.asnumpy(Ncc_iter))
            if Ncc_iter < flag_best:
                flag_best = Ncc_iter
                self.field_best = self.field.copy()

            v[0], v[1] = self.cal_vector(self.fixed, tempy, alpha, fx, fy, flag)

            v=self.field2diffeomorphic(v)

            self.field[0] = self.binli_iterpolator(self.field[0], v)
            self.field[1] = self.binli_iterpolator(self.field[1], v)

            self.field+=v/cp.abs(v.max())
            self.field[0] = gaus_flter(self.field[0], self.sigma)
            self.field[1] = gaus_flter(self.field[1], self.sigma)
            #self.field = self.field2diffeomorphic(self.field)


if __name__ == "__main__":
    path_fixed='/home/xuchacha/GUOKE/recurrent/diff_demons/demons/demons2d/data/heart-64.png'
    path_moving='/home/xuchacha/GUOKE/recurrent/diff_demons/demons/demons2d/data/heart-110.png'
    path_fixed='/home/xuchacha/GUOKE/recurrent/diff_demons/demons/demons2d/data/lenag2.png'
    path_moving='/home/xuchacha/GUOKE/recurrent/diff_demons/demons/demons2d/data/lenag1.png'
    fixed = cv2.imread(path_fixed,0)
    moving = cv2.imread(path_moving,0)
    fixed = cp.asarray(fixed)
    moving = cp.asarray(moving)

    energy_list=[]
    for i in range(10):
        dd = diff_demons(fixed, moving)
        dd.optimize(100, True)
        energy_list.extend(dd.energies)
        moving = dd.binli_iterpolator(dd.moving, dd.field_best)
        plt.imshow(cp.asnumpy(moving),"gray")
        plt.show()


    np.save("/home/xuchacha/graduate/demons算法/diffeomorphic_demons_2.npy",np.asarray(energy_list))
    abroke = 1