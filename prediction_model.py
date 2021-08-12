# 利用RC， 对Lorenz系统拟合
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import os
from my_function import get_lorenz_data, initial_W_in, \
    load_matrix_A, get_R, get_X, get_P, Predict_V, \
    plot_for_creat_model, get_next_True_Pred, plot_True_Pred, generate_file
import time
################################################################################
# 参数， 具体见Jaideep Pathak PRL 2017 附录
D_in = 3   # 输入维度
D_r = 900  # RC的维度,  根据generate_matrix_A.py生成的矩阵A, 选取合适的维度
D_out = D_in  # 输出等于输出维度

sigma = 1.0   # 输入矩阵的元素按均匀分布【-sigma, sigma】采样
rho = 0.6     # 谱半径
beta = 1e-4   # 正则化系数



# 输出结果文件
out_file = './out_fig/plt_fig'
out_file_bar = './out_fig/plt_err_bar'
generate_file(path=out_file)
generate_file(path=out_file_bar)
################################################################################
#产生3维lorenz系统时间序列
state0 = np.array([1.0, 1.0, 1.0])    # 初始状态
t = np.arange(0.0, 40.0, 0.005)       # 设置采样时间
len_t = len(t)
All_U = get_lorenz_data(state0, t)    # 生成Lorenz系统的时间序列
print('U shape: ', All_U.shape)

################################################################################
################-------模型搭建与拟合---------################
# 按PRL 2017 生成输入矩阵W_in, 具体看文章或相应的函数
W_in = initial_W_in([D_r, D_in], -sigma, sigma)

# 导入生成的RC中的recurrent矩阵A
path = './matrix_A/size_{}/A_0.npy'.format(D_r)
A = load_matrix_A(path)
print('max_eigvalue=',np.max(LA.eigvals(A)))


TT = [10, 50, 100, 150, 200, 300, 500, 1000, 3000, 5000] # 训练数据的步长设置
Length = 50  # 预测训练数据接下来的50个步长数据
figsize = (18, 9)
fontsize = 18

def plot_prediction_fig():
    start = time.time()
    for fig_i, T in enumerate(TT):
        U = All_U[:, -T:]  # 利用最后面T个步长的时间序列数据作为训练数据

        # 根据PRL 2017的附录中的方法，求解
        R = get_R(A, W_in, U, T)
        X = get_X(R)
        P, P1, P2 = get_P(U, X, beta)
        V = np.matmul(P1, R) + np.matmul(P2, R**2)

        # 得到训练后输出层的参数P, P1, P2，进行预测训练数据接下来的Length个步长数据
        True_V, Pred_V = get_next_True_Pred(Length, t, U, V, R, P1, P2, A, W_in)

        plt.figure(fig_i + 1, figsize=figsize)


        #画出误差图像和预测图像bar
        # plot_True_Pred(True_V, Pred_V, T, Length)

        # 画出前3维的预测曲线, 更直观
        now_t = t[-T:]
        plot_for_creat_model(now_t, U, V, R, P1, P2, A, W_in)


        plt.savefig(out_file + '/figure_' + str(fig_i + 1) + '.jpg')
        print('time used: %.2f s' % (time.time() - start))


    end = time.time()
    print('用时 %.2fs' % (end - start))
    plt.show()

def plot_prediction_error_bar():
    plt.figure(1, figsize=figsize)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

    plt.suptitle('RC -- Loren 3 dim. Different Training Length for  '
                 'predicting the \n following %d steps with %d neurons'
                 % (Length, D_r), fontsize=fontsize + 4)

    start = time.time()
    for fig_i, T in enumerate(TT):
        U = All_U[:, -T:]  # 利用最后面T个步长的时间序列数据作为训练数据

        # 根据PRL 2017的附录中的方法，求解
        R = get_R(A, W_in, U, T)
        X = get_X(R)
        P, P1, P2 = get_P(U, X, beta)
        V = np.matmul(P1, R) + np.matmul(P2, R ** 2)

        # 得到训练后输出层的参数P, P1, P2，进行预测训练数据接下来的Length个步长数据
        True_V, Pred_V = get_next_True_Pred(Length, t, U, V, R, P1, P2, A, W_in)

        # Plot 预测的值和真实值的error图
        # plt.figure(fig_i + 1, figsize=figsize)
        plt.subplot(5, 2, fig_i + 1)
        plt.imshow(Pred_V - True_V, cmap=plt.jet(), vmin=-2, vmax=2)
        plt.ylabel('x', fontsize=fontsize)
        plt.xlabel('time steps', fontsize=fontsize)
        plt.title('trianning length=%d' % (T), fontsize=fontsize)

        print('time used: %.2f s' % (time.time() - start))

    cax = plt.axes([0.85, 0.3, 0.015, 0.4])
    # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)

    plt.savefig(out_file_bar + '/err_bar.jpg')

    end = time.time()
    print('用时 %.2fs' % (end - start))
    plt.show()


if __name__ == '__main__':
    plot_prediction_error_bar()  #   预测和真实的error bar
    # plot_prediction_fig()  # 预测和真实的曲线图