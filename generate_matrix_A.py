'''
生成RC中的recurrent矩阵A，并保存在文件matrix_A中
'''
from my_function import generate_A, generate_file
import numpy as np

N = 10
rho = 0.6
path = './matrix_A'  # 生成的矩阵存储文件
generate_file(path=path)
for D_r in [400, 900, 1600, 2500]:  #
    for i in range(N):  # 生成N个shape=（D_r， D_r）的随机矩阵A
        sub_path = '{}/size_{}'.format(path, D_r)
        generate_file(path=sub_path)

        res_A, max_eigval = generate_A(shape=(D_r, D_r), rho=rho, D_r=D_r)
        np.save('{}/A_{}.npy'.format(sub_path, i), res_A)
        print('#' * 50)
        print(i)
        print('max_eig:{}'.format(max_eigval))
print('done!')