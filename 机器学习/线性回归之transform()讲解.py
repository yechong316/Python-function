# 导入必备的包
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

# 预先定义数据
x = pd.DataFrame(data =[[0 , 21.89,  40.000000,  20.926667],
                        [1 , 19.00,  40.000000,  17.500000],
                        [2 , 20.00,  44.433333,  19.100000],
                        [3 , 19.00,  42.700000,  19.100000],
                        [4 , 18.50,  42.500000,  17.890000]],columns=['lights','T1','RH_1','T2'])


ss = PolynomialFeatures(degree=2) # 定义多项式回归模型,阶次定义为2

'''
仅传入一个x时,fit_transform() 等价于 transform()
注意,在没有做拟合前,不可以直接使用transform(),必须使用fit_transform()
'''
x_transform = ss.fit_transform(x)

'''
等号左侧:转换后的数据尺寸
等号右侧:样本数 * (特征数 + 1) * (特征数 + 2) /2
'''
print(
    x_transform.size ==
      x.shape[0] * (x.shape[1] + 1) * (x.shape[1] + 2) /2
)
print(x_transform)