import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.layers import TimeDistributed
from keras import regularizers
from keras.layers import LSTM,Dense


column = 10
step =50
df=pd.read_csv("prediction_wry_3.csv",parse_dates=["time"],index_col=[0])


df_for_training=df[:1998]#取前4/5作为训练集
df_for_testing=df[1998:]#取后1/5作为验证集
scaler = MinMaxScaler(feature_range=(-1,1))#把数据集做归一化处理的函数
df_for_training_scaled = scaler.fit_transform(df_for_training)#把训练集转化为向量形式
df_for_testing_scaled=scaler.transform(df_for_testing)#把验证集转化为向量形式
print(df_for_training_scaled.shape)
print(df_for_testing_scaled.shape)


def createXY(dataset,n_past):#划分预测范围，用前100个数据预测下一个值，定义划分函数
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 1:dataset.shape[1]])
            dataY.append(dataset[i,0])#预测的是e0_5,这里数的时候从第一个列（不含time）记为0
    return np.array(dataX),np.array(dataY)

trainX,trainY=createXY(df_for_training_scaled,step)#训练集
testX,testY=createXY(df_for_testing_scaled,step)#测试集,分为两部分，第一部分是testX，这部分是大数据集中最后2000行的全部列，testY是LSTM输出结果

print("trainX Shape-- ",trainX.shape)#input from train data set
print("trainY Shape-- ",trainY.shape)#output from train data set
print("testX Shape-- ",testX.shape)#input from test data set
print("testY Shape-- ",testY.shape)#output from test data set

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.layers import Activation, Dense

def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(20,return_sequences=True,input_shape=(step,column)))
    grid_model.add(LSTM(20))
    grid_model.add(Activation('relu'))
   # grid_model.add(Dense(10,  kernel_regularizer=regularizers.l1(0.01)))
   # grid_model.add(Dense(5, input_dim=5,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
    grid_model.add(Dropout(0.2))#在训练过程中拿掉20%的神经元，防止过拟合
    grid_model.add(Dense(1))#全连接，输出数组的尺寸为1
    grid_model.compile(loss = 'mae',optimizer = optimizer)#loss是损失函数的名字，这里用的mse，optimizer是优化器名字，这里可以选择adam：tf.keras.optimizers.Adam(lr = 学习率，decay = 学习率衰减率）
    return grid_model

grid_model = KerasRegressor(build_fn=build_model,verbose=1,validation_data=(testX,testY))
parameters = {'batch_size' : [32],#批大小，就是在每个训练集中取batch_size个样本训练
              'epochs' : [30],#8或10次迭代
              'optimizer' : ['adam'] }#优化器

#grid_search的意义在于自动调参，其缺陷在于面对大数据集时非常耗时！
grid_search  = GridSearchCV(estimator = grid_model,#选择使用的分类器
                            param_grid = parameters,#需要最优化的参数的取值，值为字典或者列表，本文中用的字典
                            cv = 2)#交叉验证参数，默认使用五折交叉验证


grid_search = grid_search.fit(trainX,trainY)#把数据拟合进去
print(grid_search.best_params_)

my_model=grid_search.best_estimator_.model#将最优参数给到我们的模型中去
prediction=my_model.predict(trainX)#根据测试集测试模型
print("prediction\n", prediction)#输出我们的预测结果
print("\nPrediction Shape-",prediction.shape)

my_model.save('Model_current.h5')
print('Model Saved!')

prediction_copies_array = np.repeat(prediction,column+1, axis=-1)#改变形状
prediction_copies_array.shape#看下我们copy了13列以后的形状，哦，现在对劲了
pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),column+1)))[:,0]#逆变换，而且把第一列提取了，也就是我们需要的那部分

testX_1 = df['result']
testX_1.shape
testX_1_1=np.array(testX_1)
testX_1_1_1 = np.repeat(testX_1_1,column, axis=0)
original_copies_array = np.repeat(testY,column, axis=-1)
original_copies_array.shape
original=np.reshape(testX_1_1_1,(1999,column))[:,0]
original=original[step:1999]

prediction_copies_array = np.repeat(prediction,column+1, axis=-1)#改变形状
prediction_copies_array.shape#看下我们copy了13列以后的形状，哦，现在对劲了
pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),column+1)))[:,0]#逆变换，而且把第一列提取了，也就是我们需要的那部分

testX_1 = df['result']
testX_1.shape
testX_1_1=np.array(testX_1)
testX_1_1_1 = np.repeat(testX_1_1,column, axis=0)
original_copies_array = np.repeat(testY,column, axis=-1)
original_copies_array.shape
original=np.reshape(testX_1_1_1,(1999,column))[:,0]
original=original[step:1999]
print(original)
print(len(original))
print(len(pred))
x=list(range(0, 1949))
print(x)
print(len(x))
x2=list(range(0, 1948))
plt.scatter(x,original,label='real')
plt.scatter(x2,pred,label='predict')
#plt.plot(original, color = 'red', label = 'real')
#plt.plot(pred, color = 'blue', label = 'predict')
#plt.scatter(x,original)
plt.title(' current prediction ')
plt.xlabel('Time')
plt.ylabel(' current')
plt.legend()
plt.xlim((0, 1950))
plt.ylim((0, 400))
plt.savefig('current_prediction.jpg', dpi=500)
plt.show()

def Normalization2(x):
    return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]

original2_m = Normalization2(original)
pred2_m = Normalization2(pred)
print(len(original2_m))
print(len(pred2_m))

original2_m=original2_m[:1948]
0
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score

from sklearn.metrics import r2_score
print('以下是训练集误差')
print('R^2决定系数：',r2_score(original2_m,pred2_m))#R^2决定系数：反映模型拟合优度
print('均方误差MSE为：',mean_squared_error(original2_m,pred2_m))#均方误差
print('RMSE为：',np.sqrt(mean_squared_error(original2_m,pred2_m)))
print('median_absolute_error:',median_absolute_error(original2_m,pred2_m))#中位数绝对值误差
print('mean_absolute_error(MAE)',mean_absolute_error(original2_m,pred2_m))#平均值绝对值误差
print('解释方差：',explained_variance_score(original2_m,pred2_m))






