import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mping
import math
from multilayer_perceptron import MultilayerPerceptron

##导入数据集
data = pd.read_csv('../data/mnist-demo.csv')#dataframe格式，行数：10000，列数：785

##可视化展示
#注意：：：！！！！！！！！numpy中数组每一行用一个[]
numbers_to_display = 64#要展示的图片数量
num_cells = math.ceil(math.sqrt(numbers_to_display))#math.ceil(x)返回大于等于参数x的最小整数，即对浮点数向上取整
plt.figure(figsize=(10,10))
for plot_index in range(numbers_to_display):#for in range（3）从0到3，不包含3，即0,1,2。
    digit = data[plot_index:plot_index+1].values#1行785列
    digit_label = digit[0][0]#一个数
    digit_pixels = digit[0][1:]#784行1列
    image_size = int(math.sqrt(digit_pixels.shape[0]))#28=sqrt(784)
    frame = digit_pixels.reshape((image_size, image_size))#行数28，列数28
    plt.subplot(num_cells, num_cells, plot_index + 1)#plt.subplot(2，2，1)表示将整个图像窗口分为2行2列, 当前位置为1.
    plt.imshow(frame, cmap='Greys')
    plt.title(digit_label)#图片标题
plt.subplots_adjust(wspace=0.5,hspace =0.2)#调间距
plt.show()

##我们从10000个图片中，挑选80%（8000个）作为训练集train，由于时间的限制，我们再从这些样本中挑选num_training_examples（100）个用于训练
##另外20%（2000个）作为测试集test。

train_data1 = data.sample(frac =0.8)#调用 sample(frac=0.8) 表示从 data 中随机选择 80% 的数据进行采样。data.sample(5):随机查看5行数据,结果：dataframe格式行数8000，列数785
test_data1 = data.drop(train_data1.index)#drop()函数可以删除某一列/行,结果：行数2000，列数785

train_data = train_data1.values#将dataframe格式转为ndarray格式
test_data = test_data1.values#将dataframe格式转为ndarray格式


num_training_examples = 100
x_train = train_data[:num_training_examples,1:]#结果是一个行数：num_training_examples，列数：784，因为每个图片由784（28*28）个特征点组成
y_train = train_data[:num_training_examples,[0]]#行数：num_training_examples,列数：1
x_test = test_data[:,1:]#行数2000，列数784
y_test = test_data[:,[0]]#行数2000，列数1
layers=[784,25,10]#[784,25,10]:3个层，输入层是784个神经元，中间层是25个神经元，输出层是10个神经元
normalize_data = True#是否标准化数据
max_iterations = 300#最大迭代次数
alpha = 0.1

multilayer_perceptron = MultilayerPerceptron(x_train,y_train,layers,normalize_data)#给类
(thetas,costs)=multilayer_perceptron.train(max_iterations,alpha)#cost是一个list，里面有max_iterations（300）个项
print("行数：",len(thetas[1]))
print("列数：",len(thetas[1][0]))
#返回的thetas是一个字典，字典里包含2个矩阵，第一个矩阵是25行 785列；第二个矩阵是10行26列
plt.plot(range(len(costs)),costs)
plt.xlabel('Gradient steps')
plt.ylabel('costs')
plt.show()

##将模型拿来做预测
y_train_predictions=multilayer_perceptron.predict(x_train)#输出是每个图片的预测数值，0-9之间，由于共有num_training_examples个图片，因此结果是一个行数为num_training_examples（100）行，列数为1的多维数组
y_test_predictions=multilayer_perceptron.predict(x_test)#输出是每个图片的预测数值，0-9之间，由于共有num_training_examples个图片，因此结果是一个行数为num_training_examples（100）行，列数为1的多维数组

##计算准确率
train_p = np.sum(y_train_predictions==y_train)/y_train.shape[0]*100
test_p = np.sum(y_test_predictions==y_test)/y_test.shape[0]*100
print("训练集准确率：",train_p)
print("测试集准确率：",test_p)

numbers_to_display = 64
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(15,15))
for plot_index in range(numbers_to_display):
    digit_label = y_test[plot_index,0]
    digit_pixels = x_test[plot_index,:]
    predicted_label = y_test_predictions[plot_index][0]
    image_size = int(math.sqrt(digit_pixels.shape[0]))

    frame = digit_pixels.reshape((image_size, image_size))

    color_map = 'Greens' if predicted_label == digit_label else 'Reds'

    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap= color_map)
    plt.title(predicted_label)
    plt.tick_params(axis='both',which='both',bottom=False,labelbottom=False,labelleft=False)

plt.subplots_adjust(wspace=0.5,hspace =0.2)
plt.show()
