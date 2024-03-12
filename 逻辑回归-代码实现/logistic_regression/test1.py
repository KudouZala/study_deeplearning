import numpy as np
#labels=[['a']['b']['a']['c']]  # labels是y_train
#labels=np.mat('a';'b';'a';'c']
labels=np.array([['red'], ['blue'], ['blue'], ['blue'], ['red'], ['green']])
#labels=[['SETOSA'] ['SETOSA'] ['SETOSA'] ['VERSICOLOR'] ['VERSICOLOR']];
unique_labels = np.unique(labels);
print(unique_labels)

for label_index, unique_label in enumerate(unique_labels):  # 输出：0，红色；1，蓝色；2，绿色...
    current_lables = (labels == unique_label).astype(float)
    print(current_lables)



'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression


data = pd.read_csv('../data/iris.csv')
iris_types = ['SETOSA','VERSICOLOR','VIRGINICA']

x_axis = 'petal_length'
y_axis = 'petal_width'

for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class']==iris_type],
                data[y_axis][data['class']==iris_type],
                label = iris_type
                )
plt.show()

num_examples = data.shape[0]
x_train = data[[x_axis,y_axis]].values.reshape((num_examples,2))
y_train = data['class'].values.reshape((num_examples,1))
print(y_train)
print(type(y_train))
'''