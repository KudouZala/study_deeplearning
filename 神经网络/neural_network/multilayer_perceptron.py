
##导入库
import numpy as np
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid, sigmoid_gradient
#from sklearn.neural_network import multilayer_perceptron

##编写类-MultilayerPerceptron多层感知机
class MultilayerPerceptron:
    #类的初始化一定要注意init前后是两个__,data-数据；labels-标签；layers-层[784,25,10]
    def __init__(self,data,labels,layers,normalize_data =False):
        data_processed = prepare_for_training(data,normalize_data = normalize_data)[0]#num_training_examples（100）行785列
        self.data = data_processed#num_training_examples（100）行785列，100行表示100个图像，每个图像有784个数据点，wx+b*1，因此偏置项是100行 1列。w（包含b）是25行785列，数据预处理中，由于考虑到后面要有偏置项，因此在多加了一列1（加到了第一列）
        self.labels = labels#100行1列
        self.layers = layers #784 25 10
        self.normalize_data = normalize_data#True
        self.thetas = MultilayerPerceptron.thetas_init(layers)#thetas类似权重w，但是需要包含偏置项，因此返回的thetas是字典，由于总共是3层[784 25 10]，因此有2个index：0和1,0对应25*785的矩阵，1对应10*26的矩阵

    #预测函数，用于将训练好的模型拿来预测，因此函数的输入是数据，输出是预测的分类0-9中间的一个数字
    def predict(self,data):
        data_processed=prepare_for_training(data,normalize_data=self.normalize_data)[0] # 对输入的数据进行预处理
        num_examples = data_processed.shape[0]# data_processed是100行785列，因此获取数据的数量：100
        predictions = MultilayerPerceptron.feedforward_propagation(data_processed,self.thetas,self.layers)# 100行10列，调用feedforward_propagation静态方法进行前向传播计算，得到预测结果。
        leibie=np.argmax(predictions,axis=1).reshape((num_examples,1))#100行1列 返回预测结果，这里使用np.argmax取得每行预测概率最大的值的索引，即预测的类别0-9其中的一个数字所对应的索引值，索引值对应的就是0-9的真实数字值。
        return  leibie

    #训练函数，用于训练模型，输入的是：最大迭代次数和学习率
    def train(self, max_iterations = 100, alpha = 0.1):
        unrolled_theta = MultilayerPerceptron.thetas_unroll(self.thetas)# 对权重参数进行展开操作，因为优化算法通常需要将多个参数合并成一个长向量。
        """
        在机器学习和深度学习中，神经网络通常由多个层构成，每个层都有自己的参数，或称为权重(weight)和偏置(bias)。这些参数在实现时往往是以多维数组（通常是矩阵）的形式存储。在优化过程中，为了方便进行数学运算（例如梯度下降），有时需要将这些多维数组“展开”成一个长的一维数组。这样做可以将所有参数视为一个长向量，从而简化与参数相关的计算，比如梯度计算、参数更新等。
        在此上下文中，MultilayerPerceptron.thetas_unroll 方法的作用可能是将 self.thetas，即多个矩阵表示的网络参数，展开成一个一维数组。unrolled_theta 将是一个包含所有参数的一维数组，这样就可以传递给优化器进行优化。
        例如，如果你有一个三层的神经网络，每层的参数是：
        第一层 Theta1 形状为 (m, n)
        第二层 Theta2 形状为 (o, p)
        第三层 Theta3 形状为 (q, r)
        调用 thetas_unroll 方法可能会得到一个形状为 (mn + op + q*r, ) 的一维数组 unrolled_theta。
        """
        (optimized_theta,cost_history)=MultilayerPerceptron.gradient_descent(self.data, self.labels, unrolled_theta, self.layers,max_iterations,alpha)  # 调用gradient_descent静态方法进行梯度下降训练。
        #这里需要注意：我们传入的unrolled_theta仅仅是一个初始化的权重，而非最终的结果
        #optimized_theta是1行，19885列，layers是[784,25,10]
        self.thetas = MultilayerPerceptron.thetas_roll(optimized_theta,self.layers)# 对优化后的权重参数进行卷积操作，转换回原来的形状。

        return self.thetas,cost_history
        #返回的thetas是一个字典，字典里包含2个矩阵，第一个矩阵是25行 785列；第二个矩阵是10行26列
        # cost是一个list，里面有max_iterations（300）个项

    @staticmethod
    def thetas_init(layers):#[784,25,10]，根据layers返回需要的thetas的字典，该字典包含权重矩阵
        num_layers = len(layers)#层的个数，3个层
        thetas = {}#定义为字典
        for layer_index in range(num_layers - 1):#若为3层则layer_index遍历0，1
            """
            会执行两次，得到两组参数矩阵：25*785 ，10*26
            """
            in_count = layers[layer_index]#第一遍是784.第二遍是25
            out_count = layers[layer_index+1]#第一遍是25，第二遍是10
            #需要考虑到偏置项，因此是785，而不是784
            thetas[layer_index] = np.random.rand(out_count,in_count+1)*0.05#参数w初始化,可见onenote，随机进行初始化操作，参数值尽量小一点,randn函数返回一个或一组样本，具有标准正态分布。返回值为指定维度的array
            """
            在神经网络的初始化过程中，thetas[layer_index] = np.random.rand(out_count, in_count+1)*0.05这行代码的作用是初始化给定层的权重。这里，np.random.rand(out_count, in_count+1)生成一个形状为(out_count, in_count+1)的数组，其元素值在[0, 1)范围内均匀分布，其中out_count是输出节点的数量，in_count+1是输入节点的数量加上一个偏置项。乘以0.05的操作是为了实现以下几个目的：
            缩小权重的范围：直接从[0, 1)范围内生成的权重值可能对于神经网络的初始化来说太大了。较大的权重值可能导致神经网络在训练初期时的激活函数输出值过大，这会使得梯度更新过程中出现梯度消失或梯度爆炸的问题，从而影响训练的稳定性和效率。通过乘以0.05，权重的范围被限制在[0, 0.05)，这有助于避免这类问题。
            打破对称性：在网络初始化时给权重赋予小的随机值，还有一个目的是为了打破对称性。如果所有权重都初始化为相同的值（例如0），那么在反向传播过程中每个节点的梯度更新将会相同，导致所有权重都以相同的方式更新。这会使得神经网络的每一层实际上学不到不同的特征，从而影响学习的效果。通过赋予随机的小值，每个权重都有了微小的差异，这有助于网络学习到更丰富的特征。
            促进快速学习：较小的权重初始化值有助于保持网络激活函数的输出在其线性区间内（对于Sigmoid或Tanh激活函数而言），这使得梯度较大，进而有利于网络在训练初期的快速学习。
            总结来说，*0.05的作用是为了初始化权重到一个较小的值范围内，这种做法有助于提升神经网络训练的稳定性和效率。需要注意的是，0.05只是一个例子，实际选择的缩放因子可能根据具体情况以及网络的深度、激活函数等因素有所不同。
            """
        return thetas#返回的thetas是字典，2个index：0和1,0对应25*785的矩阵，1对应10*26的矩阵

    @staticmethod
    def thetas_unroll(thetas):
        """
        在机器学习和深度学习中，神经网络通常由多个层构成，每个层都有自己的参数，或称为权重(weight)和偏置(bias)。这些参数在实现时往往是以多维数组（通常是矩阵）的形式存储。在优化过程中，为了方便进行数学运算（例如梯度下降），有时需要将这些多维数组“展开”成一个长的一维数组。这样做可以将所有参数视为一个长向量，从而简化与参数相关的计算，比如梯度计算、参数更新等。
        在此上下文中，MultilayerPerceptron.thetas_unroll 方法的作用可能是将 self.thetas，即多个矩阵表示的网络参数，展开成一个一维数组。unrolled_theta 将是一个包含所有参数的一维数组，这样就可以传递给优化器进行优化。
        例如，如果你有一个三层的神经网络，每层的参数是：
        第一层 Theta1 形状为 (m, n)
        第二层 Theta2 形状为 (o, p)
        第三层 Theta3 形状为 (q, r)
        调用 thetas_unroll 方法可能会得到一个形状为 (mn + op + q*r, ) 的一维数组 unrolled_theta。
        """
        num_theta_layers = len(thetas)#计算thetas有多少个,由于thetas是字典，0和1，因此是2
        unrolled_theta = np.array([])
        for theta_layer_index in range(num_theta_layers):
            unrolled_theta = np.hstack((unrolled_theta, thetas[theta_layer_index].flatten()))#一行，25*785+10*26=19885列；
            """
            flatten()方法: flatten()是NumPy数组对象的一个方法，它会返回一个包含数组中所有元素的一维数组，并且这些元素按行优先顺序排列。如果thetas[theta_layer_index]是一个多维数组（通常是二维的，代表一层网络中的权重），.flatten()会将它转换成一维数组。
            np.hstack()函数: np.hstack()是NumPy库中的一个函数，用于沿着水平轴（即列方向）拼接数组。给定一个数组的元组或列表作为输入，它会返回一个由所有输入数组在水平方向上拼接而成的数组。
            """
        return unrolled_theta#返回一个矩阵：一行，25*785+10*26=19885列；

    @staticmethod
    def gradient_descent(data, labels,unrolled_theta,layers,max_iterations,alpha):
        """
        第一步：计算损失值
        第二步：计算梯度值
        """
        optimized_theta = unrolled_theta#要更新的东西-权重w和偏执b组成的矩阵，一个矩阵：一行，25*785+10*26=19885列；
        cost_history = []#存储损失值，以便后期画图
        for _ in range(max_iterations):
            #下面那句用于计算当前的损失值
            cost = MultilayerPerceptron.cost_function(data,labels,MultilayerPerceptron.thetas_roll(optimized_theta,layers),layers)
            cost_history.append(cost)#将当前的损失值保存进列表中
            #gradient_step方法的目的是计算损失函数关于当前权重optimized_theta的梯度。梯度实质上是一个向量，它指向损失函数上升最快的方向，因此，负梯度方向指向的是损失函数下降最快的方向。
            theta_gradient = MultilayerPerceptron.gradient_step(data,labels,optimized_theta,layers)
            #返回一个矩阵1行19885列
            #下面这行代码执行了梯度下降步骤，通过在权重上减去梯度乘以一个学习率alpha来更新权重。这里的操作包括：
            optimized_theta = optimized_theta - alpha* theta_gradient
            #从上面的话，看出alpha学习率的作用就是调整每次w权重更新的大小，alpha越小，更新会越慢，结果也会越精准，但是要更新的次数就会增多；较大的数值则可能导致更新过快而错过最小值
        return optimized_theta,cost_history

    @staticmethod
    def gradient_step(data,labels,optimized_theta,layers):
        theta = MultilayerPerceptron.thetas_roll(optimized_theta,layers)#返回的是字典，两个键值对，是thetas_unroll的反向过程，目的就是将一行，25*785+10*26=19885列的矩阵变回两个矩阵，第一个矩25*785的矩阵，第二个矩阵10*26的矩阵
        a3=len(theta[0][0])#785
        b3=len(theta[0])#25
        thetas_rolled_gradients = MultilayerPerceptron.back_propagation(data,labels,theta,layers)#返回一个字典，2个键值对，对应2个矩阵：两个矩阵，第一个矩25*785的矩阵，第二个矩阵10*26的矩阵
        a4=len(thetas_rolled_gradients[0][0])#785
        b4=len(thetas_rolled_gradients[0])#25
        thetas_unrolled_gradients=MultilayerPerceptron.thetas_unroll(thetas_rolled_gradients)
        return thetas_unrolled_gradients#返回的是1行19885列

    @staticmethod
    def back_propagation(data, labels, thetas, layers):
        num_layers = len(layers)#3
        (num_examples,num_features) = data.shape#100个图片 每个图片有785个特征（像素点）
        num_label_types = layers[-1]#10
        deltas = {}#设置一个字典，用于存储每一层对结果的影响

        # 初始化梯度矩阵
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]# 当前层神经元数量
            out_count = layers[layer_index + 1] # 下一层神经元数量
            # 为每层的梯度设置0矩阵，25*785（隐藏层到输入层），10*26（输出层到隐藏层）
            deltas[layer_index] = np.zeros((out_count, in_count + 1))  # 25*785 10*26

        # 对每个样本进行循环
        for example_index in range(num_examples):#下面的计算要注意只涉及到1个样本：
            # 初始化存储每层的输入和激活的字典
            layers_inputs = {}
            layers_activations ={}
            # 将当前样本的输入重塑为列向量
            layers_activation = data[example_index, :].reshape((num_features, 1))# 785*1
            # 将输入层的值设置为当前样本特征
            layers_activations[0] = layers_activation#785行一列

            # 前向传播
            for layer_index in range(num_layers - 1):
                # 获取当前层的权重
                layer_theta = thetas[layer_index]#第一次是25*785，第二次是10*26
                # 计算当前层的输入，即前一层与权重矩阵的乘积然后激活值
                #明确概念，我们有3层，输入层 中间层 输出层，中间层的输入是输入层与权重矩阵的乘积然后激活值；输出层的输入是中间层结果与第二个权重矩阵的乘积然后激活值
                layer_input = np.dot(layer_theta, layers_activation)#layer_input第一次是25行1列，第二次是10行1列的矩阵
                # 应用sigmoid函数并添加偏置单元（添加了1作为偏置单元）
                layers_activation = np.vstack((np.array([[1]]), sigmoid(layer_input)))#layers_activation第一次是26行1列，第二次是11行1列的矩阵
                # 存储当前层的输入和激活值(经过激活函数后的输出结果）
                layers_inputs[layer_index + 1] = layer_input#即，当layer_index为0时候，layers_inputs[1]=25行1列的矩阵；当layer_index为1时候，layers_inputs[2]=10行1列的矩阵
                layers_activations[layer_index + 1] = layers_activation#layers_activatio是11行1列的矩阵
            #layers_activations[0]是输入层的值
            #layers_activations[1]是输入层与w的乘积的激活值（以后简称中间层的激活值）
            #layers_activations[2]是中间层与第二个w的乘积的激活值（以后简称输出层的激活值）

            # 下句是计算输出层的激活值（去掉偏置单元）
            output_layer_activation = layers_activation[1:, :]#10行1列的矩阵，即前向传播计算出的当前结果矩阵
            #通过上述前向传播，我们得到最终的输出结果矩阵，其中是每个0-9分类的概率值
            """
            在包含反向传播代码的完整程序中，可能不直接调用独立定义的前向传播函数（feedforward_propagation），主要出于以下几个考虑：
            1. 数据结构和中间结果的保存：
            在反向传播过程中，需要利用前向传播的中间结果（每层的激活值和线性组合值）来计算梯度。如果使用独立的前向传播函数，这些中间结果需要被保存和传递到反向传播函数中，这可能要求对前向传播函数进行修改，以输出这些中间结果。在提供的反向传播代码中，通过在反向传播循环内部直接进行前向传播计算，可以即时生成并使用这些中间结果，从而简化了代码结构和流程。
            2. 计算效率：
            直接在反向传播函数中计算前向传播的相关值可以减少函数调用的开销，尤其是当反向传播函数需要对每个样本独立进行计算时。这种方法避免了重复的数据传递和函数调用，可能会在一定程度上提高整体的计算效率。
            3. 定制化计算需求：
            反向传播过程可能只需要部分前向传播的结果，或者需要以一种特定方式使用这些结果（例如，添加偏置项、重新排列数据等）。将前向传播的计算直接嵌入到反向传播函数中，可以根据需要灵活调整计算过程，而不是受到独立前向传播函数接口和返回值的限制。
            4. 代码的可读性和维护性：
            虽然重复代码通常被视为不佳的编程实践，但在某些情况下，直接在反向传播中实现必要的前向传播步骤可以使整个训练过程的逻辑更加清晰和连贯。这样做可以帮助读者或使用者更好地理解模型的工作流程，尤其是在教育或演示的上下文中。
            尽管如此，是否将前向传播作为独立函数调用，还是直接在反向传播中实现，这在很大程度上取决于具体的应用场景、性能要求和个人编程风格。在某些复杂的模型中，为了代码的模块化和重用性，可能会优先选择调用独立的前向传播函数。
            """
            # 初始化用于存储每层误差的字典
            delta = {}
            #标签处理：将标签转换为one-hot编码
            bitwise_label = np.zeros((num_label_types, 1))
            bitwise_label[labels[example_index][0]] = 1#10行1列的矩阵，比方说现在是在第一个样本的循环中，由于第一个样本的真实值是9，那么第9行是1，其余行是0
            # 计算输出层和真实值的误差
            delta[num_layers-1] = output_layer_activation - bitwise_label
            #delta[2]输出层和真实值的误差
            # delta[1]中间层和输出层的误差
            """
            在这段代码中，损失函数本身并没有直接体现出来，因为它主要关注于损失函数相对于网络参数（权重）的梯度计算。然而，通过分析这段代码，我们可以从以下几个方面推断损失函数可能是交叉熵损失函数：
            损失函数的梯度计算： 在反向传播的最后一层（输出层），代码计算了delta[num_layers - 1]。这个delta代表输出层的误差，定义为output_layer_activation - bitwise_label。对于交叉熵损失函数且激活函数为sigmoid或softmax（适用于分类问题），这个梯度的形式是预测概率（经激活函数之后的值）与实际标签的差。这与交叉熵损失函数求导的结果一致。
            标签的one-hot编码： 代码中将标签转换为one-hot编码向量bitwise_label。在多分类问题中，one-hot编码通常与交叉熵损失函数结合使用，以计算每个类别的独立损失，并在所有类别上求和。
            激活函数的导数： 在反向传播的过程中，代码涉及sigmoid_gradient(layer_input)，这是sigmoid激活函数的导数。交叉熵损失函数在与sigmoid激活函数结合时，求导后的形式通常会涉及激活函数的导数。
            即便如此，要确切知道这段代码的损失函数是什么，需要查看整个训练过程中如何计算损失值和如何调用这个back_propagation函数的上下文。通常，在神经网络的训练过程中，会首先通过前向传播和损失函数计算当前参数下的损失值，然后通过反向传播计算损失函数相对于每层参数的梯度，以便更新这些参数。
            此代码片段并没有提供损失值的计算，仅仅是实现了反向传播算法的一部分，即计算梯度的部分，而损失值的计算可能在其他部分的代码中。
            """
            # 反向传播
            for layer_index in range(num_layers - 2, 0, -1):#若num_layers=3,那么相当于从num_layers - 2降序到0，但又不包含0，因此只能为1
                # 获取当前层的权重，比方说第一次循环，layer_index是1，即获取中间层到输出层的w权重
                layer_theta = thetas[layer_index]
                # 获取下一层的误差，比方说第一次循环，layer_index是1，即获取输出层和真实值的误差
                next_delta = delta[layer_index + 1]
                # 获取当前层的输入并添加偏置单元，比方说第一次循环，layer_index是1，即：中间层的输入是输入层与权重矩阵的乘积然后激活值，25行1列的矩阵
                layer_input = layers_inputs[layer_index]
                layer_input = np.vstack((np.array((1)), layer_input))#26行1列的矩阵
                # 计算当前层的误差，若layer_index=1,则是计算中间层和输出层之间的误差
                delta[layer_index] = np.dot(layer_theta.T, next_delta) * sigmoid_gradient(layer_input)
                # 去掉偏置单元的误差
                delta[layer_index] = delta[layer_index][1:, :]
            # 更新梯度
            for layer_index in range(num_layers - 1):
                # 计算梯度
                layer_delta = np.dot(delta[layer_index + 1],layers_activations[layer_index].T)
                # layers_activations[0]是输入层的值a1(a1的输入）
                # layers_activations[1]是输入层与w的乘积的激活值（以后简称a2的激活值）（我个人也叫a2的输入）
                # layers_activations[2]是中间层与第二个w的乘积的激活值（以后简称a3的激活值）（我个人也叫a3的输入）
                # delta[2]输出层a3和真实值的误差(简称为a3的误差）
                # delta[1]中间层a2和输出层a3的误差（简称为a2的误差）
                #在神经网络中，每一层的激活值是指该层的输入值经过线性变换（如权重矩阵的乘法和偏置的加法）后，再通过激活函数得到的值。所以，当我们说“隐藏层的激活值a2”，我们是指输入层的数值与w1相乘（并添加偏置，如果有的话）后再取激活函数的结果。
                """
                这行代码layer_delta = np.dot(delta[layer_index + 1],layers_activations[layer_index].T)的目标是计算损失函数相对于第layer_index+1层权重的梯度。
                在神经网络中，权重更新的规则是依据损失函数相对于权重的梯度，通过反向传播算法来计算。具体来讲，我们需要计算出在某次迭代中，每一层的权重对应的损失函数的梯度。
                对于该行代码，delta[layer_index + 1]表示第layer_index + 1层的误差，layers_activations[layer_index].T表示第layer_index层的激活值（经过激活函数的输出）的转置。对于全连接层，误差delta与前一层的激活值的矩阵乘积（np.dot）就是损失函数相对于这一层权重的梯度。
                这个计算过程的直观理解是，我们需要知道在第layer_index+1层，每个神经元的误差对于前一层layer_index的每个神经元通过多大程度反馈，即layer_index层神经元的激活值对误差的贡献，这个贡献程度就是权重的梯度。
                这个梯度将在优化算法（如梯度下降或Adam等）中用于更新第layer_index+1层的权重，以期待在下一次迭代中减小损失函数的值。
                """

                """
                根据反向传播的原理，三层分别为a1 a2 a3 ，
                a1 a2之间是w1，a2与a3之间是w2，
                如果要对w1更新，需要依靠a2的误差和a1的输入;
                如果要对w2更新，需要依靠a3的误差和a2的输入;
                """
                # 累加每个样本的梯度
                deltas[layer_index] = deltas[layer_index] + layer_delta  # 第 25 785
        # 对梯度进行平均，得到每个样本的平均梯度
        for layer_index in range(num_layers-1):
            deltas[layer_index]=deltas[layer_index]*(1/num_examples)
        a5=len(deltas[1])
        b5 = len(deltas[1][0])
        #返回一个字典，2个键值对，对应两个矩阵元素：25*785和10*26

        return deltas # 返回计算得到的梯度




    @staticmethod
    def cost_function(data, labels, thetas, layers):
        num_layers = len(layers)#结果是3
        num_examples = data.shape[0]#样本的个数（行数），即100行，即100个图片
        num_labels = layers[-1]#结果是10，即表示labels有几种，由于咱们做的相当于是10分类问题，故为10
        a2=thetas

        #下句是做前向传播
        predictions=MultilayerPerceptron.feedforward_propagation(data, thetas, layers)#返回的是100行10列 的矩阵，从这里可以看出，这是一个典型的分类问题，输出层结果表示的意思是当前时候，这100张图片，每张图片属于0-9的概率值，哪个概率值大，就认为是哪个
        #通过前向传播，得到当前的输出层结果，输出层是一个100行10列的矩阵，即表示每个照片属于0-9十分类问题中每个分类的概率值
        #前向传播之后的结果，我们需要处理一下，我们最终的目的是想将正确的那个分类的概率值接近1，其余接近0
        #根据上面那句话的分析，我们得出一个结论：我们要创建两个cost损失函数，一个用来评估正确结果是否接近1，一个用来评估错误结果是否接近0
        #下面这句话我们先初始化我们的输出的正确结果矩阵
        bitwise_labels = np.zeros((num_examples,num_labels))#100行10列的矩阵，元素全是0
        for example_index in range(num_examples):#从0-99一个一个遍历
            bitwise_labels[example_index][labels[example_index][0]] = 1
            #第一次是第0行，列是labels[0][0]，即应该所属的正确的那个分类命为1
            #第二次是第1行，列是labels[1][0]，即应该所属的正确的那个分类命为1
        a = predictions[bitwise_labels == 1]#1行100列的矩阵，将predictions（当前输出结果）中的每个图片的正确位置的数字提取出来
        b= np.log(a)#b是1行100列的矩阵，np.loge
        bit_set_cost = np.sum(np.log(predictions[bitwise_labels == 1]))#该结果用来评估我们正确的那1个数是否接近1由于取了对数，因此最后np.sum后的数字应该是越接近0越好。
        #predictions是100行1列的矩阵，即当前的输出结果矩阵
        #bit_set_cost是一个float类型
        bit_not_set_cost = np.sum(np.log(1 - predictions[bitwise_labels == 0]))#该结果用来评估我们错误的那9个数是否接近0，由于先用1-，再取了对数，因此最后np.sum后的数字应该是越接近0越好。
        # bit_not_set_cost是一个float类型
        #下面那句话的目的是为了将损失求一个平均，因此/num_examples（100）,
        cost = (-1 / num_examples)*(bit_set_cost+bit_not_set_cost)
        """
        这段代码是计算二分类交叉熵损失（binary cross-entropy loss）的过程，这是神经网络分类任务中常用的损失函数。交叉熵损失函数在二分类问题中的公式为：
        loss = -[y*log(p) + (1 - y) * log(1 - p)]
        其中，y是真实的标签（0或1），p是模型预测的概率（在0和1之间）。
        这段代码中，bit_set_cost和bit_not_set_cost分别对应于y*log(p)和(1-y)*log(1-p)。而-1的作用是因为日志函数的值域为负，对求和的结果取负号可以使得损失值为正。
        下面是这个损失函数公式的详细解释：
        np.log(predictions[bitwise_labels == 1])计算的是y*log(p)的部分，其中y等于1，对应于正类的样本。
        np.log(1 - predictions[bitwise_labels == 0])计算的是(1-y)*log(1-p)的部分，其中y等于0，对应于负类的样本。
        np.sum()函数对所有样本的损失值进行求和。
        对求和结果取负号（*-1）的目的是使损失值为正，因为日志函数的值域为负。
        最后，/ num_examples实现对损失值的平均，得到的是每个样本的平均损失。
        至于为什么使用对数函数，主要有两个原因：
        1数学上的便利性：对数函数可以转化乘法为加法，从而简化梯度计算。
        2提高模型性能：对数损失对分类错误的惩罚更大，这使得模型在训练过程中更加关注错误分类的样本，从而提高模型的性能。
        """
        return cost

    @staticmethod
    #下面这段是前向传播的代码
    def feedforward_propagation(data, thetas, layers):
        num_layers=len(layers)#结果是3
        num_examples = data.shape[0]#结果是100，即100个样本图片
        in_layer_activation = data#100行785列的矩阵

        for layer_index in range(num_layers - 1):#通过两层中间层（两次:sigmoid(wx+b），最终得到输出层的结果
            theta = thetas[layer_index]#第一次是25行785列的矩阵；第二次是10行26列的矩阵
            #下图我们的激活函数使用sigmoid函数，np.dot是矩阵乘法
            out_layer_activation = sigmoid(np.dot(in_layer_activation, theta.T))#第一次是100行25列；第二次是100行10列的矩阵
            """
            np.dot:矩阵与矩阵相乘需要满足矩阵的乘法原则，即A=mXn，B=nXp，C=AXB=mXp。A的列数等于B的行数。
            """
            # 正常计算完之后是num_examples*25.但是要考虑偏置项变成num_examples*26
            out_layer_activation = np.hstack((np.ones((num_examples, 1)), out_layer_activation))#第一次是100行26列，第一列全是1；第二次是100行11列的矩阵，第一列全是1
            in_layer_activation = out_layer_activation#第一次是100行26列，第一列全是1；第二次是100行11列的矩阵，第一列全是1
        #返回输出层结果，结果不要偏置项
        return in_layer_activation[:, 1:]#返回的是100行10列 的矩阵，从这里可以看出，这是一个典型的分类问题，输出层结果表示的意思是当前时候，这100张图片，每张图片属于0-9的概率值，哪个概率值大，就认为是哪个

    @staticmethod
    def thetas_roll(unrolled_thetas, layers):#是thetas_unroll的反向过程，目的就是将一行，25*785+10*26=19885列的矩阵变回两个矩阵，第一个矩25*785的矩阵，第二个矩阵10*26的矩阵
        num_layers = len(layers)
        thetas = {}
        unrolled_shift = 0
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            thetas_width = in_count + 1
            thetas_height = out_count
            thetas_volume = thetas_width * thetas_height
            start_index = unrolled_shift
            end_index = unrolled_shift + thetas_volume
            layer_theta_unrolled = unrolled_thetas[start_index:end_index]
            thetas[layer_index] = layer_theta_unrolled.reshape((thetas_height, thetas_width))
            unrolled_shift = unrolled_shift+thetas_volume
        return thetas








