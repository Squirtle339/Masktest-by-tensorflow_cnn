import os
import glob
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
 
 
# 图片尺寸，在图片裁剪，网络输入层会用到
input_shape = (128, 128, 3)
# 初始学习率
learning_rate = 0.001
# 学习率衰减值
lr_decay = 0.1
# 训练集占总数据集比例
ratio1 = 0.9
# 从训练集中拆分验证集，验证集占的比例
ratio2 = 0.1
# 批处理一次的图片数，一次输入128张图片到神经网络计算
batch_size = 128
# 训练迭代次数
epochs = 12
 
def MODEL():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu", input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))
    model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))
    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))
    model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))
    model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))
    model.add(Flatten())
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(units=2, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=lr_decay),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    #tf.keras.utils.plot_model(model,to_file="model",show_layer_names=True,show_shapes=True)
    return model
 
 
def split():
    # 整个dirdata是你的整个数据集的根目录
    dir_data = r"../input/mask-images/Dataset"
    # 下面三个路径就分别是我的三种数据的目录，如果你的数据只有戴口罩核没带口罩两种，那就删一个就行
    dir_mask = os.path.join(dir_data, 'with_mask')
    dir_nomask = os.path.join(dir_data, 'without_mask')
    dir_nomask_incorrect = os.path.join(dir_data, "mask_weared_incorrect")
    # 遍历文件及，将每一个图片的地址依次存入列表中
    path_mask = [os.path.abspath(fp) for fp in glob.glob(os.path.join(dir_mask, '*.png'))]
    path_nomask = [os.path.abspath(fp) for fp in glob.glob(os.path.join(dir_nomask, '*.png'))]
    # 把未正确佩戴口罩的图片加入到没带口罩的类别
    path_nomask = path_nomask + [os.path.abspath(fp) for fp in glob.glob(os.path.join(dir_nomask_incorrect, '*.png'))]
    path_all = path_mask + path_nomask
    # 生成标签，1代表佩戴了口罩，0代表未佩戴口罩
    label_all = [1] * len(path_mask) + [0] * len(path_nomask)
    # 十字交叉分割，把所有的图片路径和标签通过shuffle=True打乱顺序，然后分ratio1比例的给训练集图片和训练集标签，剩下的作为测试集
    path_train, path_test, label_train, label_test = train_test_split(path_all, label_all, shuffle=True, train_size=ratio1)
    return path_train, path_test, label_train, label_test
 
 
def createset(path, label):
    data_set = []
    label_set = []
    # 遍历路径列表中的路径以及标签列表里的标签
    for (one_path, one_label) in zip(path,label):
        # cv2读取图片为numpy矩阵
        image = cv2.imread(one_path)
        # 缩放图片为指定大小，由于我的数据集是已经全部缩放到了128*128*3，这里就不再需要了
        # image_resize = cv2.resize(image, (128, 128), cv2.INTER_AREA)
        
        # cv2读取的矩阵值在0-255，我们选择除以255将图片归一化
        image = image/255.
        data_set.append(image)
        label_set.append(one_label)
    # 转为矩阵形式返回
    return np.array(data_set), np.array(label_set)
 
 
def drawloss():
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), valid_loss, label='test_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
 
 
def drawacc():
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), valid_accuracy, label='test_accuracy',)
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()
 
if __name__ == '__main__':
    path_train,  path_test, label_train,  label_test = split()
    data_train,label_train = createset(path_train, label_train)
    data_test, label_test = createset(path_test, label_test)
    print("训练集图片数：",int(data_train.shape[0]*0.9))
    print("验证集图片数：",int(data_train.shape[0]*0.1))
    print("测试集图片数：",data_test.shape[0])
    # 初始化网络模型
    model = MODEL()
    # 进行训练
    output = model.fit(x=data_train,
                       y=label_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       validation_split=ratio2
                      )
    # 保存网络模型
    model.save("mask.h5")
    # 可视化训练过程
    history_predict = output.history
    train_loss = history_predict['loss']
    train_accuracy = history_predict['accuracy']
    valid_loss = history_predict['val_loss']
    valid_accuracy = history_predict['val_accuracy']
    drawacc()
    drawloss()
    # 计算测试集准确率 
    acc = model.evaluate(data_test,label_test, batch_size=batch_size, verbose=1)
    print("test accuracy:", acc[1])

————————————————
版权声明：本文为CSDN博主「我不会写BUG」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_45981224/article/details/120526885
