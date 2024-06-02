#导入需要的包
import sys
#import pydot
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
 
#todo:创建一个cnn模型
def define_cnn_model():
    #使用序列模型
    model = Sequential()
    #卷积层
    model.add(Conv2D(32, (3,3), activation="relu",
                    kernel_initializer='he_uniform',
                    padding="same",
                    input_shape=(200,200,3)))
    '''卷积核数量，卷积核维度，激活函数，padding，图片像素200x200'''
    #最大池化层
    model.add(MaxPooling2D((2,2)))
    #Flatten 层
    model.add(Flatten())
    #全连接层
    model.add(Dense(128, activation="relu",kernel_initializer='he_uniform' ))
    model.add(Dense(1, activation="sigmoid"))#输出层0，1，sigmoid模型实现输出值0~1之间，分别代表猫狗
 
    #编译模型
    opt = SGD(lr=0.001, momentum=0.9)#优化器，随机梯度下降，为模型找到最佳的参数
    model.compile(optimizer=opt,
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
    return model
#打印模型图片
from keras.utils import plot_model
model = define_cnn_model()
plot_model(model,
           to_file='cnn_model_basic.png',
           dpi = 100,
           show_shapes=True,
           show_layer_names=True)
 
#训练模型
def train_cnn_model():
    #实例化模型
    model = define_cnn_model()
    #创建图片生成器,产生图片并输入
    datagen = ImageDataGenerator(rescale=1.0 / 225.0)
    train_it = datagen.flow_from_directory(
        'C:\\Users\\灵\\Desktop\\人工智能\\juanji_resort\\juanji_resort\\data\\train',    #'C:\\Users\\Alixy\\Desktop\\ma1ogo3ushu4ju4ji2\\dogs_cats\\data\\train',
        class_mode='binary',
        batch_size=64,  #一次产生并输入64张图片
        target_size=(200, 200)  #缩放图片为200x200，和输入图片大小相同
        )
   #训练模型
    model.fit_generator(train_it,
                        steps_per_epoch=len(train_it),
                        epochs=20,
                        verbose=1 )
    #把模型保存到文件夹
    model.save("C:\\Users\\灵\\Desktop\\人工智能\\juanji_resort\\juanji_resort\\data\\basic_cnn_result.h5")
 
if __name__ == "__main__":
   train_cnn_model()

