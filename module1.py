from keras.models import load_model
#模型地址
model_path = 'C:\\Users\\灵\\Desktop\\人工智能\\juanji_resort\\juanji_resort\\data\\basic_cnn_result.h5'
#载入模型
model = load_model(model_path)
 
import os, random
import numpy as np
from PIL import Image
#定义函数读取测试文件夹中的照片
def read_random_image():
    folder = r'C:\\Users\\灵\\Desktop\\人工智能\\juanji_resort\\juanji_resort\\data\\test\\'
    file_path = folder + random.choice(os.listdir(folder))
    print(file_path)
    pil_im = Image.open(file_path, 'r')
    return pil_im
 
#对一个使用模型对读取出的图片进行预测
def get_predict(pil_im, model):
    #对图片进行缩放
    pil_im = pil_im.resize((200,200))
    #将格式转换为 numpy array 格式
    array_im = np.asarray(pil_im)
    array_im = np.expand_dims(array_im, axis=0)
    #对图片进行预测
    result = model.predict(array_im)
    if result[0][0] > 0.5:
        print("预测结果是：狗")
    else:
        print("预测结果是：猫")
 
pil_im = read_random_image()
get_predict(pil_im, model)
pil_im.show(np.asarray(pil_im)) #显示随机选取的照片

