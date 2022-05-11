#載入套件
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.callbacks import Callback,EarlyStopping
import numpy

# binary encode numbers
num_digits = 10 
# 四種類型：number/fizz/buzz/fizzbuzz
nb_classes = 4 
# batch_size 每一批次訓練筆數
batch_size = 128

# 設定題目規則
# 將每個十進位數轉換成二制進數組，比如 100 => [ 0,0,0,1,1,0,0,1,0,0] ,共10維。
# 其對應的標籤為：
# [0,0,0,1] => 'Fizz'
# [0,0,1,0] =>'Buzz'
# [0,1,0,0] =>'FizzBuzz'
# [1,0,0,0] =>'其它'共4維


def fb_encode(i):
    if   i % 15 == 0: return [3]
    elif i % 5  == 0: return [2]
    elif i % 3  == 0: return [1]
    else:             return [0]

def bin_encode(i):
    return [i >> d & 1 for d in range(num_digits)]

def fizz_buzz_pred(i, pred):
    return [str(i), "fizz", "buzz", "fizzbuzz"][pred.argmax()]


def fizz_buzz(i):
    if   i % 15 == 0: return "fizzbuzz"
    elif i % 5  == 0: return "buzz"
    elif i % 3  == 0: return "fizz"
    else:             return str(i)

# 劃分訓練集跟測試集
def create_dataset(start, end):
    dataX,dataY = [],[]
    for i in range(start,end):
        dataX.append(bin_encode(i))
        dataY.append(fb_encode(i))
    return numpy.array(dataX),np_utils.to_categorical(numpy.array(dataY), nb_classes)


x_train,y_train = create_dataset(101,1001)
x_test,y_test = create_dataset(1,101)


#訓練集
x_train.shape
x_train[0]

#測試集
y_train.shape
y_train[0]

#建立 keras的Sequential model
model = Sequential()

# 建立dense層，輸入為10個神經元，輸出為1000個神經元
# Activation Function為  relu
model.add(Dense(input_dim=10, units=1000))
model.add(Activation('relu'))

# 建立dense層，輸入為上層的輸出，輸出為4個神經元
# Activation Function為 softmax
model.add(Dense(units=4))
model.add(Activation('softmax'))

#定義模型訓練方式：
#設定loss function
#optimizer 設定優化器
#設定評估模型的方式為 accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#設定訓練資料參數
#設定訓言資料與驗證資料比例
#設定epoch訓練週期
model.fit(x_train,y_train,batch_size=20, nb_epoch=100)

#預測準確率
result = model.evaluate(x_test,y_test)
score = model.evaluate(x_train,y_train)
print('\nTrain Acc:', score[1])
result = model.evaluate(x_test,y_test)
print('\nTest Acc:', result[1])

