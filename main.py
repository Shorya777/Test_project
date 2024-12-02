from tensorflow.keras.datasets import mnist
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

temp = np.zeros((y_train.size, y_train.max()+1))
temp[np.arange(y_train.size), y_train] =1
y_train = temp

temp = np.zeros((y_test.size, y_test.max()+1))
temp[np.arange(y_test.size), y_test] =1
y_test= temp

print(X_train.shape, y_train.shape)
print(np.min(X_train), np.max(X_train)) #to check the range of data

X_train = X_train/255.0
X_test = X_test/255.0

import matplotlib.pyplot as plt

for i in range(9):
    plt.subplot(330+1+i)
    plt.imshow(X_train[i], cmap= plt.get_cmap('gray'))
plt.show()


from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Sequential

model = Sequential([
    Flatten(input_shape = (28,28)), 
    Dense(50, activation = 'relu'),
    Dense(50, activation = 'relu'),
    Dense(50, activation = 'relu'),
    Dense(50, activation = 'relu'),
    Dense(10, activation = 'softmax')
    ])

model.compile(optimizer = 'adam', loss = 'CategoricalCrossentropy', metrics= ['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=100)

result = model.evaluate(X_test, y_test)
print("test_loss, test_acc", result)

from tensorflow.keras.models import save_model

path = 'trained_model.h5'

save_model(model, path, overwrite= True)
print("saved model")

X_test = X_test/255.0
temp = np.zeros((y_test.size, y_test.max()+1))
temp[np.arange(y_test.size), y_test] =1
y_test= temp

X_test = X_test*255.0
from tensorflow.keras.models import load_model
model = load_model('trained_model.h5')
result = model.evaluate(X_test, y_test, batch_size =32)
print("test_loss, test_acc", result)

