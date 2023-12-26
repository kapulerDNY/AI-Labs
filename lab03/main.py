import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

data = pd.read_csv('data.csv')


X = data.iloc[:, 1:].values / 255.0  # Нормализация значений пикселей
y = data.iloc[:, 0].values


y = to_categorical(y, num_classes=10)

#Определение размера тестовой выборки
test_size = int(0.2 * len(X))


indices = np.random.permutation(X.shape[0])

#Индексы для обучающей и тестовой выборок
train_indices, test_indices = indices[test_size:], indices[:test_size]


X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]


model = Sequential()
model.add(Dense(128, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

correctly_classified_indices = []
misclassified_indices = []


model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

predictions = model.predict(X_test)

for i in range(len(X_test)):
    predicted_label = np.argmax(predictions[i])
    true_label = np.argmax(y_test[i])

    if predicted_label == true_label:
        correctly_classified_indices.append(i)
    else:
        misclassified_indices.append(i)

#где правильно распознаны 10 штук и соотвественно там где ошибка тоже 10 штук
random_correct_indices = np.random.choice(correctly_classified_indices, size=10, replace=False)
for idx in random_correct_indices:
    true_label = np.argmax(y_test[idx])
    predicted_label = np.argmax(predictions[idx])
    #print(f'Value {true_label} was ocred {predicted_label}')
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'Value {true_label} was ocred {predicted_label}')
    plt.show()

#где произошла ошибка
random_misclassified_indices = np.random.choice(misclassified_indices, size=10, replace=False)
for idx in random_misclassified_indices:
    true_label = np.argmax(y_test[idx])
    predicted_label = np.argmax(predictions[idx])
    #print(f'Value {true_label} does not equal {predicted_label}')
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'Value {true_label} does not equal {predicted_label}')
    plt.show()

#Так как данных не мало, добавил возможность чтоб 10 изображение где правильно распознало где не правильно.
#Тем самым можно понять как ии распознает числа
