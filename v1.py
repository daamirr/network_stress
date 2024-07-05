""" 
netwok.py

нейронная сеть созданная по примеру из книги http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits

"""

#### Libraries
import numpy as np
import random

import csv
import matplotlib.pyplot as plt
from datetime import date
import sys


class Network(object):
    
    def __init__(self, sizes):#, weights_file = None):
        self.num_layers = len(sizes)
        self.sizes = sizes

        # file2 = open('D:\программирую\нейронка\exp\weights_e2000_m1.txt')
        # вот тут добавить функцию чтения весов (например if уже существет такой файл txt с такой арихтектурой и тому пободным, то взять веса из него)
       
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                    for x, y in zip(sizes[:-1], sizes[1:])]
        
        self.target = 0.1 # точность на которую мы желаем обучить нейросеть, на само обучение она никак не влияет, а лишь яв порогом для кол-ва совпадений

    def feedforward(self, a): 
        ''' Выполняет работу нейронной сети '''
        for b, w in zip(self.biases, self.weights):
            a = my_sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None, train_data=None):
        """ Обучение нейронной сети методом стохастического
        градиентного спуска. 
        The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs. 
        Epochs - кол-во проходов по training_data, 
        mini_batc_size - размер мини выборки, которая берется на рандом из всей обучающей выборки для обновления весов и смещений,
        eta - скорость обучения.  
        If "test_data" is provided then the network
        will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""


        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                results = self.evaluate(test_data)
                print("Epoch {} : {} / {} ~= {}".format(j,results,n_test, round(results/n_test, 3)))
            else:
                print("Epoch {} complete".format(j))


    def update_mini_batch(self, mini_batch, eta):
        """ Обновляет веса и смещения в среднем для мини-выборки
          на основе градиентного спкуска функции backprop """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch))*nw 
                       for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]
        

    # пока не разобрался как работает эта функция
    # переписать ее самому, чтобы понять что да как, подкорректировать под себя    
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = my_sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            my_sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = my_sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    
    def evaluate(self, test_data):
        # target = 0.10
        test_results = [(self.feedforward(x), y)    
                        for (x, y) in test_data]
        
        return sum(int((y - x) ** 2 <= self.target*y) for (x, y) in test_results)
    

    def cost_derivative(self, output_activation, y):
        return (output_activation-y)


#### Function's activation
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    # производная от сигмоидальной функции 
    # return np.exp(-x) / (1 + np.exp(-x))**2
    return sigmoid(x)*(1-sigmoid(x))

def my_sigmoid(x):
    return (2 * 1.0 / (1.0 + np.exp(-2*x)) - 1)

def my_sigmoid_prime(x):
    return 4 * np.exp(-x) / (1 + np.exp(-2*x)) ** 2

#===================== вспомогательные функции =========================
# def launch(name, data):
#     result = name.feedroward(data)
#     return ()
'''
def transfer(y, lim_min, lim_max):
    "Функция приведение любого исхожного промежтука к промежутку [0;1]"
    return (y - lim_min)/(lim_max-lim_min)

def transferlist(yy):
    return [(yi - min(yy))/(max(yy) - min(yy)) for yi in yy]

def reverse_transferlist(y, lim_min, lim_max):
    "приведение обратно к естественному промежутку"
    return [yi * (lim_max-lim_min) + lim_min for yi in y]

def separation_into_data(procent_train, all_data):
    " Функция разделения данных на две подвыборки, правда пока не рандомное, исправить! (сделать хаотичным)"
    a = round(len(all_data) * procent_train / 100)
    b = len(all_data) - a
    print(f'Всего экземпляров данных: {len(all_data)} (Размер обучающей выборки: {a}, размер тестовой выборки: {b})')
    train_data = all_data[0:a]
    test_data = all_data[a:len(all_data)]
    return train_data, test_data


# ============= генерация датасета из csv файла ====================
# csv file name
filename = "D:\\программирую\нейронка\deformation\dataset\чистые данные\Книга1.csv"

# initializing the titles and rows list
fields = []
rows = []
 
# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
 
    # extracting field names through first row
    fields = next(csvreader)
 
    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)
 
    # get total number of rows
    print("Total no. of rows: %d" % (csvreader.line_num))

x1 = []
x2 = []
x3 = []
y = []
for i in range(6, len(rows)):
    x1.append(float(rows[i][1]))
    x2.append(float(rows[i][2]))
    x3.append(float(rows[i][3]))
    y.append(float(rows[i][4]))

# сведение выходных данных к промежутку от 0 до 1
X1 = np.array(transferlist(x1))
X2 = np.array(transferlist(x2))
X3 = np.array(transferlist(x3))
Y = transferlist(y)

# запоминание крайних точек
max_deform = max(y) #1.627778267
min_deform = min(y) #0.070651402

all_data = [(np.array([x1, x2, x3]).reshape(-1, 1), y) for x1,x2, x3,y in zip(X1, X2, X3, Y)]

# разделение на обучающую и тестовую выборки
train_data, test_data = separation_into_data(80, all_data)

#===================== вызов сети(обучение) =========================
### Гипер-параметры сети
epochs = 100
architecture = [3, 30, 10, 1]
mini_batch = 10
eta = 1.0 

# инициализация
exp_network = Network(architecture)#, 'D:\программирую\нейронка\exp\weights_e2_m1.txt')

# обучение
exp_network.SGD(train_data, epochs, mini_batch, eta, test_data)   #, test_data=test_data)

# ========================= сохранение (запись) весов в файл ====================== 
# сделать нормальную запись, чтобы можно было потом считывать сохраненные веса и смещения

# запоминание весов после обучения
file1 = open(f'D:\программирую\нейронка\exp\weights_e{epochs}_m{mini_batch}.txt', 'w')
file1.write(f'{"="*10} преамбула {"="*10}\n\n тут можно расписать какое кол-во эпох, архитекутру и тому подобнье (гипер параметры)\n\n{"="*30}\n')
file1.write(f'веса:\n{exp_network.weights}\n\nсмещения:\n{exp_network.biases}')

# for i in range(len(exp_network.weights)):
#     for j in range(len(exp_network.weights[i])):
#         for n in range(len(exp_network.weights[i][j])):
#             file1.write(f'{exp_network.weights[i][j][n]}\n')#str(exp_network.weights))

# file1.write(str(exp_network.weights))
file1.close()

'''