import data_loaderV3_thickness, v1

import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import os


train, test = data_loaderV3_thickness.Data_load().load(80)

# layers = 200
# thikness = 1
# step = thikness / (layers - 1)

max_y = abs(min(data_loaderV3_thickness.Data_load().y))

th1 = (len(data_loaderV3_thickness.Data_load().rows1)-7) * 198  # кол-во обучающих экземпляров для образца с толщиной в 1мм
th2 = (len(data_loaderV3_thickness.Data_load().rows2)-7) * 198  # кол-во обучающих экземпляров для образца с толщиной в 2мм
# th4 = (len(Data_load().rows4)-7) * 198
# print(th1, th2, th4)

# координаты в глубину для разных пластинок
a = 0
b = a + 198
X_1mm = data_loaderV3_thickness.Data_load().x3[a:b]
a = th1
b = a + 198
X_2mm = data_loaderV3_thickness.Data_load().x3[a:b]
a = th2 + th1
b = a + 198
X_4mm = data_loaderV3_thickness.Data_load().x3[a:b]

rows1 = data_loaderV3_thickness.Data_load().rows1
rows2 = data_loaderV3_thickness.Data_load().rows2
rows4 = data_loaderV3_thickness.Data_load().rows4




def draw_plot(network, epochs, ab=None):
    "ab - промежуток, сколько графиков вывести"

    def taking_stress(data):
        "что возвращает сеть после обучения"
        result = network.feedforward(data) 
        return result * max_y

    # создание каталога куда будут помещаться результаты
    directory_name = f'D:/программирую/нейронка/глубина распред остаточных напряжений/вывод - графики/net{network.sizes}'
    os.makedirs(directory_name, exist_ok=True)


    def random_plot(thickness):

        if thickness == 1:
            rows = rows1
            X_mm = X_1mm
        if thickness == 2:
            rows = rows2
            X_mm = X_2mm
        if thickness == 4:
            rows = rows4
            X_mm = X_4mm

        # выбор рандомного промежутка
        if ab:
            a = random.randint(7, len(rows) - ab)
            b = a + ab
        else:
            a = 7
            b = len(rows)

        
        # к чему сеть должна стермиться
        for i in range(a, b):
            # sh = data_loaderV3_thickness.Data_load().x1[i]
            # temp = data_loaderV3_thickness.Data_load().x2[i]
            sh = float(rows[i][1])
            temp = float(rows[i][2])
            # sh_trans = data_loaderV3_thickness.Data_load().X1[i]
            # temp_trans = data_loaderV3_thickness.Data_load().X2[i]
            sh_trans = (sh - data_loaderV3_thickness.Data_load().min_SH) / (data_loaderV3_thickness.Data_load().max_SH - data_loaderV3_thickness.Data_load().min_SH)
            temp_trans = (temp - data_loaderV3_thickness.Data_load().min_temp) / (data_loaderV3_thickness.Data_load().max_temp - data_loaderV3_thickness.Data_load().min_temp)
            Z = X_mm
            Y_real = []
            Y = []
            for j in range(len(Z)):
                Y_real.append(float(rows[i][j+3]))

                test_data = [[sh_trans], [temp_trans], [Z[j] / 4], [thickness]]
                stress = taking_stress(test_data)
                Y.append(stress[0][0])  # иначе получим array
        
            
            plt.plot(X_mm, Y)

            plt.plot(X_mm, Y_real)
            plt.grid()
            plt.legend([f'Network ({epochs = })', f'Real: sh = {round(sh, 1)}, temp = {round(temp, 4)}'])
            plt.title(f'Thickness - {thickness}mm')
            plt.xlabel('Глубина, мм')
            plt.ylabel('Остаточные напряжения, МПа')
            plt.savefig(directory_name + f'/data_output_th{thickness}mm_epo{epochs}_{i}.jpg')
            # plt.show()
            plt.close()

    random_plot(1)
    random_plot(2)
    random_plot(4)



net = v1.Network([4, 30, 10, 1])

# print(net.sizes)
epochs = 5
net.SGD(train, epochs, 2, 0.3, test)


draw_plot(net, epochs, 10)
