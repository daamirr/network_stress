""" 


"""


import numpy as np
import csv


class Data_load():
    # ============= генерация датасета из csv файла ====================
    def __init__(self):
        
        # filename1 = "D:/программирую/нейронка/глубина распред остаточных напряжений/dataset/от толщины/with deform и другой заделкой (лишь с одной стороны)/1mmFull.csv"
        # filename2 = "D:/программирую/нейронка/глубина распред остаточных напряжений/dataset/от толщины/with deform и другой заделкой (лишь с одной стороны)/2mmFull.csv"
        # filename4 = "D:/программирую/нейронка/глубина распред остаточных напряжений/dataset/от толщины/with deform и другой заделкой (лишь с одной стороны)/4mmFull.csv"

        filename1 = "D:/программирую/нейронка/глубина распред остаточных напряжений/dataset/от толщины/with deform и другой заделкой (лишь с одной стороны)/тест/1mmFull.csv"
        filename2 = "D:/программирую/нейронка/глубина распред остаточных напряжений/dataset/от толщины/with deform и другой заделкой (лишь с одной стороны)/тест/2mmFull.csv"
        filename4 = "D:/программирую/нейронка/глубина распред остаточных напряжений/dataset/от толщины/with deform и другой заделкой (лишь с одной стороны)/тест/4mmFull.csv"

        self.rows1 = []
        self.rows2 = []
        self.rows4 = []
        
        with open(filename1, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                self.rows1.append(row)

        with open(filename2, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                self.rows2.append(row)

        with open(filename4, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                self.rows4.append(row)

        # входные нейроны
        self.x1 = []    # specific heat
        self.x2 = []    # temp
        self.x3 = []    # Z - depth
        self.x4 = []    # thickness 1mm, 2mm or 4mm

        # выходные нейроны
        self.y = []     # sigma in Z_i

        self.layers = 200

        # заполнение входных и выходных нейронов 
        self.append_data(1, self.rows1)
        self.append_data(2, self.rows2)
        self.append_data(4, self.rows4)

        # запоминание крайних точек
        self.max_SH = max(self.x1)
        self.min_SH = min(self.x1)
        self.max_temp = max(self.x2)
        self.min_temp = min(self.x2)

        self.max_stress = max(self.y)
        self.min_stress = min(self.y) 

        # X1 = np.array(self.transfer_universal_list(self.x1))
        self.X1 = np.array(self.accommodate_list(self.x1))
        # X2 = np.array(self.transfer_universal_list(self.x2))
        self.X2 = np.array(self.accommodate_list(self.x2))
        # X3 = np.array(self.transfer_universal_list(self.x3))
        self.X3 = np.array(self.accommodate_list(self.x3))
        self.X4 = np.array(self.accommodate_list(self.x4))
        # X4 = self.x4
        self.Y = self.transfer_universal_list(self.y)

    def append_data(self, thickness, rows):
        """" функция для разеделения данных из таблицы cvs по массивам
        соответсвенно для различных толщин и коллонок"""

        step = thickness / (self.layers - 1)
        z = 0   # от 0 (поверхности) до 4mm (нижней стороны)
        for i in range(7, len(rows)):
            for j in range(1, self.layers-1):
                if j == 101:
                    z = z + 2*step  # косяк, в расчете не проставил галочку напротив вывода этих точек, поэтому нужно и здесь их исключить

                self.x1.append(float(rows[i][1]))
                self.x2.append(float(rows[i][2]))
                self.x3.append(z)
                self.x4.append(thickness)
                self.y.append(float(rows[i][j+2]))

                z = z + step
            z = 0   # обнуление высоты
        

    def transfer_universal_list(self, param):
        """ делит каждый элемент списка на максимальное значение по модолю в этом списке
        т.е. вписывает исходный массив в единичный отрезок, проходящий через точку (0, 0)
        либо в отрозок от -1 """

        if abs(max(param)) > abs(min(param)):
            modul_max_param = abs(max(param))
        else:
            modul_max_param = abs(min(param))

        return [yi / modul_max_param for yi in param]
    
    def accommodate_list(self, param):
        """ минимум массива превращается в 0, масимум в 1"""
        ymin = min(param)
        ymax = max(param)
        return [(yi - ymin) / (ymax - ymin) for yi in param]

    def separation_into_data(self, procent_train, all_data):
        """ Разделение датасета на две подвыборки: обучающую и тестовую
        !!!Правда пока не рандомное, исправить (сделать хаотичным)!!! """

        a = round(len(all_data) * procent_train / 100)
        b = len(all_data) - a
        print(f'Всего экземпляров данных: {len(all_data)} (Размер обучающей выборки: {a}, размер тестовой выборки: {b})')
        train_data = all_data[0:a]
        test_data = all_data[a:len(all_data)]
        return train_data, test_data


    def load(self, procent):
        """ Главная ф-ция выдающая на выходе две выборки (обучающую и тестовую) в зависимости от procent - процентного распределения,
        путем сначала приведением всех данных к формату промежутка от 0 до 1, где 0 соотв. min, a 1 соотв. max,
        а затем объединением и разделением датасета с помощью функции separation"""

        all_data = [(np.array([x1, x2, x3, x4]).reshape(-1, 1), y) for x1,x2,x3,x4,y in zip(self.X1, self.X2, self.X3, self.X4, self.Y)]

        # all_data = [(np.array([x1, x2, x3, x4]).reshape(-1, 1), y) for x1,x2, x3, x4,y in zip(self.x1, self.x2, self.x3, self.x4, self.y)]

        # разделение на обучающую и тестовую выборки
        train_data, test_data = self.separation_into_data(procent, all_data)
        # return (train_data, test_data)
        return train_data, test_data




# =============== Тесты ==================
# print(Data_load().accommodate_list(100))

# tr, te = Data_load().load(80)
# # n = 1200
# # a = 198 * n
# # b = a + 198
# # print((Data_load().x3[a:b]))

# th1 = (len(Data_load().rows1)-7) * 198  # кол-во обучающих экземпляров для образца с толщиной в 1мм
# th2 = (len(Data_load().rows2)-7) * 198  # кол-во обучающих экземпляров для образца с толщиной в 2мм
# th4 = (len(Data_load().rows4)-7) * 198
# print(th1, th2, th4)


# a = 0
# b = a + 198
# print((Data_load().x3[a:b]))
# a = th1
# b = a + 198
# print((Data_load().x3[a:b]))
# a = th2 + th1
# b = a + 198
# print((Data_load().x3[a:b]))
