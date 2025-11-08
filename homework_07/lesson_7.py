from random import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
# Task 1
np_array = np.zeros(10)
np_array[4] = 1
np_array = np_array.reshape(2, 5)


# Task 2
np_arange = np.arange(10, 50)
np_arange = np_arange[::-1]
even = np_arange[np_arange % 2 == 0]


# Task 3

np_matrix = np.arange(9)
np_matrix = np_matrix.reshape(3, 3)

#Task 4

np.random.seed(0)
np_matrix_min_max = np.random.random((4,3,2))
print(np_matrix_min_max)
matrix_min = np_matrix_min_max.min()
matrix_max = np_matrix_min_max.max()
print(matrix_min)
print(matrix_max)

# Task 5

np.random.seed(1)

np_matrix_a = np.random.randint(1,101, size=(6, 4))
print(np_matrix_a)
np_matrix_b = np.random.randint(1, 101, size=(4, 3))
print(np_matrix_b)

multiplication = np.dot(np_matrix_a, np_matrix_b)
print(multiplication)

# Task 6

np.random.seed(2)
random_array = np.random.random((7, 7))
print(random_array)
mean = random_array.mean()
std_value = random_array.std()
print(mean)
print(std_value)
normalized_array = (random_array - mean) / std_value
print(normalized_array)

# Task 7

tips = pd.read_csv('tips.csv')
# Task 8
print(tips.head())

# Task 9
print(tips.shape)

# Task 10
print(tips.isnull().any().any())

# Task 11
print(tips.describe())

# Task 12
max_bill = tips['total_bill'].max()
print(max_bill)

# Task 13
num_smokers = (tips['smoker'] == 'Yes').sum()
print(num_smokers)

# Task 14
avg_bill_by_day = tips.groupby('day')['total_bill'].mean()
print(avg_bill_by_day)

# Task 15
total_bill_sex = (
    tips[tips['total_bill'] > tips['total_bill'].median()]
    .groupby('sex')['tip']
    .mean()
)
print(total_bill_sex)

# Task 16
tips['smoker_binary'] = tips['smoker'].map({'No': 0, 'Yes': 1})
print(tips[['smoker', 'smoker_binary']].head())

# Task 17
sns.histplot(tips['total_bill'], bins=20, kde=True, color='skyblue')
plt.title('Распределение total_bill')
#plt.show()

# Task 18

plt.figure(figsize=(8, 6))
sns.scatterplot(data=tips, x='total_bill', y='tip')
plt.title('Взаимосвязь между суммой счёта и чаевыми')
plt.xlabel('Сумма счёта (total_bill)')
plt.ylabel('Чаевые (tip)')
plt.grid(True, linestyle='--', alpha=0.5)
#plt.show()

#Task 19
sns.pairplot(tips)
# plt.show()

# Task 20

plt.figure(figsize=(8, 6))
sns.boxplot(data=tips, x='day', y='total_bill')
plt.title('Распределение total_bill по дням недели')
plt.xlabel('День недели')
plt.ylabel('Сумма счёта (total_bill)')
plt.show()

# Task 21

plt.figure(figsize=(10, 5))

# Ужин
plt.subplot(1, 2, 1)
sns.histplot(tips[tips['time'] == 'Dinner']['tip'], bins=20, kde=True, color='skyblue')
plt.title('Ужин (Dinner)')
plt.xlabel('Чаевые')
plt.ylabel('Частота')

# Обед
plt.subplot(1, 2, 2)
sns.histplot(tips[tips['time'] == 'Lunch']['tip'], bins=20, kde=True, color='salmon')
plt.title('Обед (Lunch)')
plt.xlabel('Чаевые')
plt.ylabel('Частота')

plt.tight_layout()
plt.show()

# Task 22

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, (gender, ax) in enumerate(zip(['Female', 'Male'], axes)):
    data = tips[tips['sex'] == gender]
    sns.scatterplot(
        data=data,
        x='total_bill',
        y='tip',
        hue='smoker',
        ax=ax,
        s=60,
        alpha=0.8
    )
    ax.set_title(f'{gender}')
    if i == 0:
        ax.legend(title='Smoker')
    else:
        ax.legend_.remove()  # убрать дублирующую легенду

plt.suptitle('Связь между total_bill и tip по полу')
plt.tight_layout()
plt.show()