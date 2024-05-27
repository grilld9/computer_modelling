import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

# Определение дифференциальных уравнений в модели SIR
def sir_model(y, t, beta, gamma):
  S, I, R = y
  dS_dt = -beta * S * I
  dI_dt = beta * S * I - gamma * I
  dR_dt = gamma * I
  return [dS_dt, dI_dt, dR_dt]

# Функция для вычисления суммарной квадратичной ошибки
def compute_loss(beta, gamma, data, initial_conditions, t):
  solution = odeint(sir_model, initial_conditions, t, args=(beta, gamma))
  return np.sum((solution - data)**2)

# Градиентный спуск для оптимизации параметров
def gradient_descent(beta, gamma, data, initial_conditions, t, lr=0.001, epochs=1000):
  for _ in range(epochs):
  # Частные производные функции потерь по параметрам
    beta_grad = (compute_loss(beta + 0.01, gamma, data, initial_conditions, t) -
      compute_loss(beta - 0.01, gamma, data, initial_conditions, t)) / 0.02
    gamma_grad = (compute_loss(beta, gamma + 0.01, data, initial_conditions, t) -
      compute_loss(beta, gamma - 0.01, data, initial_conditions, t)) / 0.02

# Обновление параметров
    beta -= lr * beta_grad
    gamma -= lr * gamma_grad

  return beta, gamma

# Исходные данные и условия
N = 1000 # общее число населения
initial_conditions = [999, 1, 0] # начальные условия: почти все здоровы, один инфицирован
t = np.linspace(0, 50, 50) # время наблюдений
data = odeint(sir_model, initial_conditions, t, args=(0.4, 0.1)) # сгенерированные данные с "известными" параметрами

# Подбор параметров
beta_opt, gamma_opt = gradient_descent(0.3, 0.05, data, initial_conditions, t)
print(f"Оптимальные значения параметров: beta = {beta_opt}, gamma = {gamma_opt}")

# Визуализация результатов
optimal_solution = odeint(sir_model, initial_conditions, t, args=(beta_opt, gamma_opt))
plt.plot(t, data[:, 1], 'r', label='Infected (data)')
plt.plot(t, optimal_solution[:, 1], 'b', label='Infected (model)')
plt.xlabel('Time')
plt.ylabel('Number of Infected People')
plt.legend()
plt.show()