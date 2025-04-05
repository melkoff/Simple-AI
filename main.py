import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Вхідні дані (наприклад, години тренування)
x = np.array([1, 2, 3, 4, 5], dtype=float)
# Результати (наприклад, рівень знань)
y = np.array([1.5, 3.0, 4.5, 6.0, 7.5], dtype=float)

# Проста модель з 1 шаром і 1 нейроном
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mean_squared_error')

print("🔧 Навчаємо модель...")
model.fit(x, y, epochs=500, verbose=0)

# Прогноз
result = model.predict([7.0])
print(f"📈 Прогноз для 7 годин навчання: {result[0][0]:.2f}")
