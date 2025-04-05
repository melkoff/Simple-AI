import numpy as np
import tensorflow as tf
from tensorflow import keras

# Створення моделі
model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])

# Компиляція моделі
model.compile(optimizer='sgd', loss='mean_squared_error')

# Навчання моделі (передаємо NumPy масиви)
print("🔧 Навчаємо модель...")
model.fit(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), epochs=10)

# Прогнозування з новим значенням
result = model.predict(np.array([7.0]))
print(f"Прогноз для 7.0: {result}")
