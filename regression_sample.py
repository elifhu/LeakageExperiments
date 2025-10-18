import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    def __init__(self):
        self.slope = 0
        self.intercept = 0

    def fit(self, X, y):
        # Ortalama hesapları
        x_mean = np.mean(X)
        y_mean = np.mean(y)

        # Katsayı hesaplama
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)

        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean

    def predict(self, X):
        return self.slope * X + self.intercept

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return mse

# Örnek veri
X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 4, 2, 5, 6])

# Modeli oluştur ve eğit
model = SimpleLinearRegression()
model.fit(X, y)

# Tahmin yap
y_pred = model.predict(X)

# Sonuçları çiz
plt.scatter(X, y, color='blue', label='Gerçek Değerler')
plt.plot(X, y_pred, color='red', label='Tahmin Edilen Doğru')
plt.title('Basit Lineer Regresyon')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show(block=False)

# Model değerlendirmesi
mse = model.evaluate(X, y)
print(f"Ortalama Kare Hata (MSE): {mse:.2f}")
print(f"Eğim (slope): {model.slope:.2f}")
print(f"Y-kesişim (intercept): {model.intercept:.2f}")