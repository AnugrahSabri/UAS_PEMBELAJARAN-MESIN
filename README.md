# UAS_PEMBELAJARAN-MESIN
202231509_ANUGRAHSABRI_UAS_PEMESIN
# Import pustaka yang diperlukan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
# Membaca dataset dari file CSV
url = 'https://raw.githubusercontent.com/username/repository/main/iris.csv'
data = pd.read_csv(url)
# Memisahkan fitur dan label
X = data.drop('species', axis=1).values
y = data['species'].values

# Mengubah label menjadi format numerik
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# One-hot encoding untuk label
y = to_categorical(y)

# Standarisasi fitur
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Membangun model Sequential
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Kompilasi model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Melatih model
history = model.fit(X_train, y_train, epochs=50, batch_size=5, validation_data=(X_test, y_test))
# Evaluasi model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Akurasi: {accuracy*100:.2f}%')
