# Import necessary libraries
import numpy as np
import pandas as pd  # For data manipulation using DataFrames
import matplotlib.pyplot as plt  # For visualization

# Import Keras and TensorFlow libraries
from keras.datasets import fashion_mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Flatten the 28x28 images into 1D arrays
x_train_flattened = x_train.reshape(x_train.shape[0], -1)
x_test_flattened = x_test.reshape(x_test.shape[0], -1)

# Create Pandas DataFrames for train and test datasets
df_train = pd.DataFrame(x_train_flattened)
df_train['label'] = y_train  # Add labels as a separate column

df_test = pd.DataFrame(x_test_flattened)
df_test['label'] = y_test  # Add labels as a separate column

# Display DataFrame summary
print("Training DataFrame:")
print(df_train.head())

print("\nTest DataFrame:")
print(df_test.head())

print(f"Train DataFrame Shape: {df_train.shape}")

# 예측 클래스 매핑
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualization: Display a 3x3 grid of random images with class names
rows, cols = 3, 3

# Randomly select 9 images
random_indices = np.random.choice(x_train.shape[0], rows * cols, replace=False)
selected_images = x_train[random_indices]
selected_labels = y_train[random_indices]

# Plot images
fig, axes = plt.subplots(rows, cols, figsize=(7, 7))
for i, ax in enumerate(axes.flat):
    ax.imshow(selected_images[i], cmap='gray')  # Display grayscale image
    ax.axis('off')  # Hide axes
    ax.set_title(f"{class_names[selected_labels[i]]}", fontsize=8)

plt.tight_layout()
plt.show()# Preprocess data: Reshape, normalize, and encode labels
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32') / 255

input_shape = (img_rows, img_cols, 1)
num_classes = len(np.unique(y_train))

# Build CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),  # 필터 32개 적용
    BatchNormalization(),  # 배치 정규화
    Conv2D(64, kernel_size=(3, 3), activation='relu'),  # 필터 64개 적용
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),  # 특성 맵 축소
    Dropout(0.25),  # 과적합 방지용 드롭아웃

    Flatten(),  # 2D → 1D 벡터 변환
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),  # 또 한번 드롭아웃

    Dense(num_classes, activation='softmax')  # 다중 클래스 분류용 출력층
])

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

print(model.summary())

batch_size = 128
epochs = 10
history = model.fit(x_train, y_train, 
                    batch_size=batch_size, 
                    epochs=epochs,
                    validation_data=(x_test, y_test), 
                    verbose=1)

score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {score[0]}")
print(f"Test Accuracy: {score[1]}")

model.save('fashion_mnist_model.keras')

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['sparse_categorical_accuracy'], label='Train')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Predict on test data and visualize predictions
model = load_model('fashion_mnist_model.keras')
predictions = model.predict(x_test[:5])

fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, ax in enumerate(axes):
    predicted_index = np.argmax(predictions[i])
    ax.imshow(x_test[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"{class_names[predicted_index]}", fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show()