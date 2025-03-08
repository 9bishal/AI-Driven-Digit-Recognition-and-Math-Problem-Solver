Here's the **file content** describing what was done and how it was done. You can save it as **README.md** for your GitHub repository.  

---

# **MNIST Digit Classification using Neural Networks**  

This project implements a **Neural Network model** to classify handwritten digits (0-9) using the **MNIST dataset**. The model is built using **TensorFlow/Keras** and achieves **97% accuracy** after training for 25 epochs.  

## **🛠 What Was Done?**  
✔ **Loaded the MNIST dataset** containing 60,000 training images and 10,000 test images.  
✔ **Preprocessed the data** by normalizing pixel values (0-255 scaled to 0-1).  
✔ **Built a Neural Network model** with:  
   - An **input layer** (flattening 28×28 images).  
   - **Two hidden layers** (128 & 32 neurons, ReLU activation).  
   - **An output layer** (10 neurons, Softmax activation for classification).  
✔ **Compiled the model** using **Adam optimizer** and **sparse categorical cross-entropy** loss.  
✔ **Trained the model** for **25 epochs** to achieve high accuracy.  
✔ **Evaluated performance** using test data and calculated the accuracy.  
✔ **Made predictions** on test images using softmax to determine the most likely digit.  

## **⚙️ How Was It Done?**  
### **1️⃣ Loading the Dataset**  
The dataset was imported from TensorFlow/Keras:  
```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

### **2️⃣ Data Preprocessing**  
To improve model performance, the images were **normalized** by scaling pixel values from **0-255** to **0-1**:  
```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```

### **3️⃣ Building the Neural Network**  
A **Sequential model** was created with input, hidden, and output layers:  
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = Sequential([
    Flatten(input_shape=(28,28)),      # Input Layer (Flatten 28x28 images)
    Dense(128, activation="relu"),     # First Hidden Layer (128 neurons, ReLU)
    Dense(32, activation="relu"),      # Second Hidden Layer (32 neurons, ReLU)
    Dense(10, activation="softmax")    # Output Layer (10 neurons, Softmax)
])
```

### **4️⃣ Compiling the Model**  
The model was compiled using the **Adam optimizer** and **sparse categorical cross-entropy** loss function:  
```python
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
```

### **5️⃣ Training the Model**  
The model was trained for **25 epochs**, meaning it learned from the dataset **25 times**:  
```python
history = model.fit(x_train, y_train, epochs=25, validation_split=0.2)
```

### **6️⃣ Evaluating the Model**  
The trained model was tested on unseen data to check its accuracy:  
```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
```

### **7️⃣ Making Predictions**  
The model predicts the digit for a given test image by selecting the class with the highest probability:  
```python
import numpy as np

y_prob = model.predict(x_test)
y_pred = np.argmax(y_prob, axis=1)
```

To test a **single image**:  
```python
model.predict(x_test[1].reshape(1,28,28)).argmax(axis=1)
```

## **📊 Results**  
- **Final Accuracy:** **97%** on the test dataset.  
- The model correctly identifies most handwritten digits.  
- Softmax ensures the highest probability class is chosen as the predicted digit.  

## **📜 Conclusion**  
This neural network successfully classifies handwritten digits with high accuracy. The model can be further improved by adding **dropout layers, convolutional layers (CNNs), or tuning hyperparameters**.  

---

