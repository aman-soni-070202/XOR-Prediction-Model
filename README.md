# XOR Neural Network from Scratch

This repository contains a simple implementation of a Feedforward Neural Network (FNN) trained to solve the classic XOR logic gate problem — built entirely from scratch, without using any machine learning libraries like TensorFlow or PyTorch.

---

## 🧠 Project Overview

The XOR problem is a fundamental challenge in neural networks, as it cannot be solved by a single-layer perceptron. This project demonstrates how even a small neural network with a hidden layer can learn to model non-linear relationships.

---

## 📁 Files

- `trainer.py` – Contains the code to build, train, and save the XOR neural network model.
- `predictor.py` – Loads the trained model and makes predictions on new inputs.

---

## 🔧 How It Works

- The model has:
  - **2 input neurons** (for the two binary inputs)
  - **2 hidden layer** with configurable neurons
  - **1 output neuron** using sigmoid activation
- Training is done using forward propagation and backpropagation manually coded from scratch.

---

## 🎛️ Experimentation

This project includes support for multiple activation and loss functions to experiment with:

- **Activation Functions:**
  - `Sigmoid` – squashes values between 0 and 1
  - `Tanh` – squashes values between -1 and 1 (symmetric and often better for hidden layers)

- **Loss Functions:**
  - `Mean Squared Error (MSE)`
  - `Binary Cross Entropy (BCE)`

You can easily toggle between them in the code to observe how they impact training performance and learning speed.

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/aman-soni-070202/XOR-Prediction-Model.git
   cd XOR-Prediction-Model
2. Train the model:

   `python trainer.py`

3. Run predictions:

   `python predictor.py`

---

## 🧪 Sample Predictions

After training, the network should correctly predict the XOR outputs:

Input A | Input B | Predicted Output
  0     |    0    |   0
  0     |    1    |   1
  1     |    0    |   1
  1     |    1    |   0

---

## 📚 Learning Purpose

This project is intended to help understand the core building blocks of a neural network:

    Manual weight updates

    Backpropagation logic

    Activation and loss functions

Perfect for beginners learning how deep learning really works under the hood!

---

## 📌 Author

Made with 💡 and curiosity by Aman Soni.

---

## 📃 License

Feel free to use or modify this for educational purposes.
