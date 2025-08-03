
⚠️ The `archive` folder is **not included** in this repository due to its large size.

---

## 📊 Project Structure

- Image Loading & Labeling**: All images are read and labeled (`0` for cats, `1` for dogs).
- Data Augmentation: Applied using `ImageDataGenerator`.
- Model Architecture: Simple CNN with 3 convolutional layers.
- Training: Model is trained for 10 epochs with data generators.
- Visualization: Example images and accuracy graphs are plotted.

---
# 🐶 Cats vs 🐱 Dogs Image Classifier with CNN (TensorFlow & Keras)

This is a deep learning project that uses Convolutional Neural Networks (CNNs) to classify images as either **cats** or **dogs**. The model is built using **TensorFlow**, **Keras**, and **Matplotlib**, and trained on a labeled dataset of pet images.

---

## 📁 Dataset

You can download the image dataset from Kaggle:

👉 **[Cats and Dogs Image Classification – by Samuel Cortinhas](https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification)**

> After downloading, extract the contents and place the `archive/` folder inside your project directory:


- One More thing: Don't forget to change the location of the basepath variable to the exact location of where you saved the the Dog and Cat Archive file
## 🖼️ Sample Output Images

> 🐶 Dogs Dataset Preview (25 random images):

![Dogs Preview](https://github.com/Bereket-Belachew/Image-Processing-Cats-Or-Dogs/blob/2d014b213e6da818d13780595ac2afa27406bb9e/Screenshot%202025-08-03%20at%2016.43.29.png)

> 🐱 Cats Dataset Preview (25 random images):

![Cats Preview](https://github.com/Bereket-Belachew/Image-Processing-Cats-Or-Dogs/blob/2d00a469937aa67958a3968fa29982804d39e25c/Screenshot%202025-08-03%20at%2016.43.47.png)

---

## 📈 Training Accuracy Graph

> Accuracy over 10 epochs:

![Accuracy Data](https://github.com/Bereket-Belachew/Image-Processing-Cats-Or-Dogs/blob/44a0615ecb9ec591c16a6b1cdccd94a2ffef9c69/Screenshot%202025-08-03%20at%2016.45.13.png)
![Accuracy Graph](https://github.com/Bereket-Belachew/Image-Processing-Cats-Or-Dogs/blob/8c5dfa59d8c800993c359336b27dff8c1389c73b/Screenshot%202025-08-03%20at%2016.44.51.png)
---

## 🛠️ How to Run

1. Install dependencies:

```bash
pip install tensorflow matplotlib pandas numpy seaborn kaggle
