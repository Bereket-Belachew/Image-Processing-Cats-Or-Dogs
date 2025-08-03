
⚠️ The `archive` folder is **not included** in this repository due to its large size.

---

## 📊 Project Structure

- **Image Loading & Labeling**: All images are read and labeled (`0` for cats, `1` for dogs).
- **Data Augmentation**: Applied using `ImageDataGenerator`.
- **Model Architecture**: Simple CNN with 3 convolutional layers.
- **Training**: Model is trained for 10 epochs with data generators.
- **Visualization**: Example images and accuracy graphs are plotted.

---

## 🖼️ Sample Output Images

> 🐶 Dogs Dataset Preview (25 random images):

![Dogs Preview](PLACEHOLDER_FOR_DOGS_IMAGE)

> 🐱 Cats Dataset Preview (25 random images):

![Cats Preview](PLACEHOLDER_FOR_CATS_IMAGE)

---

## 📈 Training Accuracy Graph

> Accuracy over 10 epochs:

![Accuracy Graph](https://github.com/Bereket-Belachew/Image-Processing-Cats-Or-Dogs/blob/44a0615ecb9ec591c16a6b1cdccd94a2ffef9c69/Screenshot%202025-08-03%20at%2016.45.13.png)

---

## 🛠️ How to Run

1. Install dependencies:

```bash
pip install tensorflow matplotlib pandas numpy seaborn kaggle
