import numpy as np
import pandas as pd
import kagglehub
import numpy
import matplotlib.pyplot as plt
import warnings

import seaborn as sns
import random
from tensorflow.keras.preprocessing.image import (ImageDataGenerator, load_img)
from keras import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D


#for line 14
import os

warnings.filterwarnings('ignore')

#Getting the data into Table by Creating a DataFrame(a panda functionality
input_path= []
Label = []

basepath = '/Users/Newstandard/PycharmProjects/MachinelearningProjects/CatsAndDogs/archive/train'



#To access the dog and Cat files we need to import OS
for image_name in os.listdir(basepath):
    image_folder_path = os.path.join(basepath, image_name)
    if not os.path.isdir(image_folder_path):
        continue  # Skip .DS_Store or any non-folder

    for file_name in os.listdir(image_folder_path):
        if image_name == 'cats':
            Label.append(0)
        else:
            Label.append(1)
        input_path.append(os.path.join(image_folder_path, file_name))

print(Label[0],input_path[0])#This is just to check if the images are loaded correctly, if you have an error dont forget to install tensorflow in the terminal using "pip install tensorflow"


#Let's Create a table using panda
df = pd.DataFrame()
df['images']=input_path
df['label']=Label
df['label']=df['label'].astype('str')
df=df.sample(frac=1).reset_index()  #shuffle


print(df.head())

#Matplotlib part
plt.figure(figsize=(25,25))
temp = df[df['label']=='1']['images']
start = random.randint(0,len(temp))
files =temp[start:start+25 ]

for index, file in enumerate(files):
    plt.subplot(5,5, index+1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title('dogs')
    plt.axis('off')
#Here we display the images of the dogs
plt.tight_layout()
plt.show()

#For the cats
plt.figure(figsize=(25,25))
temp = df[df['label']=='0']['images']
start = random.randint(0,len(temp))
files =temp[start:start+25 ]

for index, file in enumerate(files):
    plt.subplot(5,5, index+1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title('cats')
    plt.axis('off')
#Here we display the images of the dogs
plt.tight_layout()
plt.show()

#DataGenerator
train_generator = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_iterator = train_generator.flow_from_dataframe(
    df,
    x_col='images',
    y_col='label',
    target_size=(64,64),
    batch_size=64,
    class_mode='binary'
)


#Modelling the data
model = Sequential([
    Conv2D(16,(3,3),activation='relu',input_shape=(64,64,3)),#3 means RGB
    MaxPooling2D((2,2)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512,activation='relu'),
    Dense(1,activation='sigmoid')


]
)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'] )
model.summary()

#Train
history = model.fit(train_iterator ,epochs=10)


#Result
acc = history.history['accuracy']
epochs_range= range(len(acc))
plt.plot(epochs_range,acc,'b',label="Training accuracy")
plt.title("Accuracy Graph")
plt.figure(figsize=(8,6))
plt.grid(True)
plt.legend()  # Add this
plt.tight_layout()
plt.show()
print("Accuracy list:", acc)
print("Epochs range:", list(epochs_range))
#preditction Algorithm
