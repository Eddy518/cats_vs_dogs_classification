import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import random
import os

# Image Size Constants
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_SIZE = (IMG_WIDTH,IMG_HEIGHT) # 28 by 28 Image
IMG_CHANNEL = 3 #RGB

filenames = os.listdir("../data/train/train")

categories = []
for f_name in filenames:
    # print(f_name)
    category = f_name.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

# print(categories)

df = pd.DataFrame({
    'filename':filenames,
    'category':categories
})
# print(pd)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_WIDTH,IMG_HEIGHT,IMG_CHANNEL)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(2,activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()

# Don't compile model again if already exist (too much time consuming)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience = 10)
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',patience=2, verbose=1, factor=0.5, min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

df["category"] = df['category'].replace({0:'cat',1:'dog'})
train_df,validate_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train=train_df.shape[0]
total_validate = validate_df.shape[0]

batch_size = 15

train_datagen = ImageDataGenerator(rotation_range=15,
    rescale = 1./255, shear_range = 0.1, zoom_range = 0.2, horizontal_flip = True, width_shift_range = 0.1, height_shift_range = 0.1)

train_generator = train_datagen.flow_from_dataframe(train_df, "../data/train/train/", x_col='filename', y_col='category',
    target_size=IMG_SIZE, class_mode='categorical',batch_size=batch_size)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(validate_df, "../data/train/train/", x_col='filename', y_col='category', target_size=IMG_SIZE, class_mode='categorical',batch_size=batch_size)

if not os.path.isfile("../output/cats_vs_dogs_1.keras"):
    epochs = 10
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, validation_steps=total_validate//batch_size, steps_per_epoch=total_train//batch_size,callbacks=callbacks)

    model.save("../output/cats_vs_dogs_1.keras")

test_filenames = os.listdir("../data/test1/")
test_df = pd.DataFrame({
    'filename':test_filenames,
    'category':['']*len(test_filenames) # Changed to empty strings for categorical mode
})
nb_samples = test_df.shape[0]

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory="../data/test1/",
    x_col='filename',
    y_col='category',  # Add category column
    class_mode='categorical',
    target_size=IMG_SIZE,
    batch_size=batch_size,
    shuffle=False
)

predict = model.predict(test_generator, steps=int(np.ceil(nb_samples/batch_size)))

test_df['category'] = np.argmax(predict, axis=-1)

label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)

test_df['category'] = test_df['category'].replace({'dog':1, 'cat':0})

sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12,24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("../data/test1/"+filename,target_size=IMG_SIZE)
    plt.subplot(6,3,index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')
plt.tight_layout()
plt.show()

results = {
    0:'cat',
    1:'dog'
}

from PIL import Image
im = Image.open("../verify/cat.jpg")
im = im.resize(IMG_SIZE)
im = np.array(im)
im = np.expand_dims(im, axis=0)
im = im/255
pred = np.argmax(model.predict([im]), axis=-1)[0]
print(pred,results[pred])
