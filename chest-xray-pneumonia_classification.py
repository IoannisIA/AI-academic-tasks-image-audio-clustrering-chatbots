import os
import imgaug as aug
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from pathlib import Path
from skimage.io import imread
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
color = sns.color_palette()
import tensorflow as tf
import cv2


# Set the seed for hash based operations in python
os.environ['PYTHONHASHSEED'] = '0'

# Set the numpy seed
np.random.seed(111)

# Set the random seed in tensorflow at graph level
tf.random.set_seed(111)

# Define a tensorflow session with above session configs
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=tf.compat.v1.ConfigProto())

# Set the session in keras
tf.compat.v1.keras.backend.set_session(sess)

# Make the augmentation sequence deterministic
aug.seed(111)


"""The dataset is divided into three sets: 
1) train set 2) validation set and 3) test set. Let's grab the dataset

"""

# Define path to the data directory
data_dir = Path('chest_xray')

# Path to train directory (Fancy pathlib...no more os.path!!)
train_dir = data_dir / 'train'

# Path to validation directory
val_dir = data_dir / 'val'

# Path to test directory
test_dir = data_dir / 'test'


"""We will first go through the training dataset. 
We will do some analysis on that, look at some of the samples, 
check the number of samples for each class, etc. 
Lets' do it.

Each of the above directory contains two sub-directories:

NORMAL: These are the samples that describe the normal (no pneumonia) case.
PNEUMONIA: This directory contains those samples that are the pneumonia cases.
"""

# Get the path to the normal and pneumonia sub-directories
normal_cases_dir = train_dir / 'NORMAL'
pneumonia_cases_dir = train_dir / 'PNEUMONIA'

# Get the list of all the images
normal_cases = normal_cases_dir.glob('*.jpeg')
pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')

# An empty list. We will insert the data into this list in (img_path, label) format
train_data = []

# Go through all the normal cases. The label for these cases will be 0
for img in normal_cases:
    train_data.append((img,0))

# Go through all the pneumonia cases. The label for these cases will be 1
for img in pneumonia_cases:
    train_data.append((img, 1))

# Get a pandas dataframe from the data we have in our list
train_data = pd.DataFrame(train_data, columns=['image', 'label'],index=None)

# Shuffle the data
train_data = train_data.sample(frac=1.).reset_index(drop=True)

# How the dataframe looks like?
train_data.head()



# Get the counts for each class
cases_count = train_data['label'].value_counts()
print("counts for each class: ",cases_count)

# Plot the results
plt.figure(figsize=(10,8))
sns.barplot(x=cases_count.index, y= cases_count.values)
plt.title('Number of cases', fontsize=14)
plt.xlabel('Case type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(range(len(cases_count.index)), ['Normal(0)', 'Pneumonia(1)'])
plt.show()


"""
As you can see the data is highly imbalanced. 
We have almost with thrice pneumonia cases here as compared to the normal cases. 
This situation is very normal when it comes to medical data. 
The data will always be imbalanced. 
either there will be too many normal cases or 
there will be too many cases with the disease.

Let's look at how a normal case is different from that of a pneumonia case. 
We will look at somes samples from our training data itself.

"""


# Get few samples for both the classes
pneumonia_samples = (train_data[train_data['label']==1]['image'].iloc[:5]).tolist()
normal_samples = (train_data[train_data['label']==0]['image'].iloc[:5]).tolist()

# Concat the data in a single list and delete the above two list
samples = pneumonia_samples + normal_samples
del pneumonia_samples, normal_samples

# Plot the data
f, ax = plt.subplots(2,5, figsize=(30,10))
for i in range(10):
    img = imread(samples[i])
    ax[i//5, i%5].imshow(img, cmap='gray')
    if i<5:
        ax[i//5, i%5].set_title("Pneumonia")
    else:
        ax[i//5, i%5].set_title("Normal")
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_aspect('auto')
plt.show()


"""
If you look carefully, then there are some cases where you won't be able
 to differentiate between a normal case and a pneumonia case with the naked eye.
  There is one case in the above plot, at least for me ,which is too much confusing. 
  If we can build a robust classifier, it would be a great assist to the doctor too.

Preparing validation data
We will be defining a generator for the training dataset later in the notebook
 but as the validation data is small, so I can read the images and can load the data 
 without the need of a generator. 
 This is exactly what the code block given below is doing.
 """

# Get the path to the sub-directories
normal_cases_dir = val_dir / 'NORMAL'
pneumonia_cases_dir = val_dir / 'PNEUMONIA'

# Get the list of all the images
normal_cases = normal_cases_dir.glob('*.jpeg')
pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')

# List that are going to contain validation images data and the corresponding labels
valid_data = []
valid_labels = []

# Some images are in grayscale while majority of them contains 3 channels. So, if the image is grayscale, we will convert into a image with 3 channels.
# We will normalize the pixel values and resizing all the images to 224x224

# Normal cases
for img in normal_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    label = to_categorical(0, num_classes=2)
    valid_data.append(img)
    valid_labels.append(label)

# Pneumonia cases
for img in pneumonia_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    label = to_categorical(1, num_classes=2)
    valid_data.append(img)
    valid_labels.append(label)

# Convert the list into numpy arrays
valid_data = np.array(valid_data)
valid_labels = np.array(valid_labels)

print("Total number of validation examples: ", valid_data.shape)
print("Total number of labels:", valid_labels.shape)



"""
Augmentation
Data augmentation is a powerful technique which helps in almost every case 
for improving the robustness of a model. 
But augmentation can be much more helpful where the dataset is imbalanced. 
You can generate different samples of undersampled class in order to try 
to balance the overall distribution.

I like imgaug a lot. It comes with a very clean api and you can do 
hell of augmentations with it. It's worth exploring!! 
In the next code block, I will define a augmentation sequence.
 You will notice Oneof and it does exactly that. 
 At each iteration, it will take one augmentation technique out of the three 
 and will apply that on the samples
 """


# Augmentation sequence
seq = iaa.OneOf([
    iaa.HorizontalFlip(), # horizontal flips
    iaa.Affine(rotate=40), # roatation
    iaa.Multiply((1.2, 1.5))]) #random brightness


"""
Training data generator
Here I will define a very simple data generator. 
You can do more than this if you want but I think at this point, 
this is more than enough I need.
"""


def data_gen(data, batch_size):
    # Get total number of samples in the data
    n = len(data)
    steps = n // batch_size

    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size, 2), dtype=np.float32)

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n)

    # Initialize a counter
    i = 0
    while True:
        np.random.shuffle(indices)
        # Get the next batch
        count = 0
        next_batch = indices[(i * batch_size):(i + 1) * batch_size]
        for j, idx in enumerate(next_batch):
            img_name = data.iloc[idx]['image']
            label = data.iloc[idx]['label']

            # one hot encoding
            encoded_label = to_categorical(label, num_classes=2)
            # read the image and resize
            img = cv2.imread(str(img_name))
            img = cv2.resize(img, (224, 224))

            # check if it's grayscale
            if img.shape[2] == 1:
                img = np.dstack([img, img, img])

            # cv2 reads in BGR mode by default
            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # normalize the image pixels
            orig_img = img.astype(np.float32) / 255.

            batch_data[count] = orig_img
            batch_labels[count] = encoded_label

            # generating more samples of the undersampled class
            if label == 0 and count < batch_size - 2:
                aug_img1 = seq.augment_image(img)
                aug_img2 = seq.augment_image(img)
                aug_img1 = cv2.cvtColor(aug_img1, cv2.COLOR_BGR2RGB)
                aug_img2 = cv2.cvtColor(aug_img2, cv2.COLOR_BGR2RGB)
                aug_img1 = aug_img1.astype(np.float32) / 255.
                aug_img2 = aug_img2.astype(np.float32) / 255.

                batch_data[count + 1] = aug_img1
                batch_labels[count + 1] = encoded_label
                batch_data[count + 2] = aug_img2
                batch_labels[count + 2] = encoded_label
                count += 2

            else:
                count += 1

            if count == batch_size - 1:
                break

        i += 1
        yield batch_data, batch_labels

        if i >= steps:
            i = 0


"""Model
This is the best part. If you look at other kernels on this dataset, 
everyone is busy doing transfer learning and fine-tuning. 
You should transfer learn but wisely. 
We will be doing partial transfer learning and rest of the model will 
be trained from scratch. I will explain this in detail but before that, 
I would love to share one of the best practices when it comes to building 
deep learning models from scratch on limited data.

Choose a simple architecture.
Initialize the first few layers from a network that is pretrained on imagenet. 
This is because first few layers capture general details like color blobs, patches, 
edges, etc. Instead of randomly initialized weights for these layers, 
it would be much better if you fine tune them.
Choose layers that introduce a lesser number of parameters. 
For example, Depthwise SeparableConv is a good replacement for Conv layer. 
It introduces lesser number of parameters as compared to normal convolution 
and as different filters are applied to each channel, it captures more information. 
Xception a powerful network, is built on top of such layers only. 
You can read about Xception and Depthwise Separable Convolutions in this paper.
Use batch norm with convolutions. As the network becomes deeper, 
batch norm start to play an important role.
Add dense layers with reasonable amount of neurons.
Train with a higher learning rate and experiment with the number of neurons in 
the dense layers. Do it for the depth of your network too.
Once you know a good depth, start training your network with a lower learning
 rate along with decay.
This is all that I have done in the next code block."""


def build_model():
    input_img = Input(shape=(224, 224, 3), name='ImageInput')
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # Second conv block
    x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # Third conv block
    x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # Fourth conv block
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.2)(x)

    # Fifth conv block
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.2)(x)

    # FC layer
    x = Flatten()(x)
    x = Dense(units=256, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dropout(rate=0.3)(x)

    # Output layer
    output = Dense(units=2, activation='sigmoid')(x)
    model = Model(inputs=input_img, outputs=output)
    return model


model =  build_model()
model.summary()




# opt = RMSprop(lr=0.0001, decay=1e-6)
es = EarlyStopping(patience=5)
chkpt = ModelCheckpoint(filepath='best_model_todate', save_best_only=True, save_weights_only=True)
model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=Adam(lr=0.0001, decay=1e-5))

batch_size = 16
nb_epochs = 50

# Get a train data generator
train_data_gen = data_gen(data=train_data, batch_size=batch_size)

# Define the number of training steps
nb_train_steps = train_data.shape[0]//batch_size

print("Number of training and validation steps: {} and {}".
      format(nb_train_steps, len(valid_data)))

"""
I have commented out the training step as of now as it will train the network 
again while rendering the notebook and I would have to wait for an hour or so which 
I don't want to. I have uploaded the weights of the best model I achieved so far. 
Feel free to use it for further fine-tuning of the network. 
We will load those weights and will run the inference on the test set using 
those weights only. But...but for your reference, 
I will attach the screenshot of the training steps here.
"""

# # Fit the model
history = model.fit_generator(train_data_gen,
                              epochs=nb_epochs, steps_per_epoch=nb_train_steps,
                               validation_data=(valid_data, valid_labels),
                               callbacks=[es, chkpt],
                               class_weight={0:1.0, 1:0.4})

# Load the model weights
model.load_weights("best_model_todate")


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Preparing test data
normal_cases_dir = test_dir / 'NORMAL'
pneumonia_cases_dir = test_dir / 'PNEUMONIA'

normal_cases = normal_cases_dir.glob('*.jpeg')
pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')

test_data = []
test_labels = []

for img in normal_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    label = to_categorical(0, num_classes=2)
    test_data.append(img)
    test_labels.append(label)

for img in pneumonia_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    label = to_categorical(1, num_classes=2)
    test_data.append(img)
    test_labels.append(label)

test_data = np.array(test_data)
test_labels = np.array(test_labels)

print("Total number of test examples: ", test_data.shape)
print("Total number of labels:", test_labels.shape)


# Evaluation on test dataset
test_loss, test_score = model.evaluate(test_data, test_labels, batch_size=16)
print("Loss on test set: ", test_loss)
print("Accuracy on test set: ", test_score)


# Get predictions
preds = model.predict(test_data, batch_size=16)
preds = np.argmax(preds, axis=-1)

# Original labels
orig_test_labels = np.argmax(test_labels, axis=-1)

print(orig_test_labels.shape)
print(preds.shape)


"""
When a particular problem includes an imbalanced dataset, 
then accuracy isn't a good metric to look for. 
For example, if your dataset contains 95 negatives and 5 positives, 
having a model with 95% accuracy doesn't make sense at all. 
The classifier might label every example as negative and still 
achieve 95% accuracy. Hence, we need to look for alternative metrics. 
Precision and Recall are really good metrics for such kind of problems.

We will get the confusion matrix from our predictions and see what 
is the recall and precision of our model."""


# Get the confusion matrix
cm  = confusion_matrix(orig_test_labels, preds)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.show()

# Calculate Precision and Recall
tn, fp, fn, tp = cm.ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn)

print("Recall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))



"""Nice!!! So, our model has a 98% recall. 
In such problems, a good recall value is expected. 
But if you notice, the precision is only 80%. 
This is one thing to notice. 
Precision and Recall follows a trade-off, and you need to 
find a point where your recall, as well as your precision,
 is more than good but both can't increase simultaneously.

That's it folks!! I hope you enjoyed this kernel. Happy Kaggling!!"""