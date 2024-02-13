import os
#os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,70).__str__()
import cv2 # import after setting OPENCV_IO_MAX_IMAGE_PIXELS
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np


### horovod and GPU setup #################
import horovod.tensorflow.keras as hvd
import tensorflow as tf

hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    
#################

def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
    plt.title(title_2)
    plt.show()
    
# https://youtu.be/XyX5HNuv-xE
# https://youtu.be/q-p8v1Bxvac
"""
Standard Unet
Model not compiled here, instead will be done externally to make it
easy to test various loss functions and optimizers. 
"""


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

################################################################
def multi_unet_model(n_classes=3, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    p5 = MaxPooling2D(pool_size=(2, 2))(c5)
     
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p5)
    c6 = Dropout(0.3)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    #Expansive path
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c5])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c4])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c3])
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    u10 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c9)
    u10 = concatenate([u10, c2])
    c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u10)
    c10 = Dropout(0.1)(c10)
    c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c10)
     
    u11 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c10)
    u11 = concatenate([u11, c1], axis=3)
    c11 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u11)
    c11 = Dropout(0.1)(c11)
    c11 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c11)
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c11)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    #NOTE: Compile the model in the main program to make it easy to test with various loss functions
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    #model.summary()
    
    return model
 


#from simple_multi_unet_model import multi_unet_model #Uses softmax 

from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt



#Resizing images, if needed
SIZE_X = 256 #128 
SIZE_Y = 256 #128
n_classes = 3 #4 #Number of classes for segmentation

#Capture training image info as a list
train_images = []

#for directory_path in glob.glob("train_data/"):
for directory_path in glob.glob("train_images_4032/"):
    for img_path in sorted(glob.glob(os.path.join(directory_path, "*.png"))):
        img = cv2.imread(img_path, 0)       
        #img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)
        
# for directory_path in glob.glob("train_data/"):
#     for img_path in sorted(glob.glob(os.path.join(directory_path, "*.png"))):
#         img = cv2.imread(img_path, 0)       
#         #img = cv2.resize(img, (SIZE_Y, SIZE_X))
#         train_images.append(img)
       
#Convert list to array for machine learning processing        
train_images = np.array(train_images)

#Capture mask/label info as a list
train_masks = [] 

#for directory_path in glob.glob("train_masks/"):
for directory_path in glob.glob("train_masks_4032/"):
    for mask_path in sorted(glob.glob(os.path.join(directory_path, "*.png"))):
        mask = cv2.imread(mask_path, 0)       
        #mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        train_masks.append(mask)
        
# for directory_path in glob.glob("train_masks/"):
#     for mask_path in sorted(glob.glob(os.path.join(directory_path, "*.png"))):
#         mask = cv2.imread(mask_path, 0)       
#         #mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
#         train_masks.append(mask)
        
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)

#print(np.unique(train_masks))

#plot_image(train_images[5000], train_masks[5000])

#plot_image(train_images[1], train_masks[1])

###############################################
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)


#print(np.unique(train_masks_encoded_original_shape))

#print(train_images.shape)

#print(train_masks.shape)

#################################################
train_images = np.expand_dims(train_images, axis=3)
train_images = normalize(train_images, axis=1)

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

#print(train_images.shape)

#print(train_masks_input.shape)

#Create a subset of data for quick testing
#Picking 10% for testing and remaining for training
from sklearn.model_selection import train_test_split
#X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size = 0.20, random_state = 0)

#Further split training data t a smaller subset for quick testing of models
#X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.1, random_state = 0)

from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))


print("Class values in the dataset are ... ", np.unique(y_train_cat))  # 0 is the background/few unlabeled 
print(X_train.shape, " shape ", y_train_cat.shape)

BATCH_SIZE_PER_REPLICA = 16
#tf.data and pipeline data
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
print("len dataset ",len(dataset))
print("len list dataset",len(list(dataset)))
dataset = dataset.repeat().shuffle(10000).batch(BATCH_SIZE_PER_REPLICA)


# if hvd.rank() == 0:
#     timeline_dir = "./hvd"
#     os.makedirs(timeline_dir)
#     os.environ['HOROVOD_TIMELINE'] = timeline_dir + "/timeline.json"


#print(train_masks_cat.shape)

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()

### horovod changes to compile model ###################
# log the model
if hvd.rank() == 0:
    print(model.summary())
    
opt = tf.keras.optimizers.Adam()
opt = hvd.DistributedOptimizer(opt, backward_passes_per_step=1, average_aggregated_gradients=True)
#opt = tf.keras.optimizers.SGD(learning_rate=0.01 * hvd.size(), momentum=0.9)
#opt = hvd.DistributedOptimizer(opt)

# Horovod: adjust learning rate based on number of GPUs.
#opt = tf.optimizers.Adam(0.001 * hvd.size())

# setup training parameters
model.compile(
    optimizer=opt, 
    loss='categorical_crossentropy', 
    #metrics=['accuracy'],
    metrics=['accuracy', f1_m,precision_m, recall_m],
    experimental_run_tf_function=False #for horovod # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow # uses hvd.DistributedOptimizer() to compute ...
)

class Throughput(tf.keras.callbacks.Callback):
    def __init__(self, total_data=0):
        self.total_data = total_data
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    def on_epoch_end(self, epoch, logs=None):
        if hvd.rank() == 0:
            epoch_time = time.time() - self.epoch_start_time
            print("Epoch time : {}".format(epoch_time))
            data_per_sec = round(self.total_data/epoch_time, 2)
            print("Data/sec : {}".format(data_per_sec))
            #print("Horovod size : {}".format(hvd.size()))


callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    Throughput(total_data=len(y_train_cat))
]


# Define the checkpoint directory to store the checkpoints.
#checkpoint_dir = './training_checkpoints'
# Define the name of the checkpoint files.
#checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

verbose = 1 if hvd.rank() == 0 else 0

print("before distributed model training with horovod of size ", hvd.size())

##########################################################

### old code before hvd ###
#model.compile(
#    optimizer='adam', 
#    loss='categorical_crossentropy', 
#    metrics=['accuracy']
#)
#print(model.summary())
#######

#If starting with pre-trained weights. 
#model.load_weights('???.hdf5')
import time

###### start horovod model training #######################
t0 = time.time()

history = model.fit(dataset, 
                    steps_per_epoch=len(y_train_cat) // (BATCH_SIZE_PER_REPLICA*hvd.size()), 
                    verbose=verbose, 
                    epochs=50, 
                    callbacks = callbacks
                    #validation_data=(X_test, y_test_cat), 
                    #class_weight=class_weights,
                    #shuffle=False
                   )
t1 = time.time()
print(f"The Training time is {t1-t0} seconds.")


if hvd.rank() == 0:
    # test the model
    begin = time.time()
    #_, acc = model.evaluate(X_test, y_test_cat)
    #print("Accuracy is = ", (acc * 100.0),  "% and hvd rank is", hvd.rank())
    test_loss, test_acc, f1_score, precision, recall = model.evaluate(X_test, y_test_cat, verbose=1)
    end = time.time()
    #print(f"The testing time is {end - begin} seconds, testing accuracy is {test_acc}, loss is {test_loss}, hvd rank {hvd.rank()}")
    print(f"The testing time is {end - begin} seconds, testing accuracy is {test_acc}, loss is {test_loss}")
    print(f"F1_score is {f1_score}, precision is {precision}, recall is {recall}")


##########################################################
#model_name = 's2_arctic_antarctic_day_night_50_batch_32.hdf5'
#if hvd.rank() == 0:
#    model.save(model_name)
#model.save('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')
############################################################
#Evaluate the model
	# evaluate model
# _, acc = model.evaluate(X_test, y_test_cat)
# print("Accuracy is = ", (acc * 100.0),  "% and hvd rank is", hvd.rank())

##################################
# #model = get_model()
# #model.load_weights('s2_multi_with_cloud.hdf5')
# model.load_weights(model_name)
# #model.load_weights('s2_multi_with_cloud_auto_labeled_100.hdf5')
# #model.load_weights('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')  

# #IOU
# y_pred=model.predict(X_test)
# y_pred_argmax=np.argmax(y_pred, axis=3)

# _, acc = model.evaluate(X_test, y_test_cat)
# print("Accuracy is = ", (acc * 100.0), "% and hvd rank is", hvd.rank())

# ###
# #plot the training and validation accuracy and loss at each epoch
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# plt.plot(epochs, acc, 'y', label='Training Accuracy')
# plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
# plt.title('Training and validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()