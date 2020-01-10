from google.colab import drive
drive.mount("/content/gdrive/", force_remount=True)

#!ls
%cd /content/gdrive/My Drive/MURA2
#!ls

 model = Sequential()

 model.add(Convolution2D(filters=64,input_shape=(224,224,3),kernel_size=(3, 3), activation='relu',padding = 'same'))
 model.add(Convolution2D(filters=64,kernel_size=(3, 3), activation='relu',padding='same'))
 model.add(MaxPooling2D((2,2), strides=(2,2)))

    
 model.add(Convolution2D(filters=128,kernel_size=(3, 3), activation='relu',padding='same'))
 model.add(Convolution2D(filters=128,kernel_size=(3, 3), activation='relu',padding='same'))
 model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

 model.add(Convolution2D(filters=256,kernel_size=(3, 3), activation='relu',padding='same'))
 model.add(Convolution2D(filters=256,kernel_size=(3, 3), activation='relu',padding='same'))
 model.add(Convolution2D(filters=256,kernel_size=(3, 3), activation='relu',padding='same'))
 model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


 model.add(Convolution2D(filters=512,kernel_size=(3, 3), activation='relu',padding='same'))
 model.add(Convolution2D(filters=512,kernel_size=(3, 3), activation='relu',padding='same'))
 model.add(Convolution2D(filters=512,kernel_size=(3, 3), activation='relu',padding='same'))
 model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

 model.add(Convolution2D(filters=512,kernel_size=(3, 3), activation='relu',padding='same'))
 model.add(Convolution2D(filters=512,kernel_size=(3, 3), activation='relu',padding='same'))
 model.add(Convolution2D(filters=512,kernel_size=(3, 3), activation='relu',padding='same'))
 model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

 model.add(Flatten())
 model.add(Dense(units=4096, activation='relu'))
 model.add(Dense(units=4096, activation='relu'))
 model.add(Dense(units=7, activation='softmax'))
    
 print ("Create model successfully")
 sgd = SGD(lr=0.01, decay=1e-6, nesterov=True)
 model.compile(optimizer=sgd,loss='categorical_crossentropy', metrics=['accuracy'])
    
#from keras.utils.np.utils import to_categorical
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[1.5,1.5])

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('train',
                                                    target_size=(224, 224),
                                                    batch_size=64,
                                                    class_mode='categorical')

test_set = test_datagen.flow_from_directory('vaild',
                                            target_size=(224,224),
                                                        batch_size=32,
                                                        class_mode='categorical')
#training_set = to_categorical(training_set)
#test_set = to_categorical(test_set)

model.fit_generator(training_set,
                         steps_per_epoch=75,
                         epochs=32,
                        validation_data=test_set,
                         validation_steps=25)

model.save("mura_train87_val80.h5")
