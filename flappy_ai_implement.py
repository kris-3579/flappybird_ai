from __future__ import division, print_function
import tensorflow as tf
from keras.layers import Activation, Dense, Flatten, Conv2D
from keras.optimizers import Adam
from skimage.transform import resize
import collections
import numpy as np
import os

import flappy_ai

def preprocess_images(images):
    if images.shape[0] < 4:
        # single image
        img = images[0]
        img = resize(img, (80, 80))
        img = img.astype("float")
        img /= 255.0
        new_images = np.stack((img, img, img, img), axis=2)
    else:
        # 4 images
        img_list = []
        for i in range(images.shape[0]):
            img = resize(images[i], (80, 80))
            img = img.astype("float")
            img /= 255.0
            img_list.append(img)
        new_images = np.stack((img_list[0], img_list[1], img_list[2], img_list[3]),
                       axis=2)
    new_images = np.expand_dims(new_images, axis=0)
    return new_images


def get_next_batch(experience, model, num_actions, gamma, batch_size):
    batch_indices = np.random.randint(low=0, high=len(experience),
                                      size=batch_size)
    batch = [experience[i] for i in batch_indices]
    X = np.zeros((batch_size, 80, 80, 4))
    Y = np.zeros((batch_size, num_actions))
    for i in range(len(batch)):
        new_images, a_t, r_t, new_imagesp1, game_over = batch[i]
        X[i] = new_images
        Y[i] = model.predict(new_images)[0]
        Q_sa = np.max(model.predict(new_imagesp1)[0])
        if game_over:
            Y[i, a_t] = r_t
        else:
            Y[i, a_t] = r_t + gamma * Q_sa
    return X, Y

# initialize parameters
data_path = "/Users/kristofreid/Desktop/flapbird_ai"
num_actions = 2  # number of valid actions (left, stay, right)
GAMMA = 0.99  # decay rate of past observations
initial_epsilon = 0.1  # starting value of epsilon
final_epsilon = 0.0001  # final value of epsilon
mem_size = 50000  # number of previous transitions to remember
num_epochs_observe = 100
num_epochs_train = 2000

BATCH_SIZE = 32
num_epochs = num_epochs_observe + num_epochs_train

# build the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=8, strides=4,
                 kernel_initializer="normal",
                 padding="same",
                 input_shape=(80, 80, 4)))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Conv2D(64, kernel_size=4, strides=2,
                 kernel_initializer="normal",
                 padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=1,
                 kernel_initializer="normal",
                 padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, kernel_initializer="normal"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Dense(3, kernel_initializer="normal"))

model.compile(optimizer= "adam", loss="mse")

# train network
game = flappy_ai.flappy_game()
experience = collections.deque(maxlen=mem_size)

fout = open(os.path.join(data_path, "rl-network-results.tsv"), "r")
num_games, num_wins = 0, 0
epsilon = initial_epsilon
for e in range(num_epochs):
    loss = 0.0
    game.reset()

    # get first state
    a_0 = 1  # (0 = left, 1 = stay, 2 = right)
    img, r_0, game_over = game.step(a_0)
    new_images = preprocess_images(img)

    while not game_over:
        new_imagesm1 = new_images
        # next action
        if e <= num_epochs_observe:
            a_t = np.random.randint(low=0, high=num_actions, size=1)[0]
        else:
            if np.random.rand() <= epsilon:
                a_t = np.random.randint(low=0, high=num_actions, size=1)[0]
            else:
                q = model.predict(new_images)[0]
                a_t = np.argmax(q)

        # apply action, get reward
        img, r_t, game_over = game.step(a_t)
        new_images = preprocess_images(img)
        # if reward, increment num_wins
        if r_t == 1:
            num_wins += 1
        # store experience
        experience.append((new_imagesm1, a_t, r_t, new_images, game_over))

        if e > num_epochs_observe:
            # finished observing, now start training
            # get next batch
            X, Y = get_next_batch(experience, model, num_actions,
                                  GAMMA, BATCH_SIZE)
            loss += model.train_on_batch(X, Y)

    # reduce epsilon gradually
    if epsilon > final_epsilon:
        fout = open(os.path.join(data_path, "rl-network-results.tsv"), "w")
        epsilon -= (initial_epsilon - final_epsilon) / num_epochs

    print("Epoch {:04d}/{:d} | Loss {:.5f} | Win Count: {:d}"
          .format(e + 1, num_epochs, loss, num_wins))
    fout.write("{:04d}\t{:.5f}\t{:d}\n".format(e + 1, loss, num_wins))

    if e % 100 == 0:
        model.save(os.path.join(data_path, "rl-network.h5"), overwrite=True)

fout.close()
model.save(os.path.join(data_path, "rl-network.h5"), overwrite=True)
