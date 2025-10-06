"""
Author: Sheena Shaha
Assignment 1 - Problem 2: Logistic Regression on Fashion-MNIST
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

print("Eager execution:", tf.executing_eagerly())

learning_rate = 0.001
batch_size = 128
n_epochs = 10
n_classes = 10
img_shape = (28, 28)

# Load Fashion-MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0
train_images = train_images.reshape(-1, 28 * 28)
test_images = test_images.reshape(-1, 28 * 28)

train_X, val_X, train_y, val_y = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)
train_y_oh = tf.one_hot(train_y, depth=n_classes)
val_y_oh = tf.one_hot(val_y, depth=n_classes)
test_y_oh = tf.one_hot(test_labels, depth=n_classes)

train_data = tf.data.Dataset.from_tensor_slices((train_X, train_y_oh)).shuffle(10000).batch(batch_size)
val_data = tf.data.Dataset.from_tensor_slices((val_X, val_y_oh)).batch(batch_size)
test_data = tf.data.Dataset.from_tensor_slices((test_images, test_y_oh)).batch(batch_size)

W = tf.Variable(tf.random.normal([784, n_classes], stddev=0.01))
b = tf.Variable(tf.zeros([n_classes]))

def model(x):
    return tf.matmul(x, W) + b

def loss_fn(y_true, logits):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits))

optimizer = tf.optimizers.Adam(learning_rate)

def compute_accuracy(logits, labels):
    preds = tf.argmax(logits, axis=1)
    true = tf.argmax(labels, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(preds, true), tf.float32))

print("\nStarting training...\n")
for epoch in range(n_epochs):
    epoch_loss = 0.0
    n_batches = 0
    start = time.time()

    for x_batch, y_batch in train_data:
        with tf.GradientTape() as tape:
            logits = model(x_batch)
            loss_val = loss_fn(y_batch, logits)
        grads = tape.gradient(loss_val, [W, b])
        optimizer.apply_gradients(zip(grads, [W, b]))
        epoch_loss += loss_val.numpy()
        n_batches += 1

    val_logits = model(val_X)
    val_acc = compute_accuracy(val_logits, val_y_oh).numpy()
    avg_loss = epoch_loss / n_batches
    print(f"Epoch {epoch+1:02d}/{n_epochs} | Loss={avg_loss:.4f} | Val Acc={val_acc*100:.2f}% | Time={time.time()-start:.2f}s")

print("\nTraining complete ✅\n")

test_logits = model(test_images)
test_acc = compute_accuracy(test_logits, test_y_oh).numpy()
print(f"✅ Final Test Accuracy: {test_acc*100:.2f}%")

def plot_images(images, y_true, y_pred=None):
    assert len(images) == len(y_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        xlabel = f"True: {y_true[i]}" if y_pred is None else f"True: {y_true[i]}, Pred: {y_pred[i]}"
        ax.set_xlabel(xlabel)
        ax.set_xticks([]); ax.set_yticks([])
    plt.show()

sample_imgs = test_images[:9]
y_true = test_labels[:9]
y_pred = tf.argmax(tf.nn.softmax(model(sample_imgs)), axis=1).numpy()
plot_images(sample_imgs, y_true, y_pred)

def plot_weights(W):
    W = W.numpy()
    w_min, w_max = np.min(W), np.max(W)
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        if i < 10:
            image = W[:, i].reshape(img_shape)
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')
            ax.set_xlabel(f"Class {i}")
        ax.set_xticks([]); ax.set_yticks([])
    plt.show()

plot_weights(W)

rf = RandomForestClassifier(n_estimators=50)
rf.fit(train_X[:5000], train_y[:5000])
print("RandomForest test accuracy:", rf.score(test_images[:2000], test_labels[:2000]))

svm = SVC(kernel="linear")
svm.fit(train_X[:2000], train_y[:2000])
print("SVM test accuracy:", svm.score(test_images[:2000], test_labels[:2000]))

W_np = W.numpy().T
kmeans = KMeans(n_clusters=10, random_state=0).fit(W_np)
tsne = TSNE(n_components=2, random_state=0).fit_transform(W_np)
plt.scatter(tsne[:, 0], tsne[:, 1], c=kmeans.labels_, cmap='tab10')
plt.title("t-SNE of Class Weights (Clustered)")
plt.show()
