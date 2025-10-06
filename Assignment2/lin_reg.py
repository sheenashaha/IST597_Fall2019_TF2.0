"""
author:-aam35
"""
import time
 
import tensorflow as tf
import matplotlib.pyplot as plt
 
# Create data
NUM_EXAMPLES = 500
 
#define inputs and outputs with some noise
X = tf.random.normal([NUM_EXAMPLES])  #inputs
noise = tf.random.normal([NUM_EXAMPLES]) #noise
y = X * 3 + 2 + noise  #true output
 
# Create variables.
W = tf.Variable(0., name='W')  # Initializing W
b = tf.Variable(0., name='b')  # Initializing b
 
 
train_steps = 2500  # Number of training iterations
learning_rate = 0.001  # Step size
 
# Define the linear predictor.
def prediction(x):
  return W * x + b
 
# Define loss functions
def squared_loss(y, y_predicted):
  return tf.reduce_mean(tf.square(y_predicted - y))
 
def huber_loss(y, y_predicted, m=1.0):
  """Huber loss."""
  error = y_predicted - y
  abs_error = tf.abs(error)
  quadratic = 0.5 * tf.square(error)
  linear = m * abs_error - 0.5 * m**2
  return tf.reduce_mean(tf.where(abs_error < m, quadratic, linear))
 
# Choose a loss function (e.g., Huber loss)
loss_type = "Huber" # You can change this to "MSE" or other defined losses
 
# Initialize variables for early stopping/learning rate scheduling
best_loss = float('inf')
patience = 100 # How many steps to wait for improvement
patience_counter = 0
lr_decay_factor = 0.9 # Factor to reduce learning rate
 
for i in range(train_steps):
    with tf.GradientTape() as tape:
        # Forward pass: compute predicted y (yhat)
        yhat = prediction(X)
 
        # Compute loss based on the selected loss function
        if loss_type == "MSE":
            loss = squared_loss(y, yhat)
        elif loss_type == "Huber":
            loss = huber_loss(y, yhat)
        # Add other loss functions here if needed
        else:
            raise ValueError("Unknown loss type selected.")
 
    # Compute gradients of loss with respect to W and b
    dW, db = tape.gradient(loss, [W, b])
 
    # Update parameters using gradient descent
    W.assign_sub(learning_rate * dW)
    b.assign_sub(learning_rate * db)
 
    # --- Learning Rate Scheduling ---
    # If the loss improves, reset the patience counter; otherwise, increase it.
    current_loss = loss.numpy()
    if current_loss < best_loss:
        best_loss = current_loss
        patience_counter = 0
    else:
        patience_counter += 1
 
    # If the loss hasn't improved for 'patience' steps, reduce the learning rate.
    if patience_counter >= patience:
        learning_rate *= lr_decay_factor
        print(f"Reducing learning rate to {learning_rate:.6f} at step {i}")
        patience_counter = 0  # Reset the counter after reducing LR
 
    # Print training progress every 500 steps
    if i % 500 == 0:
        print(f"Step {i}, Loss: {current_loss:.4f}, W: {W.numpy():.4f}, b: {b.numpy():.4f}")
 
print(f"\nFinal Model: W = {W.numpy():.4f}, b = {b.numpy():.4f}, Final Loss: {loss.numpy():.4f}")
 
plt.plot(X, y, 'bo',label='org')
plt.plot(X, W.numpy() * X + b.numpy(), 'r',
         label=f"{loss_type} regression")
plt.legend()
plt.show()
