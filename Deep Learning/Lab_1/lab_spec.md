# Lab 1: The Bare-Metal Perceptron & The Great Keras Race

## 1. The Core Objective

Your goal is to write a custom object-oriented neural network class using **only NumPy**, use it to classify a non-linear dataset, and then write a tiny TensorFlow script to see if your custom training path matches Keras's optimization trajectory.

## 2. Architectural Specification

- **Layers:** 1 Input Layer, 1 Hidden Layer, 1 Output Layer.

- **Activations:** Hidden layer uses **ReLU**; Output layer uses **Sigmoid**.

- **Loss Function:** **Binary Cross-Entropy (BCE)**.

- **Optimizer:** Standard **Gradient Descent** (no momentum, no Adam—just raw updates).

## 3. The Structural Blueprints & Math Cheatsheet

To make your matrix operations run smoothly without shape errors, use this blueprint to track your tensor dimensions.

Let $m$ represent your batch size (e.g., 400 samples), $d$ represent your input dimensions (2 features), and $h$ represent your hidden layer size (let's use 4 hidden neurons).

### Forward Pass Mechanics

1. **Hidden Layer Linear Step:** Calculate $Z_1 = XW_1 + b_1$

    - _Shape Check:_ $X$ is $(m, d)$. To output a hidden matrix $Z_1$ of shape $(m, h)$, what must the shape of $W_1$ and $b_1$ be?

2. **Hidden Layer Activation:** Calculate $A_1 = \max(0, Z_1)$

3. **Output Layer Linear Step:** Calculate $Z_2 = A_1W_2 + b_2$

    - _Shape Check:_ $A_1$ is $(m, h)$. To output a final pre-activation scalar per sample $Z_2$ of shape $(m, 1)$, what must the shape of $W_2$ and $b_2$ be?

4. **Output Layer Activation:** Calculate $A_2 = \frac{1}{1 + e^{-Z_2}}$

### Backward Pass Mechanics (The Calculus Equations)

Your backward function needs to compute the gradients of the loss with respect to all four parameters ($W_1, b_1, W_2, b_2$). Work backward using these steps:

1. **Output Error Vector ($dZ_2$):**

    $$dZ_2 = A_2 - Y \quad (\text{Shape: } (m, 1))$$

2. **Output Gradients ($dW_2, db_2$):**

    $$dW_2 = \frac{1}{m} A_1^T \cdot dZ_2$$

    $$db_2 = \frac{1}{m} \sum_{\text{rows}} dZ_2 \quad (\text{Keep dimensions as a 2D array!})$$

3. **Hidden Activation Error ($dA_1$):** Pass the error backward through the output weights:

    $$dA_1 = dZ_2 \cdot W_2^T$$

4. **Hidden Layer Error Vector ($dZ_1$):** Backpropagate through the ReLU function. The derivative of ReLU is $1$ if the corresponding element in $Z_1 > 0$, else it is $0$.

    $$dZ_1 = dA_1 \odot \text{ReLU}'(Z_1)$$

5. **Hidden Gradients ($dW_1, db_1$):**

    $$dW_1 = \frac{1}{m} X^T \cdot dZ_1$$

    $$db_1 = \frac{1}{m} \sum_{\text{rows}} dZ_1$$

## 4. Step-by-Step Implementation Guide

### Milestone 1: Synthesize Data & Handle Shapes

1. Use `sklearn.datasets.make_moons(n_samples=400, noise=0.2, random_state=42)` to get your $X$ and $y$ arrays.

2. **Crucial Step:** Check the shape of $y$. By default, scikit-learn outputs a 1D array of shape `(400,)`. If you perform matrix operations with this, NumPy will broadcast incorrectly and silently ruin your math. Explicitly reshape it to a 2D column vector of shape `(400, 1)`.

3. Use `matplotlib` to plot $X$ to visualize the two interlocking moons.

### Milestone 2: Initialize Your Weights Correctly

Create a class called `ScratchMLP` with an `__init__` method.

1. Initialize `W1` using **He Initialization**: sample from a standard normal distribution (`np.random.randn`) and multiply it by $\sqrt{\frac{2}{\text{input\_dim}}}$.

2. Initialize `b1` as a matrix of zeros with shape `(1, hidden_dim)`.

3. Initialize `W2` using **Xavier Initialization**: sample from a normal distribution and multiply it by $\sqrt{\frac{1}{\text{hidden\_dim}}}$.

4. Initialize `b2` as a matrix of zeros with shape `(1, 1)`.

### Milestone 3: Build the Layer Methods

Inside your class, implement these foundational math helper functions:

1. `relu(Z)`: Returns $\max(0, Z)$.

2. `relu_derivative(Z)`: Returns an array of $1$s where $Z > 0$ and $0$s elsewhere.

3. `sigmoid(Z)`: Returns the standard sigmoid formula. _Tip:_ Wrap your input with `np.clip(Z, -500, 500)` before applying the exponent to prevent numerical overflow errors if weights grow large.

### Milestone 4: Write the Training Loop

1. **`forward(X)`**: Compute $Z_1, A_1, Z_2, A_2$ sequentially. Store these variables as instance attributes (e.g., `self.A1`) because your backward pass will need to read them.

2. **`backward(X, Y)`**: Implement the equations from the math section above. Return the final four gradient matrices (`dW1`, `db1`, `dW2`, `db2`).

3. **`fit(X, Y, epochs, lr)`**: Loop through the number of epochs. In each iteration:

    - Run the forward pass to get predictions ($A_2$).

    - Calculate the Binary Cross-Entropy loss: $-\frac{1}{m} \sum [Y \log(A_2) + (1-Y) \log(1-A_2)]$. _Tip:_ Add a tiny epsilon value like `1e-15` inside the logs to avoid taking $\log(0)$.

    - Call your backward pass to fetch gradients.

    - Update your parameters using gradient descent: $Param = Param - (lr \times dParam)$.

    - Append the loss to a list so you can return your training history.

### Milestone 5: The Keras Validation

1. Build an identical network using `tf.keras.Sequential`.

2. Add a `Dense` layer with 4 units and `'relu'` activation, followed by a `Dense` layer with 1 unit and `'sigmoid'` activation.

3. Compile it using the standard `tf.keras.optimizers.SGD(learning_rate=0.1)` optimizer and `'binary_crossentropy'` loss.

4. Train both models for 1,500 epochs with a learning rate of 0.1. Plot both loss logs on a single line chart to verify if your scratch math mirrors Keras!

## 5. Your Interview Defense Targets

While coding this, keep these common interview questions in mind. Once you finish writing the code, make sure you can answer them clearly:

- _Why do we use `keepdims=True` or `axis=0` when computing the bias gradients ($db_1, db_2$)? What happens to the matrix shapes if we forget?_

- _If your loss curve flatlines at exactly 0.693 (which is $\ln(2)$) from the very first epoch, what structural flaw does that indicate about your network's capacity to pass gradients?_
