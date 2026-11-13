import numpy as np

np.set_printoptions(edgeitems=100, linewidth=20000)


def sigmoid(x):
    # Sigmoid activation function: sigmoid(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    # Derivative of sigmoid activation function: d_sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    """
    A neural network with:
      - 2 inputs
      - a hidden layer with 2 neurons (h1, h2)
      - an output layer with 1 neuron (o1)
    """

    def __init__(self):
        # Initialize weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Initialize biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def forward(self, x):
        self.h1_sum = self.w1 * x[0] + self.w2 * x[1] + self.b1
        self.h1 = sigmoid(self.h1_sum)
        self.h2_sum = self.w3 * x[0] + self.w4 * x[1] + self.b2
        self.h2 = sigmoid(self.h2_sum)
        self.o1_sum = self.w5 * self.h1 + self.w6 * self.h2 + self.b3
        self.y_pred = sigmoid(self.o1_sum)
        return self.y_pred

    def backward(self, x, y_true):
        # Calculate gradients for OUTPUT layer
        dL_dy_pred = -2 * (y_true - self.y_pred)

        self.dL_dw5 = dL_dy_pred * d_sigmoid(self.o1_sum) * self.h1
        self.dL_dw6 = dL_dy_pred * d_sigmoid(self.o1_sum) * self.h2
        self.dL_db3 = dL_dy_pred * d_sigmoid(self.o1_sum)

        # Calculate gradients for HIDDEN layer
        self.dL_dw1 = (
            dL_dy_pred
            * self.w5
            * d_sigmoid(self.o1_sum)
            * d_sigmoid(self.h1_sum)
            * x[0]
        )
        self.dL_dw2 = (
            dL_dy_pred
            * self.w5
            * d_sigmoid(self.o1_sum)
            * d_sigmoid(self.h1_sum)
            * x[1]
        )
        self.dL_dw3 = (
            dL_dy_pred
            * self.w6
            * d_sigmoid(self.o1_sum)
            * d_sigmoid(self.h2_sum)
            * x[0]
        )
        self.dL_dw4 = (
            dL_dy_pred
            * self.w6
            * d_sigmoid(self.o1_sum)
            * d_sigmoid(self.h2_sum)
            * x[1]
        )
        self.dL_db1 = (
            dL_dy_pred * self.w5 * d_sigmoid(self.o1_sum) * d_sigmoid(self.h1_sum)
        )
        self.dL_db2 = (
            dL_dy_pred * self.w6 * d_sigmoid(self.o1_sum) * d_sigmoid(self.h2_sum)
        )

    def update_weights(self, lr):
        self.w1 -= lr * self.dL_dw1
        self.w2 -= lr * self.dL_dw2
        self.w3 -= lr * self.dL_dw3
        self.w4 -= lr * self.dL_dw4
        self.w5 -= lr * self.dL_dw5
        self.w6 -= lr * self.dL_dw6

        self.b1 -= lr * self.dL_db1
        self.b2 -= lr * self.dL_db2
        self.b3 -= lr * self.dL_db3

    # Loss function
    def loss(self, y_trues, y_preds):
        losses = (y_trues - y_preds) ** 2
        return losses.min(), losses.mean(), losses.max()

    def train(self, ground_truth_dataset, epoch, lr):
        # ground_truth_dataset: has shape (n, 3), where n is the number of items.
        # epoch: the number of times to loop through the entire ground truth dataset
        # lr: learning rate

        epochs = []
        min_losses = []
        avg_losses = []
        max_losses = []
        training_samples = np.array(ground_truth_dataset)
        y_trues = training_samples[:, 2]

        for ep in range(epoch):
            losses = []
            np.random.shuffle(training_samples)
            for item in training_samples:
                # input
                x = item[:2]

                # real result
                y_true = item[2]

                self.forward(x)
                self.backward(x, y_true)
                self.update_weights(lr)

            y_preds = np.apply_along_axis(self.forward, 1, training_samples[:, 0:2])
            min_loss, avg_loss, max_loss = self.loss(y_trues, y_preds)

            epochs.append(ep)
            min_losses.append(min_loss)
            avg_losses.append(avg_loss)
            max_losses.append(max_loss)

            if ep % 100 == 0:
                print(
                    "Epoch {}: min_loss = {}, avg_loss = {}, max_loss = {}".format(
                        ep, min_losses[ep], avg_losses[ep], max_losses[ep]
                    )
                )

            if max_loss < 1e-3:
                break

        return (epochs, min_losses, avg_losses, max_losses)


def main():
    ground_truth_dataset = [
        [15, 3, 1],
        [10, 5, 1],
        [20, 1, 1],
        [1, 5, 0],
        [5, 0, 0],
        [30, 0, 1],
        [2, 1, 0],
        [5, 5, 1],
        [7, 10, 0],
        [25, 6, 1],
    ]

    n = NeuralNetwork()
    n.train(ground_truth_dataset, 1000001, 0.001)


if __name__ == "__main__":
    main()
