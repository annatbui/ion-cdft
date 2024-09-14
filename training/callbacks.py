import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from IPython import display


params = {"axes.labelsize": 14,
          "axes.titlesize": 16,}
plt.rcParams["axes.linewidth"] = 1
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
plt.rcParams['figure.dpi'] = 100
plt.rcParams.update(params)


# Check whether TensorFloat-32 execution is currently enabled
tf.config.experimental.enable_tensor_float_32_execution(False)

def exp_decay(epoch):
    initial_lr = 0.001
    k = 0.1
    lr = initial_lr * tf.math.exp(-k * epoch)
    return float(lr)

def lrschedule2(epoch, lr=1e-3):
    """
    Define a learning rate schedule that exponentially 
    decreases the learning rate after the first few epochs
    """
    return lr * 0.95 if epoch > 10 else lr

def lrschedule(epoch, lr):
    """
    Define a learning rate schedule that exponentially 
    decreases the learning rate after the first few epochs
    """
    return lr * 0.95 if epoch > 5 else lr

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_losses = []
        self.val_losses = []

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # Append the training loss for the epoch
        self.train_losses.append(logs.get("loss"))

        # Append the validation loss for the epoch
        self.val_losses.append(logs.get("val_loss"))

        # Clear the plot
        plt.clf()

        # Plot training loss
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Training", color="red")

        # Plot validation loss
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Validation", ls='--', color="black")

        # Label the axes
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        plt.tick_params(direction="in", which="minor", length=3)
        plt.tick_params(direction="in", which="major", length=5, labelsize=13)
        plt.grid(which="major", ls="dashed", dashes=(1, 3), lw=0.8, zorder=0)
        plt.legend(frameon=True, loc="best", fontsize=12,edgecolor="black")

        # Add a legend
        #plt.legend()

        # Clear the output and display the plot
        display.clear_output(wait=True)
        display.display(plt.gcf())

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass