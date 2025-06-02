# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

class TrainingHistoryPlotter:
    """
    Class for plotting training history for both phases.
    """
    @staticmethod
    def plot(history_initial, history_fine=None):
        """
        Plot training history for both phases.

        :param history_initial: Initial training history object
        :param history_fine: Fine-tuning history object (optional)
        """
        if history_initial is None:
            print("No training history available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Combine histories if fine-tuning was performed
        if history_fine is not None:
            acc = history_initial.history['accuracy'] + history_fine.history['accuracy']
            val_acc = history_initial.history['val_accuracy'] + history_fine.history['val_accuracy']
            loss = history_initial.history['loss'] + history_fine.history['loss']
            val_loss = history_initial.history['val_loss'] + history_fine.history['val_loss']
            transition_epoch = len(history_initial.epoch)
        else:
            acc = history_initial.history['accuracy']
            val_acc = history_initial.history['val_accuracy']
            loss = history_initial.history['loss']
            val_loss = history_initial.history['val_loss']
            transition_epoch = None

        epochs = range(1, len(acc) + 1)

        # Plot accuracy
        axes[0, 0].plot(epochs, acc, 'bo-', label='Training Accuracy')
        axes[0, 0].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
        if transition_epoch:
            axes[0, 0].axvline(x=transition_epoch, color='g', linestyle='--', label='Fine-tuning starts')
        axes[0, 0].set_title('Training and Validation Accuracy')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()

        # Plot loss
        axes[0, 1].plot(epochs, loss, 'bo-', label='Training Loss')
        axes[0, 1].plot(epochs, val_loss, 'ro-', label='Validation Loss')
        if transition_epoch:
            axes[0, 1].axvline(x=transition_epoch, color='g', linestyle='--', label='Fine-tuning starts')
        axes[0, 1].set_title('Training and Validation Loss')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()

        plt.tight_layout()
        plt.show()