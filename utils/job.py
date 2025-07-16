import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns


class Job:
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        model: nn.Module,
        device: str,
        loss_fn,
        metrics_fn,
        optimizer,
        verbose=False,
        step=5
    ) -> None:
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn
        self.optimizer = optimizer
        self.verbose = verbose
        self.step=step

    def _train(self, epoch):
        self.model.train()

        total_loss = 0
        total_acc = 0

        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.long().to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(X)
            loss = self.loss_fn(logits, y)

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            accuracy = self.metrics_fn(preds, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy

            if batch % self.step == 0:
                loss, current = loss.item(), batch
                step = batch // 100 * (epoch + 1)

                # Log to MLFlow
                # mlflow.log_metric("train_loss", loss, step=step)
                # mlflow.log_metric("train_accuracy", accuracy, step=step)

                # Log to terminal
                if self.verbose:
                    print(
                        f"loss: {loss:2f} accuracy: {accuracy:2f} [{current + self.step} / {len(self.train_dataloader)}]"
                    )

        avg_train_loss = total_loss / len(self.train_dataloader)
        avg_train_acc = total_acc / len(self.train_dataloader)

        mlflow.log_metric("train_loss", avg_train_loss, step=step)
        mlflow.log_metric("train_accuracy", avg_train_acc, step=step)

        return avg_train_acc, avg_train_loss

    def _validate(self, epoch):
        num_batches = len(self.val_dataloader)

        self.model.eval()
        eval_loss, eval_accuracy = 0, 0

        with torch.no_grad():
            for X, y in self.val_dataloader:
                X, y = X.to(self.device), y.long().to(self.device)

                logits = self.model(X)
                preds = torch.argmax(F.softmax(logits, dim=1), dim=1)

                eval_loss += self.loss_fn(logits, y).item()
                eval_accuracy += self.metrics_fn(preds, y)

        eval_loss /= num_batches
        eval_accuracy /= num_batches
        mlflow.log_metric("eval_loss", eval_loss, step=epoch)
        mlflow.log_metric("eval_accuracy", eval_accuracy, step=epoch)

        if self.verbose:
            print(
                f"Eval metrics: Accuracy: {eval_accuracy:.2f}, Avg loss: {eval_loss:2f} "
            )

        return eval_accuracy, eval_loss

    def _log_learning_curves(self, train_accs, train_losses, eval_accs, eval_losses):
        epochs = range(1, len(train_accs) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        train_accs = [acc.cpu().item() if torch.is_tensor(acc) else acc for acc in train_accs]
        eval_accs = [acc.cpu().item() if torch.is_tensor(acc) else acc for acc in eval_accs]

        # Accuracy
        axes[0].plot(epochs, train_accs, label="Train Acc")
        axes[0].plot(epochs, eval_accs, label="Eval Acc")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Training & Evaluation Accuracy")
        axes[0].legend()

        train_losses = [acc.cpu().item() if torch.is_tensor(acc) else acc for acc in train_losses]
        eval_losses = [acc.cpu().item() if torch.is_tensor(acc) else acc for acc in eval_losses]

        # Loss
        axes[1].plot(epochs, train_losses, label="Train Loss")
        axes[1].plot(epochs, eval_losses, label="Eval Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Training & Evaluation Loss")
        axes[1].legend()

        fig.tight_layout()

        mlflow.log_figure(fig, "learning_curves.png")

        plt.close(fig)

    def fit(self, epochs):
        train_accuracies = []
        train_losses = []
        val_accuracies = []
        val_losses = []

        for t in range(epochs):
            if self.verbose:
                print(f"Epoch {t + 1} -------------------------------")

            train_acc, train_loss = self._train(epoch=t)
            val_acc, val_loss = self._validate(epoch=t)

            train_accuracies.append(train_acc)
            train_losses.append(train_loss)
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)

            if self.verbose:
                print("\n")

        self._log_learning_curves(
            train_accuracies, train_losses, val_accuracies, val_losses
        )

        return val_loss

    def test(self, model):
        model.eval()
        test_preds, test_labels = [], []

        with torch.no_grad():
            for batch in self.test_dataloader:
                inputs, labels = batch
                labels = labels.long()
                logits = model(inputs)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        # Accuracy
        acc = accuracy_score(test_labels, test_preds)
        print(f"Test Accuracy: {acc:.4f}")

        # Classification Report
        test_dataset = self.test_dataloader.dataset
        target_names = test_dataset.string_classes  # type: ignore
        print(classification_report(test_labels, test_preds, target_names=target_names))

        # Confusion Matrix
        cm = confusion_matrix(test_labels, test_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=target_names,
            yticklabels=target_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - Test Set")
        plt.tight_layout()
        plt.show()
