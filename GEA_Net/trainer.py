# trainer.py

import os
import time
import tensorflow as tf
from sklearn.metrics import classification_report
from data_utils import generate_batches


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.sess = None
        self.saver = None

    def train_epoch(self, train_X, train_y, adj_init, epoch):
        batches, num_batches = generate_batches(
            train_X, train_y, self.config.batch_size, is_train=True
        )
        lr = self.config.learning_rate * (0.95 ** epoch)
        total_loss = 0
        total_acc = 0

        for batch_X, batch_y in batches:
            loss, acc = self.model.train_step(
                self.sess, batch_X, batch_y, adj_init, lr, self.config.dropout_rate
            )
            total_loss += loss
            total_acc += acc

        return total_loss / num_batches, total_acc / num_batches

    def evaluate(self, test_X, test_y, adj_init):
        batches, num_batches = generate_batches(test_X, test_y, 256, is_train=False)
        total_loss = 0
        total_acc = 0
        all_preds = []
        all_labels = []

        for batch_X, batch_y in batches:
            loss, acc, preds = self.model.eval_step(self.sess, batch_X, batch_y, adj_init)
            total_loss += loss
            total_acc += acc
            all_preds.extend(preds)
            all_labels.extend(batch_y)

        return total_loss / num_batches, total_acc / num_batches, all_preds, all_labels

    def train(self, train_X, train_y, val_X, val_y, adj_init):
        best_val_acc = 0.0

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            train_loss, train_acc = self.train_epoch(train_X, train_y, adj_init, epoch)
            val_loss, val_acc, _, _ = self.evaluate(val_X, val_y, adj_init)

            epoch_time = time.time() - epoch_start

            print(f"Epoch {epoch + 1:3d}/{self.config.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Time: {epoch_time:.2f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.saver.save(self.sess, self.config.check_points, write_meta_graph=False)
                print(f"  -> Best model saved! (Val Acc: {val_acc:.4f})")

        return best_val_acc

    def test(self, test_X, test_y, adj_init):
        self.saver.restore(self.sess, self.config.check_points)
        test_loss, test_acc, test_preds, test_labels = self.evaluate(test_X, test_y, adj_init)

        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds, digits=4))

        return test_acc, test_preds

    def run(self, train_X, train_y, val_X, val_y, test_X, test_y, adj_init):
        os.makedirs(os.path.dirname(self.config.check_points), exist_ok=True)

        with tf.Session(graph=self.model.graph, config=self.model.config_proto) as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            print("\n" + "=" * 70)
            print("Starting training...")
            print("=" * 70)

            best_val_acc = self.train(train_X, train_y, val_X, val_y, adj_init)

            print("\n" + "=" * 70)
            print("Evaluating on test set...")
            print("=" * 70)

            test_acc, _ = self.test(test_X, test_y, adj_init)

            return best_val_acc, test_acc