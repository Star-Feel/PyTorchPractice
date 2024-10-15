import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import pandas as pd
import torch.utils
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os

def plot_training_loss_acc(runner, fig_name=None,
                           fig_size=(16, 6),
                           sample_step=20,
                           loss_legend_loc="upper right",
                           acc_legend_loc="lower right",
                           train_color="#8E004D",
                           val_color="#E20079",
                           fontsize='x-large',
                           train_linestyle='-',
                           val_linestyle='--'):
    plt.figure(figsize=fig_size)

    plt.subplot(1, 2, 1)
    train_items = runner.train_step_losses[::sample_step]
    train_steps = [x[0] for x in train_items]
    train_losses = [x[1] for x in train_items]

    plt.plot(train_steps, train_losses, color=train_color, linestyle=train_linestyle, label="Train loss")
    if len(runner.val_losses) > 0:
        val_steps = [x[0] for x in runner.val_losses]
        val_losses = [x[1] for x in runner.val_losses]
        plt.plot(val_steps, val_losses, color=val_color, linestyle=val_linestyle, label="Validation loss")
    
    plt.ylabel("loss", fontsize=fontsize)
    plt.xlabel("step", fontsize=fontsize)
    plt.legend(loc=loss_legend_loc, fontsize=fontsize)

    if len(runner.val_scores) > 0:
        plt.subplot(1, 2, 2)
        plt.plot(val_steps, runner.val_scores, color=val_color, linestyle=val_linestyle, label="Validation accuracy")
        plt.ylabel("score", fontsize=fontsize)
        plt.xlabel("step", fontsize=fontsize)
        plt.legend(loc=acc_legend_loc, fontsize=fontsize)  
    
    plt.show()

class RunnerV3(object):
    def __init__(self, model, optimizer, metric, loss_fn, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.metric = metric
        self.loss_fn = loss_fn
        
        self.val_scores = []

        self.train_epoch_losses = []
        self.train_step_losses = []
        self.val_losses = []

        self.best_score = 0

    def train(self, train_dataloader, val_dataloader=None, **kwargs):
        self.model.train()
        num_epochs = kwargs.get("num_epochs", 0)
        log_steps = kwargs.get("log_steps", 100)
        eval_steps = kwargs.get("eval_steps", 0)
        save_dir = kwargs.get("save_dir", "./checkpoint_auto")
        custom_print_log = kwargs.get("custom_print_log", None)
        grad_clip = kwargs.get("grad_clip", None)

        num_training_steps = num_epochs * len(train_dataloader)

        if eval_steps:
            if self.metric is None:
                raise RuntimeError('Error: Metric can not be None!')
            if val_dataloader is None:
                raise RuntimeError('Error: val_loader can not be None!')

        global_step = 0

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for epoch in range(num_epochs):
            total_loss = 0
            for step, data in enumerate(train_dataloader):
                X, y = data
                logits = self.model(X)
                # print("logits shape:", logits.shape, "y shape:", y.shape) #del
                loss = self.loss_fn(logits, y)
                total_loss += loss
                self.train_step_losses.append((global_step, loss.item()))

                if log_steps and global_step % log_steps == 0:
                    print(f"[Train] epoch: {epoch}/{num_epochs}, step: {global_step}/{num_training_steps}, loss: {loss.item():.5f}")

                loss.backward()

                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

                if custom_print_log:
                    custom_print_log(self.model)

                self.optimizer.step()
                self.optimizer.zero_grad()

                if eval_steps > 0 and global_step != 0 and \
                    (global_step % eval_steps == 0 or global_step == (num_training_steps - 1)):
                    val_score, val_loss = self.evaluate(val_dataloader, global_step=global_step)
                    print(f"[Evaluate] val score: {val_score:.5f}, val loss: {val_loss:.5f}")
                    self.model.train()

                    if val_score > self.best_score:
                        self.save_model(save_dir)
                        print(f"[Evaluate] best accuracy performance has been updated: {self.best_score:.5f}, val loss: {val_score:.5f}")
                        self.best_score = val_score

                global_step += 1
            train_loss = (total_loss / len(train_dataloader)).item()
            self.train_epoch_losses.append(train_loss)
        
        print("[Train] Training done!")

    @torch.no_grad()
    def evaluate(self, val_dataloader, **kwargs):
        assert self.metric is not None
        self.model.eval()
        global_step = kwargs.get("global_step", -1)
        total_loss = 0.
        self.metric.reset()
        
        for batch_id, data in enumerate(val_dataloader):
            X, y = data
            logits = self.model(X)
            loss = self.loss_fn(logits, y).item()
            total_loss += loss
            self.metric.update(logits, y)

        val_loss = total_loss / len(val_dataloader)
        print(f"val_loss: {val_loss}, total_loss: {total_loss}, len: {len(val_dataloader)}")
        self.val_losses.append((global_step, val_loss))
        val_score = self.metric.accumulate()
        self.val_scores.append(val_score)
        # print(self.val_scores)

        return val_score, val_loss
    
    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        return self.model(X)
    
    def save_model(self, save_dir):
        torch.save(self.model, os.path.join(save_dir, "model.pt"))
    
    def load_model(self, model_dir):
        self.model = torch.load(model_dir)

class Accuracy(object):
    def __init__(self, is_logist=True):
        # is_logist: outputs是对率还是激活后的值
        self.num_correct =  0
        self.num_count = 0
        self.is_logist = is_logist
    
    def update(self, outputs, labels):
        """
        input:
            outputs: shape=[N, class_num]
            labels: shape=[N, 1]
        """
        if outputs.shape[1] == 1:
            outputs = torch.squeeze(outputs, -1)
            if self.is_logist:
                preds = torch.tensor(outputs >= 0, dtype=torch.float32)
            else:
                preds = torch.tensor(outputs >= 0.5, dtype=torch.float32)
        else:
            preds = torch.argmax(outputs, dim=1)
        
        labels = torch.squeeze(labels, -1)
        batch_correct = torch.sum(preds==labels).item()
        batch_count = labels.shape[0]
        # print(batch_correct, batch_count)

        self.num_correct += batch_correct
        self.num_count += batch_count

        # print("total:", self.num_correct, self.num_count)
    
    def accumulate(self):
        if self.num_count == 0:
            return 0
        # print("total:", self.num_correct, self.num_count)
        return self.num_correct / self.num_count
    
    def reset(self):
        self.num_correct = 0
        self.num_count = 0
    
    def name(self):
        return "Accuracy"