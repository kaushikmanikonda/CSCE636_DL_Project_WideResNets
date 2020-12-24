### YOUR CODE HERE
import os, time
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from tqdm import tqdm

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, model_configs, checkpoint_path = None):
        cudnn.benchmark = True
        self.model_configs = model_configs
        self.network = MyNetwork(model_configs)
        self.network = self.network.cuda()

        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                raise Exception("Checkpoint path does not exit: ", checkpoint_path)

            self.network.load_state_dict(torch.load(checkpoint_path))


    def model_setup(self, training_configs):
        self.network_loss = nn.CrossEntropyLoss().cuda()

        self.network_optimizer = torch.optim.SGD(self.network.parameters(), training_configs["learning_rate"],
                                        momentum=0.9, nesterov=True, weight_decay=5e-4)

        self.scheduler = MultiStepLR(self.network_optimizer, milestones=[80, 140, 200], gamma=0.2)


    def train(self,  train_loader, training_configs, test_loader):
        for epoch in range(training_configs["epochs"]):
            avg_loss = 0.
            correct = 0.
            total = 0.

            progress_bar = tqdm(train_loader)
            for i, (images, labels) in enumerate(progress_bar):
                progress_bar.set_description('Epoch ' + str(epoch))
                images, labels = images.cuda(), labels.cuda()

                self.network.zero_grad()
                pred = self.network(images)
                loss = self.network_loss(pred, labels)
                loss.backward()

                self.network_optimizer.step()
                avg_loss += loss.item()

                # Calculate running average of accuracy
                pred = torch.max(pred.data, 1)[1]
                total += labels.size(0)
                correct += (pred == labels.data).sum().item()
                accuracy = correct / total

                progress_bar.set_postfix(
                xentropy='%.3f' % (avg_loss / (i + 1)),
                acc='%.3f' % accuracy)


            test_acc = self.evaluate(test_loader)
            tqdm.write('test_acc: %.3f' % (test_acc))

            self.scheduler.step()

            row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
            print(row)

            if epoch % 10 == 0:
                name = "chkp_epoch" + str(epoch) + ".pt"
                path = f'/content/drive/MyDrive/chkp'
                checkpoint_path = os.path.join(path,name)
                if not os.path.exists(path):
                    os.makedirs(path)
                print(path)
                torch.save(self.network.state_dict(), checkpoint_path)


    def evaluate(self, test_loader):
        self.network.eval()
        correct = 0.
        total = 0.

        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()

            with torch.no_grad():
                pred = self.network(images)

            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        val_acc = correct / total
        self.network.train()
        return val_acc

    def predict_prob(self, private_test_loader):
        self.network.eval()
        pred_list = []

        for images in private_test_loader:
            images = images.cuda()

            with torch.no_grad():
                pred_list.extend(self.network(images).cpu().numpy())

        pred_linear = np.asarray(pred_list)

        # softmax to get the probablities
        pred_exp = np.exp(pred_linear)
        pred = pred_exp/np.sum(pred_exp, axis=-1, keepdims=True)
        return pred

### END CODE HERE