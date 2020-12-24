### YOUR CODE HERE
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs, preprocess_configs

import torch

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", help="path to the checkpoint file")
parser.add_argument("--save_dir", help="path to save the results")
parser.add_argument("mode", help="train, test or predict")
parser.add_argument("data_dir", help="path to the data")
args = parser.parse_args()

if __name__ == '__main__':

    setup_seed(0)

    if args.mode == 'train':
        train_loader, test_loader = load_data(args.data_dir, preprocess_configs)

        # initialize the model for training
        model = MyModel(model_configs)
        model.model_setup(training_configs)

        # test loader is sent to get testing accuracy even while training
        model.train(train_loader, training_configs, test_loader)
        model.evaluate(test_loader)

    elif args.mode == 'test':
        model = MyModel(model_configs, args.checkpoint)
        
        # Testing on public testing dataset
        _, test_loader = load_data(args.data_dir, preprocess_configs)
        test_acc = model.evaluate(test_loader)
        print("Test Accuracy = ", test_acc)

    elif args.mode == 'predict':
        model = MyModel(model_configs, args.checkpoint)
        
        # Predicting and storing results on private testing dataset 
        private_test_loader = load_testing_images(args.data_dir, preprocess_configs)
        predictions = model.predict_prob(private_test_loader)
        results_path = os.path.join(args.save_dir,"predictions")
        np.save(results_path, predictions)
        print("Predictions saved!")

### END CODE HERE

