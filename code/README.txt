===========================================================================
Training >>
-----------

Training will be done on the public training dataset.
If you want to skip downloading the data, place the 'cifar-10-batches-py' folder inside the 'data' folder.


To train this model run the following command.

$ !python main.py train data

===========================================================================
Testing on Public dataset >>
----------------------------

Testing will be done on the public testing dataset.
If you want to skip downloading the data, place the 'cifar-10-batches-py' folder inside the 'data' folder.
For testing, we need to provide the saved checkpoint of the model. Download the checkpoint file from the
gdrive path provided in the 'saved_models' folder and store it in the same folder.


To test this model using the saved_model submitted with this code, run the following command.

$ !python main.py test data --checkpoint ../saved_models/chkp_epoch240.pt

[The checkpoint file chkp_epoch240.pt should be present in the saved_models folder]

===========================================================================
Prediction on private dataset >>
--------------------------------

For predictions on the private dataset, please provide the data 'private_test_images.npy'
in the 'data' folder. Also, make sure the checkpoint file is present in the saved model directory. 

$ !python main.py predict data --checkpoint ../saved_models/chkp_epoch240.pt --save_dir ./results

===========================================================================

