                                         DISEASE RECOGNITION USING NAIL SAMPLES


DESCRIPTION:

Collaborated with a startup GAIA, Microsoft Research Lab and undertook a project to build a multi-classification algorithm to predict diseases a person is prone to at a very early stage by analyzing his/her nails based on a Chinese research.

Applied computer vision to the model for reshaping images which are then used by Tensorflow for data flow.

Modeled Convolutional Neural Network algorithm to train and test datasets, proposed system achieved an accuracy of 84.3%






PROCESS:

The data is extracted from Kaggle. Computer vision is used for resizing and labeling the images.

Convolutional Neural Networks are used for training and testing of data. The layers Convolution, ReLu, and Pooling are added two times to shrink the images. At last, a Fully connected layer is added where all the shrunk and filtered images are put in a single list.

Calculation of loss function is done and Adam optimizer is used to reduce the loss. The training of the Model is done for 4 Epochs to get higher accuracy.


