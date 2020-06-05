                                         DISEASE RECOGNITION USING NAIL SAMPLES


DESCRIPTION:

Collaborated with a startup GAIA, Microsoft Research Lab and undertook a project to build a multi-classification algorithm to predict diseases a person is prone to at a very early stage by analyzing his/her nails based on a Chinese research.

Applied computer vision to the model for reshaping images which are then used by Tensorflow for data flow.

Modeled Convolutional Neural Network algorithm to train and test datasets, proposed system achieved an accuracy of 84.3%






PROCESS:

The data is extracted from Kaggle. Computer vision is used for resizing and labeling the images.

Convolutional Neural Networks are used for training and testing of data. The layers Convolution, ReLu, and Pooling are added two times to shrink the images. At last, a Fully connected layer is added where all the shrunk and filtered images are put in a single list.

Calculation of loss function is done and Adam optimizer is used to reduce the loss. The training of the Model is done for 4 Epochs to get higher accuracy.

The model gives an accuracy of 84.3% with loss 15.7%

AIM:

The project is based on finding the disease a human may be prone to just by analyzing his/her nails. It uses machine learning and convolutional neural networks.

The main aim of this system design is to provide an application for use in healthcare domain for prediction of diseases.The proposed system will take nail image as an input and will perform processing on input image.Then finally it will predict probable  diseases.
In health care domain many diseases can be predicted by observing color of human nails. Doctors observe nails of patient to get assistance in disease identification. Usually pink nails indicate healthy human.The need of system to analyze nails for disease prediction is because human eye is having subjectivity about colors, having limitation in resolution and small amount of color change in few pixels on nail would not be highlighted to human eyes which may lead to wrong result where as computer recognizes small color changes on nail. 

In this system human nail image is captured using camera. Captured image is uploaded onto a system and region of interest from nail area is selected from uploaded image manually. The selected area is then processed further for extracting features of nail such as color of nail. This color feature of nail is matched using simple matcher algorithm for disease prediction. In this way the system is useful in prediction of diseases in their initial stages. 

In health care domain doctors observe human nails as supporting information or symptoms for certain disease prediction. The same task is defined by the proposed model without  any human intervention. 

The model gives more accurate results than human vision, errors because it overcomes the limitations of human eye like subjectivity and resolution power.

