                                         DISEASE RECOGNITION USING NAIL SAMPLES

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

ALGORITHM:

The name “convolutional neural network” indicates that the network employs a mathematical operation called convolution.Convolution is a specialized kind of linear operation. Convolutional networks are simply neural networks that use convolution in place of general matrix multiplication in at least one of their layers.

A convolutional neural network consists of an input and an output layer, as well as multiple hidden layers.  The activation function is commonly a RELU layer, and is subsequently followed by additional convolutions such as pooling layers, fully connected layers and normalization layers, referred to as hidden layers because their inputs and outputs are masked by the activation function and final convolution.

When programming a CNN, the input is a tensor with shape (number of images) x (image width) x (image height) x (image depth). Then after passing through a convolutional layer, the image becomes abstracted to a feature map, with shape (number of images) x (feature map width) x (feature map height) x (feature map channels). 

A convolutional layer within a neural network should have the following attributes:
      1. Convolutional kernels defined by a width and height (hyper-parameters).
      2.  The number of input channels and output channels (hyper-parameter).
      3.  The depth of the Convolution filter (the input channels) must be equal to the  number channels (depth) of the input feature map.
Convolutional layers convolve the input and pass its result to the next layer. This is similar to the response of a neuron in the visual cortex to a specific stimulus.

Pooling: Convolutional networks may include local or global pooling layers to streamline the underlying computation. Pooling layers reduce the dimensions of the data by combining the outputs of neuron clusters at one layer into a single neuron in the next layer. Local pooling combines small clusters, typically 2 x 2. Global pooling acts on all the neurons of the convolutional layer.

In addition, pooling may compute a max or an average. Max pooling uses the maximum value from each of a cluster of neurons at the prior layer.

ReLU layer:ReLU is the abbreviation of rectified linear unit, which applies the non-saturating activation function {\textstyle f(x)=\max(0,x)}.

The proposed system guides in such scenario to take decision in disease diagnosis. The input to the proposed system is person nail image. The system will process an image of nail and extract features of nail which is used for disease diagnosis. 

Human nail consist of various features, out of which proposed system uses nail color changes for disease diagnosis. Here, first training set data is prepared using CNN from nail images of patients of specific diseases. A feature extracted from input nail image is compared with the training data set to get result. 

Human fingernail image analysis is procedure consists of image capturing, pre-processing of image, image segmentation, segmentation of image, feature extraction

The nail features such as color, shape and texture used to predict diseases.










