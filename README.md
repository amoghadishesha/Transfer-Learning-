# Transfer-Learning-

Visit this link for detailed report:
https://sites.google.com/view/transferlearning/home

In this project, we modified and retrained an existing pre-trained CNN (vgg16) to detect airplanes in the images. We run the code in Python, using vgg16 pre-trained network and subset of Caltech-101 dataset which are available online. While retraining the vgg16 network, We froze most of the layers in it and modified the last 3 layers (fully connected layer, fully connected layer, and softmax). In the end the new CNN can distinguish planes (class 1) from all other object (class 0) in the images with a reasonably good accuracy.
Framework
We used Tensorflow via Keras for implementing the Project. 
Pre-trained CNN
We chose vgg16, which is already trained on ImageNet database,  as the pre-trained Convolutional Neural Network  for this project. vgg16 takes images of size 224x224.
Data pre-processing and Data Augmentation
The pre-processing part of the project takes care of certain properties like rotation invariance, shift invariance and skew invariance. The rotation and resizing issues are taken care in the pre-trained model. Modifying the last few layers of the convolutional neural network will not be dependent on 3 features of the images while execution. With this we are making sure the network does not overfit.
Data Augmentation via number of transformations helps to avoid over-fitting and to build a generalized model.
Fine Tuning VGG16 
To "Fine-Tune" the last convolutional block of the VGG16 model alongside the top-level classifier. Fine-tuning consist in starting from a trained network, then re-training it on a new dataset using very small weight updates. In our case, this can be done in 3 steps:
Call the convolutional base of VGG16 by instantiating it.
Load the weights of the VGG16 model 
Except the last convolutional block, freeze the layers of the VGG16 model
We choose to only fine-tune the last block consisting of two fully-connected and logit layers, rather than the entire network. We aim at implementing transfer learning by using different dataset as well as we avoid overfit of the model.  Since our pre-trained network is trained on a generalized dataset, implying that low-level blocks are trained for general parameters we freeze these blocks and train the top layer.  The top-layer of  the CNN model is fine tuned for specialized features as a result we only train the top-layer of the model. 
Retraining
For the given project, we fine tune the network using SGD(stochastic gradient descent) optimizer just to keep our magnitude as low as possible. This helps in retaining the  the previously learned features for the frozen part of the network.


Pre-Trained Dataset:
Image-net Database, Stanford Vision Lab, Stanford University, Princeton University 
Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015.

We used Caltech101 dataset to create a new dataset for our project. We divided 2159 sample images into 2 classes (planes V.S. non planes) and used 1015 images to retrain the CNN, 674 images for validation and 470 images for testing our newly trained model. (validation VS testing)
www.vision.caltech.edu/Image_Datasets/Caltech101/#Description
