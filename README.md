# Using Convolutional Neural Networks (CNN, ConvNet) for Image Classification.

## Pre-trained Models

* [Keras models for image classification with weights trained on ImageNet](https://keras.io/applications/)
* [Matlab: Pretrained Convolutional Neural Networks](https://www.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html)
  
## Useful References

  * [Francois Chollet: Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) (2016-06-05)
  * [Adrian Rosebrock: ImageNet: VGGNet, ResNet, Inception, and Xception with Keras](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/) (2017-03-20)
  * [CNN Architectures : VGG, Resnet, InceptionNet, XceptionNet @Kaggle](https://www.kaggle.com/shivamb/cnn-architectures-vgg-resnet-inception-tl) (2018-Sep)

## Datasets
  * [ImageNet/LSVRC 2012](http://image-net.org/challenges/LSVRC/2012/): 1.2 million images in 1,000 classes. Additional 150,000 validation and test images. [Dataset @Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge)
  * [COCO - Common Objects in Context](http://cocodataset.org/)
  * [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)
    * **CIFAR-10**: 60,000 32x32 colour images in **10 classes**. 6,000 images per class, 5,0000 training, 10,000 test images.
      [CIFAR-10@Kaggle](https://www.kaggle.com/c/cifar-10/data)
    * **CIFAR-100**: 60,000 32x32 colour images in **100 classes**. 600 images each. There are 500 training images and 100 testing images per class. 20 superclasses.
  * [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#data): 11,530 images in 20 classes.
  * [MNIST](http://yann.lecun.com/exdb/mnist/) - handwritten digits, 70,000 28x28 greyscale images in 10 classes. 60,000 training and 10,000 test examples. (1998)
  * [@Kaggle Cats & Dogs](https://www.kaggle.com/c/dogs-vs-cats/data) 25,000 images
  * [@Kaggle Fashion MNIST Zalando](https://www.kaggle.com/zalando-research/fashionmnist/home): 70,000 greyscale images in 10 classes. 60,000 training and 10,000 test images
  * [@Kaggle Fruits-360](https://www.kaggle.com/moltean/fruits/home): 62,037 100x100 images in 90 classes (one fruit per image). 46,371 training and 15,563 test images.
  * [VGG datasets](http://www.robots.ox.ac.uk/~vgg/data/)

## Publications

* [ResNet - Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Dec 2015.
* [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861). Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam. April 2017.
* [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381). Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. Apr 2018.
* [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626). Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, Quoc V. Le. Jul. 2018.
* [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946). Mingxing Tan, Quoc Le. May 2019.
