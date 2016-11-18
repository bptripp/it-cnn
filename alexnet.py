__author__ = 'bptripp'

# Testing responses of pre-trained AlexNet for comparison with IT

from keras.optimizers import SGD
from keras.layers import Flatten
import keras
import numpy as np
from convnetskeras.convnets import preprocess_image_batch, convnet


def load_net(remove_last_layer=True, weights_path='weights/alexnet_weights.h5', units_to_keep=None, remove_level=1):

    sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    model = convnet('alexnet', weights_path=weights_path, heatmap=False)

    if remove_last_layer:
        pop(model) #activation

        if remove_level > 0:
            pop(model) #dense
            pop(model) #dropout

        if remove_level > 1:
            pop(model) #dense
            pop(model) #dropout

        # https://github.com/fchollet/keras/issues/2640
        # although it looks like this is done correctly already
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]

        # this code is from Container.__init__
        model.internal_output_shapes = [x._keras_shape for x in model.outputs]

        # for layer in model.layers:
        #     print(layer)

    model.compile(optimizer=sgd, loss='mse')

    return model


def load_vgg(remove_last_layer=True, weights_path='weights/vgg16_weights.h5', units_to_keep=None, remove_level=1):

    sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    model = convnet('vgg_16', weights_path=weights_path, heatmap=False)

    if remove_last_layer:
        pop(model) #activation

        if remove_level > 0:
            pop(model) #dense
            pop(model) #dropout

        if remove_level > 1:
            pop(model) #dense
            pop(model) #dropout

        if remove_level > 2:
            pop(model) #dense
            pop(model) #flatten
            model.add(Flatten(name='flatten'))

        if remove_level > 3:
            pop(model) #flatten added above
            pop(model) #maxpool
            pop(model) #conv
            pop(model) #zeropad
            pop(model) #conv
            pop(model) #zeropad
            pop(model) #conv
            pop(model) #zeropad
            model.add(Flatten(name='flatten'))

        if remove_level > 4:
            pop(model) #flatten added above
            pop(model) #maxpool
            pop(model) #conv
            pop(model) #zeropad
            pop(model) #conv
            pop(model) #zeropad
            pop(model) #conv
            pop(model) #zeropad
            model.add(Flatten(name='flatten'))


        # https://github.com/fchollet/keras/issues/2640
        # although it looks like this is done correctly already
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]

        # this code is from Container.__init__
        model.internal_output_shapes = [x._keras_shape for x in model.outputs]

        # for layer in model.layers:
        #     print(layer)

    model.compile(optimizer=sgd, loss='mse')

    return model

# adapted from https://github.com/fchollet/keras/issues/2371
def pop(model):
    '''Removes a layer instance on top of the layer stack.
    '''
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    else:
        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-1].outbound_nodes = []
            model.outputs = [model.layers[-1].output]
        model.built = False


def preprocess(image_files, use_vgg=False):
    """
    :param image_files:
    :return: array of image data with default params
    """
    if use_vgg:
        return preprocess_image_batch(image_files, img_size=(256,256), crop_size=(224,224), color_mode="bgr")
    else:
        return preprocess_image_batch(image_files, img_size=(256,256), crop_size=(227,227), color_mode="rgb")

    # foo = np.moveaxis(foo, 1, 3)
    # print(foo.shape)


if __name__ == '__main__':
    load_net(weights_path='./weights/alexnet_weights.h5', units_to_keep=100)
