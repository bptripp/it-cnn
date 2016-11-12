__author__ = 'bptripp'

# Testing responses of pre-trained AlexNet for comparison with IT

from keras.optimizers import SGD
import keras
from convnetskeras.convnets import preprocess_image_batch, convnet


def load_net(remove_last_layer=True, weights_path='weights/alexnet_weights.h5', units_to_keep=None):
    sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    model = convnet('alexnet', weights_path=weights_path, heatmap=False)

    if remove_last_layer:
        pop(model) #activation
        pop(model) #dense
        pop(model) #dropout

        # https://github.com/fchollet/keras/issues/2640
        # although it looks like this is done correctly already
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]

        # this code is from Container.__init__
        model.internal_output_shapes = [x._keras_shape for x in model.outputs]

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


def preprocess(image_files):
    """
    :param image_files:
    :return: array of image data with default params
    """
    return preprocess_image_batch(image_files, img_size=(256,256), crop_size=(227,227), color_mode="rgb")


if __name__ == '__main__':
    load_net(weights_path='./weights/alexnet_weights.h5', units_to_keep=100)
