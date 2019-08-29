
import keras.backend as K
from keras.applications.vgg19 import VGG19
import tensorflow as tf

def binary_crossentropy(y_true, y_pred):
    #the input tensors are expected to be logits (not passed through softmax)
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                   logits=y_pred)
    
    
def vgg_loss(y_true, y_pred):
    input_tensor = K.concatenate([y_true, y_pred], axis=0)
    model = VGG19(input_tensor=input_tensor,weights='imagenet', include_top=False)
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    layer_features = outputs_dict["block2_conv2"]
    y_true_features = layer_features[0, :, :, :]
    y_pred_features = layer_features[1, :, :, :]
     
    return K.mean(K.square(y_true_features - y_pred_features)) 


def total_variation(y_true, y_pred):
    
    images=y_pred
    
    pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
    pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
      
    a = K.square(pixel_dif1)
    b = K.square(pixel_dif2)
    
    sum_axis = [1, 2, 3]
    
    return (K.mean(a, axis = sum_axis)+K.mean(b, axis = sum_axis))


"""
def total_variation(y_true, y_pred):
    
    x=y_pred
    assert K.ndim(x) == 4
    
    img_nrows=x.shape[1]
    img_ncols=x.shape[2]
    
    a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    
    return K.sum(K.pow(a + b, 1.25))
"""