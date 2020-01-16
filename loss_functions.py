
import keras.backend as K
from keras.applications.vgg19 import VGG19
import tensorflow as tf
from skimage.measure import compare_ssim

import keras_contrib.backend as KC


class SSIM(object):
    """Difference of Structural Similarity (DSSIM loss function).
    """

    def __init__(self, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0):
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2
        self.dim_ordering = K.image_data_format()
        self.backend = K.backend()

    def shape(self, x):
        return K.int_shape(x) if self.backend == 'tensorflow' else K.shape(x)

    def compute(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a
        # gradient definition in the Theano tree
        #   and cannot be used for learning
        y_true = K.variable(y_true)
        y_pred = K.variable(y_pred)
        
        kernel = [self.kernel_size, self.kernel_size]
        y_true = K.reshape(y_true, [-1] + list(self.shape(y_pred)[1:]))
        y_pred = K.reshape(y_pred, [-1] + list(self.shape(y_pred)[1:]))

        patches_pred = KC.extract_image_patches(y_pred, kernel, kernel, 'valid',
                                                self.dim_ordering)
        patches_true = KC.extract_image_patches(y_true, kernel, kernel, 'valid',
                                                self.dim_ordering)

        # Reshape to get the var in the cells
        bs, w, h, c1, c2, c3 = self.shape(patches_pred)
        patches_pred = K.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
        patches_true = K.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
        # Get mean
        u_true = K.mean(patches_true, axis=-1)
        u_pred = K.mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c1) * (2 * covar_true_pred + self.c2)
        denom = ((K.square(u_true)
                  + K.square(u_pred)
                  + self.c1) * (var_pred + var_true + self.c2))
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        
        result = K.mean(ssim)
        return K.eval(result)


def binary_crossentropy(y_true, y_pred):
    #the input tensors are expected to be logits (not passed through softmax)
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                   logits=y_pred)

def L2(y_true, y_pred):
    
    size=K.shape(y_true)
    
    
    x = K.sqrt(K.sum(K.square(y_true-y_pred), axis=[1,2,3]))/(K.cast(size[1], tf.float32)*K.cast(size[2], tf.float32)*K.cast(size[3], tf.float32))
    result = K.mean(x)
    
    return result
    
def vgg_loss(y_true, y_pred):
    #print(y_true.shape)
    #y_true = preprocess_input(y_true)
    #y_pred = preprocess_input(y_pred)
    #y_true=255*y_true
    #y_pred=y_pred*255
    input_tensor = K.concatenate([y_true, y_pred], axis=0)
    model = VGG19(input_tensor=input_tensor,weights='imagenet', include_top=False)
    
    model.trainable = False
    for l in model.layers:
        l.trainable = False

    model.compile(optimizer='rmsprop', loss='mse')
    
    print(model.summary())
    
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    layer_features = outputs_dict["block2_conv2"]
    y_true_features = layer_features[0, :, :, :]
    y_pred_features = layer_features[1, :, :, :]
     
    return K.mean(K.square(y_true_features - y_pred_features)) 

def ssim(y_true, y_pred):
    return compare_ssim(y_true, y_pred, multichannel=True)


def total_variation(y_true, y_pred):
    
    x=y_pred
    
    img_nrows=x.shape[1]
    img_ncols=x.shape[2]
    
    a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    
    #return K.sum(K.pow(a + b, 1.25))
    return K.mean(a+b)

def hinge_G_loss(y_true, y_pred):
    return K.mean(-1*y_pred)

def hinge_D_real_loss(y_true, y_pred):
    return K.mean(K.relu(1-y_pred))

def hinge_D_fake_loss(y_true, y_pred):
    return K.mean(K.relu(1+y_pred))

def hinge_D_loss(y_true, y_pred):
    return K.mean(K.relu(1-y_true*y_pred))
    

    
