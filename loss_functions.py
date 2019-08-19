
import keras.backend as K
from keras.applications.vgg19 import VGG19
# Define custom loss
def vgg_loss(y_true, y_pred):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    
    vggmodel = VGG19(include_top=False)
    f_p = vggmodel(y_pred)  
    f_t = vggmodel(y_true)  
    return K.mean(K.square(f_p - f_t)) 