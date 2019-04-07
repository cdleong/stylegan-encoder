import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import keras.backend as K


def load_images(images_list, img_size):
    loaded_images = list()
    for img_path in images_list:
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img = np.expand_dims(img, 0)
        loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    preprocessed_images = preprocess_input(loaded_images)
    return preprocessed_images


class PerceptualModel:
    def __init__(self, img_size, layer=9, batch_size=1, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.img_size = img_size
        self.layer = layer
        self.batch_size = batch_size

        self.perceptual_model = None
        self.ref_img_features = None
        self.features_weight = None
        self.loss = None

    def build_perceptual_model(self, generated_image_tensor):
        vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        self.perceptual_model = Model(vgg16.input, vgg16.layers[self.layer].output)
        generated_image = preprocess_input(tf.image.resize_images(generated_image_tensor,
                                                                  (self.img_size, self.img_size), method=1))
        generated_img_features = self.perceptual_model(generated_image)

        self.ref_img_features = tf.get_variable('ref_img_features', shape=generated_img_features.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.features_weight = tf.get_variable('features_weight', shape=generated_img_features.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
        
        self.loss = tf.losses.mean_squared_error(self.features_weight * self.ref_img_features,
                                                 self.features_weight * generated_img_features) / 82890.0
                                                 
        # trying to get Adam working
        


    def set_reference_images(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image = load_images(images_list, self.img_size)
        image_features = self.perceptual_model.predict_on_batch(loaded_image)

        # in case if number of images less than actual batch size
        # can be optimized further
        weight_mask = np.ones(self.features_weight.shape)
        if len(images_list) != self.batch_size:
            features_space = list(self.features_weight.shape[1:])
            existing_features_shape = [len(images_list)] + features_space
            empty_features_shape = [self.batch_size - len(images_list)] + features_space

            existing_examples = np.ones(shape=existing_features_shape)
            empty_examples = np.zeros(shape=empty_features_shape)
            weight_mask = np.vstack([existing_examples, empty_examples])

            image_features = np.vstack([image_features, np.zeros(empty_features_shape)])

        self.sess.run(tf.assign(self.features_weight, weight_mask))
        self.sess.run(tf.assign(self.ref_img_features, image_features))

    def optimize(self, vars_to_optimize, iterations=500, learning_rate=1.):
        # Colin: more optimizer choices exist besides GradientDescent. 
        # http://cs231n.github.io/neural-networks-3/
        # http://ruder.io/optimizing-gradient-descent/index.html#whichoptimizertochoose
        
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        
        
        
        
        # Default optimizer is just straight Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        
        # Colin: Let's give Adam a try?
        # Apparently AdamOptimizer has extra magic. https://stackoverflow.com/questions/47765595/tensorflow-attempting-to-use-uninitialized-value-beta1-power?rq=1
        # https://github.com/tensorflow/tensorflow/issues/8057
        # https://stackoverflow.com/questions/34001922/failedpreconditionerror-attempting-to-use-uninitialized-in-tensorflow
        # can't seem to get it working.
        # https://github.com/openai/universe-starter-agent/issues/31
        # https://stackoverflow.com/questions/41533489/how-to-initialise-only-optimizer-variables-in-tensorflow/45624533
        # https://stackoverflow.com/questions/33788989/tensorflow-using-adam-optimizer
        # https://pythonprogramming.net/tensorflow-neural-network-session-machine-learning-tutorial/        
        # Finally got it working by adding self.sess.run(tf.global_variables_initializer())
        # AFTER the definition of the minimize function.
        
        
        # Adam works OK. 
        # It keeps improving for longer, but converges slower. 
        # it got MG_4139 to 0.24 loss after 10k iterations
#         optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
        
        # AdaGRAD seems to work pretty well!
        # It gets MG_4139 to .09 in 10k iterations
#         optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)  
                
        # according to CS231n (http://cs231n.github.io/neural-networks-3/), "SGD+Nesterov Momentum" one is good: 
        # https://www.tensorflow.org/api_docs/python/tf/train/MomentumOptimizer
        # I saw momentum=0.9 in an example once
        # Momentum Optimizer got MG_4139 to .14 in 10k iterations
#         optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
        
        
        # According to http://ruder.io/optimizing-gradient-descent/index.html,
        # " RMSprop, Adadelta, and Adam are very similar algorithms that do well in similar circumstances."
        # So it should perform similar to Adam
        # https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
        
        # Also, in that same article, AdaDelta seemed to find the minimum fastest.
        # https://www.tensorflow.org/api_docs/python/tf/train/AdadeltaOptimizer
        
        min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])  # This part, with fancy optimizers, 
                                                                             # makes more vars that 
                                                                             # also need initializing...
        
        
        # IF YOU USE A FANCY OPTIMIZIER YOU NEED TO PUT THIS HERE
        # It has to go _after_ you both define the optimizer, 
        # and define the minimize operation
        # initialize optimizer variables so it don't crash.
        self.sess.run(tf.variables_initializer(optimizer.variables()))  # initialize only the optimizer vars. 
        
        print(f"About to minimize using optimizer {optimizer}")        
        
        for _ in range(iterations):
            
            _, loss = self.sess.run([min_op, self.loss])
            yield loss

