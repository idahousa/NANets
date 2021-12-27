import tensorflow as tf
"""=====================================================================================================================
- type_output indicates the type of activiation function used in the output layer of a network.
- type_output can be: 'sigmoid' => Network with single object
                      'softmax' => Network with multiple objects
                      'raw'     => Network with multiple objects
====================================================================================================================="""
class SegNetLosses:
    def __init__(self):
        pass
    @staticmethod
    def get_ce(i_type_output='sigmoid'):
        """i_type_output: the activation used in the output of network"""
        assert isinstance(i_type_output,str)
        assert i_type_output in ('sigmoid','softmax','raw')
        def ce(y_true,y_pred):
            if i_type_output == 'sigmoid':#Single object with sigmoid activation at the last layer
                return tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true,y_pred)
            elif i_type_output=='softmax':#Multiple objects with sigmoid activation at the last layer
                return tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y_true,y_pred)
            else:#Multiple objects with None activation at the last layer
                return tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true,y_pred)
        return ce
    @staticmethod
    def get_wce(i_weight=None,i_type_output='sigmoid'):
        assert isinstance(i_weight,(list,tuple))
        assert isinstance(i_type_output, str)
        assert i_type_output in ('sigmoid', 'softmax', 'raw')
        epsilon = tf.keras.backend.epsilon()
        weight  = tf.convert_to_tensor(i_weight)
        def wce(y_true,y_pred):
            y_true = tf.cast(y_true, tf.float32)
            if i_type_output=='sigmoid':
                wce_val = -(y_true*tf.math.log(y_pred)*weight + (1.0-y_true)*tf.math.log(1.0-y_pred)*(1.0-weight))
            else:
                if i_type_output=='softmax':
                    pass
                else:
                    y_pred = tf.nn.softmax(y_pred, axis=-1)               # Result: (None, height, width, num_classes)
                y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)  # Result: (None, height, width, num_classes)
                wce_val = -y_true * tf.math.log(y_pred) * weight              # Broast-casting. #Result: (None, height, width, num_classes)
            return tf.reduce_mean(tf.reduce_sum(wce_val, axis=-1), axis=None)
        return wce
    @staticmethod
    def get_dice_coef(i_type_output=None):
        assert isinstance(i_type_output, str)
        assert i_type_output in ('sigmoid', 'softmax', 'raw')
        epsilon = tf.keras.backend.epsilon()
        def dice_coef(y_true, y_pred):
            """https://stackoverflow.com/questions/65125670/implementing-multiclass-dice-loss-function
               https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py"""
            """y_true and y_pred must be same shape"""
            y_true = tf.cast(y_true, tf.dtypes.float32)
            if i_type_output=='sigmoid':
                print('Output is after sigmoid')
                y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            else:
                if i_type_output=='softmax':
                    print('Output is after softmax')
                else:
                    y_pred = tf.cast(tf.nn.softmax(y_pred, axis=-1), tf.dtypes.float32)
                y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
                """Remove the background class label 0)"""
                y_true   = tf.split(y_true, num_or_size_splits=[1, -1], axis=-1)[-1]  # Shape: (None, height, width, num_class-1)
                y_pred   = tf.split(y_pred, num_or_size_splits=[1, -1], axis=-1)[-1]  # Shape: (None, height, width, num_class-1)
            y_true_f = tf.reshape(y_true, [-1])
            y_pred_f = tf.reshape(y_pred, [-1])
            intersect = tf.reduce_sum(y_true_f * y_pred_f)
            denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
            return (2. * intersect + epsilon) / (denominator + epsilon)
        return dice_coef
    @staticmethod
    def get_dice_loss(i_type_output=None):
        assert isinstance(i_type_output, str)
        assert i_type_output in ('sigmoid', 'softmax', 'raw')
        def dice_loss(y_true,y_pred):
            """https://stackoverflow.com/questions/65125670/implementing-multiclass-dice-loss-function
               https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py"""
            dice_coef = SegNetLosses.get_dice_coef(i_type_output=i_type_output)(y_true, y_pred)
            return 1.0 - dice_coef
        return dice_loss
    @staticmethod
    def get_focal_loss(i_type_output=None,i_alpha=0.25,i_gamma=2.0):
        assert isinstance(i_type_output, str)
        assert isinstance(i_alpha,float)
        assert isinstance(i_gamma,float)
        assert i_alpha>0
        assert i_gamma>0
        assert i_type_output in ('sigmoid', 'softmax', 'raw')
        epsilon = tf.keras.backend.epsilon()
        def focal_loss(y_true,y_pred):
            y_true = tf.cast(y_true, tf.dtypes.float32)
            if i_type_output=='sigmoid':
                y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
                # Calculate p_t
                p_t = tf.where(tf.math.equal(y_true, 1), y_pred, 1 - y_pred)
                # Calculate alpha_t
                alpha_factor = tf.ones_like(y_true) * i_alpha
                alpha_t      = tf.where(tf.math.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
                # Calculate cross entropy
                cross_entropy = -tf.math.log(p_t)
                weight        = alpha_t * tf.math.pow((1 - p_t), i_gamma)
                # Calculate focal loss
                loss = weight * cross_entropy
                return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
            else:
                if i_type_output=='softmax':
                    pass
                else:
                    y_pred = tf.nn.softmax(y_pred, axis=-1)
                y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
                # Calculate Cross Entropy
                cross_entropy = -y_true * tf.math.log(y_pred)
                # Calculate Focal Loss
                loss = i_alpha * tf.math.pow(1 - y_pred, i_gamma) * cross_entropy
                # Compute mean loss in mini_batch
                return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        return focal_loss
    @staticmethod
    def get_dice_fl(i_type_output=None,i_alpha=0.25,i_gamma=2.0):
        dice_fn = SegNetLosses.get_dice_loss(i_type_output=i_type_output)
        fl_fn   = SegNetLosses.get_focal_loss(i_type_output=i_type_output,i_alpha=i_alpha,i_gamma=i_gamma)
        def dice_fl(i_y_true,i_y_pred):
            dice_loss = dice_fn(y_true=i_y_true,y_pred=i_y_pred)
            fl_loss   = fl_fn(y_true=i_y_true,y_pred=i_y_pred)
            return dice_loss + fl_loss #Check the weight again (datnt)
        return dice_fl
    @staticmethod
    def get_custom_objects(i_type_output=None):
        custom_objects = {'ce'         : SegNetLosses.get_ce(i_type_output=i_type_output),
                          'wce'        : SegNetLosses.get_wce(i_type_output=i_type_output,i_weight=[0.45, 0.55]),
                          'focal_loss' : SegNetLosses.get_focal_loss(i_type_output=i_type_output,i_gamma=2.0, i_alpha=0.25),
                          'dice_loss'  : SegNetLosses.get_dice_loss(i_type_output=i_type_output),
                          'dice_coef'  : SegNetLosses.get_dice_coef(i_type_output=i_type_output),
                          'dice_fl'    : SegNetLosses.get_dice_fl(i_type_output=i_type_output,i_gamma=2.0,i_alpha=0.25)}
        return custom_objects
    @staticmethod
    def compile(i_net=None, i_type_output=None, i_lr=0.001, i_loss_name='wCE', i_weights=None):
        assert isinstance(i_loss_name, str)
        assert i_loss_name in ('CE','wCE', 'swCE', 'FL', 'Dice', 'swCEnDice', 'Dice_FL')
        assert isinstance(i_weights, (list, tuple, tf.Tensor, tf.SparseTensor))
        if i_loss_name == 'CE':
            loss = SegNetLosses.get_ce(i_type_output=i_type_output)
        elif i_loss_name == 'wCE':
            loss = SegNetLosses.get_wce(i_type_output=i_type_output,i_weight=i_weights)
        elif i_loss_name == 'FL':
            loss = SegNetLosses.get_focal_loss(i_type_output=i_type_output,i_gamma=2.0, i_alpha=0.25)
        elif i_loss_name == 'Dice':
            loss = SegNetLosses.get_dice_loss(i_type_output=i_type_output)
        elif i_loss_name == 'Dice_FL':
            loss = SegNetLosses.get_dice_fl(i_type_output=i_type_output,i_gamma=2.0,i_alpha=0.25)
        else:
            loss = SegNetLosses.get_ce(i_type_output=i_type_output)
        i_net.compile(optimizer = tf.keras.optimizers.Adam(lr=i_lr),
                      loss      = loss,
                      metrics   = ['accuracy', SegNetLosses.get_dice_coef(i_type_output=i_type_output)])
        return i_net
if __name__ == '__main__':
    print('This module is to implement loss functions for segmentation networks')
"""=================================================================================================================="""