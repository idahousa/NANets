import numpy as np
import tensorflow as tf
"""=================================================================================================================="""
class LearningRateScheduler:
    def __init__(self):
        pass
    """0.Time decay schedule"""
    @staticmethod
    def time_decay_fn(i_max_epoch = 100, i_decay_rate=0.99):
        assert isinstance(i_max_epoch,int)
        assert isinstance(i_decay_rate,float)
        assert 0<i_decay_rate<=1.
        assert i_max_epoch>0
        def time_decay(epoch, lr):
            if 0<epoch < i_max_epoch:
                return lr * i_decay_rate
            else:
                return lr
        return time_decay
    """1.Step decay schedule"""
    @staticmethod
    def step_decay_fn(i_step=10,i_decay_rate=0.5):
        assert isinstance(i_step,int)
        assert i_step >= 0
        assert isinstance(i_decay_rate,float)
        assert 0<i_decay_rate<=1.
        if i_step<=0:
            i_step = 10
        else:
            pass
        def step_decay(epoch,lr):
            coef     = epoch//i_step
            reminder = epoch - coef*i_step
            if reminder==0 and epoch>0:
                lr_new = lr*i_decay_rate
            else:
                lr_new = lr
            return lr_new
        return step_decay
    """2.Exponential decay schedule"""
    @staticmethod
    def exponential_decay_fn(i_decay_rate=0.1):
        assert isinstance(i_decay_rate,float)
        assert 0<i_decay_rate<=1.
        def exponential_decay(epoch,lr):
            return lr*np.exp(-i_decay_rate*epoch)
        return exponential_decay
    """3.Cyclical schedule"""
    @staticmethod
    def cyclical_schedule_fn(i_step=10, i_base_lr=0.001,i_custom=False):
        """Change from i_base_lr to i_max_lr cyclically"""
        i_max_lr = i_base_lr
        i_min_lr = i_max_lr*0.01 #Reduce 100 times
        def cyclical_schedule(epoch, lr):
            if i_custom: #DATNT - Custom cyclic schedule as datnt's idea
                if epoch<=i_step:
                    return i_base_lr
                else:
                    pass
            else:
                pass
            coef     = epoch // (i_step * 2)
            reminder = epoch - coef * 2 * i_step
            max_lr   = i_min_lr + (i_max_lr - i_min_lr) * np.exp(-0.5*coef) #Define by DATNT
            size     = (max_lr - i_min_lr) / i_step
            if reminder < i_step:
                return lr+size
                #return reminder * size + i_min_lr
            else:
                return lr-size
                #return (i_step - reminder) * size + max_lr
        return cyclical_schedule
    """Call function"""
    def __call__(self, i_params=None):
        """
        @param i_params: Dictionary of parameters. Format: i_params = {'decay_rule':0,'step': 10, 'decay_rate': 0.9}
        @return: Decay function object
        @rtype:
        """
        if i_params is None:
            i_params = {'decay_rule':0,'step': 10, 'decay_rate': 0.9,'base_lr':0.0001,'max_lr':0.01}
        else:
            pass
        assert isinstance(i_params,dict)
        if i_params['decay_rule']==0:
            print('Using Time Learning Rate Schedule')
            return tf.keras.callbacks.LearningRateScheduler(self.time_decay_fn(i_decay_rate=i_params['decay_rate']),verbose=0)
        elif i_params['decay_rule']==1:
            print('Using Step Learning Rate Schedule')
            return tf.keras.callbacks.LearningRateScheduler(self.step_decay_fn(i_step=i_params['step'],i_decay_rate=i_params['decay_rate']),verbose=0)
        elif i_params['decay_rule']==2:
            print('Using Exponential Learning Rate Schedule')
            return tf.keras.callbacks.LearningRateScheduler(self.exponential_decay_fn(i_decay_rate=i_params['decay_rate']),verbose=0)
        elif i_params['decay_rule']==3:
            print('Using Cyclical Learning Rate Schedule')
            return tf.keras.callbacks.LearningRateScheduler(self.cyclical_schedule_fn(i_step=i_params['step'],i_base_lr=i_params['base_lr']), verbose=0)
        elif i_params['decay_rule']==4:
            print('Using MODIFIED Cyclical Learning Rate Schedule')
            return tf.keras.callbacks.LearningRateScheduler(self.cyclical_schedule_fn(i_step=i_params['step'], i_base_lr=i_params['base_lr'],i_custom=True), verbose=0)
        else:
            raise Exception('Invalid decay schedule plan')
"""=================================================================================================================="""
if __name__=="__main__":
    print('This module is to implement various learning rate schedule method for training DNN models')
    lr_schuler = LearningRateScheduler()
    lr_schuler_params = {'decay_rule': 0, 'step': 100, 'decay_rate': 0.999, 'base_lr': 0.0001}
    callback = [lr_schuler(lr_schuler_params)]#Use this callback in the model.fit as follows:
    """
    net.fit(x            = train_db.batch(batch_size),
            epochs       = num_epochs,
            verbose      = 1,
            shuffle      = True,
            validation_data = val_db.batch(batch_size),
            callbacks    = callback)
    """
"""=================================================================================================================="""