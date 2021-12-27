import os
import tensorflow as tf
from libs.logs import Logs
"""========================Define custom callback for training procedure============================================="""
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self,i_model_path=None):
        super(CustomCallback, self).__init__()
        self.model_path = i_model_path
        self.save_path,self.model_name = os.path.split(i_model_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        else:
            pass
        self.accuracy = 0.0
    def on_epoch_begin(self, epoch, logs=None):
        pass
    def on_epoch_end(self, epoch, logs=None):
        """
        @param epoch: Current epoch index
        @type epoch: Integer
        @param logs: include acc and loss, and optionally include val_loss (if validation is enabled), and val_acc (if validation and accuracy monitoring are enabled)
        @type logs: Dictionary
        @return:
        @rtype:
        """
        """Save model to disk after every epoch"""
        if 'accuracy' in logs.keys():
            if self.accuracy < logs['accuracy']:
                Logs.log('Accuracy improved from {} to {}'.format(self.accuracy, logs['accuracy']))
                self.accuracy = logs['accuracy']
                self.model.save(filepath=self.model_path)
            else:
                pass
        else:
            self.model.save(filepath=self.model_path)
        summary_str = 'Epoch: {} '.format(epoch)
        for key in logs.keys():
            summary_str = '{} {} = {:3.6f}'.format(summary_str, key, logs[key])
        summary_str +='\n'
        Logs.log(summary_str)
    def on_batch_begin(self, batch, logs=None):
        pass
    def on_batch_end(self, batch, logs=None):
        #print('Batch end: {}'.format(logs))
        pass
    def on_train_begin(self, logs=None):
        print('Train Begin: ',logs)
        Logs.log('Start training our model...')
    def on_train_end(self, logs=None):
        print('Train End: ',logs)
        Logs.log('Finished training our model!')
"""==================================================================================="""
if __name__ == "__main__":
    print('This module is to implement custom callbacks for keras models')
    net_save_path = os.getcwd()
    log_infor     = CustomCallback(i_model_path=net_save_path)
    """
    net.fit(x            = train_db.batch(batch_size),
            epochs       = num_epochs,
            verbose      = 1,
            shuffle      = True,
            validation_data = val_db.batch(batch_size),
            callbacks  = [log_infor])
    """
"""==================================================================================="""