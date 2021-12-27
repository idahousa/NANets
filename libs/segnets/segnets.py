import os
import time
import numpy as np
import tensorflow as tf
from libs.logs import Logs
from libs.commons import SupFns
import matplotlib.pyplot as plt
from libs.segnets.TSFNet import TSFNets
from libs.segnets.metrics import SegMetrics
from libs.datasets.dataset import CSDataset
from libs.segnets.metrics import SegMetrics3D
from libs.segnets.segnetlosses import SegNetLosses
from libs.segnets.unets import UNets,NestedUnets
from libs.callbacks.callbacks import CustomCallback
from libs.callbacks.lrs import LearningRateScheduler
"""=====================================================================================================================
- The bellow class only valid if the network has one output or multiple output but all the outputs have same dimension.
(Check the type_output for detal).
- For the case of using multiple output with different dimension => Lets customize the compile function. It cannot be
generalized because of difference in outputs.
- By default, the output of network always uses activation (sigmoid for single channel, softmax for multiple channels).
====================================================================================================================="""
class ImageSegNets:
    cv_debug = False
    cv_threshold = 0.5
    def __init__(self,
                 i_net_id      = 0,
                 i_save_path   = None,
                 i_image_shape = (256,256,3),
                 i_num_classes = 1,
                 i_continue    = False,**kwargs):
        assert isinstance(i_image_shape,(list,tuple))
        assert len(i_image_shape)==3
        assert isinstance(i_num_classes,int)
        assert i_num_classes>0
        assert isinstance(i_continue,bool)
        assert isinstance(i_net_id,int)
        assert i_net_id>=0
        self.net_id      = i_net_id
        self.input_shape = i_image_shape
        self.mask_shape  = (self.input_shape[0],self.input_shape[1],1)
        self.num_classes = i_num_classes
        self.tcontinue   = i_continue
        if self.num_classes==1:
            self.type_output = 'sigmoid'
        else:
            self.type_output = 'softmax'
        """Init the save path to save model"""
        if i_save_path is None:
            self.save_path = os.path.join(os.getcwd(),'ckpts')
        else:
            assert isinstance(i_save_path,str)
            self.save_path = i_save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        else:
            pass
        self.model_path = os.path.join(self.save_path,'segnet_{}.h5'.format(self.net_id))
        """Init the model"""
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(filepath=self.model_path,custom_objects=SegNetLosses.get_custom_objects(i_type_output=self.type_output))
        else:
            self.model = None
        """Inference training parameters"""
        if 'seg_lr' in kwargs.keys():
            self.lr = kwargs['seg_lr']
        else:
            self.lr = 0.0001
        if 'seg_loss' in kwargs.keys():
            self.loss = kwargs['seg_loss']
        else:
            self.loss = 'CE'
        if 'seg_repeat' in kwargs.keys():
            self.db_repeat = kwargs['seg_repeat']
        else:
            self.db_repeat = 1
        if 'seg_epochs' in kwargs.keys():
            self.num_epochs = kwargs['seg_epochs']
        else:
            self.num_epochs = 10
        if 'seg_weight' in kwargs.keys():
            self.weight = kwargs['seg_weight']
        else:
            self.weight = [0.25,0.75]
        if 'seg_lsm_factor' in kwargs.keys():
            self.lsm_factor = kwargs['seg_lst_factor']
        else:
            self.lsm_factor = 0.0
        if 'seg_batch_size' in kwargs.keys():
            self.batch_size = kwargs['seg_batch_size']
        else:
            self.batch_size = 16
        if 'seg_ori_output' in kwargs.keys():
            self.get_original_output = kwargs['seg_ori_output']
        else:
            self.get_original_output = False
    @staticmethod
    def pipeline(i_record=None,i_ori_shape=(256,256,1),i_lsm_factor=0.0,i_num_classes=2,i_train_flag=True):
        """i_record is the output of tf.data.Dataset after doing pipeline()"""
        """i_ori_shape is the original shape of input image"""
        assert isinstance(i_record, (list, tuple, dict))
        assert isinstance(i_ori_shape, (list, tuple))
        assert isinstance(i_train_flag, bool)
        assert len(i_ori_shape) in (2, 3)
        assert isinstance(i_lsm_factor,float)
        assert 0.0<=i_lsm_factor<=1.0
        assert isinstance(i_num_classes,int)
        assert i_num_classes>0
        if len(i_ori_shape) == 2:
            image_shape = (i_ori_shape[0], i_ori_shape[1], 1)
        else:
            assert i_ori_shape[-1] in (1, 3)
            image_shape = i_ori_shape
        """Init the original shape of mask"""
        mask_shape = (i_ori_shape[0], i_ori_shape[1], 1)
        if isinstance(i_record, (list, tuple)):
            image,mask = i_record
        else:
            image,mask = i_record['image'],i_record['mask']
        assert isinstance(image, (tf.Tensor, tf.SparseTensor))
        assert isinstance(mask, (tf.Tensor, tf.SparseTensor))
        assert image.dtype in (tf.dtypes.uint8,)
        assert mask.dtype in (tf.dtypes.uint8,)
        image = tf.reshape(tensor=image, shape=image_shape)
        mask  = tf.reshape(tensor=mask, shape=mask_shape)
        if i_train_flag:
            images = tf.concat(values=(image, mask), axis=-1)
            images = tf.image.random_flip_up_down(images)
            images = tf.image.random_flip_left_right(images)
            image, mask = tf.split(value=images, num_or_size_splits=[image_shape[-1], mask_shape[-1]],axis=-1)
            """Normalization"""
            image = tf.reshape(tensor=tf.cast(image, tf.dtypes.float32), shape=image_shape) / 255.0
            mask  = tf.cast(mask, tf.dtypes.uint8)
            if i_num_classes==1:
                pass
            else:
                mask  = tf.one_hot(indices=tf.squeeze(tf.reshape(tensor=mask, shape=mask_shape)),depth=i_num_classes)
                """Label smoothing"""
                mask = mask * (1.0 - i_lsm_factor)
                mask = mask + i_lsm_factor / i_num_classes
        else:
            """Normalization"""
            image = tf.cast(image, tf.dtypes.float32) / 255.0
            mask  = tf.cast(mask, tf.dtypes.uint8)
            if i_num_classes==1:
                pass
            else:
                mask  = tf.one_hot(indices=tf.squeeze(tf.reshape(tensor=mask, shape=mask_shape)),depth=i_num_classes)
        print('image ',image.shape,mask.shape)
        return image, mask
    def init_network(self):
        if self.net_id==0:#Conventional UNet
            return UNets(i_net_name      = 'unet',
                         i_shortcut_mani = False,
                         i_block_name    = 'conv',
                         i_img_height    = self.input_shape[0],
                         i_img_width     = self.input_shape[1],
                         i_img_depth     = self.input_shape[2],
                         i_num_classes   = self.num_classes,
                         i_filters       = (64,128,256),#(32,64,128,256)
                         i_supervision   = False).build()
        elif self.net_id==1:#Conventional UNetPlus
            return UNets(i_net_name        = 'unetplus',
                         i_shortcut_mani   = False,
                         i_block_name      = 'conv',
                         i_img_height      = self.input_shape[0],
                         i_img_width       = self.input_shape[1],
                         i_img_depth       = self.input_shape[2],
                         i_num_classes     = self.num_classes,
                         i_filters         = (32,64,128,256,512),
                         i_supervision     = False).build()
        elif self.net_id==2:#Attention Unet
            return NestedUnets(i_net_name        = 'unet',
                               i_shortcut_mani   = False,
                               i_block_name      = 'conv',
                               i_img_height      = self.input_shape[0],
                               i_img_width       = self.input_shape[1],
                               i_img_depth       = self.input_shape[2],
                               i_num_classes     = self.num_classes,
                               i_shallow_filters = (32,64,128),
                               i_deep_filters    = (32,64,128,256,512),
                               i_supervision     = False).build_multiple()
        elif self.net_id==3:#Attention UNetPlus
            return NestedUnets(i_net_name        = 'unetplus',
                               i_shortcut_mani   = False,
                               i_block_name      = 'conv',
                               i_img_height      = self.input_shape[0],
                               i_img_width       = self.input_shape[1],
                               i_img_depth       = self.input_shape[2],
                               i_num_classes     = self.num_classes,
                               i_shallow_filters = (32, 64, 128),
                               i_deep_filters    = (32, 64, 128, 256, 512),
                               i_supervision     = False).build_multiple()
        elif self.net_id==4:#Residual UNet
            return UNets(i_net_name      = 'unet',
                         i_shortcut_mani = False,
                         i_block_name    = 'residual',
                         i_img_height    = self.input_shape[0],
                         i_img_width     = self.input_shape[1],
                         i_img_depth     = self.input_shape[2],
                         i_num_classes   = self.num_classes,
                         i_filters       = (64,128,256),#(64,128,128,256)
                         i_supervision   = False).build()
        elif self.net_id==5:#Residual UNetPlus
            return UNets(i_net_name        = 'unetplus',
                         i_shortcut_mani   = False,
                         i_block_name      = 'conv',
                         i_img_height      = self.input_shape[0],
                         i_img_width       = self.input_shape[1],
                         i_img_depth       = self.input_shape[2],
                         i_num_classes     = self.num_classes,
                         i_filters         = (64,128,256),
                         i_supervision     = False).build()
        elif self.net_id==6:#Attention Residual Unet
            return NestedUnets(i_net_name        = 'unet',
                               i_shortcut_mani   = False,
                               i_block_name      = 'residual',
                               i_img_height      = self.input_shape[0],
                               i_img_width       = self.input_shape[1],
                               i_img_depth       = self.input_shape[2],
                               i_num_classes     = self.num_classes,
                               i_shallow_filters = (32,64,128),
                               i_deep_filters    = (32,64,128,256),
                               i_supervision     = False).build_multiple()
        elif self.net_id==7:#Attention Residual UNetPlus
            return NestedUnets(i_net_name        = 'unetplus',
                               i_shortcut_mani   = False,
                               i_block_name      = 'residual',
                               i_img_height      = self.input_shape[0],
                               i_img_width       = self.input_shape[1],
                               i_img_depth       = self.input_shape[2],
                               i_num_classes     = self.num_classes,
                               i_shallow_filters = (32, 64, 128),
                               i_deep_filters    = (32, 64, 128, 256),
                               i_supervision     = False).build_multiple()
        elif self.net_id == 8:#Arsalan net (add)
            return TSFNets(i_input_shape=self.input_shape,i_base_filters=32,i_num_scales=3,i_fusion_rule='add').build()
        elif self.net_id == 9:  # Arsalan net (concat)
            return TSFNets(i_input_shape=self.input_shape, i_base_filters=32, i_num_scales=3,i_fusion_rule='concat',i_blocks='conv').build()
        elif self.net_id == 10:
            return TSFNets(i_input_shape=self.input_shape, i_base_filters=32, i_num_scales=3,i_fusion_rule='concat',i_blocks='residual').build()
        elif self.net_id == 11:
            return UNets(i_net_name      = 'unet',
                         i_shortcut_mani = True,
                         i_block_name    = 'residual',
                         i_img_height    = self.input_shape[0],
                         i_img_width     = self.input_shape[1],
                         i_img_depth     = self.input_shape[2],
                         i_num_classes   = self.num_classes,
                         i_filters       = (64, 128, 256),
                         i_supervision   = False).build()
        elif self.net_id==12:
            shallow_network = ImageSegNets(i_net_id      = 0,
                                           i_save_path   = self.save_path,
                                           i_image_shape = self.input_shape,
                                           i_num_classes = 1,
                                           i_continue    = False).model
            return NestedUnets(i_net_name        = 'unet',
                               i_shortcut_mani   = True,
                               i_block_name      = 'residual',  # 'conv','sk','atrous','gatrous'
                               i_img_height      = self.input_shape[0],
                               i_img_width       = self.input_shape[1],
                               i_img_depth       = self.input_shape[2],
                               i_num_classes     = self.num_classes,
                               i_shallow_filters = (16, 32, 64),
                               i_deep_filters    = (16, 32, 64, 128),
                               i_supervision     = False,
                               model             = shallow_network).build_multiple()
        elif self.net_id==13:
            shallow_network = ImageSegNets(i_net_id      = 4,
                                           i_save_path   = self.save_path,
                                           i_image_shape = self.input_shape,
                                           i_num_classes = 1,
                                           i_continue    = False).model
            return NestedUnets(i_net_name                = 'unet',
                               i_shortcut_mani           = True,
                               i_block_name              = 'residual',  # 'conv','sk','atrous','gatrous'
                               i_img_height              = self.input_shape[0],
                               i_img_width               = self.input_shape[1],
                               i_img_depth               = self.input_shape[2],
                               i_num_classes             = self.num_classes,
                               i_shallow_filters         = (16, 32, 64),
                               i_deep_filters            = (16, 32, 64, 128),
                               i_supervision             = False,
                               model                     = shallow_network).build_multiple()
        elif self.net_id == 14:
            shallow_network = ImageSegNets(i_net_id      = 9,
                                           i_save_path   = self.save_path,
                                           i_image_shape = self.input_shape,
                                           i_num_classes = 1,
                                           i_continue    = False).model
            return NestedUnets(i_net_name                = 'unet',
                               i_shortcut_mani           = True,
                               i_block_name              = 'residual',  # 'conv','sk','atrous','gatrous'
                               i_img_height              = self.input_shape[0],
                               i_img_width               = self.input_shape[1],
                               i_img_depth               = self.input_shape[2],
                               i_num_classes             = self.num_classes,
                               i_shallow_filters         = (16, 32, 64),
                               i_deep_filters            = (16, 32, 64, 128),
                               i_supervision             = False,
                               model                     = shallow_network).build_multiple()
        elif self.net_id == 15:#Conventional UNet
            return UNets(i_net_name      = 'unet',
                         i_shortcut_mani = False,
                         i_block_name    = 'conv',
                         i_img_height    = self.input_shape[0],
                         i_img_width     = self.input_shape[1],
                         i_img_depth     = self.input_shape[2],
                         i_num_classes   = self.num_classes,
                         i_filters       = (32, 64, 128, 256, 512),#32, 64, 128, 256, 512
                         i_supervision=False).build()
        elif self.net_id == 16:#Residual UNet
            return UNets(i_net_name      = 'unet',
                         i_shortcut_mani = False,
                         i_block_name    = 'residual',
                         i_img_height    = self.input_shape[0],
                         i_img_width     = self.input_shape[1],
                         i_img_depth     = self.input_shape[2],
                         i_num_classes   = self.num_classes,
                         i_filters       = (32, 64, 128, 256, 512),#32, 64, 128, 256, 512
                         i_supervision=False).build()
        elif self.net_id == 17:#Conventionial UNet++ without supervision
            return UNets(i_net_name      = 'unetplus',
                         i_shortcut_mani = False,
                         i_block_name    = 'conv',
                         i_img_height    = self.input_shape[0],
                         i_img_width     = self.input_shape[1],
                         i_img_depth     = self.input_shape[2],
                         i_num_classes   = self.num_classes,
                         i_filters       = (32, 64, 128, 256, 512),#32, 64, 128, 256, 512
                         i_supervision   = False).build()
        elif self.net_id == 18:#Conventional UNet++ with supervision
            return UNets(i_net_name      = 'unetplus',
                         i_shortcut_mani = False,
                         i_block_name    = 'conv',
                         i_img_height    = self.input_shape[0],
                         i_img_width     = self.input_shape[1],
                         i_img_depth     = self.input_shape[2],
                         i_num_classes   = self.num_classes,
                         i_filters       = (32, 64, 128, 256, 512),
                         i_supervision   = True).build()
        elif self.net_id == 19:
            shallow_network = ImageSegNets(i_net_id      = 15,
                                           i_save_path   = self.save_path,
                                           i_image_shape = self.input_shape,
                                           i_num_classes = 1,
                                           i_continue    = False).model
            return NestedUnets(i_net_name                = 'unet',
                               i_shortcut_mani           = True,
                               i_block_name              = 'residual',  # 'conv','sk','atrous','gatrous'
                               i_img_height              = self.input_shape[0],
                               i_img_width               = self.input_shape[1],
                               i_img_depth               = self.input_shape[2],
                               i_num_classes             = self.num_classes,
                               i_shallow_filters         = (16, 32, 64),
                               i_deep_filters            = (16, 32, 64, 128),
                               i_supervision             = False,
                               model                     = shallow_network).build_multiple()
        elif self.net_id == 20:
            shallow_network = ImageSegNets(i_net_id      = 16,
                                           i_save_path   = self.save_path,
                                           i_image_shape = self.input_shape,
                                           i_num_classes = 1,
                                           i_continue    = False).model
            return NestedUnets(i_net_name        = 'unet',
                               i_shortcut_mani   = True,
                               i_block_name      = 'residual',  # 'conv','sk','atrous','gatrous'
                               i_img_height      = self.input_shape[0],
                               i_img_width       = self.input_shape[1],
                               i_img_depth       = self.input_shape[2],
                               i_num_classes     = self.num_classes,
                               i_shallow_filters = (16, 32, 64),
                               i_deep_filters    = (16, 32, 64, 128),
                               i_supervision     = False,
                               model             = shallow_network).build_multiple()
        elif self.net_id == 21:
            shallow_network = ImageSegNets(i_net_id      = 17,
                                           i_save_path   = self.save_path,
                                           i_image_shape = self.input_shape,
                                           i_num_classes = 1,
                                           i_continue    = False).model
            return NestedUnets(i_net_name                = 'unet',
                               i_shortcut_mani           = True,
                               i_block_name              = 'residual',  # 'conv','sk','atrous','gatrous'
                               i_img_height              = self.input_shape[0],
                               i_img_width               = self.input_shape[1],
                               i_img_depth               = self.input_shape[2],
                               i_num_classes             = self.num_classes,
                               i_shallow_filters         = (16, 32, 64),
                               i_deep_filters            = (16, 32, 64, 128),
                               i_supervision             = False,
                               model                     = shallow_network).build_multiple()
        elif self.net_id == 22:
            shallow_network = ImageSegNets(i_net_id      = 0,
                                           i_save_path   = self.save_path,
                                           i_image_shape = self.input_shape,
                                           i_num_classes = 1,
                                           i_continue    = False).model
            return NestedUnets(i_net_name                = 'unet',
                               i_shortcut_mani           = False,
                               i_block_name              = 'conv',  # 'conv','sk','atrous','gatrous'
                               i_img_height              = self.input_shape[0],
                               i_img_width               = self.input_shape[1],
                               i_img_depth               = self.input_shape[2],
                               i_num_classes             = self.num_classes,
                               i_shallow_filters         = (16, 32, 64),
                               i_deep_filters            = (32, 64, 128, 256, 512),
                               i_supervision             = False,
                               model                     = shallow_network).build_multiple()
        elif self.net_id == 23:
            shallow_network = ImageSegNets(i_net_id      = 5,
                                           i_save_path   = self.save_path,
                                           i_image_shape = self.input_shape,
                                           i_num_classes = 1,
                                           i_continue=False).model
            return NestedUnets(i_net_name      = 'unet',
                               i_shortcut_mani = True,
                               i_block_name    = 'residual',  # 'conv','sk','atrous','gatrous'
                               i_img_height    = self.input_shape[0],
                               i_img_width     = self.input_shape[1],
                               i_img_depth     = self.input_shape[2],
                               i_num_classes   = self.num_classes,
                               i_shallow_filters = (16, 32, 64),
                               i_deep_filters    = (16, 32, 64, 128),
                               i_supervision     = False,
                               model             = shallow_network).build_multiple()
        elif self.net_id == 24:#Conventional UNet
            return UNets(i_net_name      = 'unet',
                         i_shortcut_mani = False,
                         i_block_name    = 'conv',
                         i_img_height    = self.input_shape[0],
                         i_img_width     = self.input_shape[1],
                         i_img_depth     = self.input_shape[2],
                         i_num_classes   = self.num_classes,
                         i_filters       = (64, 128, 256, 512, 1024),
                         i_supervision=False).build()
        elif self.net_id == 25:#Residual UNet
            return UNets(i_net_name      = 'unet',
                         i_shortcut_mani = False,
                         i_block_name    = 'residual',
                         i_img_height    = self.input_shape[0],
                         i_img_width     = self.input_shape[1],
                         i_img_depth     = self.input_shape[2],
                         i_num_classes   = self.num_classes,
                         i_filters       = (64, 128, 256, 512, 1024),
                         i_supervision=False).build()
        elif self.net_id == 26:#Conventionial UNet++ without supervision
            return UNets(i_net_name      = 'unetplus',
                         i_shortcut_mani = False,
                         i_block_name    = 'conv',
                         i_img_height    = self.input_shape[0],
                         i_img_width     = self.input_shape[1],
                         i_img_depth     = self.input_shape[2],
                         i_num_classes   = self.num_classes,
                         i_filters       = (64, 128, 256, 512, 1024),
                         i_supervision   = False).build()
        else:#NestedNetwork but train shallow and deep network separately.
            shallow_network = ImageSegNets(i_net_id      = 9,
                                           i_save_path   = self.save_path,
                                           i_image_shape = self.input_shape,
                                           i_num_classes = 1,
                                           i_continue    = False).model
            return NestedUnets(i_net_name        = 'unet',
                               i_shortcut_mani   = True,
                               i_block_name      = 'residual',  # 'conv','sk','atrous','gatrous'
                               i_img_height      = self.input_shape[0],
                               i_img_width       = self.input_shape[1],
                               i_img_depth       = self.input_shape[2],
                               i_num_classes     = self.num_classes,
                               i_shallow_filters = (16,32,64),
                               i_deep_filters    = (16,32,64,128),
                               i_supervision     = False,
                               model             = shallow_network).build_multiple()
    def train(self,i_train_db=None,i_val_db=None):
        """i_train_db can be (list, tuple, None)"""
        """i_val_db can be (list, tuple, None)"""
        """If i_train_db is None,then we will read the saved tfrecord file at the self.save_path location."""
        """If i_val_db is None,then we will read the saved tfrecord file at the self.save_path location."""
        """Process the dataset"""
        db = CSDataset(i_save_path=self.save_path,i_target_shape=self.input_shape)
        train_db = db.prepare(i_db=i_train_db,i_train_flag=True,i_lsm_factor=self.lsm_factor,i_num_classes=self.num_classes,i_pipeline_fn=self.pipeline)
        val_db   = db.prepare(i_db=i_val_db,i_train_flag=False,i_lsm_factor=self.lsm_factor,i_num_classes=self.num_classes,i_pipeline_fn=self.pipeline)
        train_db = train_db.filter(lambda x,y:tf.reduce_sum(y)>0)
        val_db   = val_db.filter(lambda x,y:tf.reduce_sum(y)>0)
        train_db = train_db.batch(self.batch_size)
        val_db   = val_db.batch(self.batch_size)
        if self.cv_debug:
            for batch in val_db:#train_db
                debug_images,debug_masks = batch
                for index, image in enumerate(debug_images):
                    mask = debug_masks[index]
                    print('debug_count = {} - {} vs {}'.format(index, image.shape, mask.shape))
                    if self.type_output=='sigmoid':
                        print('Using sigmoid activation function')
                    else:
                        mask = tf.argmax(mask,axis=-1)
                    plt.subplot(1, 2, 1)
                    plt.imshow(image, cmap='gray')
                    plt.title('Image')
                    plt.subplot(1, 2, 2)
                    plt.imshow(mask, cmap='gray')
                    plt.title('Mask')
                    plt.show()
                    if index>3:
                        break
                    else:
                        pass
                break
        else:
            pass
        """Model initialization"""
        if os.path.exists(self.model_path):
            if self.tcontinue:
                network = tf.keras.models.load_model(filepath=self.model_path,custom_objects=SegNetLosses.get_custom_objects(i_type_output=self.type_output))
            else:
                return False
        else:
            Logs.log("Train a new model from scratch")
            network = self.init_network()
        assert isinstance(network,tf.keras.models.Model)
        network.summary()
        net = SegNetLosses.compile(i_net=network, i_type_output=self.type_output,i_lr=self.lr, i_loss_name=self.loss, i_weights=self.weight)
        """Training"""
        log_infor  = CustomCallback(i_model_path=self.model_path)
        lr_schuler = LearningRateScheduler()
        lr_params  = {'decay_rule': 1, 'step': int(self.num_epochs / 10), 'decay_rate': 0.90, 'base_lr': self.lr}
        schedule   = lr_schuler(lr_params)
        callbacks  = [schedule, log_infor]
        network.fit(x               = train_db.repeat(self.db_repeat),
                    epochs          = self.num_epochs,
                    verbose         = 1,
                    shuffle         = True,
                    validation_data = val_db,
                    callbacks       = callbacks)
        """Update the nework"""
        self.model = tf.keras.models.load_model(self.model_path, custom_objects=SegNetLosses.get_custom_objects(i_type_output=self.type_output))
        return net
    def eval(self,i_db=None,i_debug=False):
        assert isinstance(i_db,(list,tuple,tf.data.Dataset))
        assert isinstance(i_debug,bool)
        labels, predictions = [],[]
        #ori_output = self.get_original_output
        self.get_original_output=False
        evaluer = SegMetrics(i_num_classes=(self.num_classes + 1) if self.type_output == 'sigmoid' else self.num_classes,i_care_background=False)
        for index, element in enumerate(i_db):
            print("(ImageSegNets) Evaluating element: {}".format(index))
            assert isinstance(element,(list,tuple,dict))
            """Extract data"""
            if isinstance(element,(list,tuple)):
                image,mask = element
            else:
                image,mask = element['image'],element['mask']
            """Preprocess image data"""
            if isinstance(image,(tf.Tensor,tf.SparseTensor)):
                image = image.numpy()
            else:
                assert isinstance(image,np.ndarray)
            if len(image.shape)==2:
                image = np.expand_dims(np.expand_dims(image,-1),0)
            else:
                if len(image.shape)==3:
                    if image.shape[-1] in (1,3):
                        image = np.expand_dims(image,axis=0)
                    else:
                        image = np.expand_dims(image,axis=-1)
                else:
                    pass
            if isinstance(mask,(tf.Tensor,tf.SparseTensor)):
                mask = mask.numpy()
            else:
                assert isinstance(mask,np.ndarray)
            if len(mask.shape)==2:
                mask = np.expand_dims(np.expand_dims(mask,axis=-1),axis=0)
            else:
                if len(mask.shape)==3:
                    if mask.shape[-1]==1:
                        mask = np.expand_dims(mask,axis=0)
                    else:
                        mask = np.expand_dims(mask,axis=-1)
                else:
                    assert len(mask.shape)==4
            """Prediction"""
            preds = self.predict(i_image=image)
            for pred_index,pred in enumerate(preds):
                current_mask = mask[pred_index]
                current_mask = SupFns.scale_mask(i_mask=current_mask, i_tsize=self.mask_shape)
                labels.append(current_mask)
                predictions.append(pred)
                if i_debug:
                    current_image = image[pred_index]
                    print(current_image.shape,current_mask.shape, pred.shape)
                    plt.subplot(1,3,1)
                    plt.imshow(current_image,cmap='gray')
                    plt.title('Image')
                    plt.subplot(1,3,2)
                    plt.imshow(current_mask,cmap='gray')
                    plt.title('Mask')
                    plt.subplot(1,3,3)
                    plt.imshow(pred,cmap='gray')
                    dice = evaluer.get_metrics(i_label=current_mask,i_pred=pred)[0]
                    plt.title('Prediction - Dice = {:2.2f}'.format(dice))
                    plt.show()
                else:
                    pass
        """Performance measurement"""
        #Logs.log('Using entire dataset')
        #measures, measure_mean, measure_std = evaluer.eval(i_labels=labels, i_preds=predictions, i_object_care=False)
        #Logs.log('Measure shape = {}'.format(measures.shape))
        #Logs.log('Measure mean  = {}'.format(measure_mean))
        #Logs.log('Measure std   = {}'.format(measure_std))
        #Logs.log('Using sub dataset that only consider images containing objects')
        #measures, measure_mean, measure_std = evaluer.eval(i_labels=labels, i_preds=predictions, i_object_care=True)
        #Logs.log('Measure shape = {}'.format(measures.shape))
        #Logs.log('Measure mean  = {}'.format(measure_mean))
        #Logs.log('Measure std   = {}'.format(measure_std))
        #self.get_original_output = ori_output
        Logs.log('3D Dice measurement')
        Logs.log('Suppose that all eval images are from a same 3D volume')
        global_labels = [np.array(labels)]
        global_preds = [np.array(predictions)]
        three_dim_evaluer = SegMetrics3D()
        three_dim_evaluer.measures(i_labels=global_labels, i_preds=global_preds, i_object_index=1)
        Logs.log('***'*100)
        return labels, predictions
    def predict(self,i_image=None):
        assert isinstance(i_image,np.ndarray)
        assert isinstance(self.model,tf.keras.models.Model)
        assert len(i_image.shape) in (2,3,4)
        """Make image batch"""
        image_shape = i_image.shape
        if len(image_shape) in (2,3):
            if len(image_shape)==2:
                images = np.expand_dims(i_image,axis=-1)
            else:
                images = i_image.copy()
            images = np.expand_dims(images,axis=0)
        else:
            images = i_image.copy()
        assert images.shape[-1] in (1,3)
        """Size and Color adjustment"""
        norm_images = []
        for image in images:
            assert isinstance(image,np.ndarray)
            """Color adjustment"""
            if image.shape[-1]==self.input_shape[-1]:
                pass
            else:
                if image.shape[-1]==1:
                    image = np.concatenate((image,image,image),axis=-1)
                else:
                    image = np.mean(image,axis=-1,keepdims=True).astype(image.dtype)
            """Size adjustment"""
            image = SupFns.imresize(i_image=image,i_tsize=self.input_shape[0:2])
            norm_images.append(image)
        images = np.array(norm_images)
        assert images.dtype in (np.uint8,)
        """Gray level normalization"""
        images = images.astype(np.float)/255.0
        """Prediction"""
        start_time = time.time_ns()
        pred = self.model.predict(images)
        proc_time = (time.time_ns()-start_time)/1000000
        #print(pred.shape,np.min(pred),np.max(pred))
        if isinstance(pred,np.ndarray):
            pass
        else:
            assert isinstance(pred,(list,tuple))
            pred = pred[-1]#Get the last output
        if self.type_output=='sigmoid':
            if self.get_original_output:
                print('(SegNets) Returning original mask')
                Logs.log('Prediction time = {}'.format(proc_time))
                return pred
            else:
                print('(SegNets) Returning binary mask')
                return (pred>= self.cv_threshold).astype(np.uint8)
        else:
            return np.expand_dims(np.argmax(pred,axis=-1),axis=-1) #Shape: (None, height, width, 1)
if __name__ == '__main__':
    print('This module is to implement a general-purpose for a segmentation problem')
    print('Possible params: "lr","loss","repeat","epochs","weight","lsm_factor","batch_size"')
    tr_db, va_db = SupFns.get_sample_db(i_tsize=(256, 256), i_num_train_samples=1000, i_num_val_samples=100)
    segnet = ImageSegNets(i_net_id      = 0,
                          i_save_path   = None,
                          i_image_shape = (256,256,1),
                          i_num_classes = 2,
                          i_continue    = False,
                          seg_loss      = 'Dice',
                          seg_epochs    = 10,
                          seg_repeat    = 10)
    tr_db = list(zip(tr_db[0],tr_db[1]))
    va_db = list(zip(va_db[0],va_db[1]))
    segnet.train(i_train_db=tr_db,i_val_db=va_db)
    segnet.eval(i_db=tr_db)
    segnet.eval(i_db=va_db)
    for item in va_db:
        simage,smask = item
        spred = segnet.predict(i_image=simage)[0]
        print(spred.shape,np.sum(spred))
        plt.subplot(1,3, 1)
        plt.imshow(simage,cmap='gray')
        plt.subplot(1, 3, 2)
        plt.imshow(smask,cmap='gray')
        plt.subplot(1, 3, 3)
        plt.imshow(spred,cmap='gray')
        plt.show()
"""=================================================================================================================="""