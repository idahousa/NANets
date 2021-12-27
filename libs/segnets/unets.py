import tensorflow as tf
class UNets:
    cv_debug = False
    def __init__(self,
                 i_net_name      = 'unet',
                 i_shortcut_mani = True,
                 i_block_name    = 'conv',#'conv','sk','atrous','gatrous'
                 i_img_height    = 256,
                 i_img_width     = 256,
                 i_img_depth     = 1,
                 i_num_classes   = 1,
                 i_filters       = (32, 64, 128, 256, 512),
                 i_supervision   = False,
                 i_name          = None):
        assert isinstance(i_img_height,int)
        assert isinstance(i_img_width,int)
        assert isinstance(i_img_depth,int)
        assert isinstance(i_num_classes,int)
        assert isinstance(i_filters,(list,tuple))
        assert isinstance(i_supervision,bool)
        assert i_img_height>0
        assert i_img_width>0
        assert i_img_depth>0
        assert len(i_filters)>0
        assert isinstance(i_net_name,str)
        assert i_net_name in ('unet','unetplus')
        assert isinstance(i_shortcut_mani,bool)
        self.net_name      = i_net_name
        self.shortcut_mani = i_shortcut_mani
        self.block_name    = i_block_name
        self.input_shape   = (i_img_height,i_img_width,i_img_depth)
        self.network_depth = len(i_filters)
        self.filters       = i_filters
        self.supervision   = i_supervision if self.net_name=='unetplus' else False
        self.num_classes   = i_num_classes
        if self.num_classes==1:
            self.activation = 'sigmoid'
        else:
            self.activation = 'softmax'
        if i_name is None:
            self.name = 'output'
        else:
            assert isinstance(i_name,str)
            self.name = i_name
    @staticmethod
    def conv_block(i_block_name=None,i_input=None, i_nb_filter=32, i_kernel_size=3,**kwargs):
        if i_block_name=='conv':
            return UNets.norm_conv_block(i_input=i_input, i_nb_filter=i_nb_filter, i_kernel_size=i_kernel_size)
        elif i_block_name=='atrous':
            return UNets.dilated_conv_block(i_input=i_input, i_nb_filter=i_nb_filter, i_kernel_size=i_kernel_size)
        elif i_block_name=='gatrous':
            if 'i_num_groups' in kwargs.keys():
                i_num_groups = kwargs['i_num_groups']
            else:
                i_num_groups = 4
            return UNets.group_dilated_conv_block(i_input=i_input, i_nb_filter=i_nb_filter, i_kernel_size=i_kernel_size,i_num_groups=i_num_groups)
        elif i_block_name=='sk':
            return UNets.sk_block(i_input=i_input, i_nb_filter=i_nb_filter, i_kernel_size=i_kernel_size)
        elif i_block_name=='conv_bn':
            return UNets.norm_conv_bn_block(i_input=i_input, i_nb_filter=i_nb_filter, i_kernel_size=i_kernel_size)
        elif i_block_name=='residual':
            return UNets.residual_block(i_input=i_input, i_nb_filter=i_nb_filter, i_kernel_size=i_kernel_size)
        else:
            raise Exception('Invalid block name!')
    @staticmethod
    def group_dilated_conv_block(i_input=None, i_nb_filter=32, i_kernel_size=3,i_num_groups=4):
        """Reference: Figure 2: Semantic segmentation by multi-scale feature extraction based on grouped dilated convolution"""
        outputs    = []
        num_groups = i_num_groups
        segments   = tf.split(value=i_input, num_or_size_splits=num_groups, axis=-1)
        nb_filter  = i_nb_filter//num_groups
        for index, segment in enumerate(segments):
            output = tf.keras.layers.Conv2D(filters=nb_filter, kernel_size=(i_kernel_size, i_kernel_size),activation='relu', dilation_rate=(index + 1, index + 1), padding='same')(segment)
            outputs.append(output)
        if len(outputs)>1:
            output = tf.keras.layers.Concatenate()(outputs)
        else:
            output = outputs[0]
        return output
    @staticmethod
    def dilated_conv_block(i_input=None, i_nb_filter=32, i_kernel_size=3):
        """Reference: Figure 2: Semantic segmentation by multi-scale feature extraction based on grouped dilated convolution"""
        nb_filter = i_nb_filter//4
        x_a = tf.keras.layers.Conv2D(filters=nb_filter, kernel_size=(i_kernel_size, i_kernel_size), activation='relu',dilation_rate=(1, 1), padding='same')(i_input)
        x_b = tf.keras.layers.Conv2D(filters=nb_filter, kernel_size=(i_kernel_size, i_kernel_size), activation='relu',dilation_rate=(2, 2), padding='same')(i_input)
        x_c = tf.keras.layers.Conv2D(filters=nb_filter, kernel_size=(i_kernel_size, i_kernel_size), activation='relu',dilation_rate=(3, 3), padding='same')(i_input)
        x_d = tf.keras.layers.Conv2D(filters=nb_filter, kernel_size=(i_kernel_size, i_kernel_size), activation='relu',dilation_rate=(4, 4), padding='same')(i_input)
        output = tf.keras.layers.Concatenate()([x_a, x_b, x_c, x_d])
        return output
    @staticmethod
    def norm_conv_block(i_input=None, i_nb_filter=32, i_kernel_size=3):
        x = tf.keras.layers.Conv2D(i_nb_filter, (i_kernel_size, i_kernel_size), activation='relu',kernel_initializer='he_normal', padding='same')(i_input)
        #x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.layers.Conv2D(i_nb_filter, (i_kernel_size, i_kernel_size), activation='relu',kernel_initializer='he_normal', padding='same')(x)
        #x = tf.keras.layers.Dropout(rate=0.5)(x)
        return x
    @staticmethod
    def norm_conv_bn_block(i_input=None, i_nb_filter=32, i_kernel_size=3):
        outputs = tf.keras.layers.Conv2D(i_nb_filter, (i_kernel_size, i_kernel_size),kernel_initializer='he_normal', padding='same')(i_input)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)
        outputs = tf.keras.layers.Conv2D(i_nb_filter, (i_kernel_size, i_kernel_size),kernel_initializer='he_normal', padding='same')(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)
        return outputs
    @staticmethod
    def residual_block(i_input=None, i_nb_filter=32, i_kernel_size=3):
        """First CONV block"""
        outputs = tf.keras.layers.Conv2D(filters=i_nb_filter, kernel_size=(1, 1), strides=(1, 1), padding='same')(i_input)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)
        """Second CONV block"""
        outputs = tf.keras.layers.Conv2D(filters=i_nb_filter, kernel_size=(i_kernel_size, i_kernel_size),strides=(1,1), padding='same')(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)
        """Third CONV block"""
        outputs = tf.keras.layers.Conv2D(filters=i_nb_filter, kernel_size=(1, 1), padding='same')(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        """Shortcut CONV block"""
        shortcut = tf.keras.layers.Conv2D(filters=i_nb_filter,kernel_size=(1, 1), strides=(1,1), padding='same')(i_input)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        """Aggregation"""
        outputs = tf.keras.layers.Add()([outputs, shortcut])
        outputs = tf.keras.layers.Activation('relu')(outputs)
        return outputs
    @staticmethod
    def sk_block(i_input=None,i_nb_filter=32,i_kernel_size=3):
        """Reference: Breast mass segmentation in ultrasound with selective kernel UNet convolutional neural network"""
        """1. Dilated convolution branch"""
        branch_1 = tf.keras.layers.Conv2D(filters=i_nb_filter,kernel_size=(i_kernel_size,i_kernel_size),dilation_rate=2,padding='same')(i_input)
        #branch_1 = tf.keras.layers.Dropout(rate=0.5)(branch_1)
        branch_1 = tf.keras.layers.BatchNormalization()(branch_1)
        branch_1 = tf.keras.layers.Activation('relu')(branch_1)
        """2. Normal convolution branch"""
        branch_2 = tf.keras.layers.Conv2D(filters=i_nb_filter, kernel_size=(i_kernel_size, i_kernel_size), padding='same')(i_input)
        #branch_2 = tf.keras.layers.Dropout(rate=0.5)(branch_2)
        branch_2 = tf.keras.layers.BatchNormalization()(branch_2)
        branch_2 = tf.keras.layers.Activation('relu')(branch_2)
        fusion = tf.keras.layers.Add()([branch_1,branch_2])
        fusion = tf.keras.layers.GlobalAveragePooling2D()(fusion)
        fusion = tf.keras.layers.Dense(units=i_nb_filter//2)(fusion)
        fusion = tf.keras.layers.BatchNormalization()(fusion)
        fusion = tf.keras.layers.Activation('relu')(fusion)
        fusion = tf.keras.layers.Dense(units=i_nb_filter)(fusion)
        fusion = tf.keras.layers.Activation('sigmoid')(fusion)
        weight_1 = tf.keras.layers.Lambda(lambda x:x)(fusion)
        weight_2 = tf.keras.layers.Lambda(lambda x:1.0-x)(fusion)
        """Making output of each branch"""
        output_1 = tf.keras.layers.Multiply()([branch_1, weight_1])
        output_2 = tf.keras.layers.Multiply()([branch_2, weight_2])
        return tf.keras.layers.Add()([output_1,output_2])
    def build(self):
        buffers = []
        inputs  = tf.keras.layers.Input(shape=self.input_shape)
        """Warming-up layer"""
        outputs = self.conv_block(i_block_name=self.block_name,i_input=inputs,i_nb_filter=self.filters[0],i_kernel_size=3,i_num_groups=1)
        for col in range(self.network_depth):
            buffer = []
            for row in range(self.network_depth-col):
                if self.cv_debug:
                    print('col = {}, row = {}'.format(col, row))
                else:
                    pass
                """1.Basic line (backbone)"""
                if col==0:
                    if row==0:
                        buffer.append(outputs)
                    else:
                        ts = self.conv_block(i_block_name=self.block_name,i_input=buffer[-1],i_nb_filter=self.filters[row],i_kernel_size=3)
                        ts = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(ts)
                        buffer.append(ts)
                else:
                    """Inference lines"""
                    concat = []
                    for index in range(0, col):
                        ts = buffers[index][row]
                        if ts is None:
                            continue
                        else:
                            pass
                        concat.append(ts)
                        if self.shortcut_mani:
                            ts = self.conv_block(i_block_name=self.block_name,i_input=ts,i_nb_filter=self.filters[row],i_kernel_size=3)
                            concat.append(ts)
                        else:
                            pass
                        if self.cv_debug:
                            print('x = ', ts.shape)
                        else:
                            pass
                    assert len(concat)>0
                    ts = buffers[col - 1][row + 1]
                    if ts is None:
                        buffer.append(None)
                        continue
                    else:
                        pass
                    if self.cv_debug:
                        print('y = ', ts.shape)
                    else:
                        pass
                    ts = tf.keras.layers.Conv2DTranspose(self.filters[row], (2, 2), strides=(2, 2), padding='same')(ts)
                    concat.append(ts)
                    ts = tf.keras.layers.Concatenate(axis=-1)(concat)
                    ts = self.conv_block(i_block_name=self.block_name,i_input=ts, i_nb_filter=self.filters[row], i_kernel_size=3)
                    if self.net_name=="unet":
                        if row==(self.network_depth-col-1):
                            buffer.append(ts)
                        else:
                            buffer.append(None)
                    else:
                        assert self.net_name=="unetplus"
                        buffer.append(ts)
            buffers.append(buffer)
        if self.supervision:
            final_outputs = []
            for index in range(1,self.network_depth):
                final_outputs.append(tf.keras.layers.Conv2D(filters            = self.num_classes,
                                                            kernel_size        = (1, 1),
                                                            kernel_initializer = 'he_normal',
                                                            padding            = 'same',
                                                            activation         = self.activation,
                                                            name               = '{}_{}'.format(self.name,index))(buffers[index][0]))
            model = tf.keras.models.Model(inputs=inputs, outputs=final_outputs)
        else:
            final_output = tf.keras.layers.Conv2D(filters            = self.num_classes,
                                                  kernel_size        = (3,3),
                                                  kernel_initializer = 'he_normal',
                                                  padding            = 'same',
                                                  activation         = self.activation,
                                                  name               = self.name)(buffers[-1][0])
            model = tf.keras.models.Model(inputs=inputs, outputs=[final_output])
        return model
class NestedUnets:
    cv_debug = False
    def __init__(self,
                 i_net_name        = 'unet',
                 i_shortcut_mani   = True,
                 i_block_name      = 'conv',  # 'conv','sk','atrous','gatrous'
                 i_img_height      = 256,
                 i_img_width       = 256,
                 i_img_depth       = 3,
                 i_num_classes     = 1,
                 i_shallow_filters = (32,64,128),
                 i_deep_filters    = (32,64,128,256,512),
                 i_supervision     = False,**kwargs):
        assert isinstance(i_img_height, int)
        assert isinstance(i_img_width, int)
        assert isinstance(i_img_depth, int)
        assert isinstance(i_num_classes, int)
        assert isinstance(i_shallow_filters, (list, tuple))
        assert isinstance(i_deep_filters, (list, tuple))
        """Current code is correct if num_class =1. For other value, please check again"""
        assert i_num_classes == 1
        assert isinstance(i_net_name, str)
        assert i_net_name in ('unet', 'unetplus')
        assert isinstance(i_shortcut_mani,bool)
        self.net_name        = i_net_name
        self.shortcut_mani   = i_shortcut_mani
        self.block_name      = i_block_name
        self.num_classes     = i_num_classes
        self.shallow_filters = i_shallow_filters
        self.deep_filters    = i_deep_filters
        self.input_shape     = (i_img_height, i_img_width, i_img_depth)
        self.network_depth   = len(self.deep_filters)
        self.supervision     = i_supervision if self.net_name=='unetplus' else False
        if self.num_classes==1:
            self.activation = 'sigmoid'
        else:
            self.activation = 'softmax'
        if 'model' in kwargs.keys():
            if isinstance(kwargs['model'],tf.keras.models.Model):
                self.model = kwargs['model']
            else:
                self.model = None
        else:
            self.model = None
    @staticmethod
    def attention(i_feature_maps=None,i_salient_maps=None,i_stage=1,i_pool_size=2):
        """Reference: Attention enriched deep learning model for breast tumor segmentation in ultrasound images"""
        pool_size = pow(i_pool_size,i_stage)
        print(i_feature_maps.shape, i_salient_maps.shape,pool_size)
        """Feature maps manipulation"""
        f_b1 = tf.keras.layers.MaxPooling2D(pool_size=(i_pool_size,i_pool_size),strides=(i_pool_size,i_pool_size))(i_feature_maps)
        f_b2 = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=(i_pool_size,i_pool_size),padding='same')(i_feature_maps)
        """Salient maps manipulation"""
        s_b1 = tf.keras.layers.MaxPooling2D(pool_size=(pool_size,pool_size),strides=(pool_size,pool_size))(i_salient_maps)
        s_b1 = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same')(s_b1)
        """Fusion of feature and salient"""
        fusion = tf.keras.layers.Add()([f_b2, s_b1])
        fusion = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(fusion)
        fusion = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(fusion)
        fusion = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(fusion)
        fusion = tf.keras.layers.Activation('sigmoid')(fusion)
        """Final output"""
        output = tf.keras.layers.Multiply()([f_b1,fusion])
        return output
    def build_attention(self):
        """Check the Figure 1 of the reference paper: """
        """Reference: Attention enriched deep learning model for breast tumor segmentation in ultrasound images"""
        input_a = tf.keras.layers.Input(shape = self.input_shape)
        input_b = tf.keras.layers.Input(shape = self.input_shape)
        """Build UNet-based Network based on two inputs : Gray image and salient maps"""
        buffers = []
        """Warming-up layer"""
        outputs = UNets.conv_block(i_block_name=self.block_name,i_input=input_a,i_nb_filter=self.deep_filters[0],i_kernel_size=3,i_num_groups=1)
        for col in range(self.network_depth):
            buffer = []
            for row in range(self.network_depth-col):
                if self.cv_debug:
                    print('col = {}, row = {}'.format(col, row))
                else:
                    pass
                """1.Basic line (backbone)"""
                if col==0:
                    if row==0:
                        buffer.append(outputs)
                    else:
                        ts = self.attention(i_feature_maps=buffer[-1],i_salient_maps=input_b,i_stage=row,i_pool_size=2)
                        ts = UNets.conv_block(i_block_name=self.block_name,i_input=ts,i_nb_filter=self.deep_filters[row],i_kernel_size=3)
                        buffer.append(ts)
                else:
                    """Inference lines"""
                    concat = []
                    for index in range(0, col):
                        ts = buffers[index][row]
                        if ts is None:
                            continue
                        else:
                            pass
                        concat.append(ts)
                        if self.shortcut_mani:
                            ts = UNets.conv_block(i_block_name=self.block_name,i_input=ts,i_nb_filter=self.deep_filters[row],i_kernel_size=3)
                            concat.append(ts)
                        else:
                            pass
                        if self.cv_debug:
                            print('x = ', ts.shape)
                        else:
                            pass
                    assert len(concat) > 0
                    ts = buffers[col - 1][row + 1]
                    if ts is None:
                        buffer.append(None)
                        continue
                    else:
                        pass
                    if self.cv_debug:
                        print('y = ', ts.shape)
                    else:
                        pass
                    ts = tf.keras.layers.Conv2DTranspose(self.deep_filters[row], (2, 2), strides=(2, 2), padding='same')(ts)
                    concat.append(ts)
                    ts = tf.keras.layers.Concatenate(axis=-1)(concat)
                    ts = UNets.conv_block(i_block_name=self.block_name,i_input=ts, i_nb_filter=self.deep_filters[row], i_kernel_size=3)
                    if self.net_name == "unet":
                        if row == (self.network_depth - col - 1):
                            buffer.append(ts)
                        else:
                            buffer.append(None)
                    else:
                        assert self.net_name == "unetplus"
                        buffer.append(ts)
            buffers.append(buffer)
        if self.supervision:
            final_outputs = []
            for index in range(1,self.network_depth):
                final_outputs.append(tf.keras.layers.Conv2D(filters            = self.num_classes,
                                                            kernel_size        = (1, 1),
                                                            kernel_initializer = 'he_normal',
                                                            padding            = 'same',
                                                            activation         = self.activation,
                                                            name               = 'deep_{}'.format(index))(buffers[index][0]))
            model = tf.keras.models.Model(inputs=[input_a,input_b], outputs=final_outputs)
        else:
            final_output = tf.keras.layers.Conv2D(filters            = self.num_classes,
                                                  kernel_size        = (1,1),
                                                  kernel_initializer = 'he_normal',
                                                  padding            = 'same',
                                                  activation         = self.activation,
                                                  name               = 'deep')(buffers[-1][0])
            model = tf.keras.models.Model(inputs=[input_a,input_b], outputs=[final_output])
        return model
    def build_single(self):
        inputs = tf.keras.layers.Input(shape = self.input_shape)
        if self.model is None:
            shallow_model = UNets(i_net_name     = self.net_name,
                                  i_shortcut_mani= self.shortcut_mani,
                                  i_block_name   = self.block_name,
                                  i_img_height   = self.input_shape[0],
                                  i_img_width    = self.input_shape[1],
                                  i_img_depth    = self.input_shape[2],
                                  i_num_classes  = self.num_classes,
                                  i_filters      = self.shallow_filters,
                                  i_supervision  = False,
                                  i_name         = 'shallow_output').build()
        else:
            shallow_model = self.model
            assert isinstance(shallow_model,tf.keras.models.Model)
            shallow_model.trainable = False
        salient_maps = shallow_model(inputs)
        salient_maps = tf.keras.layers.Lambda(lambda x:x,name='shallow')(salient_maps)
        fusion       = self.attention(i_feature_maps=inputs,i_salient_maps=salient_maps,i_stage=1,i_pool_size=1)
        deep_model   = UNets(i_net_name     = self.net_name,
                             i_shortcut_mani= self.shortcut_mani,
                             i_block_name   = self.block_name,
                             i_img_height   = self.input_shape[0],
                             i_img_width    = self.input_shape[1],
                             i_img_depth    = self.input_shape[2],
                             i_num_classes  = self.num_classes,
                             i_filters      = self.deep_filters,
                             i_supervision  = False,
                             i_name         = 'deep_output').build()
        output = deep_model(fusion)
        output = tf.keras.layers.Lambda(lambda x: x, name='deep')(output)
        model  = tf.keras.models.Model(inputs=inputs, outputs=[salient_maps, output])
        return model
    def build_multiple(self):
        """Check the Figure 1 of the reference paper: """
        """Reference: Attention enriched deep learning model for breast tumor segmentation in ultrasound images"""
        inputs = tf.keras.layers.Input(shape = self.input_shape)
        if self.model is None:
            shallow_model = UNets(i_net_name     = self.net_name,
                                  i_shortcut_mani= self.shortcut_mani,
                                  i_block_name   = self.block_name,
                                  i_img_height   = self.input_shape[0],
                                  i_img_width    = self.input_shape[1],
                                  i_img_depth    = self.input_shape[2],
                                  i_num_classes  = self.num_classes,
                                  i_filters      = self.shallow_filters,
                                  i_supervision  = False,
                                  i_name         = 'shallow_output').build()
        else:
            shallow_model = self.model
            assert isinstance(shallow_model,tf.keras.models.Model)
            shallow_model.trainable = False
        salient_maps = shallow_model(inputs)
        salient_maps = tf.keras.layers.Lambda(lambda x: x, name='shallow')(salient_maps)
        """Build UNet-based Network based on two inputs : Gray image and salient maps"""
        buffers = []
        """Warming-up layer"""
        outputs = UNets.conv_block(i_block_name=self.block_name,i_input=inputs,i_nb_filter=self.deep_filters[0],i_kernel_size=3,i_num_groups=1)
        for col in range(self.network_depth):
            buffer = []
            for row in range(self.network_depth-col):
                if self.cv_debug:
                    print('col = {}, row = {}'.format(col, row))
                else:
                    pass
                """1.Basic line (backbone)"""
                if col==0:
                    if row==0:
                        buffer.append(outputs)
                    else:
                        ts = self.attention(i_feature_maps=buffer[-1],i_salient_maps=salient_maps,i_stage=row,i_pool_size=2)
                        ts = UNets.conv_block(i_block_name=self.block_name,i_input=ts,i_nb_filter=self.deep_filters[row],i_kernel_size=3)
                        buffer.append(ts)
                else:
                    """Inference lines"""
                    concat = []
                    for index in range(0, col):
                        ts = buffers[index][row]
                        if ts is None:
                            continue
                        else:
                            pass
                        concat.append(ts)
                        if self.shortcut_mani:
                            ts = UNets.conv_block(i_block_name=self.block_name,i_input=ts,i_nb_filter=self.deep_filters[row],i_kernel_size=3)
                            concat.append(ts)
                        else:
                            pass
                        if self.cv_debug:
                            print('x = ', ts.shape)
                        else:
                            pass
                    assert len(concat) > 0
                    ts = buffers[col - 1][row + 1]
                    if ts is None:
                        buffer.append(None)
                        continue
                    else:
                        pass
                    if self.cv_debug:
                        print('y = ', ts.shape)
                    else:
                        pass
                    ts = tf.keras.layers.Conv2DTranspose(self.deep_filters[row], (2, 2), strides=(2, 2), padding='same')(ts)
                    concat.append(ts)
                    ts = tf.keras.layers.Concatenate(axis=-1)(concat)
                    ts = UNets.conv_block(i_block_name=self.block_name,i_input=ts, i_nb_filter=self.deep_filters[row], i_kernel_size=3)
                    if self.net_name == "unet":
                        if row == (self.network_depth - col - 1):
                            buffer.append(ts)
                        else:
                            buffer.append(None)
                    else:
                        assert self.net_name == "unetplus"
                        buffer.append(ts)
            buffers.append(buffer)
        if self.supervision:
            if self.model is None:
                final_outputs = [salient_maps]
            else:
                final_outputs = []
            for index in range(1,self.network_depth):
                final_outputs.append(tf.keras.layers.Conv2D(filters            = self.num_classes,
                                                            kernel_size        = (1, 1),
                                                            kernel_initializer = 'he_normal',
                                                            padding            = 'same',
                                                            activation         = self.activation,
                                                            name               = 'deep_{}'.format(index))(buffers[index][0]))
            model = tf.keras.models.Model(inputs=inputs, outputs=final_outputs)
        else:
            final_output = tf.keras.layers.Conv2D(filters            = self.num_classes,
                                                  kernel_size        = (1,1),
                                                  kernel_initializer = 'he_normal',
                                                  padding            = 'same',
                                                  activation         = self.activation,
                                                  name               = 'deep')(buffers[-1][0])
            if self.model is None:
                model = tf.keras.models.Model(inputs=inputs, outputs=[salient_maps,final_output])
            else:
                model = tf.keras.models.Model(inputs=inputs, outputs=[final_output])
        return model
"""=================================================================================================================="""
if __name__ == '__main__':
    print("This module is to implement general UNetPlus network")
    networks = UNets(i_net_name='unet',i_shortcut_mani=False,i_img_depth=1).build()
    #network = UNets(i_net_name='unet', i_shortcut_mani=True).build()
    #network = UNets(i_net_name='unet', i_shortcut_mani=False).build()
    #network = UNets(i_net_name='unetplus', i_shortcut_mani=True).build()
    #network = UNets(i_net_name='unetplus', i_shortcut_mani=False).build()
    #network = NestedUnets(i_net_name='unetplus',i_shortcut_mani=True).build_single()
    #network = NestedUnets(i_net_name='unet', i_shortcut_mani=True).build_multiple()
    #network = NestedUnets(i_net_name='unetplus',i_shortcut_mani=False).build_single()
    network = NestedUnets(i_net_name='unet', i_img_depth=1,i_shortcut_mani=False,model=networks).build_multiple()
    #network = NestedUnets(i_net_name='unet', i_shortcut_mani=False).build_attention()
    #network = UNets(i_net_name = 'unet',i_shortcut_mani = True,i_block_name = 'gatrous').build()
    network.summary()
    tf.keras.utils.plot_model(network, to_file='model-attention.png')
"""=================================================================================================================="""