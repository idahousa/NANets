import math
import tensorflow as tf
class TSFNets:
    def __init__(self,
                 i_input_shape  = (480,720,3),
                 i_base_filters = 32,
                 i_num_scales   = 3,
                 i_fusion_rule  = 'add',
                 i_blocks       = 'conv'):
        self.input_shape  = i_input_shape
        self.base_filters = i_base_filters
        self.num_scales   = i_num_scales
        self.fusion_rule  = i_fusion_rule
        self.block        = i_blocks
    @staticmethod
    def residual_block(i_input=None, i_nb_filter=32, i_kernel_size=(3,3),i_stride=(1,1)):
        """First CONV block"""
        outputs = tf.keras.layers.Conv2D(filters=i_nb_filter, kernel_size=(1, 1), strides=(1, 1), padding='same')(i_input)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)
        """Second CONV block"""
        outputs = tf.keras.layers.Conv2D(filters=i_nb_filter, kernel_size=i_kernel_size,strides=(1, 1), padding='same')(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)
        """Third CONV block"""
        outputs = tf.keras.layers.Conv2D(filters=i_nb_filter, kernel_size=(1, 1), padding='same')(outputs)
        #outputs = tf.keras.layers.BatchNormalization()(outputs)
        """Shortcut CONV block"""
        shortcut = tf.keras.layers.Conv2D(filters=i_nb_filter, kernel_size=(1, 1), strides=(1, 1), padding='same')(i_input)
        #shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        """Aggregation"""
        outputs = tf.keras.layers.Add()([outputs, shortcut])
        #outputs = tf.keras.layers.AveragePooling2D(pool_size=i_stride,strides=i_stride)(outputs)
        outputs = tf.keras.layers.MaxPool2D(pool_size=i_stride, strides=i_stride)(outputs)
        return outputs
    @staticmethod
    def Conv2D(i_block_name='conv',i_input=None,i_filters=32,i_kernel_size=(3,3),i_stride=(1,1)):
        if i_block_name=='conv':
            return tf.keras.layers.Conv2D(filters=i_filters,kernel_size=i_kernel_size,strides=i_stride,padding='same')(i_input)
        elif i_block_name=='residual':
            return TSFNets.residual_block(i_input=i_input,i_nb_filter=i_filters,i_kernel_size=i_kernel_size,i_stride=i_stride)
        else:
            return tf.keras.layers.Conv2D(filters=i_filters, kernel_size=i_kernel_size, strides=i_stride,padding='same')(i_input)
    def build(self):
        inputs  = tf.keras.layers.Input(shape=self.input_shape)
        """Warming-up layer"""
        #output = tf.keras.layers.Conv2D(filters=self.base_filters,kernel_size=(3,3),strides=(1,1),padding='same')(inputs)
        output = self.Conv2D(i_block_name=self.block,i_input=inputs,i_filters=self.base_filters,i_kernel_size=(3,3),i_stride=(1,1))
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.Activation('relu')(output)
        temp   = output
        #output = tf.keras.layers.Conv2D(filters=self.base_filters,kernel_size=(3,3),strides=(2,2),padding='same')(output)
        output = self.Conv2D(i_block_name=self.block,i_input=output,i_filters=self.base_filters,i_kernel_size=(3,3),i_stride=(2,2))
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.Activation('relu')(output)
        """Scales"""
        output_a, output_aa = self.build_branch(i_input=output,i_num_layers  = 5,i_nb_filter=self.base_filters,i_stride=1,i_df=1)
        output_b, output_bb = self.build_branch(i_input=output, i_num_layers = 5, i_nb_filter=self.base_filters, i_stride=2,i_df=2)
        output_c, output_cc = self.build_branch(i_input=output, i_num_layers = 5, i_nb_filter=self.base_filters, i_stride=4,i_df=4)
        """Fusion_A"""
        #output_aa= tf.keras.layers.Conv2D(filters=self.base_filters * 16, strides=(1, 1), kernel_size=(1, 1), padding='same')(output_aa)
        output_aa= self.Conv2D(i_block_name=self.block,i_input=output_aa, i_filters=self.base_filters*16,i_kernel_size=(1,1),i_stride=(1,1))
        if self.fusion_rule=='add':
            fusion_a = tf.keras.layers.Add()([output_a,output_b,output_c,output_aa])
        else:
            fusion_a = tf.keras.layers.Concatenate()([output_a,output_b,output_c,output_aa])
        fusion_a = tf.keras.layers.BatchNormalization()(fusion_a)
        fusion_a = tf.keras.layers.Activation('relu')(fusion_a)
        #fusion_a = tf.keras.layers.Conv2D(filters=self.base_filters * 4, strides=(1, 1), kernel_size=(1, 1), padding='same')(fusion_a)
        fusion_a = self.Conv2D(i_block_name=self.block,i_input=fusion_a,i_filters=self.base_filters*4,i_kernel_size=(1,1),i_stride=(1,1))
        fusion_a = tf.keras.layers.BatchNormalization()(fusion_a)
        fusion_a = tf.keras.layers.Activation('relu')(fusion_a)
        fusion_a = tf.keras.layers.AvgPool2D(pool_size=(2,2),strides=(2,2))(fusion_a)
        #fusion_a = tf.keras.layers.Conv2D(filters=self.base_filters * 2, strides=(1, 1), kernel_size=(3, 3), padding='same')(fusion_a)
        fusion_a = self.Conv2D(i_block_name=self.block,i_input=fusion_a,i_filters=self.base_filters*2,i_kernel_size=(3,3),i_stride=(1,1))
        """Fusion_B"""
        if self.fusion_rule=='add':
            fusion_b = tf.keras.layers.Add()([fusion_a,output_bb])
        else:
            fusion_b = tf.keras.layers.Concatenate()([fusion_a, output_bb])
        fusion_b = tf.keras.layers.BatchNormalization()(fusion_b)
        fusion_b = tf.keras.layers.Activation('relu')(fusion_b)
        fusion_b = tf.keras.layers.Conv2DTranspose(self.base_filters,kernel_size=(2, 2), strides=(2, 2), padding='same')(fusion_b)
        fusion_b = tf.keras.layers.BatchNormalization()(fusion_b)
        fusion_b = tf.keras.layers.Activation('relu')(fusion_b)
        fusion_b = tf.keras.layers.Conv2DTranspose(self.base_filters,kernel_size=(2, 2), strides=(2, 2), padding='same')(fusion_b)
        fusion_b = tf.keras.layers.BatchNormalization()(fusion_b)
        fusion_b = tf.keras.layers.Activation('relu')(fusion_b)
        """Output"""
        if self.fusion_rule=='add':
            output = tf.keras.layers.Add()([fusion_b,temp])
        else:
            output = tf.keras.layers.Concatenate()([fusion_b, temp])
        #output = tf.keras.layers.Conv2D(filters=1, strides=(1, 1), kernel_size=(1, 1),padding='same')(output)
        output = self.Conv2D(i_block_name=self.block,i_input=output,i_filters=1,i_kernel_size=(1,1),i_stride=(1,1))
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.Activation('sigmoid')(output)
        model = tf.keras.models.Model(inputs=inputs, outputs=[output])
        model.summary()
        return model

    def build_branch(self,i_input=None,i_num_layers=5,i_nb_filter=32, i_stride=1,i_df=1):
        assert isinstance(i_num_layers,int)
        assert isinstance(i_nb_filter,int)
        assert isinstance(i_stride,int)
        assert isinstance(i_df,int)
        assert i_num_layers>0
        assert i_nb_filter>0
        assert i_stride>0
        assert i_df>0
        num_t_conv = int(math.log2(i_stride))
        num_n_conv = i_num_layers - num_t_conv - 1
        num_filters= i_nb_filter
        """First layer: Dilated convolution"""
        output = tf.keras.layers.Conv2D(filters=i_nb_filter, strides=(1,1),kernel_size=(3, 3),dilation_rate=(i_df,i_df), padding='same')(i_input)
        #output = tf.keras.layers.AveragePooling2D(pool_size=(i_stride,i_stride),strides=(i_stride,i_stride))(output)
        output = tf.keras.layers.MaxPool2D(pool_size=(i_stride, i_stride), strides=(i_stride, i_stride))(output)
        foutput= output
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.Activation('relu')(output)
        """Normal convolution"""
        for layer in range(num_n_conv):
            """Second layer: Normal 3-by-3 convolution"""
            num_filters = num_filters*2
            #output      = tf.keras.layers.Conv2D(filters=num_filters, strides=(1, 1), kernel_size=(3, 3), dilation_rate=(1, 1), padding='same')(output)
            output      = self.Conv2D(i_block_name=self.block,i_input=output,i_filters=num_filters,i_kernel_size=(3,3),i_stride=(1,1))
            if i_stride>1:
                if layer==0:
                    foutput = output
                else:
                    pass
            else:
                pass
            if (i_stride == 1) and layer==(num_n_conv-1):
                return output,foutput
            else:
                output = tf.keras.layers.BatchNormalization()(output)
                output = tf.keras.layers.Activation('relu')(output)
        for layer in range(num_t_conv):
            num_filters = num_filters * 2
            output      = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(output)
            if layer==(num_t_conv-1):
                return output,foutput
            else:
                output = tf.keras.layers.BatchNormalization()(output)
                output = tf.keras.layers.Activation('relu')(output)
"""=================================================================================================================="""
if __name__ == '__main__':
    print('This module implement Asalan net!')
    network = TSFNets().build()
    tf.keras.utils.plot_model(network,to_file='model.png')
"""=================================================================================================================="""