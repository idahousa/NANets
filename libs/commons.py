import zlib
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from pycocotools.mask import encode as coco_encode #Please use: pip3 install pycocotools to install coco tools
from pycocotools.mask import decode as coco_decode #Please use: pip3 install pycocotools to install coco tools
class SupFns:
    def __init__(self):
        pass
    @classmethod
    def compare(cls, i_x=None, i_y=None):
        """Compare two list or tuple of number"""
        assert isinstance(i_x, (list, tuple))
        assert isinstance(i_y, (list, tuple))
        assert len(i_x) == len(i_y)
        cmps = []
        for index, x in enumerate(i_x):
            y = i_y[index]
            if x == y:
                cmps.append(0)
            else:
                cmps.append(1)
        """cmps serves as register that indicates the similarity between two lists or tuples"""
        return cmps
    """Support function for resizing image"""
    @classmethod
    def imresize(cls, i_image=None, i_tsize=(512,512),i_min=None,i_max=None):
        """i_tsize in format (height, width). But, PIL.Image.fromarray.resize(width, height). So, we must invert i_tsize before scaling"""
        assert isinstance(i_image, np.ndarray)
        assert isinstance(i_tsize,(list,tuple))
        assert len(i_tsize)==2
        assert i_tsize[0]>0
        assert i_tsize[1]>0
        tsize = (i_tsize[1],i_tsize[0])
        if i_image.dtype != np.uint8:#Using min-max scaling to make unit8 image
            if i_min is None:
                min_val = np.min(i_image)
            else:
                assert isinstance(i_min, (int, float))
                min_val = i_min
            if i_max is None:
                max_val = np.max(i_image)
            else:
                assert isinstance(i_max, (int, float))
                max_val = i_max
            assert min_val <= max_val
            image   = (i_image-min_val)/(max_val-min_val+1e-10)
            image   = (image*255.0).astype(np.uint8)
        else:
            image   = i_image.copy()
        assert image.dtype == np.uint8
        image = np.array(Image.fromarray(np.squeeze(image)).resize(tsize))
        if len(image.shape)==2:
            image = np.expand_dims(image,-1)
        else:
            assert len(image.shape)==3
        return image
    @classmethod
    def scale_mask(cls,i_mask=None,i_tsize=None):#Remove i_num_labels later as it affects to some of other files.
        assert isinstance(i_mask,np.ndarray), 'Got type: {}'.format(type(i_mask))
        assert isinstance(i_tsize,(list,tuple,int)), 'Got type: {}'.format(type(i_tsize))
        assert len(i_mask.shape) in (2, 3), 'Got shape: {}'.format(i_mask.shape)
        if isinstance(i_tsize,int):
            assert i_tsize>0, 'Got value: {}'.format(i_tsize)
            tsize = (i_tsize,i_tsize)
        else:
            assert len(i_tsize) in (2,3), 'Got values: {}'.format(i_tsize)
            assert 0 < i_tsize[0], 'Got values: {}'.format(i_tsize)
            assert 0 < i_tsize[1], 'Got values: {}'.format(i_tsize)
            tsize = (i_tsize[0], i_tsize[1])
        if len(i_mask.shape)==2:
            mask = np.expand_dims(i_mask,-1)
        else:
            mask = i_mask.copy()
        assert mask.shape[-1] in (1, )
        num_labels = int(np.max(mask)) + 1
        masks = [np.zeros(shape=(tsize[0],tsize[1],1),dtype=np.uint8)]
        for index in range(1, num_labels):
            cmask = (mask==index).astype(np.uint8)
            cmask = cls.imresize(i_image=cmask,i_tsize=tsize)
            cmask = (cmask>0).astype(np.uint8)
            cmask = (cmask*index).astype(np.uint8)
            masks.append(cmask)
        masks = np.concatenate(masks,axis=-1)
        assert len(masks.shape) == 3, 'Got shape: {}'.format(masks.shape)
        assert masks.shape[-1] == num_labels
        mask  = np.expand_dims(np.max(masks,axis=-1),axis=-1)  # Selecting the object with small index. Alternative is reduce_max
        return mask #Return shape: (i_tsize,i_tsize,1)
    @classmethod
    def encode_binary_mask(cls,i_mask=None):
        """Converts a binary mask into OID challenge encoding ascii text."""
        assert isinstance(i_mask, np.ndarray)
        assert len(i_mask.shape) in (2, 3)
        if len(i_mask.shape) == 3:
            assert i_mask.shape[-1] == 1
            mask = i_mask.copy()
        else:
            mask = np.expand_dims(i_mask, axis=-1)
        assert mask.dtype in (np.bool, np.uint8)
        if mask.dtype == np.uint8:
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
        assert mask.dtype == np.uint8
        # convert input mask to expected COCO API input --
        mask_to_encode = np.asfortranarray(mask)
        # RLE encode mask --
        encoded_mask = coco_encode(mask_to_encode)[0]["counts"]
        # compress and base64 encoding --
        binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
        base64_str = base64.b64encode(binary_str)
        return base64_str
    @classmethod
    def decode_mask(cls,i_compressed_str=None, i_height=None, i_width=None):
        """i_compressed_str is the result of encode_binary_mask"""
        """i_height,i_width is the height and width of original mask"""
        assert isinstance(i_compressed_str, bytes)
        assert isinstance(i_height, int)
        assert isinstance(i_width, int)
        assert i_height > 0
        assert i_width > 0
        base64_str = base64.b64decode(i_compressed_str)
        binary_str = zlib.decompress(base64_str)
        rleObjects = [{'size': [i_height, i_width], 'counts': binary_str}]
        mask = coco_decode(rleObjects)
        return mask
    @classmethod
    def get_sample_db(cls,i_tsize=(224,224),i_num_train_samples=None,i_num_val_samples=None):
        assert isinstance(i_tsize,(list,tuple)), 'Got type: {}'.format(type(i_tsize))
        assert len(i_tsize)==2, 'Got lenght: {}'.format(len(i_tsize))
        assert i_tsize[0] > 0, 'Got value: {}'.format(i_tsize[0])
        assert i_tsize[1] > 0, 'Got value: {}'.format(i_tsize[1])
        def make_mask(i_image=None):
            assert isinstance(i_image,np.ndarray), 'Got type: {}'.format(type(i_image))
            assert i_image.dtype == np.uint8, 'Got type: {}'.format(i_image.dtype)
            image = (i_image>127).astype(np.uint8)
            return image
        (train_images, train_labels), (val_images, val_labels) = tf.keras.datasets.mnist.load_data()
        if i_num_train_samples is  None:
            i_num_train_samples = train_images.shape[0]
        else:
            assert isinstance(i_num_train_samples,int)
            assert i_num_train_samples>0
        if i_num_val_samples is None:
            i_num_val_samples = val_images.shape[0]
        else:
            assert isinstance(i_num_val_samples,int)
            assert i_num_val_samples>0
        train_images = [cls.imresize(i_image=image,i_tsize=i_tsize) for index, image in enumerate(train_images) if index <=i_num_train_samples]
        train_labels = [label for index, label in enumerate(train_labels) if index <= i_num_train_samples]
        val_images   = [cls.imresize(i_image=image,i_tsize=i_tsize) for index, image in enumerate(val_images) if index <=i_num_val_samples]
        val_labels   = [label for index, label in enumerate(val_labels) if index <= i_num_val_samples]
        train_masks  = [make_mask(i_image=image) for image in train_images]
        val_masks    = [make_mask(i_image=image) for image in val_images]
        train_db     = (train_images,train_masks,train_labels)
        val_db       = (val_images,val_masks,val_labels)
        return train_db,val_db
"""=================================================================================================================="""
if __name__ == '__main__':
    print('This module contain general support functions used in various programs')
    tr_db,va_db = SupFns.get_sample_db(i_tsize=(224,224),i_num_train_samples=10,i_num_val_samples=1)
    example_image = tr_db[1][0]
    plt.subplot(1,3,1)
    plt.imshow(example_image,cmap='gray')
    plt.title('Original image')
    rimage = SupFns.scale_mask(i_mask=example_image,i_tsize=128)
    plt.subplot(1,3,2)
    plt.imshow(rimage,cmap='gray')
    plt.title('Scale mask')
    print(rimage.shape,example_image.shape)
    ecoded_mask = SupFns.encode_binary_mask(i_mask=rimage)
    print('Encoded mask = ', ecoded_mask)
    decoded_mask = SupFns.decode_mask(i_compressed_str=ecoded_mask,i_height=rimage.shape[0],i_width=rimage.shape[1])
    plt.subplot(1,3,3)
    plt.imshow(decoded_mask,cmap='gray')
    plt.title('Decoded Mask')
    plt.show()
"""=================================================================================================================="""