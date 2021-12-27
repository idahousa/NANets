import os
import random
import numpy as np
from pydicom import dcmread
from libs.commons import SupFns
db_path = r'G:\roots\segmentations\thyroid\thyroid'
class ThreeDimThyroid:
    def __init__(self,i_db_path=None,i_image_height=256,i_image_width=256):
        if i_db_path is None:
            self.db_path = db_path
        else:
            assert isinstance(i_db_path,str)
            assert os.path.exists(i_db_path)
            self.db_path     = db_path
        self.num_files       = 16 #Fixed
        self.image_height    = i_image_height
        self.image_width     = i_image_width
        self.index           = 0
        self.num_masks       = 0
        self.remove_boundary = True
    def read_file(self,i_file_path=None):
        assert isinstance(i_file_path,str)
        assert os.path.exists(i_file_path)
        npy_path = i_file_path[0:len(i_file_path)-4]+'_npy.npy'
        if os.path.exists(npy_path):
            simages = np.load(npy_path)
        else:
            images  = dcmread(i_file_path).pixel_array.astype(np.float)
            simages = []
            for item in images:
                if i_file_path.find('groundtruth')>0:
                    simages.append(SupFns.scale_mask(i_mask=item, i_tsize=(self.image_height, self.image_width)))
                else:
                    simages.append(SupFns.imresize(i_image=item, i_tsize=(self.image_height, self.image_width)))
            simages = np.array(simages)
            np.save(npy_path,simages)
        return simages
    def read_pair(self,i_data_path=None,i_mask_path=None):
        images = self.read_file(i_file_path=i_data_path)
        masks  = self.read_file(i_file_path=i_mask_path)
        self.index += images.shape[0]
        self.num_masks += np.sum(np.sum(masks,axis=(1,2))>0)
        print('Shape = ',images.shape,masks.shape,np.sum(masks),self.index,self.num_masks)
        if self.remove_boundary:
            """This is to remove some boundary points to reduce imballance effects"""
            crop_images,crops_masks = [],[]
            for index,image in enumerate(images):
                mask = masks[index]
                ul_x,ul_y,br_x,br_y = self.find_boundary(i_image=image)
                image = image[ul_x:br_x,ul_y:br_y,:]
                mask  = mask[ul_x:br_x,ul_y:br_y,:]
                crop_images.append(SupFns.imresize(i_image=image,i_tsize=(self.image_height,self.image_width)))
                crops_masks.append(SupFns.scale_mask(i_mask=mask,i_tsize=(self.image_height,self.image_width)))
            return np.array(crop_images),np.array(crops_masks)
        else:
            pass
        return images,masks
    def find_boundary(self,i_image=None):
        assert isinstance(i_image,np.ndarray)
        def find_range(i_profile=None,i_size = 192):
            profile = 100*i_profile/(np.max(i_profile)+1)
            profile_length = profile.shape[0]
            margin = profile_length - i_size
            begin_point,end_point = 0, 0
            for index, item in enumerate(i_profile):
                if item >= 1.0 or index>=margin:
                    begin_point = index
                    end_point   = begin_point + i_size
                    break
                else:
                    continue
            return begin_point,end_point
        image = np.squeeze(i_image)
        vertical_line   = np.sum(image,axis=0)
        horizontal_line = np.sum(image,axis=1)
        size = int(0.75*max(self.image_height,self.image_width))
        ul_x,br_x = find_range(vertical_line,size)
        ul_y,br_y = find_range(horizontal_line,size)
        return ul_x,ul_y,br_x,br_y
    def get_data(self,i_fold_index=0,i_2d_image=True,i_semi_auto=False):
        """Dataset size is just 16 files, then we should perform a leave-one-out procedure"""
        assert isinstance(i_fold_index,int)
        assert 0<i_fold_index<=self.num_files
        train_data_paths = [os.path.join(self.db_path,'data','D{:02d}.dcm'.format(i)) for i in range(1,self.num_files+1) if i!=i_fold_index]
        train_mask_paths = [os.path.join(self.db_path,'groundtruth','D{:02d}.dcm'.format(i)) for i in range(1,self.num_files+1) if i!=i_fold_index]
        val_data_path    = os.path.join(self.db_path,'data','D{:02d}.dcm'.format(i_fold_index))
        val_mask_path    = os.path.join(self.db_path, 'groundtruth','D{:02d}.dcm'.format(i_fold_index))
        train_images,train_masks = [],[]
        for index,item in enumerate(train_data_paths):
            images,masks = self.read_pair(i_data_path=train_data_paths[index],i_mask_path=train_mask_paths[index])
            print('Final = ',images.shape,masks.shape,' Sum Mask = ' ,np.sum(masks))
            if i_2d_image:
                for img_index,image in enumerate(images):
                    train_images.append(image)
                    train_masks.append(masks[img_index])
            else:
                train_images.append(images)
                train_masks.append(masks)
        train_images = np.array(train_images)
        train_masks  = np.array(train_masks)
        val_images,val_masks = self.read_pair(i_data_path=val_data_path,i_mask_path=val_mask_path)
        print('Final = ', val_images.shape, val_masks.shape, ' Sum Mask = ', np.sum(val_masks))
        print('Training   DB:', train_images.shape, train_masks.shape)
        print('Training   DB:', np.min(train_images), np.max(train_images), np.min(train_masks), np.max(train_masks))
        print('Evaluating DB:', val_images.shape, val_masks.shape)
        print('Evaluating DB:', np.min(val_images), np.max(val_images), np.min(val_masks), np.max(val_masks))
        train_db = list(zip(train_images,train_masks))
        val_db   = list(zip(val_images,val_masks))
        for _ in range(10):
            random.shuffle(train_db)
        #random.shuffle(val_db)
        if i_semi_auto:
            modify_train_db, modify_val_db = [],[]
            for item in train_db:
                image,mask = item
                if np.sum(mask)>100:#Set to >0 to remove too small objects
                    modify_train_db.append(item)
                else:
                    pass
            for item in val_db:
                image, mask = item
                if np.sum(mask) > 0:
                    modify_val_db.append(item)
                else:
                    pass
            print('Length of semi-auto: {} vs {}'.format(len(modify_train_db),len(modify_val_db)))
            return modify_train_db,modify_val_db,modify_val_db
        else:
            pass
        return train_db,val_db,val_db
    """Train, val, test sets"""
    def get_data_train_val_test(self,i_fold_index=0,i_2d_image=True,i_semi_auto=False):
        """Dataset size is just 16 files, then we should perform a leave-one-out procedure"""
        assert isinstance(i_fold_index,int)
        assert 0<i_fold_index<=self.num_files
        train_data_paths  = [os.path.join(self.db_path,'data','D{:02d}.dcm'.format(i)) for i in range(1,self.num_files+1) if i!=i_fold_index and i!=(i_fold_index+1)]
        train_mask_paths  = [os.path.join(self.db_path,'groundtruth','D{:02d}.dcm'.format(i)) for i in range(1,self.num_files+1) if i!=i_fold_index and i!=(i_fold_index+1)]
        if i_fold_index%2==1:
            test_data_path    = os.path.join(self.db_path,'data','D{:02d}.dcm'.format(i_fold_index))
            test_mask_path    = os.path.join(self.db_path, 'groundtruth','D{:02d}.dcm'.format(i_fold_index))
            val_data_path     = os.path.join(self.db_path, 'data', 'D{:02d}.dcm'.format(i_fold_index + 1))
            val_mask_path     = os.path.join(self.db_path, 'groundtruth', 'D{:02d}.dcm'.format(i_fold_index + 1))
        else:
            test_data_path    = os.path.join(self.db_path,'data','D{:02d}.dcm'.format(i_fold_index+1))
            test_mask_path    = os.path.join(self.db_path, 'groundtruth','D{:02d}.dcm'.format(i_fold_index+1))
            val_data_path     = os.path.join(self.db_path, 'data', 'D{:02d}.dcm'.format(i_fold_index))
            val_mask_path     = os.path.join(self.db_path, 'groundtruth', 'D{:02d}.dcm'.format(i_fold_index))
        train_images,train_masks = [],[]
        for index,item in enumerate(train_data_paths):
            images,masks = self.read_pair(i_data_path=train_data_paths[index],i_mask_path=train_mask_paths[index])
            print('Final = ',images.shape,masks.shape,' Sum Mask = ' ,np.sum(masks))
            if i_2d_image:
                for img_index,image in enumerate(images):
                    train_images.append(image)
                    train_masks.append(masks[img_index])
            else:
                train_images.append(images)
                train_masks.append(masks)
        train_images = np.array(train_images)
        train_masks  = np.array(train_masks)
        val_images,val_masks = self.read_pair(i_data_path=val_data_path,i_mask_path=val_mask_path)
        test_images, test_masks = self.read_pair(i_data_path=test_data_path, i_mask_path=test_mask_path)
        print('Final = ', val_images.shape, val_masks.shape, ' Sum Mask = ', np.sum(val_masks))
        print('Training   DB:', train_images.shape, train_masks.shape)
        print('Training   DB:', np.min(train_images), np.max(train_images), np.min(train_masks), np.max(train_masks))
        print('Testing    DB:', test_images.shape, test_masks.shape)
        print('Testing    DB:', np.min(test_images), np.max(test_images), np.min(test_masks), np.max(test_masks))
        print('Evaluating DB:', val_images.shape, val_masks.shape)
        print('Evaluating DB:', np.min(val_images), np.max(val_images), np.min(val_masks), np.max(val_masks))
        train_db = list(zip(train_images,train_masks))
        val_db   = list(zip(val_images,val_masks))
        test_db  = list(zip(test_images, test_masks))
        for _ in range(10):
            random.shuffle(train_db)
        if i_semi_auto:
            modify_train_db, modify_val_db,modify_test_db = [],[],[]
            for item in train_db:
                image,mask = item
                if np.sum(mask)>100:#Set to >0 to remove too small objects
                    modify_train_db.append(item)
                else:
                    pass
            for item in val_db:
                image, mask = item
                if np.sum(mask) > 0:
                    modify_val_db.append(item)
                else:
                    pass
            for item in test_db:
                image, mask = item
                if np.sum(mask) > 0:
                    modify_test_db.append(item)
                else:
                    pass
            print('Length of semi-auto: {} vs {} vs {}'.format(len(modify_train_db),len(modify_val_db),len(modify_test_db)))
            return modify_train_db,modify_val_db,modify_test_db
        else:
            pass
        return train_db,val_db,test_db
"""=================================================================================================================="""
if __name__ == '__main__':
    import imageio
    import matplotlib.pyplot as plt
    print('This module is to read 3D thyroid dataset')
    print('Reference link: https://opencas.webarchiv.kit.edu/?q=node/29')
    db    = ThreeDimThyroid()
    x, y,z  = db.get_data(i_fold_index=1,i_2d_image=True,i_semi_auto=True)
    show  = True
    index = 0
    save_path = os.path.join(os.getcwd(), 'threedim_images')
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        pass
    for (img,msk) in y:
        index = index + 1
        if show:
            plt.subplot(1, 3, 1)
            plt.imshow(img,cmap='gray')
            plt.title('Image')
            plt.subplot(1, 3, 2)
            plt.imshow(msk, cmap='gray')
            plt.title('Mask')
            plt.subplot(1, 3, 3)
            fusion = np.concatenate((img,img,np.maximum(img,msk*255)),axis=-1)
            plt.imshow(fusion,cmap='gray')
            plt.title('Fusion')
            plt.show()
        else:
            pass
        imageio.imwrite('threedim_images/{}_image.jpg'.format(index), img)
        imageio.imwrite('threedim_images/{}_msk.jpg'.format(index), msk*255)
        print(np.min(img),np.max(img),' vs ',np.min(msk),np.max(msk))
"""=================================================================================================================="""