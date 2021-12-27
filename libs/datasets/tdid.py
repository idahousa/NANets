import os
import imageio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from libs.datasets.tdid_roi import TDID_ROI
from libs.datasets.tdid_mask import TDID_Mask
from libs.datasets.tdid_parser import TDID_Parser
"""TDID Thyroid dataset. Link at: http://cimalab.intec.co/applications/thyroid/"""
db_path   = r'G:\roots\classifications\thyroid\tdid\images'
"""====================================================================================================="""
class TDID_DB:
    num_labels    = 2
    benign_scores = ('2','3')              #Fixed. Label 0 for binary classification
    malign_scores = ('4a','4b','4c','5')   #Fixed. Label 1 for binary classification
    scores = benign_scores + malign_scores #Fixed.
    image_size    = (256,256)              #Fixed.
    num_samples = (1, 1, 1, 1, 1, 1)
    assert isinstance(num_labels,int)
    assert num_labels in (2,6)
    """Init function for every class instance"""
    def __init__(self, **kwargs):
        if 'i_db_path' in kwargs.keys():
            self.db_path = kwargs['i_db_path']
        else:
            self.db_path = db_path
        self.tsize = self.image_size
    """Evaluate the label for images"""
    @classmethod
    def get_label(cls,i_tirad=None):
        assert isinstance(i_tirad,str)
        assert i_tirad in cls.scores
        if cls.num_labels == 2:
            if i_tirad in cls.benign_scores:
                rtn_label = 0
            elif i_tirad in cls.malign_scores:
                rtn_label = 1
            else:
                rtn_label = None
        else:
            if i_tirad is None:
                rtn_label = None
            else:
                rtn_label = cls.scores.index(i_tirad)
        print('rtn_label = ',rtn_label)
        return rtn_label
    """Extract ROI from TDID dataset"""
    @classmethod
    def resize(cls,i_image=None,i_tsize=(224,224)):
        """Resize an input image to a specified size"""
        assert isinstance(i_image,np.ndarray)
        assert len(i_image.shape) in (2,3)
        assert isinstance(i_tsize,(list,tuple))
        assert len(i_tsize) == 2
        if i_image.dtype in (np.uint8,):
            return np.array(Image.fromarray(i_image).resize(size=i_tsize))
        elif i_image.dtype in (np.float32,np.float64,np.float):
            return np.array(Image.fromarray((255*i_image).astype(np.uint8)).resize(size=i_tsize))
        else:
            print(i_image.dtype)
            raise Exception('Invalid datatype')
    """Extract ROI for an image"""
    @classmethod
    def extract_roi(cls,i_db_path=None,i_num_fold=2,i_fold_index=1,i_train_set=True):
        """Dont use directly. Use extract_roi_with_mask instead."""
        assert isinstance(i_db_path,str)
        assert os.path.exists(i_db_path)
        assert isinstance(i_num_fold,int)
        assert 0<i_num_fold<=5
        assert isinstance(i_fold_index,int)
        assert 0<i_fold_index<=i_num_fold
        TDID_Parser.benign_scores = cls.benign_scores
        TDID_Parser.malign_scores = cls.malign_scores
        TDID_Parser.scores        = cls.scores
        TDID_Parser.num_samples   = cls.num_samples
        parser = TDID_Parser(i_db_path    = i_db_path,
                             i_num_fold   = i_num_fold,
                             i_fold_index = i_fold_index,
                             i_train_set  = i_train_set)
        print('Number of patients: ',len(parser.xmls))
        data,labels = [],[]
        for xml in parser.xmls:
            rois = TDID_ROI.get_roi(i_xml_path=xml)
            for roi in rois:
                image_path  = roi['image']
                image_tirad = roi['tirad']
                image_rois  = roi['rois']
                ipath,iname = os.path.split(image_path)
                iname       = iname[0:len(iname)-3]
                iname      +='npy'
                npy_name = os.path.join(ipath,iname)
                if os.path.exists(npy_name):
                    image = np.load(npy_name)
                else:
                    image = imageio.imread(image_path)
                    np.save(npy_name,image)
                for image_roi in image_rois:
                    iul,ibr = image_roi
                    iroi = image[iul[0]:ibr[0],iul[1]:ibr[1],:]
                    iroi = cls.resize(i_image=iroi,i_tsize=cls.image_size)
                    ilabel = cls.get_label(image_tirad)
                    data.append(iroi)
                    labels.append(ilabel)
        return np.array(data),np.array(labels)
    """Extract ROI with mask from TDID dataset"""
    @classmethod
    def extract_roi_with_mask(cls,i_db_path=None,i_num_fold=2,i_fold_index=1,i_train_set=True):
        assert isinstance(i_db_path,str)
        assert os.path.exists(i_db_path)
        TDID_Parser.benign_scores = cls.benign_scores
        TDID_Parser.malign_scores = cls.malign_scores
        TDID_Parser.scores        = cls.scores
        TDID_Parser.num_samples   = cls.num_samples
        parser = TDID_Parser(i_db_path    = i_db_path,
                             i_num_fold   = i_num_fold,
                             i_fold_index = i_fold_index,
                             i_train_set  = i_train_set)
        print('Number of patients: ',len(parser.xmls))
        data,gmasks, labels = [],[],[]
        for xml in parser.xmls:
            xml_path,xml_name = os.path.split(xml)
            xml_name = xml_name[0:len(xml_name)-4]
            data_name = xml_name + '_data.npy'
            mask_name = xml_name + '_mask.npy'
            label_name= xml_name + '_label_{}.npy'.format(cls.num_labels)
            cdata_path = os.path.join(xml_path,data_name)
            cmask_path = os.path.join(xml_path,mask_name)
            clabel_path= os.path.join(xml_path,label_name)
            read_image_again = False
            if os.path.exists(cdata_path) and os.path.exists(cmask_path) and os.path.exists(clabel_path):
                cdata  = np.load(cdata_path)
                cmasks = np.load(cmask_path)
                clabels= np.load(clabel_path)
                for index, item in enumerate(cdata):
                    if item.shape[0]==cls.image_size[0]:
                        read_image_again = False
                    else:
                        read_image_again = True
                        break
                    if i_train_set:
                        if np.sum(cmasks[index])<=0:
                            continue
                        else:
                            pass
                    else:
                        pass
                    data.append(item)
                    gmasks.append(cmasks[index])
                    labels.append(clabels[index])
            else:
                read_image_again = True
            if read_image_again:#Different image size => read data again
                pass
            else:
                continue
            cdata  = []
            cmasks = []
            clabels= []
            rois        = TDID_ROI.get_roi(i_xml_path=xml)
            names,masks = TDID_Mask.create_mask(i_xml_path=xml,i_train_flag=i_train_set)
            for roi in rois:
                image_path  = roi['image']
                image_tirad = roi['tirad']
                image_rois  = roi['rois']
                ipath,iname = os.path.split(image_path)
                iname       = iname[0:len(iname)-3]
                iname +='npy'
                npy_name = os.path.join(ipath,iname)
                if os.path.exists(npy_name):
                    image = np.load(npy_name)
                else:
                    image = imageio.imread(image_path)
                    np.save(npy_name,image)
                image_name = os.path.split(image_path)[1]
                mask_index = names.index(image_name)
                mask = masks[mask_index]
                for image_roi in image_rois:
                    iul,ibr = image_roi
                    iroi  = image[iul[0]:ibr[0],iul[1]:ibr[1],:]
                    iroi  = cls.resize(i_image=iroi,i_tsize=cls.image_size)
                    imask = mask[iul[0]:ibr[0],iul[1]:ibr[1],:]
                    imask = cls.resize(i_image=imask,i_tsize=cls.image_size)
                    imask[imask > 0] = 255
                    ilabel = cls.get_label(image_tirad)
                    cdata.append(iroi)
                    cmasks.append(np.mean(imask,axis=-1,keepdims=True))
                    clabels.append(ilabel)
            np.save(cdata_path,np.array(cdata))
            np.save(cmask_path,np.array(cmasks))
            np.save(clabel_path,np.array(clabels))
            for index, item in enumerate(cdata):
                if i_train_set:
                    if np.sum(cmasks[index]) <= 0:
                        continue
                    else:
                        pass
                else:
                    pass
                data.append(item)
                gmasks.append(cmasks[index])
                labels.append(clabels[index])
                """
                if np.sum(data)>0:
                    data.append(item)
                    gmasks.append(cmasks[index])
                    labels.append(clabels[index])
                else:
                    pass
                """
        data    = np.array(data,dtype=object).astype(np.uint8)
        gmasks  = np.array(gmasks,dtype=object).astype(np.uint8)
        labels  = np.array(labels).astype(np.int)
        print('Original range TDID dataset:----(tdid.py at 205)')
        print(data.shape,gmasks.shape,labels.shape)
        print('Data : ',np.min(data),np.max(data))
        print('Mask : ',np.min(gmasks),np.max(gmasks))
        print('Label: ',np.min(labels),np.max(labels))
        print('-'*25)
        return data,gmasks,labels
    """Main function"""
    def get_roi(self,i_image=None,i_mask_flag=True,i_step=1):
        assert isinstance(i_image,np.ndarray)
        assert isinstance(i_mask_flag,bool)
        assert isinstance(i_step,int)
        assert i_step>=1
        shift  = int(0.1*self.image_size[0])
        images = []
        for x in range(0,shift,i_step):
            for y in range(0,shift,i_step):
                image = np.squeeze(i_image[y:,x:,:].copy())
                image = self.resize(image,i_tsize=self.tsize)
                if i_mask_flag:
                    image = (image>0).astype(np.float)
                    image = (image*255).astype(np.uint8)
                else:
                    pass
                if len(image.shape)==2:
                    image = np.expand_dims(image,-1)
                else:
                    assert len(image.shape)==3
                images.append(image)
        return np.array(images)
    def __call__(self,i_num_folds=2,i_fold_index=1,i_train_set=True):
        assert isinstance(i_num_folds,int)
        assert i_num_folds>1
        assert isinstance(i_num_folds,int)
        assert i_fold_index>0
        assert isinstance(i_train_set,bool)
        def do_aug(i_images,i_masks,i_labels):
            assert isinstance(i_images,(list,tuple,np.ndarray))
            assert isinstance(i_masks, (list, tuple, np.ndarray))
            assert isinstance(i_labels, (list, tuple, np.ndarray))
            do_aug_images,do_aug_masks,do_aug_labels = [],[],[]
            if isinstance(i_images,(list,tuple)):
                num_samples = len(i_images)
            else:
                num_samples = i_images.shape[0]
            for index,image in enumerate(i_images):
                print('Data Augmentation (TDID): {}/{} - {}'.format(index,num_samples,image.shape))
                mask  = i_masks[index]
                label = i_labels[index]
                #image = np.mean(image,axis=-1,keepdims=True)
                #print(mask.shape, image.shape)
                aug_images = self.get_roi(i_image=image,i_mask_flag=False,i_step=5)
                aug_masks  = self.get_roi(i_image=mask,i_mask_flag=True,i_step=5)
                for aug_index,aug_image in enumerate(aug_images):
                    do_aug_images.append(aug_image)
                    do_aug_masks.append(aug_masks[aug_index])
                    do_aug_labels.append(label)
            return np.array(do_aug_images),np.array(do_aug_masks),np.array(do_aug_labels)
        """Start making images"""
        images,masks,labels =  self.extract_roi_with_mask(i_db_path    = self.db_path,
                                                          i_num_fold   = i_num_folds,
                                                          i_fold_index = i_fold_index,
                                                          i_train_set  = i_train_set)
        """Mixed Image Data augmentation"""
        print('Image Range = ',np.min(images),np.max(images))
        print('Mask Range  = ',np.min(masks),np.max(masks))
        print('Label Range = ',np.min(labels),np.max(labels))
        print('Details     = ',images.shape,masks.shape,labels.shape)
        """Check the ground-truth confusion matrix of the dataset"""
        num_labels = len(np.unique(labels))
        num_images = 0
        conf_maxtrix = np.zeros(shape=(num_labels,num_labels))
        for item in labels:
            indice = int(item+0.5) #0.5 for the case of using label smoothing
            conf_maxtrix[indice][indice]+=1
            num_images +=1
        print('-'*50)
        print('Confusion matrix: ')
        print(conf_maxtrix)
        print('-' * 50)
        if i_train_set:
            #images, masks, labels = do_aug(images, masks, labels)
            pass
        else:
            pass
        return images,masks,labels
    def get_data(self,i_num_folds=5,i_fold_index=1):
        train_images, train_masks, train_labels = self(i_num_folds=i_num_folds,i_fold_index=i_fold_index,i_train_set=True)
        val_images, val_masks, val_labels       = self(i_num_folds=i_num_folds,i_fold_index=i_fold_index,i_train_set=False)
        """Making gray images"""
        train_images = np.mean(train_images,axis=-1,keepdims=True).astype(np.uint8)
        val_images   = np.mean(val_images,axis=-1,keepdims=True).astype(np.uint8)
        print(train_images.shape, np.min(train_images),np.max(train_images),np.min(train_masks),np.max(train_masks))
        print(val_images.shape, np.min(val_images), np.max(val_images), np.min(val_masks), np.max(val_masks))
        return (train_images, train_masks, train_labels), (val_images, val_masks, val_labels)
    def get_train_val_test_data(self,i_num_folds=5,i_fold_index=1):
        train_db,test_db = self.get_data(i_num_folds=i_num_folds,i_fold_index=i_fold_index)
        train_images, train_masks, train_labels = train_db
        test_images, test_masks, test_labels    = test_db
        num_train_images = train_images.shape[0]
        num_test_images  = test_images.shape[0]
        print('Number of val_images = {}'.format(num_test_images))
        """Manually select some images for eval purpose"""
        eval_images,eval_masks,eval_labels = [],[],[]
        tr_images, tr_masks, tr_labels = [], [], []
        start_index = i_fold_index - 1
        segment     = i_num_folds - 1
        for val_index in range(0,num_train_images):
            image = train_images[val_index]
            mask  = train_masks[val_index]
            label = train_labels[val_index]
            if val_index%segment == 0 and val_index>=start_index:
                eval_images.append(image)
                eval_masks.append(mask)
                eval_labels.append(label)
            else:
                tr_images.append(image)
                tr_masks.append(mask)
                tr_labels.append(label)
        train_images = np.array(tr_images)
        train_masks  = np.array(tr_masks)
        train_labels = np.array(tr_labels)
        eval_images = np.array(eval_images)
        eval_masks  = np.array(eval_masks)
        eval_labels = np.array(eval_labels)
        train_masks = (train_masks>0).astype(np.uint8)
        eval_masks  = (eval_masks>0).astype(np.uint8)
        test_masks  = (test_masks>0).astype(np.uint8)
        print('Information....')
        print('Training   DB:', train_images.shape,train_masks.shape,train_labels.shape)
        print('Training   DB:', np.min(train_images), np.max(train_images),np.min(train_masks),np.max(train_masks), np.min(train_labels),np.max(train_labels))
        print('Evaluating DB:', eval_images.shape, eval_masks.shape, eval_labels.shape)
        print('Evaluating DB:', np.min(eval_images), np.max(eval_images), np.min(eval_masks), np.max(eval_masks),np.min(eval_labels), np.max(eval_labels))
        print('Testing    DB:', test_images.shape, test_masks.shape, test_labels.shape)
        print('Testing    DB:', np.min(test_images), np.max(test_images), np.min(test_masks), np.max(test_masks),np.min(test_labels), np.max(test_labels))
        train_db = list(zip(train_images,train_masks))
        eval_db  = list(zip(eval_images,eval_masks))
        test_db  = list(zip(test_images,test_masks))
        return train_db,eval_db,test_db
"""====================================================================================================="""
if __name__ == "__main__":
    print("This module is to set general parameters for TDID dataset")
    show                = True
    TDID_DB.num_samples = (1,1,1,1,1,1)
    TDID_DB.num_labels  = 2
    example_db          = TDID_DB(i_db_path=db_path)
    example_images, example_masks, example_labels = example_db(i_num_folds=5,i_fold_index=1,i_train_set=False)
    index = 0
    save_path = os.path.join(os.getcwd(),'images')
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        pass
    for item_db in list(zip(example_images, example_masks, example_labels)):
        index = index +1
        img,msk,lb = item_db
        lb = np.argmax(lb)
        imageio.imwrite('images/{}_image.jpg'.format(index), img)
        imageio.imwrite('images/{}_msk.jpg'.format(index), msk)
        print(img.shape, msk.shape, lb.shape)
        if show:
            plt.subplot(1,2,1)
            plt.imshow(img,cmap='gray')
            plt.title('Label = {}'.format(lb))
            plt.subplot(1,2,2)
            plt.imshow(msk,cmap='gray')
            plt.title('Label = {}'.format(lb))
            plt.show()
            plt.close()
"""====================================================================================================="""