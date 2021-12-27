import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters as skfilter
from skimage.measure import label as sklabel
from libs.datasets.tdid_parser import TDID_Parser
"""This class is for extract ROI (with processing step) to extract Thyroid region.
Donot use this class directly! This class is used by TDID class"""
class TDID_ROI:
    def __init__(self, *args, **kwargs):
        pass
    """ptile method for adaptive thresholding"""
    @classmethod
    def ptile(cls,i_image=None, i_ptile=0.25):
        assert isinstance(i_image,np.ndarray)
        assert len(i_image.shape)==2
        assert isinstance(i_ptile,float)
        assert 0.0<=i_ptile<=1.0
        hist = np.zeros(shape=(256,))
        height, width = i_image.shape
        for h in  range(height):
            for w in range(width):
                pixel = int(i_image[h,w])
                hist[pixel,] += 1
        sum_hist = np.sum(hist)
        assert sum_hist == height*width
        threshold = 0
        for threshold in range(256):
            ratio = np.sum(hist[0:threshold])
            ratio = ratio/sum_hist
            if ratio >= i_ptile:
                break
            else:
                pass
        return hist,threshold
    """"Small object removal and disclosing method"""
    @classmethod
    def n_connection(cls,i_image=None,i_size=3):
        assert isinstance(i_image,np.ndarray)
        assert len(i_image.shape)==2
        def check(image,i_h,i_w,i_margin=2,evidence=False):
            if image[i_h,i_w]:
                pass
            else:
                return False
            if evidence:
                w_begin = i_w + i_margin
            else:
                w_begin = i_w - i_margin
            h_begin = i_h - i_margin
            h_end   = i_h + i_margin + 1
            w_end   = i_w + i_margin + 1
            for i in range(h_begin,h_end):
                for j in range(w_begin,w_end):
                    if image[i,j]:
                        continue
                    else:
                        return False
            return True
        margin       = i_size//2
        height,width = i_image.shape
        rtn_image    = np.zeros_like(i_image)
        for h in range(margin,height-margin):
            is_connected = False
            for w in range(margin,width-margin):
                is_connected = check(i_image,h,w,margin,is_connected)
                if is_connected:
                    rtn_image[h,w]=1
                else:
                    continue
        return rtn_image
    @classmethod
    def first_stage(cls,i_image=None,i_ul=None,i_br=None):
        assert isinstance(i_image,np.ndarray)
        def get_coordinate(i_label_image,i_object_index):
            y_cor, x_cor = np.where(i_label_image == i_object_index)
            height,width = i_label_image.shape[0:2]
            if i_ul is not None and i_br is not None:
                margin = 5
            else:
                margin = 0
            min_x  = max(0,min(x_cor))      + margin
            max_x  = min(width,max(x_cor))  - margin
            min_y  = max(0,min(y_cor))      + margin
            max_y  = min(height,max(y_cor)) - margin
            print(min_y,min_x,max_y,max_x,margin)
            if i_ul is not None and i_br is not None:#For the second box
                return (min_y+i_ul[0],min_x+i_ul[1]),(max_y+i_ul[0],max_x+i_ul[1]) #Format: (y,x) ~ (h, w).
            else:#For the first box
                return (min_y,min_x),(max_y,max_x) #Format: (y,x) ~ (h, w).
        """Get gray image"""
        if len(i_image.shape)==2:
            gray_image = i_image.copy()
        else:
            gray_image = np.average(i_image,2)
        if i_ul is not None and i_br is not None:
            gray_image = gray_image[i_ul[0]:i_br[0],i_ul[1]:i_br[1]]
            ptile = 0.01
            ksize = 3
        else:#Call the first-stage using only input image
            ptile = 0.25
            ksize = 7
        """Get adaptive threshold"""
        hist, threshold = cls.ptile(gray_image,ptile)
        bin_image   = gray_image>=threshold
        bin_image   = cls.n_connection(bin_image,ksize)
        """Find the connected objects"""
        label_image, num_labels = sklabel(bin_image, return_num=True)
        sizes = [0] #Note: 0 is used for background
        for index in range(1,num_labels+1):#Note: 0 is used for background
            size = np.sum(label_image==index)
            sizes.append(size)
        sizes = np.array(sizes)
        """Find the two largest object. Only correct for TDID dataset"""
        first_object       = np.argmax(sizes)
        first_object_size  = sizes[first_object]
        sizes[first_object]= 0
        second_object = np.argmax(sizes)
        second_object_size = sizes[second_object]
        if first_object_size>2*second_object_size:#Single object
            return get_coordinate(label_image,first_object),False
        else:#Two separated object
            return get_coordinate(label_image,first_object),get_coordinate(label_image,second_object),True
    @classmethod
    def second_stage(cls,i_image,i_roi):
        assert isinstance(i_image,np.ndarray)
        ul,br = i_roi[0] #As my design
        image = i_image[ul[0]:br[0],ul[1]:br[1],:]
        """Define vertical mask"""
        mask_v = np.zeros(shape=(10, 5))
        mask_v[:, 0] =  1
        mask_v[:, 1] =  3
        mask_v[:, 2] =  0
        mask_v[:, 3] = -3
        mask_v[:, 4] = -1
        """"Convolution"""
        if len(i_image.shape)==2:
            gray_image = image.copy()
        else:
            gray_image = np.sum(image, 2).astype(int)
        fimage = abs(skfilter.edges.convolve(input=gray_image, weights=mask_v))
        """Accumulate projection histogram"""
        hist    = np.sum(fimage,axis=0)
        max_pos = np.argmax(hist)
        max_val = hist[max_pos]
        avg_val = np.average(hist)
        hist[max_pos] = avg_val
        avg_val = np.average(hist)
        std_val = np.std(hist)
        print(avg_val,std_val,max_val)
        if max_val > avg_val+5*std_val: #Outlier detection using 5-sigma rule
            pt1_ul = ul
            pt1_br = (br[0],max_pos+ul[1])
            pt2_ul = (ul[0],max_pos+ul[1])
            pt2_br = br
            roi1_size = (pt1_br[0]-pt1_ul[0])*(pt1_br[1]-pt1_ul[1])
            roi2_size = (pt2_br[0]-pt2_ul[0])*(pt2_br[1]-pt2_ul[1])
            print(roi1_size,roi2_size)
            if roi1_size>roi2_size:
                if roi1_size > 1.5* roi2_size:
                    #Invalid
                    return cls.first_stage(i_image,i_roi[0][0],i_roi[0][1])#i_roi
                else:
                    pass
            else:
                if roi2_size>1.5*roi1_size:
                    #Invalid
                    return cls.first_stage(i_image,i_roi[0][0],i_roi[0][1])#i_roi
                else:
                    pass
            """Exist two ROIs that cannot be detected using the first-stage"""
            roi1   = cls.first_stage(i_image,pt1_ul,pt1_br)
            roi2   = cls.first_stage(i_image,pt2_ul,pt2_br)
            return roi1[0],roi2[0],False
        else:#It correctly contains single ROI in the given image
            return cls.first_stage(i_image,i_roi[0][0],i_roi[0][1])#i_roi
    @classmethod
    def draw_rectangle(cls,i_image,i_ul,i_br,i_value=255):#Format (y,x)
        assert isinstance(i_image,np.ndarray)
        assert isinstance(i_ul, (list, tuple))
        assert isinstance(i_br, (list, tuple))
        height, width = i_image.shape[0:2]
        assert 0<=i_ul[0]<=height
        assert 0<=i_ul[1]<=width
        assert 0<=i_br[0]<=height
        assert 0<=i_br[1]<=width
        for index in range(i_ul[0], i_br[0]):
            i_image[index, i_ul[1]] = i_value
            i_image[index, i_br[1]] = i_value
        for index in range(i_ul[1], i_br[1]):
            i_image[i_ul[0], index] = i_value
            i_image[i_br[0], index] = i_value
        return i_image
    """Example usage"""
    @classmethod
    def show(cls,i_xml_path=None,i_show=True):
        assert isinstance(i_xml_path,str)
        assert os.path.exists(i_xml_path)
        assert isinstance(i_show,bool)
        rois = cls.get_roi(i_xml_path=i_xml_path)
        for roi in rois:
            image_path = roi['image']
            image = imageio.imread(image_path)
            image_path,image_name = os.path.split(image_path)
            dst_dir = os.path.join(os.getcwd(),'images','rois')
            if os.path.exists(dst_dir):
                pass
            else:
                os.makedirs(dst_dir)
            dst_path = os.path.join(dst_dir,image_name)
            imageio.imsave(dst_path,image)
            """Start showing"""
            #image = skfilter.median(image)
            crois = roi['rois']
            for croi in crois:
                image = cls.draw_rectangle(image,croi[0],croi[1])
            if i_show:
                plt.imshow(image)
                plt.show()
            else:
                pass
            """Save image to disk"""
            dst_path = os.path.join(dst_dir,image_name[0:len(image_name)-4]+'_temp'+'.JPG')
            imageio.imsave(dst_path,image)
            print('Save imae to {}'.format(dst_path))
    """Main function"""
    @classmethod
    def get_roi(cls,i_xml_path=None):
        assert isinstance(i_xml_path,str)
        assert os.path.exists(i_xml_path)
        assert i_xml_path.endswith('.xml')
        """landmarks is a list of dictionaries"""
        xml_path,xml_name = os.path.split(i_xml_path)
        xml_name          = xml_name[0:len(xml_name)-4]
        save_path         = os.path.join(xml_path,'{}_ROIs.npy'.format(xml_name))
        if os.path.exists(save_path):
            print('Load ROIs from: {}'.format(save_path))
            return np.load(save_path,allow_pickle=True)
        else:
            pass
        landmarks = TDID_Parser.parse_xml(i_xml_path=i_xml_path)
        rois      = []
        processed_images = []
        for landmark in landmarks:
            image_path = landmark['image']
            tirad      = landmark['tirad']
            if image_path in processed_images:
                continue
            else:
                processed_images.append(image_path)
            print(landmark)
            print(processed_images)
            image      = imageio.imread(image_path)
            c_rois     = cls.first_stage(i_image=image)
            if c_rois[-1]:#Exist two objects in the image.
                pass
            else:#Exist one object => Check in advance to see whether it contains more than one object.
                c_rois = cls.second_stage(image,c_rois)
            rois.append({'image':image_path,'rois':c_rois[0:len(c_rois)-1],'tirad':tirad})
        rois = np.array(rois)
        np.save(save_path,rois)
        return rois
"""====================================================================================================="""
if __name__ == "__main__":
    print("This module is to extract roi for thyroid images")
    print("This module is to produce the ROI in which Thyroid regions exist!")
    db_path = r'G:\roots\classifications\thyroid\tdid\images'
    parser = TDID_Parser(i_db_path = db_path,i_train_set=False)
    item = parser.xmls[10]
    TDID_ROI.get_roi(item)
    TDID_ROI.show(i_xml_path=item)
    for item in parser.xmls:
        TDID_ROI.show(i_xml_path=item,i_show = False)
"""====================================================================================================="""
