import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters as skfilter
from libs.datasets.tdid_parser import TDID_Parser
from skimage.draw import polygon,ellipse,rectangle #Library for drawing polygon
"""======================================================================
This class is to create mask for thyroid region using provided annotation.
Donot use this class directly! This class is used by TDID class"""
class TDID_Mask:
    def __init__(self):
        pass
    @classmethod
    def draw_polygon(cls,i_landmark,i_height,i_width):
        def convert_landmark(i_ldmark):
            assert isinstance(i_ldmark, (list, tuple))
            ixs,iys = [],[]
            for ld_point in i_ldmark:
                pt_x, pt_y = ld_point
                ixs.append(pt_x)
                iys.append(pt_y)
            return np.array(ixs),np.array(iys)
        def get_polygon(i_xs,i_ys,i_h,i_w):
            row, col = polygon(i_ys,i_xs,shape=(i_h,i_w))
            return row,col
        xs, ys = convert_landmark(i_landmark)
        return get_polygon(xs,ys,i_height,i_width)
    @classmethod
    def draw_ellipse(cls,i_cx,i_cy,i_rx,i_ry,i_theta,i_height,i_width):
        row,col = ellipse(i_cy,i_cx,i_ry,i_rx,rotation=i_theta,shape=(i_height,i_width))
        return row,col
    @classmethod
    def draw_rect(cls,i_rect_x,i_rect_y,i_rect_height, i_rect_width,i_width,i_height):
        row, col = rectangle(start=(i_rect_y,i_rect_x),extent=(i_rect_height,i_rect_width),shape=(i_height,i_width))
        return row,col
    """The main function"""
    @classmethod
    def create_mask(cls,i_xml_path=None,i_train_flag=False):
        assert isinstance(i_xml_path,str)
        assert os.path.exists(i_xml_path)
        """landmarks is a list of dictionaries"""
        landmarks = TDID_Parser.parse_xml(i_xml_path=i_xml_path)
        rtn_names     = []
        rtn_masks     = []
        for landmark in landmarks:
            image_path = landmark['image']
            image_name = os.path.split(image_path)[1]
            image      = imageio.imread(image_path)
            height, width = image.shape[0:2]
            row,col = cls.draw_polygon(i_landmark=landmark['landmark'],i_height=height,i_width=width)
            mask    = np.zeros_like(image)
            mask[row,col]=255  #Mark as white pixels
            if i_train_flag:
                """Expanding the shape little to conver all nodule regions"""
                ifilter = np.ones(shape=(11,11,3))
                mask = 255*(abs(skfilter.edges.convolve(input=mask, weights=ifilter))>0).astype(np.uint8)
            else:
                pass
            """In some cases, one image can have two or more landmark annotations"""
            if len(rtn_names) == 0:
                pass
            else:
                if image_name in rtn_names:
                    index         = rtn_names.index(image_name)
                    rtn_masks[index]  += mask   #Element-wise addition for an imae with more than one landmark
                    continue
                else:
                    pass
            rtn_names.append(image_name)
            rtn_masks.append(mask)
        return rtn_names, rtn_masks
"""================================================================"""
if __name__ == "__main__":
    print("This module is to create groundtruth mask for thyroid images")      
    db_path = r'G:\roots\classifications\thyroid\tdid\images'
    parser = TDID_Parser(i_db_path = db_path,i_train_set=True)
    masker = TDID_Mask()
    xml_path,images = parser.show(50)
    print(xml_path)
    names,masks = masker.create_mask(i_xml_path = xml_path)
    for imask in masks:
        plt.imshow(imask)
        plt.show()
"""================================================================"""