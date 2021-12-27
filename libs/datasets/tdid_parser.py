import os
import random
import imageio
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from libs.datasets.nfolds import NFolds
"""This module is to prepared data (TDID dataset) for experiments
   Donot use this class directly! This class is used by TDID class"""
class TDID_Parser:
    benign_scores = ('2','3')            #Label 0 for binary classification
    malign_scores = ('4a','4b','4c','5') #Label 1 for binary classification
    scores = benign_scores + malign_scores
    num_samples   = (1,1,1,1,1,1)
    def __init__(self,
                 i_db_path    = None,
                 i_num_fold   = 2,
                 i_fold_index = 1,
                 i_train_set  = True):
        assert isinstance(i_db_path,str) #Path to mother director that contains TDID thyroid dataset
        assert os.path.exists(i_db_path)
        assert isinstance(i_num_fold,int)
        assert isinstance(i_fold_index,int)
        assert 0< i_num_fold<=5   # As my design
        assert 0< i_fold_index<=i_num_fold # As my design
        assert isinstance(i_train_set,bool)
        self.db_path     = i_db_path
        self.num_folds   = i_num_fold
        self.fold_index  = i_fold_index
        self.istrain_set = i_train_set
        self.xmls       = self.get_xml_files()

    """1. List all the xml file in the dataset directory"""
    def get_xml_files(self):
        """Get all the xml files (annotation files) in the TDID dataset"""
        files  = [file for file in os.listdir(self.db_path) if os.path.isfile(os.path.join(self.db_path, file))]
        files  = [os.path.join(self.db_path,file) for file in files if file.endswith('.xml')]
        files  = [file for file in files if self.parse_tirads(file) is not None]
        labels = [self.scores.index(self.parse_tirads(file)) for file in files]
        nfold  = NFolds(i_data=files,i_labels=labels,i_num_fold=self.num_folds,i_num_samples=self.num_samples)
        train_sets,val_sets = nfold(i_fold_index=self.fold_index)
        if self.istrain_set:
            xmls = [files[index] for index in train_sets]
            random.shuffle(xmls)
            return xmls
        else:
            xmls = [files[index] for index in val_sets]
            random.shuffle(xmls)
            return xmls
    @classmethod
    def parse_tirads(cls,i_xml_path = None):
        """Extract the TI-RADS score of images in xml file
           One xml file can contain information of more than one image.
        param i_xml_path: Full path to a xml file (of TDID dataset) in disk.
        :return: The TI-RADS score of images of same person.
        """
        assert isinstance(i_xml_path,str)
        assert os.path.exists(i_xml_path)
        try:
            tree = ET.parse(i_xml_path)
        except FileNotFoundError:
            return -1
        root = tree.getroot()
        tirads = []
        for item in root.iter('tirads'):
            tirads.append(item.text)
        assert len(tirads)==1 #Only one tirads for each person.
        return tirads[0]    #Because only one TI-RADS for one xml file (as I checked)
    @classmethod
    def parse_images(cls,i_xml_path = None):
        """Extract the name of images (of each person) stored in xml file (of TDID dataset)
        :param i_xml_path: Full path to a xml file (of TDID dataset) in disk.
        :return: List of images names.
        """
        assert isinstance(i_xml_path, str)
        assert os.path.exists(i_xml_path)
        assert i_xml_path.endswith('.xml')
        database_path, xml_file_name = os.path.split(i_xml_path)
        xml_file_name = xml_file_name[0:len(xml_file_name)-4] #As file name in format: ***.xml
        try:
            tree = ET.parse(i_xml_path)
        except FileNotFoundError:
            return -1
        root = tree.getroot()
        images = []
        for item in root.iter('image'):
            images.append(os.path.join(database_path,'{}_{}.jpg'.format(xml_file_name,item.text)))
        return images
    @classmethod
    def draw_landmark(cls, i_image, i_landmark):
        """Draw landmark points of thyroid regions
        :param i_image: An input thyroid ultrasound image
        :param i_landmark: A list (or tuple) of landmark points
        """
        assert isinstance(i_image, np.ndarray)
        assert isinstance(i_landmark, (list, tuple))
        height, width, depth = i_image.shape
        for point in i_landmark:
            pt_x, pt_y = point
            if pt_x < 0 or pt_x >= width or pt_y < 0 or pt_y >= height:
                pass
            else:
                i_image[pt_y, pt_x, :] = 255
        return i_image
    """Example"""
    def show(self,i_xml_index=0,i_show=True):
        assert isinstance(i_xml_index,int)
        assert 0<=i_xml_index<len(self.xmls)
        xml_path  = self.xmls[i_xml_index]
        landmarks = self.parse_xml(xml_path)
        images = []
        for landmark in landmarks:
            image_path = landmark['image']
            image_name = os.path.split(image_path)[1]
            tirad      = landmark['tirad']
            image = imageio.imread(image_path)
            image = TDID_Parser.draw_landmark(image,landmark['landmark'])
            if i_show:
                plt.imshow(image)
                plt.title('Image Name = {} with TIRAD = {}'.format(image_name,tirad))
                plt.show()
            else:
                pass
            images.append(image)
        return xml_path,images
    """Main function: Only correct with the TDID dataset. For other dataset, we must refer to the format of xml file first!(datnt)"""
    @classmethod
    def parse_xml(cls, i_xml_path = None):
        """Extract the information of images in TDID dataset
        :param i_xml_path: Full path to a xml file (of TDID dataset) in disk.
        :return:
        Note: Check the format of xml file for more detail of implementation.
        """
        def parse_annotation(i_str):
            if isinstance(i_str,str): #Exist point annotation
                pattern = '{"points":'
                cposes   = []
                while True:
                    cpos = i_str.find(pattern)
                    if cpos < 0:
                        break
                    else:
                        cposes.append(cpos)
                        i_str = i_str.replace(pattern, 'x' * len(pattern), 1)
                cposes.append(len(i_str))
                return cposes
            else:#Does not exist point annotation
                return False
        def parse_landmarks(i_str=None):
            if isinstance(i_str,str):
                pass
            else:
                return False
            rtn_land_marks = []
            while True:
                rtn = parse_point(i_str=i_str)
                if isinstance(rtn, list):
                    rtn_land_marks.append(rtn[0])
                    i_str = rtn[1]
                else:
                    break
            return rtn_land_marks
        def parse_point(i_str=None):
            assert isinstance(i_str, str)
            x_pattern = '{"x":'
            y_pattern = '"y":'
            x_value = parse_pattern(i_str=i_str, i_pattern=x_pattern)
            y_value = parse_pattern(i_str=i_str, i_pattern=y_pattern)
            if x_value > 0 and y_value > 0:
                i_str = i_str.replace(x_pattern, 'x' * len(x_pattern), 1)
                i_str = i_str.replace(y_pattern, 'x' * len(y_pattern), 1)
                return [(x_value, y_value), i_str]
            else:
                return -1
        def parse_pattern(i_str=None, i_pattern=None):
            assert isinstance(i_str, str)
            assert isinstance(i_pattern, str)
            ipos = i_str.find(i_pattern)
            i_str_len = len(i_str)
            if ipos >= 0:
                ipos += len(i_pattern)
                end_pos = ipos + 1
                if end_pos>= i_str_len:
                    return -1
                else:
                    while True:
                        if i_str[end_pos] == ',' or i_str[end_pos] == '}':
                            break
                        else:
                            end_pos += 1
                        if end_pos >= i_str_len:
                            break
                    if end_pos >= len(i_str):
                        return -1
                    else:
                        value = int(i_str[ipos:end_pos])
                    return value
            else:
                return -1
        assert isinstance(i_xml_path, str)
        assert os.path.exists(i_xml_path)
        assert i_xml_path.endswith('.xml')
        database_path, xml_file_name = os.path.split(i_xml_path)
        xml_file_name = xml_file_name[0:len(xml_file_name)-4] #As file name in format: ***.xml
        try:
            tree = ET.parse(i_xml_path)
        except FileNotFoundError:
            return -1
        root = tree.getroot()
        land_marks = []
        image_name = []
        tirads     = []
        index      = 0
        for item in root.iter('tirads'):
            tirads.append(item.text)
        assert len(tirads) == 1  # Only one tirads for each person.
        for item in root.iter('image'):
            image_name.append(item.text)
        for item in root.iter('svg'):#Note: One image can have more than one landmark annotation (Correct with only TDID dataset)
            data = item.text
            poses = parse_annotation(i_str=data)
            if isinstance(poses, list):
                for s_index in range(len(poses) - 1):
                    pos = poses[s_index]
                    next_pos   = poses[s_index + 1]
                    annotation = data[pos:next_pos]
                    land_mark  = parse_landmarks(i_str=annotation)
                    if isinstance(land_mark, list):
                        image_path = os.path.join(database_path,'{}_{}.jpg'.format(xml_file_name,image_name[index]))
                        land_marks.append({'image': image_path, 'landmark': land_mark, 'tirad': tirads[0]})#Return value format
                    else:
                        pass
            else:
                pass
            index += 1
        return land_marks
"""====================================================================================================="""
if __name__ == "__main__":
    db_path = r'G:\roots\classifications\thyroid\tdid\images'
    parser = TDID_Parser(i_db_path = db_path,i_train_set=False)
    parser.show(111)
    image_index = 0
    save_path = os.path.join(os.getcwd(),'images','nodules')
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)
    for ind, xml_item in enumerate(parser.xmls):
        print(ind,xml_item)
        cxml_path,cimages = parser.show(ind,False)
        for img in cimages:
            if parser.istrain_set:
                imageio.imwrite(os.path.join(save_path,'train_image_{}.jpg'.format(image_index)),img)
            else:
                imageio.imwrite(os.path.join(save_path,'val_image_{}.jpg'.format(image_index)),img)
            image_index +=1
"""====================================================================================================="""