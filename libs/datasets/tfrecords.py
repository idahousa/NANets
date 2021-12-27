import os
import pickle
import numpy as np
import tensorflow as tf
"""=================================================================================================================="""
class TFRecordDB:
    """
    - This class is designed to handle only four cases:
        +) input data is in integer type (python datatype) => Use to handle label, description (using integer values)
        +) input data is in float type (python datatype) => Use to handle the label, description (using float values)
        +) input data is in string (python datatype) => Use to handle image name, path, description
        +) input data is in np.ndarray => Use to handle image data
    - According to type of data, we automically choose the Types between 'FixedLenFeature', or 'VarLenFeature'
    - For image, we input it as np.ndarray. Using tf.image.encode_jpeg to encode it (convert to a series of bytes) to write to tfrecord
    - For string (name, path) => We use the .encode() method (of string class) to convert it to bytes
      => Need to use .decode() method to decode the original string. (See the main function for example usage)
    - This class will write two files:
      +) a tfrecord file that store data
      +) a pkl file (field file) that is used to read the tfrecord file.
    - Use the zip() to prepare data to the TFRecordDB.write() function as this function only accept list (tuple) of records
    - Finally, the important things is the specifying the field dicts that contain the name of fields in the data.
      Example: fields = {'image_path':[],'image':[],'label':[]}
    - Note:
        tfio is official supports from google. I used this one to implement encode_bmp for lossless compresion.
    - if lossy = True, then we use the jpeg compression method to encode and decode image => The input and the reconstruected
                       image can be different because jpeg is a lossy image compression method
    - if lossy = False, then we use the png compression method to encode and decode image => The input and the reconstructed
                       images are same because png is a lossless image compressoin method
    Note: The png only accept image in unit8 datatype, but also integer datatype (uint16)
    Note: This script is only used for images (matrix in matrixs) as it uses encode() opts.
    """
    lossy  = True #If set to False, then use the lossless encoding method (png) to encode image.
    rdtype = tf.dtypes.uint8 #Please use tf.dtypes.uint16 if the input range of image is not uint8. But this case is seldomly use.
    Types  = ['FixedLenFeature','VarLenFeature']              #Fixed this one
    DTypes = ['tf.int64','tf.float64','tf.string','tf.image'] #Fixed this one
    def __init__(self):
        pass
    @staticmethod
    def init_field_path(i_tfrecord_path=None):
        save_path,save_file = os.path.split(i_tfrecord_path)
        assert len(save_file)>8
        save_file_name = save_file[0:len(save_file)-9]
        return os.path.join(save_path,save_file_name+'_field.pkl')
    @staticmethod
    def init_tfrecord_paths(i_basic_tfrecord_file_path=None):
        """Find all name_index.tfrecord file in a directory when name.tfrecord file is given"""
        assert os.path.exists(i_basic_tfrecord_file_path)
        tfrecord_path,tfrecord_file = os.path.split(i_basic_tfrecord_file_path)
        assert len(tfrecord_file)>8
        tfrecord_file_name = tfrecord_file[0:len(tfrecord_file)-9]
        tfrecord_list = [f for f in os.listdir(tfrecord_path) if f.endswith('.tfrecord')]
        tfrecord_list = [os.path.join(tfrecord_path,f) for f in tfrecord_list if f.find(tfrecord_file_name)==0]
        return tfrecord_list
    @staticmethod
    def convert_bytes_features(i_value):
        if isinstance(i_value, type(tf.constant(0))):
            i_value = i_value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        if isinstance(i_value, bytes):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[i_value]))
        elif isinstance(i_value, (list, tuple)):
            for item in i_value:
                assert isinstance(item, bytes)
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=i_value))
        else:
            return None
    @staticmethod
    def convert_float_features(i_value):
        """Returns a float_list from a float / double."""
        assert isinstance(i_value,(float,list,tuple))
        if isinstance(i_value, float):
            return tf.train.Feature(float_list=tf.train.FloatList(value=[i_value]))
        elif isinstance(i_value, (list, tuple)):
            return tf.train.Feature(float_list=tf.train.FloatList(value=i_value))
        else:
            return None
    @staticmethod
    def convert_int64_feautures(i_value):
        """Returns an int64_list from a bool / enum / int / uint."""
        assert isinstance(i_value,(bool,int,list,tuple))
        if isinstance(i_value, (bool, int)):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[i_value]))
        elif isinstance(i_value, (list, tuple)):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=i_value))
        else:
            return None
    @classmethod
    def convert_to_features(cls,i_value):
        """Algorithm: np.ndarray => bytes => using convert_bytes_features"""
        assert i_value is not None
        #print('Writing data with DType = ',type(i_value))
        if isinstance(i_value,float):#type(i_value)==float (python dtype)
            return TFRecordDB.convert_float_features(i_value),cls.Types.index('FixedLenFeature'),cls.DTypes.index('tf.float64')
        elif isinstance(i_value,(str,bytes)):#type(i_value)==bytes (python dtype)
            if isinstance(i_value,str):
                """Lets use string.encode() to convert string to bytes"""
                i_value = i_value.encode()
            else:
                pass
            return TFRecordDB.convert_bytes_features(i_value),cls.Types.index('VarLenFeature'),cls.DTypes.index('tf.string')
        elif isinstance(i_value,(bool,int)):#type(i_value)==int (python dtype)
            return TFRecordDB.convert_int64_feautures(i_value),cls.Types.index('FixedLenFeature'),cls.DTypes.index('tf.int64')
        elif isinstance(i_value,np.ndarray):#list,tuple or numpy array
            """This option is to process image. So, its shape must be 2 (gray), or 3 (RGB)"""
            assert isinstance(i_value,np.ndarray)
            assert len(i_value.shape) in (2,3)
            if len(i_value.shape)==2:
                i_value = np.expand_dims(i_value,-1)
            assert len(i_value.shape)==3
            if cls.lossy:
                """Using encode_jpeg for lossy compression"""
                i_value = tf.io.encode_jpeg(i_value, quality=100)  # Set to 100 for best quality (lossess compression)
            else:
                i_value = tf.io.encode_png(i_value,compression=-1) #Use 9 for smallest file size (slowest).0 for largest file size(fastest). Use -1 for default (similar to 9)
            return TFRecordDB.convert_bytes_features(i_value), cls.Types.index('VarLenFeature'), cls.DTypes.index('tf.image')
        else:
            """Numpy number such as numpy.uint8, numpy.int16 etc. """
            """Check this one again for better performance and exceptions"""
            return TFRecordDB.convert_to_features(i_value.item())
    @staticmethod
    def serialize(i_features=None,i_fields=None):
        """Serialize one record"""
        assert isinstance(i_features,(list,tuple))
        assert isinstance(i_fields,dict)
        field_keys = list(i_fields.keys())
        assert len(field_keys)==len(i_features)
        feature       = {}
        serial_types  = []
        serial_dtypes = []
        for index, item in enumerate(i_features):
            key   = '{}'.format(field_keys[index])
            value,types,dtypes = TFRecordDB.convert_to_features(item)
            feature.update({key:value})
            serial_types.append(types)
            serial_dtypes.append(dtypes)
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        serial_data = example.SerializeToString()
        return serial_data,serial_types,serial_dtypes
    @staticmethod
    def write_single_file(i_n_records=None,i_fields=None,i_save_file=None):
        """i_save_file is the path to tfrecord file that we want to create, Example: C:\\...\\db.tfrecord"""
        assert isinstance(i_n_records,(list,tuple))
        assert isinstance(i_fields,dict)
        save_path,save_file = os.path.split(i_save_file)
        assert save_file.endswith('.tfrecord')
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        if os.path.exists(i_save_file):
            return False
        else:
            pass
        gtypes,gdtypes = [],[]
        with tf.io.TFRecordWriter(i_save_file) as writer:
            for record in i_n_records:
                example,types,dtypes = TFRecordDB.serialize(record,i_fields)
                writer.write(example)
                """Assertion of gtypes and gdtypes"""
                if len(gtypes)==0:
                    gtypes = types
                else:
                    for index in range(len(gtypes)):
                        assert gtypes[index]==types[index]
                if len(gdtypes)==0:
                    gdtypes = dtypes
                else:
                    for index in range(len(gdtypes)):
                        assert gdtypes[index]==dtypes[index]
        """Save i_fields,gtypes, and gdtypes to file that will be used for reading (parsig) tfrecord fields"""
        zip_types = list(zip(gtypes,gdtypes))
        for index, key in enumerate(i_fields.keys()):
            i_fields.update({key:zip_types[index]})
        save_field_path = TFRecordDB.init_field_path(i_save_file)
        if os.path.exists(save_field_path):
            print('Did not write field file (.pkl) for this case?')
        else:
            with open(save_field_path,'wb') as field_file:
                pickle.dump(i_fields,field_file)
        return i_fields
    @staticmethod
    def write(i_n_records=None,i_size = 1000, i_fields=None,i_save_file=None):
        """i_size is the number of records to be written in single tfrecord files"""
        assert isinstance(i_n_records,(list,tuple))
        num_records = len(i_n_records)
        save_paths = []
        if num_records<=i_size:
            TFRecordDB.write_single_file(i_n_records=i_n_records,i_fields=i_fields,i_save_file=i_save_file)
            save_paths.append(i_save_file)
        else:
            num_files = num_records//i_size
            if num_records%i_size==0:
                pass
            else:
                num_files +=1
            print('Num files: ',num_files)
            save_path,save_file = os.path.split(i_save_file)
            assert len(save_file)>8
            save_file_name = save_file[0:len(save_file)-9]
            for index in range(num_files):
                print('Writing {}(th) block!...'.format(index))
                if index == 0:
                    c_save_file_path = os.path.join(save_path,'{}.tfrecord'.format(save_file_name,index))
                else:
                    c_save_file_path = os.path.join(save_path,'{}_{}.tfrecord'.format(save_file_name,index))
                size_records = []
                for item in range(index*i_size,(index+1)*i_size):
                    if item>=len(i_n_records):
                        pass
                    else:
                        size_records.append(i_n_records[item])
                TFRecordDB.write_single_file(i_n_records=size_records,i_fields=i_fields,i_save_file=c_save_file_path)
                save_paths.append(c_save_file_path)
        return save_paths
    @staticmethod
    def init_file_path(i_main_tfrecord_path=None,i_segment_index=1):
        """This support function is to create file name of the n(th) tfrecord file"""
        """It is used when we need to create a tfrecord file which contain very big number of samples"""
        """See the VinBigData dataset for an example in the libs"""
        assert isinstance(i_main_tfrecord_path,str)
        assert isinstance(i_segment_index,int)
        assert i_segment_index>=0
        file_path,file_name = os.path.split(i_main_tfrecord_path)
        assert file_name.endswith('.tfrecord')
        file_name = file_name[0:len(file_name)-len('.tfrecord')]
        if i_segment_index>0:
            rtn_path = os.path.join(file_path,'{}_{}.tfrecord'.format(file_name,i_segment_index))
        else:
            rtn_path = i_main_tfrecord_path
        return rtn_path
    @classmethod
    def reads(cls,i_tfrecord_paths=None,i_original=False):
        assert isinstance(i_original, bool)
        def parse(i_record):#Return list of data of original data
            """Note: This function must be customed according to the datatype of each fields in data structure"""
            """Here is example of reading data struture with three fields of (image_path, image, label)"""
            """Here we use tf.io.parse_single_example() to Parses a single Example proto"""
            """Resulting: A dict mapping feature keys to Tensor and SparseTensor values."""
            return_vals = {} if i_original else []
            data        =  tf.io.parse_single_example(i_record,features=features)
            for record_key in fields.keys():
                record_indicator = fields[record_key]
                record_types     = TFRecordDB.Types[record_indicator[0]]
                record_dtypes    = TFRecordDB.DTypes[record_indicator[1]]
                if record_types == 'FixedLenFeature':
                    val = data[record_key]
                elif record_types  == 'VarLenFeature':
                    val = data[record_key].values[0]
                else:
                    raise Exception('Invalid')
                if record_dtypes == 'tf.image':
                    """Decode the data as image using tf.io.decode_image() ops"""
                    val = tf.io.decode_image(contents=val,dtype=cls.rdtype)
                else:
                    pass
                if i_original:
                    return_vals.update({record_key: val})
                else:
                    return_vals.append(val)
            return return_vals
        assert isinstance(i_tfrecord_paths,(list,tuple))
        fields = {}
        for index, tfrecord_path in enumerate(i_tfrecord_paths):
            field_path = TFRecordDB.init_field_path(tfrecord_path)
            print('Read field file at: ',field_path)
            assert os.path.exists(field_path)
            assert os.path.exists(tfrecord_path)
            with open(field_path,'rb') as pfile:
                c_field = pickle.load(pfile)
            if index == 0:
                fields = c_field
                assert isinstance(fields,dict)
                for key in fields.keys():
                    assert isinstance(fields[key],(list,tuple))
                    assert len(fields[key])==2
            else:
                assert isinstance(fields,dict)
                for key in fields.keys():
                    for ind,el in enumerate(fields[key]):
                        assert el == c_field[key][ind]
        assert isinstance(fields,dict)
        field_keys = list(fields.keys())
        features   = {}
        for key in field_keys:
            indicator = fields[key]
            assert isinstance(indicator,(list,tuple))
            assert len(indicator)==2
            types  = TFRecordDB.Types[indicator[0]]
            dtypes = TFRecordDB.DTypes[indicator[1]]
            if dtypes == 'tf.int64':
                dtypes = tf.int64
            elif dtypes == 'tf.float64':
                dtypes = tf.float64
            elif dtypes == 'tf.string':
                dtypes = tf.string
            elif dtypes == 'tf.image':
                dtypes = tf.string
            else:
                raise Exception('Invalid dtypes')
            if types == 'FixedLenFeature':
                features.update({key: tf.io.FixedLenFeature(shape=[], dtype=dtypes)})
            elif types == 'VarLenFeature':
                features.update({key:tf.io.VarLenFeature(dtype=dtypes)})
            else:
                raise Exception('Invalid types')
        """Loading tfrecord files and create tf.data.Dataset object"""
        tfdataset = tf.data.TFRecordDataset(i_tfrecord_paths)
        tfdataset = tfdataset.map(parse)
        return tfdataset
    @staticmethod
    def read(i_tfrecord_path=None,i_original=False):
        assert os.path.exists(i_tfrecord_path), 'Got path: {}'.format(i_tfrecord_path)
        list_of_tfrecord_files = TFRecordDB.init_tfrecord_paths(i_tfrecord_path)
        return TFRecordDB.reads(i_tfrecord_paths=list_of_tfrecord_files,i_original=i_original)
"""====================================================="""
if __name__ == "__main__":
    print('This module is to create tfrecord files')
    import matplotlib.pyplot as plt
    def make_image(i_label=0):
        mimage = np.zeros(shape=(32,32,1),dtype=np.uint8)
        mimage[10:20,10:20,:]=i_label+1
        return mimage
    """For writing, only need to specify the field's keys"""
    db_fields    = {'image_path':[],'image':[],'mask':[],'label':[]}
    to_save_path = os.path.join(os.getcwd(),'minst.tfrecord')
    """Load data"""
    val_data,train_data = tf.keras.datasets.mnist.load_data()
    train_images,train_labels=train_data
    train_paths = ['{}.jpg'.format(i) for i in range(0,train_images.shape[0])]
    """Input to the write function is the list (tuple) of fields"""
    train_masks = [make_image(label) for label in train_labels]
    train_data  = list(zip(train_paths, train_images, train_masks, train_labels))
    TFRecordDB.lossy = False
    saved_paths      = TFRecordDB.write(i_n_records=train_data,i_size = 100000, i_fields=db_fields,i_save_file=to_save_path)
    dataset = TFRecordDB.read(i_tfrecord_path=to_save_path,i_original=True) #Same as: dataset = TFRecordDB.reads(saved_paths)
    for db_item in dataset:
        if isinstance(db_item,dict):
            name  = db_item['image_path']
            image = db_item['image']
            mask  = db_item['mask']
            label = db_item['label']
        else:
            name,image,mask,label = db_item
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.title('{} with label = {}'.format(name.numpy().decode(),label.numpy()))
        plt.subplot(1,2,2)
        plt.imshow(mask)
        plt.title('{} with label = {}'.format(name.numpy().decode(), label.numpy()))
        plt.show()
"""====================================================="""