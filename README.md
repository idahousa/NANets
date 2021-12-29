# NANets
Thyroid Nodule Segmentation in Ultrasound Image Based on Information Fusion of Suggestion and Enhancement Networks

**Introduction**

This reposity is to implement our proposed network for enhancing the performance of thyroid nodule segmentation problem of conventional UNet-based network.

**Please note that, the pretrained models were obtained using our network (NANets) on either TDID [1] or 3DThyroid  [2] datasets.**

[1] Pedraza, L.; Vargas, C.; Narvaez, F.; Duran, O.; Munoz, E.; Romero, E. (2015). An open access thyroid ultrasound-image database. In Proceedings of the 10th International Symposium on Medical Information Processing and Analysis, Cartagena de Indias, Colombia, 28 January 2015; Volume 9287, pp. 1–6.

[2] Wunderling, T.; Golla, B.; Poudel, P.; Arens, C.; Friebe, M.; Hansen, C. (2017). Comparison of thyroid segmentation techniques for 3D ultrasound. Proceedings of SPIE Medical Imaging, Orlando, USA, 2017; https://doi.org/10.1117/12.2254234

**Any work that uses the code and provided pretrained network must acknowledge the authors by including the following reference.**

Dat Tien Nguyen, Jiho Choi, and Kang Ryoung Park, “Thyroid Nodule Segmentation in Ultrasound Image Based on Information Fusion of Suggestion and Enhancement Networks,” Expert Systems With Applications, in submission.

**Usage Instruction**
- To train our proposed network with a custom dataset, please use the main.py script and set the train_flag flag to True.
- To perform inference using our provided pretrained network, please provide the test dataset (itest_db in the main.py), and set the train_flag to False.
- Providing the trainining, testing, and validation set by customize the bellow part in main.py.


itrain_db, ival_db, itest_db = None, None, None

In which: 
- train_db, val_db and test_db are the training, validation, and testing datasets.
- train_db, val_db and test_db are the lists of (2d_image, 2d_mask) pairs
- 2d_image is (0,255) gray image
- 2d_mask is (0,1) label image. 


**Requiremetns**
- Python >= 3.5
- Tensorflow >= 2.1.0
- Window 10

**Example Results**

![2021-12-29-github demo image](https://user-images.githubusercontent.com/13897797/147631167-353e8303-670b-444d-b639-6c58a6c3c649.png)
