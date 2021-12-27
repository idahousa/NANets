import os
import imageio
import numpy as np
from libs.logs import Logs
import matplotlib.pyplot as plt
from libs.datasets.tdid import TDID_DB
from libs.segnets.metrics import SegMetrics
from libs.segnets.segnets import ImageSegNets
from libs.datasets.threeDThyroid import ThreeDimThyroid
def main(i_db_name='3D',i_image_shape=(256, 256, 3),i_fold_index=1,i_net_id=1,i_train=True,i_ckpts=None,i_batchsize=16):
    assert isinstance(i_ckpts,str)
    assert isinstance(i_batchsize,int)
    assert i_batchsize>0
    if i_db_name == '3D':
        db = ThreeDimThyroid()
        train_db, val_db, test_db= db.get_data(i_fold_index=i_fold_index,i_2d_image=True,i_semi_auto=True)
    else:
        db = TDID_DB()
        train_db, val_db, test_db = db.get_train_val_test_data(i_num_folds=5, i_fold_index=i_fold_index)
    """Note: 
    - train_db, val_db and test_db are the lists of (2d_image, 2d_mask) pairs
    - 2d_image is (0,255) gray image
    - 2d_mask is (0,1) label image. 
    """
    """Init the network"""
    segnet = ImageSegNets(i_net_id       = i_net_id,
                          i_save_path    = os.path.join(os.getcwd(), i_ckpts, 'ckpts_fold_{}'.format(i_fold_index)),
                          i_image_shape  = i_image_shape,
                          i_num_classes  = 1,
                          i_continue     = i_train,
                          seg_loss       = 'Dice',
                          seg_batch_size = i_batchsize,
                          seg_epochs     = 50,
                          seg_repeat     = 1)
    if i_train:
        segnet.train(i_train_db=train_db, i_val_db=val_db)
        segnet.eval(i_db=train_db, i_debug=False)
        segnet.eval(i_db=val_db, i_debug=False)
        segnet.eval(i_db=test_db, i_debug=False)
    else:
        dst_path = 'predictions/{}'.format(i_net_id)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        else:
            pass
        segnet.eval(i_db=test_db, i_debug=False)
        evaluer = SegMetrics(i_num_classes=2,i_care_background=False)
        val_images,val_masks = list(zip(*val_db))
        segnet.get_original_output = True
        num_images = len(val_images)
        for example_index in range(num_images):
            example_image = val_images[example_index]
            example_mask  = val_masks[example_index]
            example_pred  = segnet.predict(i_image=example_image)[0]
            plt.subplot(1, 4, 1)
            plt.imshow(example_image, cmap='gray')
            plt.title('Image')
            plt.subplot(1, 4, 2)
            plt.imshow(example_mask, cmap='gray')
            plt.title('Mask')
            plt.subplot(1, 4, 3)
            plt.imshow(example_pred, cmap='gray')
            plt.title('Salient maps')
            plt.subplot(1, 4, 4)
            final_pred = (example_pred >= 0.5).astype(np.float)
            plt.imshow(final_pred, cmap='gray')
            dice = evaluer.get_metrics(i_label=example_mask, i_pred=final_pred)[0]
            plt.title('Prediction - {:2.2f}'.format(dice))
            dice = '{:2.2f}'.format(dice)
            imageio.imwrite('predictions/{}/{}_image.jpg'.format(i_net_id,example_index), example_image)
            imageio.imwrite('predictions/{}/{}_mask.jpg'.format(i_net_id,example_index), example_mask*255)
            imageio.imwrite('predictions/{}/{}_pred.jpg'.format(i_net_id,example_index), example_pred)
            imageio.imwrite('predictions/{}/{}_fpred_{}.jpg'.format(i_net_id,example_index,dice), final_pred)
            #plt.show()
if __name__ == '__main__':
    print('This module is to implement segmentation network for 2D thyroid')
    fold_index    = 1
    db_name       = '2D' #Switch between '2D' or '3D'
    train_flag    = True
    ckpts = 'checkpoints_{}'.format(db_name)
    """UNet-based network"""
    main(i_db_name=db_name, i_fold_index=fold_index, i_net_id=0, i_train=train_flag, i_ckpts=ckpts, i_batchsize=8)
    main(i_db_name=db_name, i_fold_index=fold_index, i_net_id=12, i_train=train_flag, i_ckpts=ckpts, i_batchsize=2)
    """Residual UNet-based network"""
    main(i_db_name=db_name, i_fold_index=fold_index, i_net_id=4, i_train=train_flag, i_ckpts=ckpts, i_batchsize=8)
    main(i_db_name=db_name, i_fold_index=fold_index, i_net_id=13, i_train=train_flag, i_ckpts=ckpts, i_batchsize=2)
    """UNet++-based network"""
    main(i_db_name=db_name, i_fold_index=fold_index, i_net_id=5, i_train=train_flag, i_ckpts=ckpts, i_batchsize=8)
    main(i_db_name=db_name, i_fold_index=fold_index, i_net_id=23, i_train=train_flag, i_ckpts=ckpts, i_batchsize=2)
    """Save the log file to destination trained directory"""
    if train_flag:
        Logs.move_log(i_dst_path=os.path.join(os.getcwd(), ckpts, 'ckpts_fold_{}'.format(fold_index)))
    else:
        pass
"""=================================================================================================================="""