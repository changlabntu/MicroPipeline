import numpy as np
import glob, os, time
import tifffile as tiff
from PIL import Image
import scipy.ndimage
import argparse
import pandas as pd
from CSBDeep_fly import run_csbdeep, get_fusion


def crop_by_roi_info(tif, X, Y, dx, dy, margin=0):
    roi = tif[Y-margin:Y+dy+margin, X-margin:X+dx+margin]
    return roi


def concat_2d_cropped_to_3d(roi):
    cropped3d = []

    #for Z in range(roi['Z'] - 1, roi['Z'] - 1 + roi['dz']):
    for Z in range(roi['Z'], roi['Z'] + roi['dz']):
        slice_name = sorted(glob.glob(os.path.join(root, raw_location, '*C2*' + str(Z).zfill(4) + '.tif')))
        if len(slice_name) > 1:
            print('FOUND MORE THAN 1 SLICE')
        else:
            tif = tiff.imread(slice_name[0])

        cropped = crop_by_roi_info(tif=tif, X=roi['X'], Y=roi['Y'], dx=roi['dx'], dy=roi['dy'], margin=args.margin)
        cropped3d.append(cropped)
    cropped3d = np.stack(cropped3d, 0)
    return cropped3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='', help='csb or fusion')
    parser.add_argument('--margin', type=int, default=0, help='csb or fusion')
    parser.add_argument('--roirange', nargs='+', help='number of roi to deal with')
    args = parser.parse_args()

    args.roirange = [int(x) for x in args.roirange]

    # PATH
    root = '/media/ExtHDD02/yading/Dataset/04XFly'
    raw_location = '4x_VT64246-2_GFP_VMAT_DLG_z1_stitch/C2/'

    # Dataframe
    df = pd.read_csv(os.path.join(root, 'fly04x_MushroomBody.csv'))

    # ROI INFO
    if args.task == 'crop':
        for roi_number in range(args.roirange[0], args.roirange[1]):

            roi = df.iloc[roi_number]

            cropped3d = concat_2d_cropped_to_3d(roi)

            os.makedirs(save_location, exist_ok=True)
            save_location = os.path.join(root + '/MushroomBody' , 'roi_test', 'MB_roi_C2_' + str(roi_number).zfill(4) + '.tif')
            print('Cropping is done, tif size ' + str(cropped3d.shape) + ' save at: ' + save_location)
            tiff.imsave(save_location, cropped3d)

    # RUN CSBDeep
    elif (args.task == 'csb') or (args.task == 'fusion') or (args.task == 'csbfusion'):
        tif_list = sorted(glob.glob(os.path.join(root + '/MushroomBody' , 'roi', '*.tif')))

        for roi_number in range(*args.roirange):
            deconv = os.path.join(root + '/MushroomBody', 'deconv', str(roi_number).zfill(4) + '/')
            tif = tiff.imread(tif_list[roi_number])

            if (args.task == 'csb') or (args.task == 'csbfusion'):
                # RUN CSBDEEP
                os.makedirs(deconv, exist_ok=True)

                trd = df.iloc[roi_number]['Threshold']
                tif[tif >= trd] = trd

                rates = df.iloc[roi_number]['Rates']
                rates = [int(x) for x in rates.split(',')]
                rates = [x / 100 for x in range(*rates)]

                run_csbdeep(tif=tif, deconv=deconv, osize=tif.shape[1], rates=rates)

            if (args.task == 'fusion') or (args.task == 'csbfusion'):
                # FUSION
                pseudo = get_fusion(deconv=deconv, osize=tif.shape[1],
                                    threshold_scales=df.iloc[roi_number]['Threshold_Scales'])
                print(pseudo.shape)
                margin = args.margin
                pseudo_cropped = pseudo[:, margin:-margin, margin:-margin]
                os.makedirs(os.path.join(root, 'MushroomBody', 'pseudo'), exist_ok=True)
                pseudo_name = tif_list[roi_number].split('/')[-1]
                tiff.imsave(os.path.join(root, 'MushroomBody', 'pseudo', pseudo_name), pseudo_cropped)

#python get_roi.py --task crop --roirange 0 8 1
#python get_roi.py --task csbfusion --roirange 0 8 1 --margin 16   # (0-8,step 1)