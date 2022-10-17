import numpy as np
import glob, os, time
import tifffile as tiff
from PIL import Image
import scipy.ndimage
import argparse



def to_8bit(x):
    x = (x / x.max() * 255).astype(np.uint8)

    if len(x.shape) == 2:
        x = np.concatenate([np.expand_dims(x, 2)]*3, 2)
    return x


def imagesc(x, show=True, save=None):
    if isinstance(x, list):
        x = [to_8bit(y) for y in x]
        x = np.concatenate(x, 1)
        x = Image.fromarray(x)
    else:
        x = x - x.min()
        x = Image.fromarray(to_8bit(x))
    if show:
        x.show()
    if save:
        x.save(save)


def torch_upsample(x, size):
    import torch
    import torch.nn as nn
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    x = x.permute(0, 1, 3, 4, 2)
    up = nn.Upsample(size=(size, size, x.shape[4]), mode='trilinear', align_corners=True)
    x = up(x)
    x = x.permute(0, 1, 4, 2, 3).squeeze().numpy()
    x = x.astype(np.float16)
    return x


def quick_compare(x0, y0, shade=0.5):
    x = x0 - x0.min()
    x = x / x.max()
    y = y0 - y0.min()
    y = y / y.max()
    combine = np.concatenate([np.expand_dims(x, 2)] * 3, 2)
    combine[:, :, 0] = np.multiply(x, 1 - shade*y)
    combine[:, :, 2] = np.multiply(x, 1 - shade*y)
    return combine


def pil_upsample(x, size):
    y = []
    for s in range(x.shape[0]):
        temp = Image.fromarray(x[s, ::])
        temp = temp.resize(size=(size, size))
        y.append(np.expand_dims(np.array(temp), 0))
    y = np.concatenate(y, 0)
    return y


def run_csbdeep(tif, deconv, osize, rates):
    import tensorflow as tf
    model = tf.saved_model.load('../CSBDeep/param')

    for rate in rates:
        size = int((rate * osize) // 4 * 4)

        npy = pil_upsample(tif, size=size)

        npy = npy / npy.max()
        all = []
        # tini = time.time()

        for s in range(npy.shape[0]):
            print('channel',s)
            fiber = npy[s, ::]  # (H, W)
            # imagesc(fiber, False, f'see_{size}.png')
            fiber = np.expand_dims(fiber, 0)  # [RGB_channel=1, h, w]
            fiber = np.expand_dims(fiber, 3)  # [batch=1, RGB_channel=1, h, w]
            #fiber = np.transpose(fiber, (0, 2, 3, 1)) # [batch, h, w, RGB_channel=1]
            fiber = tf.constant(fiber, dtype=tf.float32)

            # print(list(model.signatures))
            result = model.signatures['serving_default'](fiber)

            model.signatures['serving_default'](tf.constant(fiber, dtype=tf.float32))

            # result have 2 channel, we want channel 0
            #result_to1 = (np.array(result['output'][:, :, :, 0]) - np.array(result['output'][:,:,:,0]).min()) / np.array(result['output'][:,:,:,0]).max()
            out = np.array(result['output'][:, :, :, :])
            all.append(out)

        # print(time.time() - tini)

        all = np.concatenate(all, 0)
        tiff.imsave(deconv + str(int(rate*100)).zfill(3) + '.tif', all[:, :, :, 0].astype(np.float16))


def get_fusion(deconv, osize, threshold_scales):
    """
    Here is the function to resemble the mask you want
    """
    deconv = sorted(glob.glob(deconv + '*'))[:]
    allx = []

    # read all the deconv tifs and then concatenate them together
    for t in deconv[:]:
        x = tiff.imread(t)
        x = torch_upsample(x.astype(np.float32), size=osize)
        x = x - x.min()
        x = x / x.max()
        if len(x.shape) == 2:
            x = np.expand_dims(x, 0)
        allx.append(np.expand_dims(x, 3))
    allx = np.concatenate(allx, 3)

    # voting = (all the deconv >= threshold).sum()
    pseudo = (allx >= threshold_scales).sum(3)
    #tiff.imsave(os.path.join(root, f'pseudo/{filename}.tif'), pseudo.astype(np.uint8))
    imagesc(pseudo[0, ::])

    return pseudo.astype(np.uint8)


def remove_deconv(path):
    os.system('rm -rf ' + path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='', help='name of the tif')
    parser.add_argument('--task', type=str, default='', help='csb or fusion')
    parser.add_argument('--trd', dest='trd', default=0, type=int, help='threshold of grayscale values before CSBdeep')
    parser.add_argument('--rate', nargs='+', help='range of rates before put into CSBdeep', type=int)
    parser.add_argument('--scaletrd', dest='scaletrd', default=0.1, type=float, help='threshold of grayscale values after CSBdeep')
    args = parser.parse_args()

    folder = 'deconv_roiI_2'
    filename = 'pseudo_roiI_2'
    root = '/media/ExtHDD02/yading/Dataset/04XFly/MushroomBody/'
    img_path = os.path.join(root, 'crop', args.name)
    deconv = root + f'/deconv/'


    # run CSSBDeep
    os.makedirs(deconv, exist_ok=True)
    tif = tiff.imread(img_path)

    if len(tif.shape) == 2:
        tif = np.expand_dims(tif, 0)

    osize = tif.shape[1]

    # thresholding
    if args.trd > 0:
        tif[tif >= args.trd] = args.trd

    # downsampling rates into CSBDeep
    rate_range = [args.rate[0], args.rate[1], args.rate[2]]
    rates = [x / 100 for x in range(*rate_range)]

    # # run csb deep
    if args.task == 'csb':
        run_csbdeep(tif, deconv, osize, rates)
    elif args.task == 'fusion':
        get_fusion(deconv, osize, threshold_scales=args.scaletrd)
    elif args.task == 'remove':
        remove_deconv(path=os.path.join(root, 'deconv'))

    # USAGE
    # CUDA_VISIBLE_DEVICES=0 python train.py
    # python CSBDeep_fly/CSBDeep_fly.py --task csb --rate 10 200 10 --name roiA.tif --trd 155
    # python CSBDeep_fly/CSBDeep_fly.py --task fusion --rate 10 200 10 --name roiA.tif --trd 167 --scaletrd 0.17
    # python CSBDeep_fly/CSBDeep_fly.py --task remove --rate 10 100 10 --name roiA.tif --trd 167 --scaletrd 0.17
