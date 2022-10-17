from imaris_ims_file_reader.ims import ims
from utils.utils import imagesc
import glob, os
import tqdm
import tifffile as tiff


class imaris_2_roi:
    def __init__(self, source):
        """
        Args:
            source: path to the source of the imaris file
        """
        self.source = ims(source)
        self.shape = self.source.shape

    def projection_multi(self, destination, zstep, **kwargs):
        os.makedirs(destination, exist_ok=True)
        for z in range(*zstep):
            kwargs['z'] = (z, z + zstep[2])
            projection = self.source.projection(**kwargs)
            tiff.imsave(os.path.join(destination, str(z) + '.tif'), projection)
        return None


if __name__ == '__main__':
    source = sorted(glob.glob('/media/ghc/GHc_data2/BRC/10xTH/*'))

    x = imaris_2_roi(source[0])
    #x.projection_multi(destination='temp/', zstep=(0, 1000, 1), projection_type='mean', resolution_level=2)