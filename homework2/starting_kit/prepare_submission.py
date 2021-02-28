import os
import json
import numpy as np
from PIL import Image


def rle_encode(img):
    '''
    img: numpy array, 1 - foreground, 0 - background
    Returns run length as string formatted
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


if __name__ == "__main__":

    # The submission file will be a zip file containing the a
    # submission.json file.

    # It must have the following structure:

    # - image name (without any extension, e.g., png, jpg, etc.)
    #   - shape: shape of the original image as list [Height, Width]
    #   - team: team of the original image. One in {'Bipbip', 'Pead', 'Roseau', 'Weedelec'}
    #   - crop: crop of the original image. One in {'Haricot', 'Mais'}
    #   - segmentation: prediction (as a dict)
    #     - crop: RLE encoded crop segmentation (no weed)
    #     - weed: RLE encoded weed segmentation (no crop)

    # Please refer to the folder structure of the given dataset to fill
    # the team and crop keys.

    # EXAMPLE:

    # Suppose that the numpy array generated with the
    # 'read_mask_example.py' script is the output of your segmentation model.
    # In addition, suppose that the team and crop of the segmented image
    # are 'Bipbip' and 'Haricot', respectively.
    # Then, in the following an example of submission file.

    img_name = 'arr_mask_example'

    # load mask <- HERE you should have your segmentation model instead
    mask_arr = np.load('./predictions/arr_mask_example.npy')

    submission_dict = {}
    submission_dict[img_name] = {}
    submission_dict[img_name]['shape'] = mask_arr.shape
    submission_dict[img_name]['team'] = 'Bipbip'
    submission_dict[img_name]['crop'] = 'Haricot'
    submission_dict[img_name]['segmentation'] = {}

    # RLE encoding
    # crop
    rle_encoded_crop = rle_encode(mask_arr == 1)
    # weed
    rle_encoded_weed = rle_encode(mask_arr == 2)

    submission_dict[img_name]['segmentation']['crop'] = rle_encoded_crop
    submission_dict[img_name]['segmentation']['weed'] = rle_encoded_weed

    # Please notice that in this example we have a single prediction.
    # For the competition you have to provide segmentation for each of
    # the test images.

    # Finally, save the results into the submission.json file
    with open('./predictions/submission.json', 'w') as f:
        json.dump(submission_dict, f)
