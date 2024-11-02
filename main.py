import glob
import importlib
import os
import cv2

path = 'Images{}*'.format(os.sep)  
all_submissions = glob.glob('./src/*')
os.makedirs('./results/', exist_ok=True)

for idx, algo in enumerate(all_submissions):
    module_name = '{}_{}'.format(algo.split(os.sep)[-1], 'stitcher')
    filepath = '{}{}stitcher.py'.format(algo, os.sep, 'stitcher.py')
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    PanaromaStitcher = getattr(module, 'PanaromaStitcher')
    inst = PanaromaStitcher()

    for impaths in glob.glob(path):
        stitched_image, homography_matrix_list = inst.make_panaroma_for_images_in(path=impaths)
        outfile = './results/{}/{}.png'.format(impaths.split(os.sep)[-1], spec.name)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        cv2.imwrite(outfile, stitched_image)
        print(f'Stitched image saved: {outfile}')