import sys, os, glob
import math
import numpy as np
from PIL import Image
import openslide
import staintools

from preprocess import apply_image_filters, tissue_percent

def get_downsampled_image(slide, scale_factor=32):
    large_w, large_h=slide.dimensions
    new_w = math.floor(large_w/scale_factor)
    new_h = math.floor(large_h/scale_factor)    
    level = slide.get_best_level_for_downsample(scale_factor)
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), Image.BILINEAR)
    return img, new_w, new_h

def get_start_end_coordinates(x, tile_size):
    start = int(x * tile_size)
    end = int((x+1) * tile_size)
    return start, end

def get_stain_normalizer(path='/path/to/reference/image', method='macenko'):
    target = staintools.read_image(path)
    target = staintools.LuminosityStandardizer.standardize(target)
    normalizer = staintools.StainNormalizer(method=method)
    normalizer.fit(target)
    return normalizer

def apply_stain_norm(tile, normalizer):
    to_transform = np.array(tile).astype('uint8')
    to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
    transformed = normalizer.transform(to_transform)
    transformed = Image.fromarray(transformed)
    return transformed

if __name__ == '__main__':
    
    '''
    example: python svs2tile.py 0 20
    '''
    
    slide_rootpath='/path/contains/whole_slide_images/'
    out_rootpath='/path/to/save/svs_tiles/'
    slide_list = [os.path.split(i)[1][:-4] for i in glob.glob(slide_rootpath+'*.svs')]

    start = sys.argv[1]
    stop = sys.argv[2]
    if int(stop) > len(slide_list):
        stop = str(len(slide_list))

    scale_factor = 32

    normalizer = get_stain_normalizer()
    
    for i in range(int(start), int(stop)):
        savepath = out_rootpath + slide_list[i] + '/'
        slide_path=slide_rootpath + slide_list[i] + '.svs'
        slide = openslide.open_slide(slide_path)
        prop = slide.properties
        
        if not os.path.exists(savepath):
            os.makedirs(savepath) 

        if prop['aperio.AppMag']=='40':
            print('AppMag is 40')
            tile_size=512
        elif prop['aperio.AppMag']=='20':
            print('AppMag is 20')
            tile_size=256
        else:
            print('AppMag is %s for %s' % (prop['aperio.AppMag'], slide_list[i]))
            continue

        img, new_w, new_h = get_downsampled_image(slide, scale_factor=scale_factor)
        tissue=apply_image_filters(np.array(img))

        small_tile_size = int(((tile_size/scale_factor)*2+1)//2)
        num_tiles_h = new_h//small_tile_size
        num_tiles_w = new_w//small_tile_size

        for h in range(num_tiles_h):
            for w in range(num_tiles_w):
                small_start_h, small_end_h = get_start_end_coordinates(h, small_tile_size)
                small_start_w, small_end_w = get_start_end_coordinates(w, small_tile_size)
                tile_region = tissue[small_start_h:small_end_h, small_start_w:small_end_w]
                if tissue_percent(tile_region)>=75:
                    try:
                        start_h, end_h = get_start_end_coordinates(h, tile_size)
                        start_w, end_w = get_start_end_coordinates(w, tile_size)
                        tile_path = savepath+slide_list[i]+'_'+str(tile_size)+'_x'+str(start_w)+'_'+str(w)+'_'+str(num_tiles_w)+'_y'+str(start_h)+'_'+str(h)+'_'+str(num_tiles_h)+'.png'
                        if os.path.exists(tile_path):
                            print('%s Alraedy Tiled' % (tile_path))
                        else:
                            tile = slide.read_region((start_w, start_h), 0, (tile_size, tile_size))
                            tile = tile.convert("RGB")
                            if prop['aperio.AppMag']=='40':
                                tile = tile.resize((256, 256), Image.LANCZOS)
                            transformed = apply_stain_norm(tile, normalizer)
                            transformed.save(tile_path)
                    except:
                        print('error for %s' % (tile_path))  
        print('Done for %s' % slide_list[i])
