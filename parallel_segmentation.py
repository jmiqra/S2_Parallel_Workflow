import cv2 # import after setting OPENCV_IO_MAX_IMAGE_PIXELS
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
import os
import multiprocessing as mp
import sys
import time
#import PIL
from PIL import Image
import math

def color_segmentation(img):
    lower_ice = (0, 0, 205)#(127, 7, 94) #increase v to specify ow
    upper_ice = (185, 255, 255)#(147, 53, 232) #increase h to specify si

    lower_tice = (0, 0, 31)#(127, 7, 94) #increase v to specify ow
    upper_tice = (185, 255, 204)#(147, 53, 232) #increase h to specify si

    lower_water = (0, 0, 0)#(127, 7, 94) #increase v to specify ow
    upper_water = (185, 255, 30)#(147, 53, 232) #increase h to specify si
    # Get a "mask" over the image for each pixel
    # if a pixel's color is between the lower and upper white, its mask is 1
    # Otherwise, the pixel's mask is 0
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mask_ice = cv2.inRange(hsv_img, lower_ice, upper_ice)
    mask_tice = cv2.inRange(hsv_img, lower_tice, upper_tice)
    mask_water = cv2.inRange(hsv_img, lower_water, upper_water)

    # duplicate the image
    seg_img = img.copy()

    #color each masked portion
    seg_img[mask_ice == 255] = [255, 0, 0]
    seg_img[mask_tice == 255] = [0, 0, 255]
    seg_img[mask_water == 255] = [0, 255, 0]
    
    return seg_img


def my_func(x):
    print(mp.current_process())
    return x**x

#pc = mp.cpu_count()
#number of processes
pc = int(sys.argv[1])
#root_image = "s2_images/"
#root_seg = "s2_seg/"
#root_image = "s2_250x250/"
#root_seg = "s2_250x250_seg/"
#root = "/Users/jmiqrah/Desktop/IS2_ML/Segmentation_Sentinel_2/"
root_image = "s2_par/s2_vis/"
root_seg = "s2_par/s2_seg/"
root_vis_split = "s2_par/s2_vis_split/"
root_seg_split = "s2_par/s2_seg_split/"
#all_files = os.listdir("/path-to-dir")    
#csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
n = 0#len( list(filter(lambda f: f.endswith('.png'), os.listdir(root_vis_split))) )
n_per_pc = 0#int(n/pc)
extra = 0#n%pc
idx = 0

def col_seg(x):
    global idx
    #created = mp.Process()
    current = mp.current_process()
    #print('running:', current.name, current._identity)
    #print('created:', created.name, created._identity)
    pid = int(str(current._identity)[1])

    print("Pid start ", pid, extra, idx, n_per_pc, n)

    file_name = []
    
    if pid == 1:
        idx = 0
    else:
        idx = extra

    print(idx)

    start_idx = ((pid-1)*n_per_pc) + idx
    end_idx = start_idx + n_per_pc + extra
    
    #for filepath in sorted(os.listdir(root_vis_split))[start_idx : end_idx]:
    for filepath in sorted(list(filter(lambda f: f.endswith('.png'), os.listdir(root_vis_split))))[start_idx: end_idx]:
        #print(pid, " Filepath ", root_image + filepath)
        file_name.append(root_vis_split + filepath)
        
    for f in sorted(file_name):
        if f.split(".")[-1].lower() in {"png"}:
            im = cv2.imread(sys.path[0]+"/"+f,1)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            sm = color_segmentation(im)
            sm = cv2.cvtColor(sm, cv2.COLOR_BGR2RGB)
            fw = f.replace(root_vis_split,root_seg_split).replace("vis","seg")
            #print(fw)
            cv2.imwrite(fw, sm) 
            #print(sm)
    return


def init(l):
    global lock
    lock = l

ori_size = 2000 #image original size
size = 250


def image_split():
    #split image to 100x100
    for filepath in sorted(os.listdir(root_image)):
        #only selecting "png" files
        if filepath.split(".")[-1].lower() in {"png"}:
            img = cv2.imread(sys.path[0]+"/"+root_image + filepath, 1)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #print(filepath)
            for r in range(0,img.shape[0], size):
                for c in range(0,img.shape[1],size):
                    filepath = filepath.replace(".png", "")
                    filename = root_vis_split + filepath + "_" + str(r).zfill(4) + "_" + str(c).zfill(4) + ".png"
                    #print(filename)
                    cv2.imwrite(filename, img[r:r+size, c:c+size,:])
    return

def image_merge():
    
    splits = int(pow((ori_size/size),2))
    filename = []

    for filepath in sorted(os.listdir(root_seg_split)):
        #only selecting "png" files
        if filepath.split(".")[-1].lower() in {"png"}:
            filename.append(root_seg_split + filepath)

    #print(len(filename))

    tot_img = int(len(filename)/splits)
    #print(tot_img)

    for f in range(tot_img):
        list_im = []
        for i in range(splits):
            list_im.append(filename[i+f*splits])
        #list_im.sort()
        #print(list_im)
        
        imgs = [Image.open(i) for i in list_im]
        new_im = Image.new('RGB', (ori_size, ori_size))

        # for a horizontal stacking it is simple: use hstack
        s = int((ori_size/size))
        imgs_comb_row = []
        for j in range(s):
            imgs_comb_row.append( np.hstack( (np.asarray( imgs[i+j*s] ) for i in range(s) ) ) )
        
        # for a vertical stacking it is simple: use vstack
        imgs_comb_col = np.vstack( (np.asarray( imgs_comb_row[i] ) for i in range(s) ) )
        
        #naming convention s2_vis_00_0_0.png
        name = list_im[0].split("/")[-1].replace("vis","seg")
        k = "_" + name.split(".")[0].split("_")[-1] + "_" + name.split(".")[0].split("_")[-2]
        name = name.replace(k, "")    

        # save that beautiful picture
        imgs_comb = Image.fromarray( imgs_comb_col)
        #print(root_seg + name)
        imgs_comb.save( root_seg + name)

    return

def main():

    t0 = time.time()

    image_split()
    print("split image done")

    t1 = time.time()

    #update n after image split
    global n, n_per_pc, extra
    n = len( list(filter(lambda f: f.endswith('.png'), os.listdir(root_vis_split))) )
    n_per_pc = int(n/pc)
    extra = n%pc

    #no. of CPU in the system
    p = mp.cpu_count()
    print("total P is", p)
    #pc = n
    #divisible by processor 
    #l = mp.Lock()
    #create process pool and call col_seg()
    pl = mp.Pool(processes=pc)#, initializer=init, initargs=(l,))
    result = pl.map(col_seg, range(pc))

    #print(result)
    pl.close()
    pl.join()

    print("par processes done")
    
    t2 = time.time()
    image_merge()
    print("merge image done")

    t3 = time.time()

    total = t3-t0
    split_time = t1-t0
    par_time = t2-t1
    merge_time = t3-t2

    print('Total {:0.2f} sec \nSplit time {:0.2f} sec \nParProc {:0.2f} sec \nMerge {:0.2f} sec'.format(total, split_time, par_time, merge_time))

if __name__ == "__main__":
    main()
