import zipfile
import cv2
import os
import re
import csv
import random
import argparse
import numpy as np
from time import sleep
from multiprocessing import Pool

# to display benchmark progress bar:
from tqdm import tqdm

class TracknetDataset:
    
    def __init__(self, path):
        # find the contents of the dataset:
        self.data = {}
        self.labels = {}
        anno_dir = os.path.join(path,'anno')
        zips_dir = os.path.join(path,'zips')

        for item in os.listdir(anno_dir):
            fullpath = os.path.join(anno_dir,item)
            name = os.path.splitext(item)[0]
            self.labels[name] = self._parse_label(fullpath)

        for item in os.listdir(zips_dir):
            fullpath = os.path.join(zips_dir,item)
            name = os.path.splitext(item)[0]
            assert(name in self.labels)
            self.data[name] = fullpath
            
        
    def load_images(self, name):
        images = []
        assert(name in self.data)
        assert(name in self.labels)
        with zipfile.ZipFile(self.data[name],'r') as img_zip:
            for item in img_zip.infolist():
                img_data = img_zip.read(item)
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8),1)
                images.append((item.filename, img))
        
        images.sort(key=lambda x : int(re.sub('\D','',x[0])))
        return images

    def load_rects(self, name):
        return self.labels[name]

    def show_example(self, name=None):
            if name is None:
                name = random.choice(list(self.data.keys()))
            imgs_example = self.load_images(name)
            rects_example = self.load_rects(name)
                
            # display image sequence:
            frame = None
            for (name, img), rect in tqdm(zip(imgs_example, rects_example)):
                #print(name)
                x,y,w,h = tuple([int(x) for x in rect])
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
                cv2.imshow("Preview (press \'q\' to quit)",img)
                keyboard = cv2.waitKey(100) & 0xFF
                if keyboard == ord("q") or keyboard == 27:
                    break

            cv2.destroyAllWindows()
        
    def _parse_label(self, path):
        label = []
        with open(path,'r') as f:
            for line in f.readlines():
                label.append([ float(s.strip()) for s in line.split(',') ])
        return label


def run_benchmark(tds, detector, metric='center_rmse', n_procs=4):
    
    # This error function simply computes the rmse of the 
    # predicted object centers:
    def center_rmse_err(imgs, rects):
        centers = np.array([[r[0],r[1]] for r in rects ])
        detector.reset()
        for img in imgs:
            pass
            


    err_fn = None
    if metric == 'center_rmse':
        err_fn = center_rmse_err
        
    pbar = tqdm(total=len(tasks))
    def eval_err(name):
        imgs = tds.load_images(name)
        rects = tds.load_rects(name)
        err = err_fn(imgs, rects)
        pbar.update(1)
        return err
    
    pool = Pool(processes=4)
    errors = pool.imap_unordered(eval_err,list(tds.data.keys()))
    err_mean = np.mean(np.array(errors))
    err_dict = { k:err for k, err in zip(tds.data.keys(),errors) }
    
    return err_mean, err_dict
            

def main():
    parser = argparse.ArgumentParser('This runs an object tracking benchmark against the TrackNet dataset')
    parser.add_argument('--dataset', required=True, help='Path to uncompressed TrackNet dataset (e.g. TRAIN_0)')
    args = parser.parse_args()
    tds = TracknetDataset(args.dataset)

    print('Number of videos: ', len(tds.data))
    with open('./selections.txt','a') as selections:
        for item in tds.data.keys():
            tds.show_example(item)
            ans = input(f'Add example {item} to selections (y/n)?')
            if ans == 'y' or ans == 'Y':
                selections.write(item + '\n')
    

if __name__ == '__main__':
    main()
