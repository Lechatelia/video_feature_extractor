import argparse
import os

import h5py
from tqdm import tqdm

from config import config
from feature_extractor import FeatureExtractor2D
import nets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_dpath', type=str, 
                        default='/nfs/zhujinguo/datasets/open_source_dataset/msvd_dataset/YouTubeClips',
                        help="The directory path of videos.")
    parser.add_argument('-m', '--model', type=str, 
                        default='resnet152',
                        help="The name of model from which you extract features.")
    parser.add_argument('-b', '--batch_size', type=int, default=128, help="The batch size.") # 32gv100
    parser.add_argument('-s', '--stride', type=int, default=1,
                        help="Extract feature from every <s> frames.")
    parser.add_argument('-o', '--out', type=str, 
                        default='feature_resnet152_s1/MSVD_ResNet152_{}.hdf5',
                        help="The file path of extracted feature.")
    return parser.parse_args()

def process_vidio(args, video_lists, subset):
    videoss = os.listdir(args.video_dpath)
    videos = []
    for video in videoss:
        if os.path.basename(video).split('.')[0] in video_lists:
            videos.append(video)
    save_path = args.out.format(subset)
    h5 = h5py.File(save_path, 'w') if not os.path.exists(subset) else h5py.File(subset, 'r+')
    for video in tqdm(videos):
        video_id = os.path.splitext(video)[0]
        if video_id in h5.keys():
            continue

        video_fpath = os.path.join(args.video_dpath, video)
        feats = extractor(video_fpath)
        if feats is not None:
            h5[video_id] = feats
    h5.close()
    
    
if __name__ == '__main__':
    args = parse_args()
    
    config = config[args.model]

    model = getattr(nets, args.model)(pretrained=True)
    model.cuda()
    model.eval()

    extractor = FeatureExtractor2D(
        stride=args.stride,
        mean=config.mean,
        std=config.std,
        resize_to=config.resize_to,
        crop_to=config.crop_to,
        model=model,
        batch_size=args.batch_size)
    
    if not os.path.exists(os.path.dirname(args.out)):
        os.makedirs(os.path.dirname(args.out))
        
    subsets = ["train", "val", "test"]    
    
    videoindex = open("/nfs/zhujinguo/datasets/open_source_dataset/msvd_dataset/txt_labels/youtube_mapping.txt", 'r').readlines()
    name2idx = dict()
    idx2name = dict()
    for v in videoindex:   
        name2idx[v.split()[0]] = v.split()[1] 
        idx2name[v.split()[1]] = v.split()[0] 

    for subset in subsets:
        txtfile = "/nfs/zhujinguo/datasets/open_source_dataset/msvd_dataset/txt_labels/sents_{}_lc_nopunc.txt".format(subset)
        capinfos = open(txtfile, 'r').readlines()
        visited_imames = set()
        for caption in capinfos:
            vidindex = caption.split('\t')[0]
            if vidindex not in visited_imames:
                visited_imames.add(idx2name[vidindex])
        visited_imames = list(visited_imames)
        print("set {} has {} video".format(subset, len(visited_imames)))
        
        process_vidio(args, visited_imames, subset)


    

