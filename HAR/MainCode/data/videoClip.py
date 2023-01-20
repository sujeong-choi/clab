import pandas as pd
import argparse
import os
import shutil
import subprocess
import pytube
from pytube.exceptions import VideoUnavailable
from joblib import delayed, Parallel
import time


REQUIRED_COLUMNS = ['label', 'youtube_id', 'time_start', 'time_end', 'split', 'is_cc']
TRIM_FORMAT = '%06d'
URL_BASE = 'https://www.youtube.com/watch?v='

VIDEO_EXTENSION = '.mp4'
VIDEO_FORMAT = 'mp4'
TOTAL_VIDEOS = 0


def create_file_structure(path, folders_names):
    """
    Creates folders in specified path.
    :return: dict
        Mapping from label to absolute path folder, with videos of this label
    """
    mapping = {}
    if not os.path.exists(path):
        os.mkdir(path)
    for name in folders_names:
        dir_ = os.path.join(path, name)
        if not os.path.exists(dir_):
            os.mkdir(dir_)
        mapping[name] = dir_
    return mapping

def do_trim(time_start, time_end,output_path, filename, label_to_dir, label ):

    input_filename = os.path.join(output_path, filename + VIDEO_EXTENSION)
    output_filename = os.path.join(label_to_dir[label],
                                    filename + '_{}_{}'.format(time_start, time_end) + VIDEO_EXTENSION)
    time_start = int(time_start)
    time_end = int(time_end)
    start = "{0:02d}".format(int(time_start/3600)) + ":" + "{0:02d}".format(int((time_start%3600)/60), '02') + ":" +  "{0:02d}".format(time_start%60, '02')
    end = str(time_end - time_start)

    if os.path.exists(output_filename):
        pass
        # print('Already trimmed: ', filename)
    else:
        print('Start trimming: ', filename)
        # Construct command to trim the videos (ffmpeg required).
        # command = 'ffmpeg -i "{input_filename}" ' \
        #             '-ss {time_start} ' \
        #             '-t {time_end} ' \
        #             '-vcodec copy -acodec copy -c copy ' \
        #             '"{output_filename}"'.format(
        #                 input_filename=input_filename,
        #                 time_start=start,
        #                 time_end=end,
        #                 output_filename=output_filename
        #             )
        command = 'ffmpeg -i "{input_filename}" ' \
                    '-ss {time_start} ' \
                    '-to {time_end} ' \
                    '-vf copy -c:v libx264 -preset:v veryfast -crf 22 -movflags +faststart ' \
                    '"{output_filename}"'.format(
                        input_filename=input_filename,
                        time_start=time_start,
                        time_end=time_end,
                        output_filename=output_filename
                    )
        print(command)
        try:
            subprocess.check_output(command, shell=True,stderr=subprocess.STDOUT)
            # subprocess.run(command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print('Error while trimming: ', filename)
            return False
        print('Finish trimming: ', filename)
            
def download_clip(row, label_to_dir, trim, count):
    """
    Download clip from youtube.
    row: dict-like objects with keys: ['label', 'youtube_id', 'time_start', 'time_end']
    'time_start' and 'time_end' matter if trim is True
    trim: bool, trim video to action ot not
    """

    label = row['label']
    filename = row['youtube_id']
    time_start = row['time_start']
    time_end = row['time_end']
     
    # if trim, save full video to tmp folder
    output_path = label_to_dir['tmp'] if trim else label_to_dir[label]
    # 예) label_to_dir.values() = dict_values(['videos/brush painting', 'videos/tmp'])

    # don't download if already exists
    if not os.path.exists(os.path.join(output_path, filename + VIDEO_EXTENSION)):
        print('Start downloading: ', filename)
        try:
            video = pytube.YouTube(URL_BASE + filename).\
                streams.filter(subtype=VIDEO_FORMAT).first().\
                download(output_path, filename+ VIDEO_EXTENSION)
            print('Finish downloading: ', filename)
        except VideoUnavailable:
            pass
            #print('videounavailable')
#            return
#         uncomment, if you want to skip any error:
#
#         except:
#             print('Don\'t know why something went wrong(')
#             return
    else:
        pass
        # print('Already downloaded: ', filename)

    if trim:
        # do_trim(time_start, time_end,output_path, filename, label_to_dir, label) 
        # Take video from tmp folder and put trimmed to final destination folder
        # better write full path to video
        if time_end-time_start >2:
            k=2
            if time_end-time_start > 61 : k = 7

            for i in range(0,time_end-time_start,k):
                do_trim(time_start+i, time_start+2+i,output_path, filename, label_to_dir, label)
        else:        do_trim(time_start, time_end,output_path, filename, label_to_dir, label) 
 

    # print('Processed %i out of %i' % (count + 1, TOTAL_VIDEOS))


def main(input_csv, output_dir, trim, num_jobs):
    global TOTAL_VIDEOS
    start_time  = time.time()
    assert input_csv[-4:] == '.csv', 'Provided input is not a .csv file'
    links_df = pd.read_csv(input_csv)
    # print(links_df)
    # label 필터링
    # spray painting (948), brush painting(950-840), drawing(769) 
    #links_df = links_df.loc[(links_df['label']==('spray painting' or 'drawing'))
    # easal painting, speaking painting, speaking
    links_df = links_df.loc[(links_df['label']==('easal painting'))]
    assert all(elem in REQUIRED_COLUMNS for elem in links_df.columns.values),\
        'Input csv doesn\'t contain required columns.'

    # Creates folders where videos will be saved later
    # Also create 'tmp' directory for temporary files
    folders_names = links_df['label'].unique().tolist() + ['tmp']
    label_to_dir = create_file_structure(path=output_dir,
                                         folders_names=folders_names)

    TOTAL_VIDEOS = links_df.shape[0]
    # Download files by links from dataframe
    Parallel(n_jobs=num_jobs)(delayed(download_clip)(row, label_to_dir, trim, count) for count, row in links_df.iterrows())

    # Clean tmp directory
    # shutil.rmtree(label_to_dir['tmp'])
    print("소요시간:" + str(time.time()-start_time))

if __name__ == '__main__':
    description = 'Script for downloading and trimming videos from Kinetics dataset.' \
                  'Supports Kinetics-400 as well as Kinetics-600.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('input_csv', type=str,
                   help=('Path to csv file, containing links to youtube videos.\n'
                         'Should contain following columns:\n'
                         'label, youtube_id, time_start, time_end, split, is_cc'))
    p.add_argument('output_dir', type=str,
                   help='Output directory where videos will be saved.\n'
                        'It will be created if doesn\'t exist')
    p.add_argument('--trim', action='store_true', dest='trim', default=False,
                   help='If specified, trims downloaded video, using values, provided in input_csv.\n'
                        'Requires "ffmpeg" installed and added to environment PATH')
    p.add_argument('--num-jobs', type=int, default=1,
                   help='Number of parallel processes for downloading and trimming.')
    main(**vars(p.parse_args()))
    
# 실행 코드 : python download.py kinetics700/validate.csv videos/validate/ --trim 
# 실행 코드 : python download.py kinetics700/train.csv videos/train/ --trim 
# 실행 코드 : python videoClip.py kinetics700/sujeong.csv videos/ --trim 