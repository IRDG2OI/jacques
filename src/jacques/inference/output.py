import pandas as pd
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class SavePredictionsDf():

    def __init__(self, files_dir, dest_path, csv_name):
        self.files_dir = files_dir
        self.dest_path = dest_path
        self.csv_name = csv_name

    def initialize_df(self):
        empty_df = pd.DataFrame(columns = ['dir', 'image', 'class'])
        return empty_df

    def append_df_rows(self, df, file, label):
        row = [self.files_dir, file, label]
        df_length = len(df)
        df.loc[df_length] = row
        return df

    def save_csv_predictions(self, df):
        df.to_csv(self.dest_path + self.csv_name + '.csv', index = False, header = True)


def display_predictions(results_df, image_nb=int):
    '''Function that displays images and predicted labels returned by the classify_useless_images function for a certain number of images.
    
    Args:
        results_df : a pandas dataframe with 3 columns 'dir', 'image' and 'class'
        image_nb : the number of images you want to display
    
    Returns:
    a view of images and their predicted label.
    '''
    samples = results_df.sample(n=image_nb)

    for i in range(image_nb):
        img = mpimg.imread(os.path.join(samples.iloc[i]['dir'], samples.iloc[i]['image']))
        plt.imshow(img)
        print(samples.iloc[i]['class'])
        plt.show()


def move_images(results_df, dest_path, who_moves=['useless', 'useful'], copy_or_cut=['copy', 'cut']):
    '''Function that copy/paste or cut/paste images predicted as useless or useful in the destination path.
    
    Args:
        results_df : a pandas dataframe with 3 columns 'dir', 'image' and 'class'
        dest_path : the path where the images will be moved or pasted
        who_moves : whether to move the useful or useless images in another directory
        copy_or_cut: whether to copy/paste or cut/paste the images
    
    Returns:
        Images are copied or cut into the destination path
    '''
    #check if dest_path exists (create if needed)
    os.makedirs(dest_path, exist_ok=True)

    #select images to move in the result dataframe
    images_to_move = results_df[results_df['class']==who_moves]
    path_images_to_move = [os.path.join(dir, image) for dir, image in zip(images_to_move['dir'], images_to_move['image'])]

    for image_src in tqdm(path_images_to_move):
        if copy_or_cut=='cut':
            os.rename(image_src, os.path.join(dest_path, os.path.basename(image_src)))
        elif copy_or_cut=='copy':
            shutil.copy(image_src, os.path.join(dest_path, os.path.basename(image_src)))
    
    