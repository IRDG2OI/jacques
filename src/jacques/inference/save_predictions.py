import pandas as pd
import os
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
    '''
    Function that displays images and predicted labels returned by the classify_useless_images function for a certain number of images.
    # Input:
    - results_df : a pandas dataframe with 3 columns 'dir', 'image' and 'class'
    - image_nb : the number of images you want to display
    # Output
    - a view of images and their predicted label.
    
    '''
    samples = results_df.sample(n=image_nb)

    for i in range(image_nb):
        img = mpimg.imread(os.path.join(samples.iloc[i]['dir'], samples.iloc[i]['image']))
        plt.imshow(img)
        print(samples.iloc[i]['class'])
        plt.show()
    