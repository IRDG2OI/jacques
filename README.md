<div align="center">

<img src="jacques.png" height="150px">

# Jacques : a cleaner for your Seatizen sessions

</div>

Jacques is a python package to detect useless images within a directory using artificial intelligence. The model that detects useless images has been trained on photos acquired with the Seatizen protocol.
Have a look to the seatizen acquisition protocol here : 
<div align="center">    

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7129736.svg)](https://doi.org/10.5281/zenodo.7129736)

</div>

## Installation
### Prerequisites

To install jacques in your working environnement, you should already have installed a pytorch (and torchvision) version adapted to your machine resources. By following the link below, you can find the optimal configuration to install pytorch in your working environnement. Make sure to install the last version available.
> https://pytorch.org/

### Jacques installation

Jacques can be installed by executing the following lines in your terminal:

```
git clone https://gitlab.ifremer.fr/jd326f6/jacques.git
cd jacques
pip install .
```

:bulb: If you are working in an environnement, don't forget to `pip install ipykernel` to make your environnement visible in your favourite IDE.

**Datarmor user** : jacques is already installed for you in the jacques_env environnement. You firstly need to append a line at the end of a conda text file as follow:

```
echo '/home/datawork-iot-nos/Seatizen/conda-env/jacques_cpu' >> ~/.conda/environments.txt
```

Just connect to the [jupyterlab IDE](datarmor-jupyterhub.ifremer.fr/) and select jacques_env environnement to execute your notebooks. If you don't see jacques_env, you might not be part of the Seatizen group and should ask access to one of the members.
For your first use of Jacques you will need to download a resnet manually from the terminal (only for datarmor users):
```
wget https://download.pytorch.org/models/resnet50-11ad3fa6.pth
mv resnet50-11ad3fa6.pth ~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth
```


## Quickstart
:man_student: 
All the tutorials (notebooks) are available here : [Jacques examples](https://github.com/6tronl/jacques-examples)

The checkpoint to load the classification model is available here : 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7822928.svg)](https://doi.org/10.5281/zenodo.7822928)



### Classify images in one directory
:man_student: 
[Tuto : classify one directory ](https://github.com/6tronl/jacques-examples/single_dir_classification.ipynb)

To classify a folder of images, you can execute the script below in a python script or a notebook:

```py
from jacques.inference import predictor
results = predictor.classify_useless_images(folder_path='/path/to/your/image/folder', ckpt_path='/path/to/your/checkpoint/')
```

Jacques will automatically selects the files that are images in your folder and predict the utility of the image thanks to deep learning. It will return a pandas dataframe with 3 columns : directory, name and label (useless or useful). Here is an example of the results provided by  `classify_useless_images()` : 

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dir</th>
      <th>image</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>/path/to/your/image/folder/</td>
      <td>session_2018_03_04_kite_Pointe_Esny_G0032421.JPG</td>
      <td>useful</td>
    </tr>
    <tr>
      <th>1</th>
      <td>/path/to/your/image/folder/</td>
      <td>session_2018_03_12_kite_Le_Morne_G0029296.JPG</td>
      <td>useful</td>
    </tr>
    <tr>
      <th>2</th>
      <td>/path/to/your/image/folder/</td>
      <td>session_2022_10_20_aldabra_plancha_body_v1A_00_1_399.jpeg</td>
      <td>useless</td>
    </tr>
    <tr>
      <th>3</th>
      <td>/path/to/your/image/folder/</td>
      <td>session_2021_01_13_Hermitage_AllRounder_image_001196.jpg</td>
      <td>useful</td>
    </tr>
    <tr>
      <th>4</th>
      <td>/path/to/your/image/folder/</td>
      <td>session_2019_09_20_kite_Le_Morne_avec_Manu_G0070048.JPG</td>
      <td>useful</td>
    </tr>
  </tbody>
</table>


### Classify images in several directories
:man_student: 
[Tuto : classify multiple directories ](https://github.com/6tronl/jacques-examples/multiple_dir_classification.ipynb)

To classify images contained in several directories just make a list containg the paths to your directories and execute the following codes:

```py
from jacques.inference import predictor
import pandas as pd

list_of_dir = ['path/to/dir/1/', 'path/to/dir/2/', 'path/to/dir/3/']

results_of_all_dir = pd.DataFrame(columns = ['dir', 'image', 'class'])

for directory in list_of_dir:
    results = predictor.classify_useless_images(folder_path=directory, ckpt_path='/checkpoint/path')
    results_of_all_dir = pd.concat([results_of_all_dir, results], axis=0, ignore_index=True)
```
### Classify multiple Seatizen sessions all at once
:man_student: 
[Tuto : classify multiple Seatizen sessions ](https://github.com/6tronl/jacques-examples/arbo_dir_classification.ipynb)

For Seatizen sessions that follows the famous and unique directory tree (written below), you can directly classify images of these sessions.

```
Seatizen tree (accepted in 02/2023) : 
session_YYYY_MM_DD_location_device_nb
│
└───DCIM
│   │
│   └───IMAGES
│   └───VIDEOS
│   
└───GPS
└───SENSORS
└───METADATA
└───PROCESSED_DATA
│   │
│   └───BATHY
│   └───FRAMES
│       │   session_YYYY_MM_DD_location_device_nb1.jpeg
│       │   session_YYYY_MM_DD_location_device_nb2.jpeg
│       │   ...
│   └───IA
│   └───PHOTOGRAMMETRY
│
session_YYYY_MM_DD_location_device_nb
│ ...
```
Use the following code lines to classify a Seatizen tree:

```py
from jacques.inference import predictor
import os
import pandas as pd

list_of_sessions = ['/path/to/session_YYYY_MM_DD_location_device_nb', '/path/to/session_YYYY_MM_DD_location_device_nb']

results_of_all_sessions = pd.DataFrame(columns = ['dir', 'image', 'class'])
for session in list_of_sessions:
    results = predictor.classify_useless_images(folder_path=os.path.join(session, '/PROCESSED_DATA/FRAMES/'), ckpt_path='/checkpoint/path')
    results_of_all_sessions = pd.concat([results_of_all_sessions, results], axis=0, ignore_index=True)
```

### Move images
Once the classification of useless and useful images has been done, you can choose to copy or paste images in another directory. If working with the seatizen tree, follow the method in the tuto provided to move images keeping the name of the session as a subdirectory.


```py
from jacques.inference import output

output.move_images(results,
           dest_path = '/destination/path/directory/to/move/images/',
           who_moves = 'useless',
           copy_or_cut = 'cut'
           )
```

### Display results (optionally)
Optionally, you can display some or all of the results using  `display_results()` and the results dataframe returned by `classify_useless_images()`.

```py
from jacques.inference import output

output.display_predictions(results, image_nb=5)
```

### Export results (optionally)
Results are returned as a pandas dataframe. For instance, if you want to export the results in a csv format just add :

```py
results.to_csv('path/to/export/results.csv', index = False, header = True)
```

# Have Fun!


















