from io import BytesIO  #Work with binary data in memory
import urllib.request  #Download files from the internet
from zipfile import ZipFile  #Open ZIP archives without extracting them to disk
import os
import torch
import torch.utils.data
from torchvision import datasets, transforms
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt


# Let's see if we have an available GPU
import numpy as np
import random


def setup_env():
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print("GPU available")
    else:
        print("GPU *NOT* available. Will use CPU (slow)")
        
        

    # Seed random generator for repeatibility | we need all these Because each library and device (CPU/GPU) has its own independent random number generator
    #to make everything fully reproducible, you have to seed all sources of randomness
    seed = 42
    random.seed(seed)  #This controls Python’s built-in random module (import random).
    np.random.seed(seed)  #This controls NumPy’s random number generator
    torch.manual_seed(seed)  #This controls PyTorch on the CPU| This ensures that any randomness in PyTorch operations (like weight initialization) is reproducible.
    torch.cuda.manual_seed_all(seed)  #This controls PyTorch on all GPUs  |Needed because CUDA (GPU) has its own random number generator



    # Download data if not present already
    download_and_extract()
    compute_mean_and_std()
    
    
    

    # Make checkpoints subdir if not existing
    os.makedirs("checkpoints", exist_ok=True)   #Creates a folder called checkpoints where trained models can be saved
    
    
    # Make sure we can reach the installed binaries. This is needed for the workspace
    if os.path.exists("/data/DLND/C2/landmark_images"):
        os.environ['PATH'] = f"{os.environ['PATH']}:/root/.local/bin"
        
        
        



def get_data_location():
    """
    Find the location of the dataset, either locally or in the Udacity workspace
    """

    if os.path.exists("landmark_images"):
        data_folder = "landmark_images"
    elif os.path.exists("/data/DLND/C2/landmark_images"):
        data_folder = "/data/DLND/C2/landmark_images"
    else:
        raise IOError("Please download the dataset first")

    return data_folder


def download_and_extract(
    url="https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip",
):
    
    try:
        
        location = get_data_location()
    
    except IOError:
        # Dataset does not exist
        print(f"Downloading and unzipping {url}. This will take a while...")

        with urllib.request.urlopen(url) as resp:  #Downloads ZIP file 

            with ZipFile(BytesIO(resp.read())) as fp: #Loads ZIP into memory using BytesIO

                fp.extractall(".")  #Extracts all files to current folder

        print("done")
                
    else:
        
        print(
            "Dataset already downloaded. If you need to re-download, "
            f"please delete the directory {location}"
        )
        return None


# Compute image normalization
def compute_mean_and_std():
    """
    Compute per-channel mean and std of the dataset (to be used in transforms.Normalize())
    """

    cache_file = "mean_and_std.pt"
    if os.path.exists(cache_file):
        print(f"Reusing cached mean and std")
        d = torch.load(cache_file)

        return d["mean"], d["std"]  # it returns a dictionary with mean and std keys and each with tensors as values 

    folder = get_data_location()
    ds = datasets.ImageFolder(
        folder, transform=transforms.Compose([transforms.ToTensor()])
    )
    # dl = torch.utils.data.DataLoader(
        # ds, batch_size=1, num_workers=multiprocessing.cpu_count()
        
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=0
    )

    mean = 0.0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing mean", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  #Reshape each image to compute mean per channel (RGB) flatten (h, w) (batch_Size, color channels , h*w) 
        mean += images.mean(2).sum(0)  #We're taking the mean across dimension 2 , which is the flattened pixel dimension (h*w)
    mean = mean / len(dl.dataset)

    var = 0.0
    npix = 0  #stands for number of pixels . It's a variable used to keep track of the total number of pixels in all the images in the dataset.
    for images, _ in tqdm(dl, total=len(ds), desc="Computing std", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])  #computes squared differences from the mean, adds them up across batches and pixels
        npix += images.nelement()  #images.nelement() gives the total number of elements (pixels × channels) in the current batch

    std = torch.sqrt(var / (npix / 3))

    # Cache results so we don't need to redo the computation
    torch.save({"mean": mean, "std": std}, cache_file)

    return mean, std


def after_subplot(ax: plt.Axes, group_name: str, x_label: str):
    """Add title xlabel and legend to single chart"""
    ax.set_title(group_name)
    ax.set_xlabel(x_label)
    ax.legend(loc="center right")

    if group_name.lower() == "loss":
        ax.set_ylim([None, 4.5])


def plot_confusion_matrix(pred, truth):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    gt = pd.Series(truth, name='Ground Truth')
    predicted = pd.Series(pred, name='Predicted')

    confusion_matrix = pd.crosstab(gt, predicted)

    fig, sub = plt.subplots(figsize=(14, 12))
    with sns.plotting_context("notebook"):
        idx = (confusion_matrix == 0)
        confusion_matrix[idx] = np.nan
        sns.heatmap(confusion_matrix, annot=True, ax=sub, linewidths=0.5, linecolor='lightgray', cbar=False)
