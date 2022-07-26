# ANCHOR Libraries
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt


# ANCHOR Print table of zeros and non-zeros count
def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        if 'weight' in name:
            nonzero += nz_count
            total += total_params
        print(f'nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%)\t| total zeros = {total_params - nz_count :7}\t| shape = {tensor.shape}\t| {name:20} ')
    print(f'weights alive: {nonzero}, weights pruned : {total - nonzero}, total weights: {total}, Compression rate : {total / nonzero:10.2f}x  ({100 * (total - nonzero) / total:6.2f}% pruned)')
    return round((nonzero / total) * 100, 1)


# ANCHOR Checks of the directory exist and if not, creates a new directory
def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
