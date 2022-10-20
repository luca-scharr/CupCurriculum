# ANCHOR Libraries
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


# Print table of zeros and non-zeros count
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


# Checks if the directory exist. If not, creates a new directory
def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# A custom plotter using the data structure defined in LTH_and_language_transformer.py
def my_plotter(styles: list, choice:list, variation:list, number:int=5, zoom:bool=False, ls=None, xlim=None, ylim=None)-> None:
    """
    Inputs:
        styles: can be subset from ["baseline", "pruning", "reintroduction"]
        choice: iff style == reintroduction then it can be any subset from ["old","rng","top"]
        variation: iff style == reintroduction then it can be any subset from ["freezing","identical","dynamic"]
        number: can be any integer up to 5
        zoom: zooms in on fixed area of the plot
    """
    if ylim is None:
        ylim = [0.050, 0.055]
    if xlim is None:
        xlim = [-1, 7]
    if ls is None:
        ls = {"freezing": "--", "dynamic": "-", "identical": "-", "pruning": "-", "baseline1": "-.", "baseline2": ":"}
    for style in styles:
        if style == "reintroduction":
            for c in choice:
                for v in variation:
                    c_v_val = np.full((number,20),np.inf)
                    for i in range(number):
                        with open(f"{os.getcwd()}/{style}/{c+'_'+v}/{i}/dumps/reint_summary_plot_data/best_val.dat", "rb") as input_file:
                            c_v_val[i]   = np.flip(pickle.load(input_file))
                    c_v_val_mean = c_v_val.mean(axis=0)
                    c_v_val_std  = c_v_val.std(axis=0)
                    plt.errorbar(range(20), c_v_val_mean,c_v_val_std,ls=ls.get(v), label=f"{c} {v}")
        elif style == "pruning":
            s_val = np.full((number,20),np.inf)
            for i in range(number):
                with open(f"{os.getcwd()}/{style}/{i}/dumps/summary_plot_data/best_val.dat", "rb") as input_file:
                    s_val[i]   = pickle.load(input_file)
            s_val_mean = s_val.mean(axis=0)
            s_val_std  = s_val.std(axis=0)
            plt.errorbar(range(20), s_val_mean, s_val_std,ls=ls.get(style), label=f"{style}")
        elif style == "baseline":
            s_val = np.full((number,),np.inf)
            for i in range(number):
                with open(f"{os.getcwd()}/{style}/{i}/dumps/summary_plot_data/best_val.dat", "rb") as input_file:
                    s_val[i]   = pickle.load(input_file).min()
            s_val_mean = s_val.mean(axis=0)
            s_val_std  = s_val.std(axis=0)
            plt.hlines(s_val_mean,xmin=0, xmax=19,color="r",ls=ls.get(f"{style}1"), label=f"baseline")
            plt.hlines(s_val_mean+s_val_std,xmin=0, xmax=19,color="r",ls=ls.get(f"{style}2"))
            plt.hlines(s_val_mean-s_val_std,xmin=0, xmax=19,color="r",ls=ls.get(f"{style}2"))
        else:
            print(f"This style is unknown.")
            pass
    with open(f"{os.getcwd()}/pruning/0/dumps/summary_plot_data/compression.dat", "rb") as input_file:
        comp = pickle.load(input_file)
    plt.xlabel("Unpruned Weights Percentage")
    plt.ylabel("Cross Entropy Loss")
    plt.xticks(range(20), comp, rotation="vertical")
    if zoom:
        plt.ylim(ylim[0], ylim[1])
        plt.xlim(xlim[0], xlim[1])
    plt.legend()
    #plt.grid(color="gray")
    plt.show()
    pass
