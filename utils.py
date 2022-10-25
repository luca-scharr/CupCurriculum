# ANCHOR Libraries
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import time


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


# Function used in my_plotter to plot errorbars
def errorbar(style:str, ls:dict, number:int=5, c:str=None, v:str=None):
    c_v_val = np.full((number, 20), np.inf)
    for i in range(number):
        if (not c is None) and (not v is None):
            with open(f"{os.getcwd()}/{style}/{c+'_'+v}/{i}/dumps/reint_summary_plot_data/best_val.dat", "rb") as input_file:
                c_v_val[i] = np.flip(pickle.load(input_file))
            label = f"{c} {v}"
        else:
            with open(f"{os.getcwd()}/{style}/{i}/dumps/summary_plot_data/best_val.dat", "rb") as input_file:
                c_v_val[i] = pickle.load(input_file)
            label = f"{style}"
    c_v_val_mean = c_v_val.mean(axis=0)
    c_v_val_std  = c_v_val.std(axis=0)
    plt.errorbar(range(20), c_v_val_mean, c_v_val_std, ls=ls.get(v), label=label)
    pass


# Function used in my_plotter to plot scatter plots
def scatter(ax, plot_type, value_b, value_l, marker, marker_style1, marker_style2, name, legend_exists):
    if ("best" in plot_type and "last" in plot_type):
        if value_b == value_l:
            ax.scatter(name, value_b, marker=marker,
                       label="Best Validation Loss at full Capacity" if not legend_exists[0] else "_",
                       **marker_style1)
            legend_exists[0]=True
        else:
            ax.scatter(name, value_b, marker=marker,
                       label="Best Validation Loss with less Capacity" if not legend_exists[1] else "_",
                       **marker_style2)
            legend_exists[1] = True
    elif "best" in plot_type:
        ax.scatter(name, value_b, marker=marker,
                   label="Best Validation Loss" if not legend_exists[0] else "_", **marker_style1)
        legend_exists[0] = True
    else:
        ax.scatter(name, value_b, marker=marker,
                   label="Last Validation Loss" if not legend_exists[0] else "_", **marker_style1)
        legend_exists[0] = True
    pass


# A custom plotter using the data structure defined in LTH_and_language_transformer.py
def my_plotter(plot_type:str ,styles: list, choice:list, variation:list, number:int=5, zoom:bool=False, ls:dict=None, xlim:list=None, ylim:list=None, markers:str=None)-> None:
    """
    Inputs:
        styles: can be subset from ["baseline", "pruning", "reintroduction"]
        choice: iff style == reintroduction then it can be any subset from ["old","rng","top"]
        variation: iff style == reintroduction then it can be any subset from ["freezing","identical","dynamic"]
        number: can be any integer up to 5
        zoom: zooms in on fixed area of the plot
    """
    if plot_type == "mean_var_graph":
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
                        errorbar(style, ls, number, c, v)
            elif style == "pruning":
                errorbar(style, ls, number)
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
    elif "runs_scattered" in plot_type:
        fig,ax = plt.subplots()
        marker_style1 = dict(color='blue', s=50)
        marker_style2 = dict(color='red',  s=50)
        if markers is None:
            markers = ["." for _ in range(number)]
        elif markers == "num":
            markers = [f"${i}$" for i in range(number)]
        else:
            markers = markers if len(markers) == number else print(f"Markers unknown.")
        legend_exists = [False,False]
        for style in styles:
            if style == "reintroduction":
                for c in choice:
                    for v in variation:
                        for i in range(number):
                            with open(f"{os.getcwd()}/{style}/{c+'_'+v}/{i}/dumps/reint_summary_plot_data/best_val.dat", "rb") as input_file:
                                data = pickle.load(input_file)
                            scatter(ax, plot_type, data.min(), data[-1], markers[i],marker_style1, marker_style2, f"{c}_{v}", legend_exists)
            elif style == "pruning":
                for i in range(number):
                    with open(f"{os.getcwd()}/{style}/{i}/dumps/summary_plot_data/best_val.dat", "rb") as input_file:
                        data = pickle.load(input_file)
                    scatter(ax, plot_type, data.min(), data[0], markers[i], marker_style1, marker_style2, f"{style}", legend_exists)
            elif style == "baseline":
                for i in range(number):
                    with open(f"{os.getcwd()}/{style}/{i}/dumps/summary_plot_data/best_val.dat", "rb") as input_file:
                        data = pickle.load(input_file)
                    scatter(ax, plot_type, data.min(), data[0], markers[i], marker_style1, marker_style2, f"{style}", legend_exists)
            else:
                print(f"This style is unknown.")
                pass
        fig.suptitle('Mathtext markers', fontsize=14)
        ax.set_xlabel("Experiment Type")
        ax.set_ylabel("Cross Entropy Loss")
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation="vertical", ha="right", rotation_mode="anchor")
        plt.legend()
        plt.show()
    pass
