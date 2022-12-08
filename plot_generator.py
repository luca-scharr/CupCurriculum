import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch as T
import utils
import argparse
from scipy import stats
import seaborn as sns

np.random.seed(3)
rng = np.random.default_rng(3)

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=str, default="small", choices=["small","big","large"], help="The Size of the Model")
parser.add_argument("--rewinding", type=str, default="init", choices=["init", "best","dont"], help="The rewinding variation")
parser.add_argument("--prune_variation", type=str, default="lth", choices=["lth","random"], help="Pruning variation for Experiments")
parser.add_argument("--choice", type=str, default="top", choices=["old", "rng", "top","org"], help="The Choice of Reintroductionscheme")
parser.add_argument("--variation", type=str, default="identical", choices=["dynamic", "freezing", "identical"], help="The Variation of subsequent Trainingscheme")
parser.add_argument("--baseline", type=bool, default=False, help="True runs Baseline, False runs Experiment or Miniature")
parser.add_argument("--miniature", type=bool, default=False, help="True runs Miniature, False runs Experiment")
args = parser.parse_args()


experiment = f"{args.choice}_{args.variation}"
if args.baseline:
    path = f"{os.getcwd()}/{args.size}_small/baseline"
elif args.miniature:
    path = f"{os.getcwd()}/{args.size}_small/miniature"
else:
    path = f"{os.getcwd()}/{args.size}_small/{args.rewinding}/{args.prune_variation}"

translation = {
    "old":"old",
    "org":"original",
    "rng":"random",
    "top":"top",
    "init":"initial",
    "best":"best",
    "dont":"no",
    "lth":"Lottery Tickets",
    "random":"Random"
}
comp = [100.0, 80.0, 64.0, 51.2, 41.0, 32.8, 26.2, 21.0, 16.8, 13.4, 10.7, 8.6, 6.9, 5.5, 4.4, 3.5, 2.8, 2.3, 1.8, 1.4]

n_trials = 1000

# Function creating a plot comparing the mean training and validation error accross epochs
def errorbar_overfitting(comp: list, path: str, experiment: str) -> None:
    a_train_l = np.full((5, 2000), np.inf)
    a_valid_l = np.full((5, 2000), np.inf)
    for i in np.arange(5):
        all_train_loss_prune = np.array([])
        all_train_loss_reint = np.array([])
        for _c in comp:
            with open(f"{path}/pruning/{i}/dumps/train_loss/all_train_loss_{_c}.dat", "rb") as input_file:
                all_train_loss_prune_temp = pickle.load(input_file)
            all_train_loss_prune = np.concatenate((all_train_loss_prune, all_train_loss_prune_temp))
        for c in comp:
            with open(f"{path}/reintroduction/{experiment}/{i}/dumps/train_loss_reint/all_train_loss_{c}.dat",
                      "rb") as input_file:
                all_train_loss_reint_temp = pickle.load(input_file)
            all_train_loss_reint = np.concatenate((all_train_loss_reint_temp, all_train_loss_reint))
        all_val_loss_prune = np.array([])
        all_val_loss_reint = np.array([])
        for _c in comp:
            with open(f"{path}/pruning/{i}/dumps/validation_loss/all_val_loss_{_c}.dat", "rb") as input_file:
                all_val_loss_prune_temp = pickle.load(input_file)
            all_val_loss_prune = np.concatenate((all_val_loss_prune, all_val_loss_prune_temp))
        for c in comp:
            with open(f"{path}/reintroduction/{experiment}/{i}/dumps/validation_loss_reint/all_val_loss_{c}.dat",
                      "rb") as input_file:
                all_val_loss_reint_temp = pickle.load(input_file)
            all_val_loss_reint = np.concatenate((all_val_loss_reint_temp, all_val_loss_reint))
        all_train_loss = np.concatenate((all_train_loss_prune, all_train_loss_reint))
        all_val_loss = np.concatenate((all_val_loss_prune, all_val_loss_reint))
        a_train_l[i] = all_train_loss
        a_valid_l[i] = all_val_loss
    a_train_l_mean = a_train_l.mean(axis=0)
    a_train_l_std  = a_train_l.std(axis=0)
    a_valid_l_mean = a_valid_l.mean(axis=0)
    a_valid_l_std  = a_valid_l.std(axis=0)

    if args.baseline:
        txt = f"Model Size: \t\t {args.size}" \
              f"Baseline"
    elif args.miniature:
        txt = f"Model Size: \t\t {args.size}" \
              f"Miniature"
    else:
        txt = f"Model Size:                         {args.size}\n" \
              f"Rewinding Scheme:                   {translation[args.rewinding]}\n" \
              f"Pruning Scheme:                     {translation[args.prune_variation]}\n" \
              f"Initialization Scheme:              {translation[args.choice]}\n" \
              f"Update Scheme:                      {args.variation}\n"
    print(txt)
    t = range(2000)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(t, a_train_l_mean, c = 'blue', label=f"training_loss")
    ax.fill_between(t, a_train_l_mean + a_train_l_std, a_train_l_mean - a_train_l_std, facecolor='blue', alpha=0.5)
    ax.errorbar(t, a_valid_l_mean, c = 'green', label=f"validation_loss")
    ax.fill_between(t, a_valid_l_mean + a_valid_l_std, a_valid_l_mean - a_valid_l_std, facecolor='green', alpha=0.5)
    fig.suptitle(f"Training and Validation Loss Vs Iterations")
    fig.text(0, 0, txt, ha='left', va='top')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cross Entropy Loss")
    ax.legend()
    if args.baseline:
        utils.checkdir(f"{os.getcwd()}/plots/{args.size}_small/baseline/")
        fig.savefig(f"{os.getcwd()}/plots/{args.size}_small/baseline/errorbar_overfitting", format="pdf")
    elif args.miniature:
        utils.checkdir(f"{os.getcwd()}/plots/{args.size}_small/miniature/")
        fig.savefig(f"{os.getcwd()}/plots/{args.size}_small/miniature/errorbar_overfitting", format="pdf")
    else:
        utils.checkdir(f"{os.getcwd()}/plots/{args.size}_small/{args.rewinding}/{args.prune_variation}/{args.choice}_{args.variation}/")
        fig.savefig(
            f"{os.getcwd()}/plots/{args.size}_small/{args.rewinding}/{args.prune_variation}/{args.choice}_{args.variation}/errorbar_overfitting", format="pdf")
    plt.close()
    pass


def median_approx(path: str, experiment: str, n_trials: int) -> list:
    best_val = np.full((5, 1), np.inf)
    for i in range(5):
        with open(
                f"{path}/reintroduction/{experiment}/{i}/dumps/reint_summary_plot_data/best_val.dat", "rb") as input_file:
            best_val[i] = pickle.load(input_file).min()
        with open(f"{path}/pruning/{i}/dumps/summary_plot_data/best_val.dat", "rb") as input_file:
            best_val[i] -= pickle.load(input_file).min()
    return [stats.bootstrap(best_val.T, np.mean, random_state=rng) for i in range(n_trials)]


def plot_confidence_interval(list_res: list, n_trials: int, version: str, stat: str) -> None:
    d_low  = np.zeros((len(list_res),1))
    d_high = np.zeros((len(list_res),1))
    for i, res in enumerate(list_res):
        ci        = res.confidence_interval
        d_low[i]  = ci[0]
        d_high[i] = ci[1]
    d_low  = np.concatenate((d_low, np.full(d_low.shape, 1)), axis=1)
    d_high = np.concatenate((d_high, np.full(d_high.shape, -1)), axis=1)
    d      = np.concatenate((d_low, d_high), axis=0)
    d      = d[np.argsort(d[:,0])]
    scale  = np.copy(d[:,1])
    d_2    = np.copy(d)
    data   = np.zeros((d.shape[0]*2,d.shape[1]))
    for i, val in enumerate(d[:,1]):
        d[i,1] = scale[:i+1].sum()/n_trials
        d_2[i,1] = scale[:i].sum()/n_trials
        data[2*i] = d_2[i]
        data[2 * i+1] = d[i]
    plt.plot(data[:,0], data[:,1])
    plt.scatter(d_low[:,0], np.zeros_like(d_low[:,1]),c = "red" , marker=',', alpha=.1)
    plt.scatter(d_high[:, 0], np.zeros_like(d_high[:,1]), c="green", marker=',', alpha=.1)
    if args.baseline:
        utils.checkdir(f"{os.getcwd()}/plots/{args.size}_small/baseline/")
        plt.savefig(f"{os.getcwd()}/plots/{args.size}_small/baseline/{version}_confidence_interval_{stat}", format="pdf")
    elif args.miniature:
        utils.checkdir(f"{os.getcwd()}/plots/{args.size}_small/miniature/")
        plt.savefig(f"{os.getcwd()}/plots/{args.size}_small/miniature/{version}_confidence_interval_{stat}", format="pdf")
    else:
        utils.checkdir(f"{os.getcwd()}/plots/{args.size}_small/{args.rewinding}/{args.prune_variation}/{args.choice}_{args.variation}/")
        plt.savefig(
            f"{os.getcwd()}/plots/{args.size}_small/{args.rewinding}/{args.prune_variation}/{args.choice}_{args.variation}/{version}_confidence_interval_{stat}", format="pdf")
    plt.close()
    pass

# Function to compute the Hodges-Lehmann estimator
def hodges_lehmann_estimator(m, n, set_m, set_n):
    dif_mat = np.full((m,n), np.inf)
    for i, x_i in enumerate(set_m):
        for j, y_j in enumerate(set_n):
            dif_mat[i,j] = y_j-x_i
    return np.median(np.sort(np.ravel(dif_mat)))


def wilcoxon_mann_whitney_two_sample_rank_sum_test():

    pass


print(f"Running plot generator for {experiment}")
#errorbar_overfitting(comp, path, experiment)
plot_confidence_interval(median_approx(path, experiment, n_trials), n_trials, "step", "median")
