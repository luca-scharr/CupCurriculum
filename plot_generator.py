import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch as T
import utils
import argparse
from scipy import stats
import seaborn as sns
import copy

np.random.seed(3)
rng = np.random.default_rng(3)

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=str, default="small", choices=["small","big","large"], help="The Size of the Model")
parser.add_argument("--rewinding", type=str, default="init", choices=["init", "best","dont", "warm"], help="The rewinding variation")
parser.add_argument("--prune_variation", type=str, default="lth", choices=["lth","random"], help="Pruning variation for Experiments")
parser.add_argument("--choice", type=str, default="old", choices=["old", "rng", "top","org"], help="The Choice of Reintroductionscheme")
parser.add_argument("--variation", type=str, default="identical", choices=["dynamic", "freezing", "identical"], help="The Variation of subsequent Trainingscheme")
parser.add_argument("--baseline", type=bool, default=False, help="True runs Baseline, False runs Experiment or Miniature")
parser.add_argument("--miniature", type=bool, default=False, help="True runs Miniature, False runs Experiment")
args = parser.parse_args()


experiment = f"{args.choice}_{args.variation}"

path_baseline = f"{os.getcwd()}/{args.size}_small/baseline"
path_miniature = f"{os.getcwd()}/{args.size}_small/miniature"
path = f"{os.getcwd()}/{args.size}_small/{args.rewinding}/{args.prune_variation}"

translation = {
    "old":"old",
    "org":"original",
    "rng":"random",
    "top":"top",
    "init":"initial",
    "best":"best",
    "dont":"no",
    "warm":"warm",
    "lth":"Lottery Tickets",
    "random":"Random"
}
comp = [100.0, 80.0, 64.0, 51.2, 41.0, 32.8, 26.2, 21.0, 16.8, 13.4, 10.7, 8.6, 6.9, 5.5, 4.4, 3.5, 2.8, 2.3, 1.8, 1.4]

n_trials = 1000

# Function creating a plot comparing the mean training and validation error accross epochs
def errorbar_overfitting(comp: list, path: str, experiment: str, v: str, ch: str) -> None:
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
    t = range(2000)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(t, a_train_l_mean, c = 'blue', label=f"training_loss")
    ax.fill_between(t, a_train_l_mean + a_train_l_std, a_train_l_mean - a_train_l_std, facecolor='blue', alpha=0.5)
    ax.plot(t, a_valid_l_mean, c = 'green', label=f"validation_loss")
    ax.fill_between(t, a_valid_l_mean + a_valid_l_std, a_valid_l_mean - a_valid_l_std, facecolor='green', alpha=0.5)
    fig.suptitle(f"Training and Validation Loss Vs Iterations for\n"
                 f"the {args.size} model with \'{translation[args.rewinding]}\' rewinding,\n"
                 f"{translation[args.prune_variation]} pruning and\n"
                 f" the {translation[ch]} {v} variation of the Cup Curriculum")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cross Entropy Loss")
    ax.legend()
    plt.tight_layout()
    if args.baseline:
        utils.checkdir(f"{os.getcwd()}/plots/{args.size}_small/baseline/")
        fig.savefig(f"{os.getcwd()}/plots/{args.size}_small/baseline/errorbar_overfitting", format="pdf")
    elif args.miniature:
        utils.checkdir(f"{os.getcwd()}/plots/{args.size}_small/miniature/")
        fig.savefig(f"{os.getcwd()}/plots/{args.size}_small/miniature/errorbar_overfitting", format="pdf")
    else:
        utils.checkdir(f"{os.getcwd()}/plots/{args.size}_small/{args.rewinding}/{args.prune_variation}/{experiment}/")
        fig.savefig(
            f"{os.getcwd()}/plots/{args.size}_small/{args.rewinding}/{args.prune_variation}/{experiment}/errorbar_overfitting", format="pdf")
    plt.close()
    pass


# Function creating a plot comparing the mean training and validation error accross epochs for the baseline
def errorbar_overfitting_baseline(path: str, argument: str) -> None:
    a_train_l = np.full((5, 2000), np.inf)
    a_valid_l = np.full((5, 2000), np.inf)
    for i in np.arange(5):
        all_train_loss = np.array([])
        all_val_loss = np.array([])
        for j in range(40):
            with open(f"{path}/{i}/dumps/train_loss/all_train_loss_{j}.dat", "rb") as input_file:
                all_train_loss_temp = pickle.load(input_file)
            all_train_loss = np.concatenate((all_train_loss, all_train_loss_temp))
            with open(f"{path}/{i}/dumps/validation_loss/all_val_loss_{j}.dat", "rb") as input_file:
                all_val_loss_temp = pickle.load(input_file)
            all_val_loss = np.concatenate((all_val_loss, all_val_loss_temp))
        a_train_l[i] = all_train_loss
        a_valid_l[i] = all_val_loss
    a_train_l_mean = a_train_l.mean(axis=0)
    a_train_l_std  = a_train_l.std(axis=0)
    a_valid_l_mean = a_valid_l.mean(axis=0)
    a_valid_l_std  = a_valid_l.std(axis=0)
    t = range(2000)
    txt = f"Vanilla Curriculum" if "baseline" in argument else f"Miniature Model"
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(t, a_train_l_mean, c = 'blue', label=f"training_loss")
    ax.fill_between(t, a_train_l_mean + a_train_l_std, a_train_l_mean - a_train_l_std, facecolor='blue', alpha=0.5)
    ax.plot(t, a_valid_l_mean, c = 'green', label=f"validation_loss")
    ax.fill_between(t, a_valid_l_mean + a_valid_l_std, a_valid_l_mean - a_valid_l_std, facecolor='green', alpha=0.5)
    fig.suptitle(f"Training and Validation Loss Vs Iterations for\n"
                 f"the {txt} using the {args.size} model")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cross Entropy Loss")
    ax.legend()
    plt.tight_layout()
    if "baseline" in argument:
        utils.checkdir(f"{os.getcwd()}/plots/{args.size}_small/baseline/")
        fig.savefig(f"{os.getcwd()}/plots/{args.size}_small/baseline/errorbar_overfitting", format="pdf")
    elif "miniature" in argument:
        utils.checkdir(f"{os.getcwd()}/plots/{args.size}_small/miniature/")
        fig.savefig(f"{os.getcwd()}/plots/{args.size}_small/miniature/errorbar_overfitting", format="pdf")
    plt.close()
    pass


def mean_var(path: str, variations: list, choices: list) -> None:
    a_valid_l = np.full((5, 20), np.inf)
    labels = []
    c_v_val = np.full((5, 20), np.inf)
    for i in range(5):
        with open(f"{path}/pruning/{i}/dumps/summary_plot_data/best_val.dat", "rb") as input_file:
            c_v_val[i] = np.flip(pickle.load(input_file))
    a_valid_l = np.concatenate((a_valid_l[None,:], c_v_val[None,:]))
    labels.append("pruning")
    for v in variations:
        for c in choices:
            c_v_val = np.full((5, 20), np.inf)
            for i in range(5):
                with open(f"{path}/reintroduction/{c+'_'+v}/{i}/dumps/reint_summary_plot_data/best_val.dat", "rb") as input_file:
                    c_v_val[i] = pickle.load(input_file)
            a_valid_l = np.concatenate((a_valid_l, c_v_val[None,:]))
            labels.append(f"{translation[c]} {v}")
    a_valid_l = a_valid_l[1:]
    a_valid_l_mean = np.asarray([a_valid_l[i].mean(axis=0) for i in range(a_valid_l.shape[0])])
    a_valid_l_std = np.asarray([a_valid_l[i].std(axis=0) for i in range(a_valid_l.shape[0])])

    s_val = np.full((5,), np.inf)
    for i in range(5):
        with open(f"{path_baseline}/{i}/dumps/summary_plot_data/best_val.dat", "rb") as input_file:
            s_val[i] = pickle.load(input_file).min()
    s_val_mean = s_val.mean(axis=0)
    s_val_std  = s_val.std(axis=0)

    t = range(20)
    fig = plt.figure()
    ax = fig.add_subplot()
    fig.suptitle(f"Validation Loss Vs Introduction step")
    ax.set_xlabel("Unpruned Weights Percentage")
    ax.set_ylabel("Cross Entropy Loss")
    ax.set_xticks(range(20), np.flip(np.asarray(comp)), rotation="vertical")
    for i,label in enumerate(labels):
        ls = "--" if "freezing" in label else "-"
        ax.plot(t, a_valid_l_mean[i], ls = ls, label=label)
        ax.fill_between(t, a_valid_l_mean[i] + a_valid_l_std[i], a_valid_l_mean[i] - a_valid_l_std[i], alpha=0.4)
    ax.hlines(s_val_mean, xmin=0, xmax=19, color="r", ls="-.", label=f"baseline")
    ax.hlines(s_val_mean + s_val_std, xmin=0, xmax=19, color="r", ls=":")
    ax.hlines(s_val_mean - s_val_std, xmin=0, xmax=19, color="r", ls=":")
    ax.legend()
    plt.tight_layout()
    utils.checkdir(f"{os.getcwd()}/plots/{args.size}_small/{args.rewinding}/{args.prune_variation}/")
    plt.savefig(f"{os.getcwd()}/plots/{args.size}_small/{args.rewinding}/{args.prune_variation}/mean_var", format="pdf")
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


def box_dif(path: str, variations: list, choices: list) -> None:
    seq = np.full((5,1), np.inf)
    s_val = np.full((5,1), np.inf)
    labels = []
    for i in range(5):
        with open(f"{path}/pruning/{i}/dumps/summary_plot_data/best_val.dat", "rb") as input_file:
            s_val[i] = pickle.load(input_file).min()
    for v in variations:
        for c in choices:
            c_v_val = np.full((5,1), np.inf)
            for i in range(5):
                with open(f"{path}/reintroduction/{c + '_' + v}/{i}/dumps/reint_summary_plot_data/best_val.dat", "rb") as input_file:
                    c_v_val[i] = pickle.load(input_file).min()
            dummy = c_v_val - s_val
            seq = np.concatenate((seq,copy.deepcopy(dummy)), axis=1)
            labels.append(f"{translation[c]} {v}")
    seq = seq.T[1:].T
    plt.boxplot(seq, labels=labels, vert=False)
    plt.yticks(range(1,len(labels)+1), labels, rotation="horizontal")
    plt.xlabel("Cross Entropy Loss Difference")
    plt.grid(color="grey", alpha = 0.7)
    plt.title(f"Difference to the Vanilla Curriculum for the {args.size} model\n"
              f"with the \'{translation[args.rewinding]}\' rewinding and {translation[args.prune_variation]} pruning")
    plt.tight_layout()
    utils.checkdir(f"{os.getcwd()}/plots/{args.size}_small/{args.rewinding}/{args.prune_variation}/")
    plt.savefig(f"{os.getcwd()}/plots/{args.size}_small/{args.rewinding}/{args.prune_variation}/box_dif", format="pdf")
    plt.close()
    pass


def box_dif_old_dyn(sizes: list, rewindings: list, prune_variations: list) -> None:
    seq = np.full((5,1), np.inf)
    s_val = np.full((5,1), np.inf)
    labels = []
    for size in sizes:
        args.size = size
        for rewinding in rewindings:
            args.rewinding = rewinding
            if "init" in rewinding:
                for prune_variation in prune_variations:
                    args.prune_variation = prune_variation
                    path = f"{os.getcwd()}/{args.size}_small/{args.rewinding}/{args.prune_variation}"
                    for i in range(5):
                        with open(f"{path}/pruning/{i}/dumps/summary_plot_data/best_val.dat", "rb") as input_file:
                            s_val[i] = pickle.load(input_file).min()
                    c_v_val = np.full((5,1), np.inf)
                    for i in range(5):
                        with open(f"{path}/reintroduction/old_dynamic/{i}/dumps/reint_summary_plot_data/best_val.dat", "rb") as input_file:
                            c_v_val[i] = pickle.load(input_file).min()
                    dummy = c_v_val - s_val
                    seq = np.concatenate((seq,copy.deepcopy(dummy)), axis=1)
                    labels.append(f"{size} {translation[rewinding]} {translation[prune_variation]}")
            else:
                if "small" in size:
                    path = f"{os.getcwd()}/{args.size}_small/{args.rewinding}/lth"
                    for i in range(5):
                        with open(f"{path}/pruning/{i}/dumps/summary_plot_data/best_val.dat", "rb") as input_file:
                            s_val[i] = pickle.load(input_file).min()
                    c_v_val = np.full((5, 1), np.inf)
                    for i in range(5):
                        with open(f"{path}/reintroduction/old_dynamic/{i}/dumps/reint_summary_plot_data/best_val.dat",
                                  "rb") as input_file:
                            c_v_val[i] = pickle.load(input_file).min()
                    dummy = c_v_val - s_val
                    seq = np.concatenate((seq, copy.deepcopy(dummy)), axis=1)
                    labels.append(f"{size} {translation[rewinding]} {translation['lth']}")
    seq = seq.T[1:].T
    plt.boxplot(seq, labels=labels, vert=False)
    plt.yticks(range(1,len(labels)+1), labels, rotation="horizontal")
    plt.xlabel("Cross Entropy Loss Difference")
    plt.grid(color="grey", alpha = 0.7)
    plt.title(f"Difference between the Old Dynamic variation of \n"
              f"the Cup Curriculum and the Vanilla Curriculum")
    plt.tight_layout()
    utils.checkdir(f"{os.getcwd()}/plots/")
    plt.savefig(f"{os.getcwd()}/plots/old_dynamic_box_dif", format="pdf")
    plt.close()
    pass

print(f"Running plot generator for {args.size}_small/{args.rewinding}/{args.prune_variation}")
#for v in ["freezing", "dynamic", "identical"]:
#    for ch in ["rng", "org", "old", "top"]:
#        exp = f"{ch}_{v}"
#        errorbar_overfitting(comp, path, exp, v, ch)
#box_dif_old_dyn(["small","big","large"], ["init", "best","dont", "warm"], ["lth","random"])
#box_dif(path, ["freezing", "dynamic", "identical"], ["rng", "org", "old", "top"])
mean_var(path, ["dynamic", "identical", "freezing"], ["old", "rng", "top","org"])
#errorbar_overfitting_baseline(path_baseline, "baseline")
#errorbar_overfitting_baseline(path_miniature, "miniature")