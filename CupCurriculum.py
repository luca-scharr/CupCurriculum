# Importing Libraries
import os
import sys
import argparse
from typing import Tuple
import copy
import time
import math
import numpy as np
import pickle
from pathlib import Path

# Progressbar
from tqdm import tqdm

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Highlevel from Pytorch
import torch as T
from torch import nn, Tensor
import torch.optim as opt
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Neural Network parts from Pytorch
from torch.nn import TransformerEncoder, TransformerEncoderLayer, init
import torch.nn.functional as F

# Pytorch's Dataset and Dataloader
from torch.utils.data import dataset
from torch.utils.data import DataLoader

# Dataset used
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Custom Libraries
import utils
import transformer_modell

# Plotting Style
sns.set_style('darkgrid')

# Setting the computing device
device = T.device("cuda:3" if T.cuda.is_available() else "cpu")

# Use a Parser to specify Hyperparams etc.
parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, default="old_freezing", help="The Name of the Experiment setting")
parser.add_argument("--seed", type=int, default=4, help="The Seed used for the Run")
parser.add_argument("--size", type=str, default="big", choices=["small","big","large"], help="The Size of the Model")
parser.add_argument("--decoder", type=str, default="small", choices=["small"], help="The Size of the Decoder")
parser.add_argument("--rewinding", type=str, default="init", choices=["init", "best","dont"], help="The rewinding variation")
# Set Hyperparams for Batches
parser.add_argument("--batch_size", type=int, default=100, help="The Batchsize used for Training")
parser.add_argument("--bptt", type=int, default=35, help="The Length of Backpropagation through Time")
# Set Hyperparams specifying the Model
# Small: nlayers = 2, nhead = 2
# Medium: nlayers = 4, nhead = 4
# Large: nlayers = 8, nhead = 8
parser.add_argument("--ntokens", type=int, default=33280, help="The Number of Tokens used by the Model")
parser.add_argument("--emsize", type=int, default=200, help="The Embedding Dimension used by the Model")
parser.add_argument("--d_hid", type=int, default=200, help="The Dimension of the FFN Model used in the Encoder")
parser.add_argument("--nlayers", type=int, default=4, help="The Number of Encoderlayers used in the Encoder")
parser.add_argument("--nhead", type=int, default=4, help="The Number of Heads used in the Multihead-Attention")
parser.add_argument("--dropout", type=float, default=0.2, help="The Dropout Probability used in the Model")
# Set Hyperparams defining the Pruning Procedure
parser.add_argument("--prune_variation", type=str, default="random", choices=["lth","random"], help="Pruning variation for Experiments")
parser.add_argument("--num_prune_cycles", type=int, default=20, help="The Number of Pruning Cycles")  # 20
parser.add_argument("--num_epochs_prune", type=int, default=50, help="The Number of Epochs per Pruning Cycle")  # 50
parser.add_argument("--prune_frac", type=float, default=0.20, help="The Fraction of remaining Weights to be pruned in each Iteration")
parser.add_argument("--print_freq_prune", type=int, default=1, help="The Printing-Frequency of Train- and Test Loss during Pruning")
parser.add_argument("--test_freq_prune", type=int, default=1, help="The Testing Frequency during Pruning")
# Set Hyperparams defining the Reintroduction Procedure
parser.add_argument("--choice", type=str, default="old", choices=["old", "rng", "top","org"], help="The Choice of Reintroductionscheme")
parser.add_argument("--variation", type=str, default="freezing", choices=["dynamic", "freezing", "identical"], help="The Variation of subsequent Trainingscheme")
parser.add_argument("--num_epochs_reint", type=int, default=50, help="The Number of Epochs per Reintroduction")  # 50
parser.add_argument("--print_freq_reint", type=int, default=1, help="The Printing Frequency of Train- and Test Loss durinig Reintroduction")
parser.add_argument("--test_freq_reint", type=int, default=1, help="The Testing Frequency during Reintroduction")
# Possible extension: optimize for lr, lr scheduler, factor used in scheduler, and other updating hyperparameters
parser.add_argument("-v", "--verbosity", action="count", default=1)
parser.add_argument("--baseline", type=bool, default=False, help="True runs Baseline, False runs Experiment or Miniature")
parser.add_argument("--miniature", type=bool, default=True, help="True runs Miniature, False runs Experiment")
args = parser.parse_args()

# Generates the iterators used for training, validation, and testing. Exchange for your dataset
train_iter, test_iter, val_iter = WikiText2.iters(batch_size=args.batch_size)  # shape [seq_len, batch_size]
# Finished preparing the Data

# Organizing file structure
path_mdl_size = f"{os.getcwd()}/{args.size}_{args.decoder}/"
path_prune_vr = f"{args.rewinding}/{args.prune_variation}/"

# Building the Model to be trained
init_file = Path(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/saves/model_state_dicts/initial_state_dict.pth.tar")
if not init_file.exists():
    model = transformer_modell.TransformerModel(args.ntokens, args.emsize, args.nhead, args.d_hid, args.nlayers, args.dropout, args.decoder)
    model.to(device)

    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/saves/model_state_dicts/")
    T.save(model, f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/saves/model_state_dicts/initial_state_dict.pth.tar")
else:
    model = T.load(init_file)
    model.eval()
    model.to(device)
    initial_state_dict = copy.deepcopy(model.state_dict())
# Finished defining/ loading the Model


# List of Modelstates
state_dict  = [initial_state_dict]
best_states = []
# reint_state_dict = []

# Specify the objective Function
criterion = nn.CrossEntropyLoss()  # Gets returned as Loss
lr        = 5.0  # Learning rate
optimizer = opt.SGD(model.parameters(), lr=lr)
scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=max(10, args.test_freq_prune))
# lr = 5.0 and factor = 0.95 gives a minimum of 333 updates to lr before the update gets smaler than 1e-8
# Finished specifying the objective Function


# Function defining the Warm-Up Procedure used
def train_base(epoch: int) -> float:
    total_loss   = 0.
    comp_loss    = 0.  # Used for comparison down below
    log_interval = 200
    start_time   = time.time()
    src_mask     = transformer_modell.generate_square_subsequent_mask(args.bptt).to(device)
    model.train()  # turn on train mode
    for batch_num, batch in enumerate(train_iter):
        optimizer.zero_grad()
        data_pts, targets = batch.text.to(device), batch.target.to(device)
        batch_size_local = data_pts.size(0)
        if batch_size_local != args.bptt:  # only on last batch
            src_mask = src_mask[:batch_size_local, :batch_size_local]
        output = (model(data_pts, src_mask)).view(-1, args.ntokens)
        t_loss = criterion(output, targets.view(output.size(0)))
        t_loss.backward()
        # Clipping Gradients
        T.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += t_loss.item()
        if batch_num % log_interval == 0 and batch_num > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = (total_loss - comp_loss) / (log_interval*args.batch_size)
            comp_loss = total_loss
            print(f'| epoch {epoch:3d} | {batch_num:5d}/{len(train_iter):5d} batches | '
                  f'lr {optimizer.param_groups[0]["lr"]:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f}')
            start_time = time.time()
    return total_loss / (args.batch_size * len(train_iter))


# Funktion executing the baseline and the miniature Experiment (as specified by var)
def comparison(var:str , num_iterations: int = 2000, num_checkpoints: int = 40) -> None:
    global model
    global optimizer
    global scheduler

    # Making Paths easier
    path_comp = f"{var}/{args.seed}"

    iter_per_checkpoint = int(num_iterations / num_checkpoints)
    iter_not_executed   = num_iterations % num_checkpoints

    # Initialising storage for performance indicators
    best_val_loss  = np.inf
    best_val       = np.full(num_checkpoints, np.inf)
    all_train_loss = np.zeros(iter_per_checkpoint, float)
    all_val_loss   = np.zeros(iter_per_checkpoint, float)

    print(f"There are {iter_not_executed} Iterations that will not be executed, as they do not fit in the Checkpoints")

    # Checkpoint
    for _ite in range(num_checkpoints):
        # Progressbar
        pbar = tqdm(range(iter_per_checkpoint))
        print()
        model_file = Path(f"{path_mdl_size}{path_comp}/saves/model_state_dicts/state_dict_{_ite}.pth.tar")
        if not model_file.exists():
            print(f"\n--- Checkpoint number [{_ite}/{num_checkpoints} of {var}.{seed}]: ---")
            # Training and Testing cycle
            for iter_ in pbar:
                # Training
                print()
                if "baseline" in var:
                    train_loss = train_base(iter_)
                elif "miniature" in var:
                    train_loss = train_prune(iter_)
                else:
                    print("That should not happen, please use valid option or teach me.")
                    pass
                # Testing
                if iter_ % args.test_freq_prune == 0:
                    val_loss = evaluate()
                    # Save Weights if best
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        utils.checkdir(f"{path_mdl_size}{path_comp}/saves/best_models/")
                        T.save(model, f"{path_mdl_size}{path_comp}/saves/best_models/{_ite}_model.pth.tar")
                # Save training- and validation Loss
                all_val_loss[iter_]   = val_loss
                all_train_loss[iter_] = train_loss
                # Print training- and validation Loss
                if iter_ % args.print_freq_prune == 0:
                    pbar.set_description(f'Loss: {train_loss:.3f} Validation: {val_loss:.3f}')
                scheduler.step(val_loss)

            best_val[_ite] = best_val_loss

            # Saving Current State
            utils.checkdir(f"{path_mdl_size}{path_comp}/saves/model_state_dicts/")
            T.save(model, f"{path_mdl_size}{path_comp}/saves/model_state_dicts/state_dict_{_ite}.pth.tar")

            # Saving State of Optimizer and Scheduler
            utils.checkdir(f"{path_mdl_size}{path_comp}/saves/optimizer/")
            T.save(optimizer.state_dict(), f"{path_mdl_size}{path_comp}/saves/optimizer/opt_{_ite}.pt")
            utils.checkdir(f"{path_mdl_size}{path_comp}/saves/scheduler/")
            T.save(scheduler.state_dict(), f"{path_mdl_size}{path_comp}/saves/scheduler/sched_{_ite}.pt")

            # Saving relevant Data
            # Plotting training and validation Loss, Iteration Curve
            # NOTE training Loss is computed for every iteration
            # while validation Loss is computed only for every {test_freq_prune} iterations
            # Therefore validation Loss saved is constant during the iterations inbetween.
            plt.plot(np.arange(1, iter_per_checkpoint + 1), all_train_loss, c="blue", label="Training Loss")
            plt.plot(np.arange(1, iter_per_checkpoint + 1), all_val_loss, c="red", label="Validation Loss")
            plt.title(f"Training and Validation Loss Vs Iterations (WikiText2, Language Model)")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(color="gray")
            utils.checkdir(f"{path_mdl_size}{path_comp}/plots/")
            plt.savefig(f"{path_mdl_size}{path_comp}/plots/TrainingVsValidationLoss_{_ite}.png", dpi=1200)
            plt.close()
            utils.checkdir(f"{path_mdl_size}{path_comp}/saves/best_val_loss/")
            T.save(best_val_loss, f"{path_mdl_size}{path_comp}/saves/best_val_loss/checkpoint{_ite}.pt")

            # Dump Plot values
            utils.checkdir(f"{path_mdl_size}{path_comp}/dumps/train_loss/")
            all_train_loss.dump(f"{path_mdl_size}{path_comp}/dumps/train_loss/all_train_loss_{_ite}.dat")
            utils.checkdir(f"{path_mdl_size}{path_comp}/dumps/validation_loss/")
            all_val_loss.dump(f"{path_mdl_size}{path_comp}/dumps/validation_loss/all_val_loss_{_ite}.dat")
        else:
            print(f"Recycling Values from the Run that generated the File: {model_file}")

            # Load the Model to the desired Device
            model = T.load(model_file)
            model.to(device)

            # Load the Optimizer and Scheduler
            optimizer = T.optim.SGD(model.parameters(), lr=1.0)  # LR here does not matter, as it will be replaced through load state dict
            scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=max(10, args.test_freq_prune))
            optimizer.load_state_dict(T.load(f"{path_mdl_size}{path_comp}/saves/optimizer/opt_{_ite}.pt"))
            scheduler.load_state_dict(T.load(f"{path_mdl_size}{path_comp}/saves/scheduler/sched_{_ite}.pt"))

            # Load the best Validation Loss of the Pruningcycle
            best_val[_ite] = T.load(f"{path_mdl_size}{path_comp}/saves/best_val_loss/checkpoint{_ite}.pt")

        # Resetting variables to 0
        best_val_loss  = np.inf
        all_train_loss = np.zeros(iter_per_checkpoint, float)
        all_val_loss   = np.zeros(iter_per_checkpoint, float)

        # Resetting Schedulers best such that the LR won't get regulated based on previous pruning cycles
        scheduler.best = float("inf")

    # Dumping Values for Plotting
    utils.checkdir(f"{path_mdl_size}{path_comp}/dumps/summary_plot_data/")
    best_val.dump(f"{path_mdl_size}{path_comp}/dumps/summary_plot_data/best_val.dat")

    # Plotting
    a = np.arange(num_checkpoints)
    plt.plot(a, best_val, c="blue", label="Best Performance Reached")
    plt.title(f"Validation Loss Vs Checkpoints (WikiText2, Language Model)")
    plt.xlabel("Checkpoint Number")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(color="gray")
    utils.checkdir(f"{path_mdl_size}{path_comp}/plots/")
    plt.savefig(f"{path_mdl_size}{path_comp}/plots/ValidationLossVsCheckpoints.png", dpi=1200)
    plt.close()
    pass


# Function defining the training of the Model during the pruning procedure
def train_prune(epoch: int) -> float:
    total_loss   = 0.
    comp_loss    = 0.    # Used for comparison down below
    log_interval = 200
    start_time   = time.time()
    src_mask     = transformer_modell.generate_square_subsequent_mask(args.bptt).to(device)
    model.train()  # turn on train mode
    for batch_num, batch in enumerate(train_iter):
        optimizer.zero_grad()
        data_pts, targets = batch.text.to(device), batch.target.to(device)
        batch_size_local  = data_pts.size(0)
        if batch_size_local != args.bptt:  # only on last batch
            src_mask = src_mask[:batch_size_local, :batch_size_local]
        output = (model(data_pts, src_mask)).view(-1, args.ntokens)
        t_loss = criterion(output, targets.view(output.size(0)))
        t_loss.backward()
        # Freezing Pruned weights by making their gradients Zero
        j = 0
        for name, p in model.named_parameters():
            if 'weight' in name:
                p.grad.data = p.grad.data * mask[j]
                j += 1
        # Clipping Gradients
        T.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += t_loss.item()
        if batch_num % log_interval == 0 and batch_num > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = (total_loss - comp_loss) / (log_interval*args.batch_size)
            comp_loss = total_loss
            print(f'| epoch {epoch:3d} | {batch_num:5d}/{len(train_iter):5d} batches | '
                  f'lr {optimizer.param_groups[0]["lr"]:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f}')
            start_time = time.time()
    return total_loss / (args.batch_size * len(train_iter))  # Loss per Datapoint, a little smaler


# Function defining the evaluation of the Model
def evaluate() -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = transformer_modell.generate_square_subsequent_mask(args.bptt).to(device)
    with T.no_grad():
        for batch in val_iter:
            data_pts, targets = batch.text.to(device), batch.target.to(device)
            batch_size_local  = data_pts.size(0)
            if batch_size_local != args.bptt:
                src_mask = src_mask[:batch_size_local, :batch_size_local]
            output = (model(data_pts, src_mask)).view(-1, args.ntokens)
            total_loss += criterion(output, targets.view(output.size(0))).item()
    return total_loss / (args.batch_size * len(val_iter))  # Loss per Datapoint, a little smaller


# Function pruning each layer by percentile
def prune_by_percentile(pruning_cycle: int, percent: float) -> None:
    j = 0
    for name, param in model.named_parameters():
        # Using terminology of "Deconstructing Lottery Tickets"
        # Not pruning bias term
        if 'weight' in name:
            w_c = param.data.cpu()                       # Current Weight
            w_i = (initial_state_dict[name]).data.cpu()  # Initial Weight
            m_w = mask[j].cpu()                          # Mask for this Weight
            if "lth" in args.prue_variation:
                dif = (abs(w_c) - abs(w_i))              # Difference by wich pruned Weights are decided
            elif "random" in args.prune_variation:
                dif = T.rand(m_w.size())
            else:
                dif = None
            dif = T.where(m_w != 0.,dif,float("-inf"))   # Make sure that all pruned Weights remain pruned
            b,i = T.sort(dif.view(w_c.numel()))          # i gives the Information needed for pruning, b is irrelevant
            m_w.view(w_c.numel())[i[:round(w_c.numel() * (1-(1-percent)**pruning_cycle))]] = 0  # int better?
            # Apply new weight and mask
            param.data = (w_c * m_w).to(device)
            mask[j] = m_w.to(device)
            j += 1
    pass


# Function to make an empty mask of the same size as the model
def make_mask() -> list:
    i = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            i = i + 1
    mask = [None] * i  # Init first Dimension of Mask
    i = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask[i] = T.ones_like(param.data).to(device)  # Complete Mask containing ones
            i = i + 1
    return mask


# Function to apply the mask to the Network and rewind to the best reached network
def rewinding(mask_temp:  list, state:  dict) -> None:
    i = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            param.data = (mask_temp[i].to(device) * state[name].to(device))
            i = i + 1
        else:
            param.data = state[name].to(device)
    if args.verbosity >= 2:
        print("rewinding complete")


# Function defining the pruning procedure used
def pruning_procedure(experiment: str = args.experiment, ) -> None:
    global model
    global mask
    global optimizer
    global scheduler

    # Compression Rate
    comp = np.zeros(args.num_prune_cycles, float)

    # Initialising storage for performance indicators
    best_val_loss  = np.inf
    best_val       = np.full(args.num_prune_cycles, np.inf)
    all_train_loss = np.zeros(args.num_epochs_prune, float)
    all_val_loss   = np.zeros(args.num_epochs_prune, float)

    # Pruning cycle
    for _ite in range(args.num_prune_cycles):
        # Progressbar
        pbar = tqdm(range(args.num_epochs_prune))

        # List fraction of remaining Weights per layer
        print()
        comp1      = utils.print_nonzeros(model)
        comp[_ite] = comp1

        model_file = Path(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/saves/model_state_dicts/state_dict_{_ite}.pth.tar")
        if not model_file.exists():
            print(f"\n--- Pruning Level [{_ite}/{args.num_prune_cycles} of {experiment}.{seed}]: ---")
            # Training and Testing cycle
            for iter_ in pbar:
                # Training
                print()
                train_loss = train_prune(iter_)
                # Testing
                if iter_ % args.test_freq_prune == 0:
                    val_loss = evaluate()
                    # Save Weights if best
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        utils.checkdir(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/saves/best_models/")
                        T.save(model, f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/saves/best_models/{_ite}_model.pth.tar")
                        best_state_dict = copy.deepcopy(model.state_dict())
                # Save training- and validation Loss
                all_val_loss[iter_]   = val_loss
                all_train_loss[iter_] = train_loss
                # Print training- and validation Loss
                if iter_ % args.print_freq_prune == 0:
                    pbar.set_description(f'Loss: {train_loss:.3f} Validation: {val_loss:.3f}')
                scheduler.step(val_loss)

            best_val[_ite] = best_val_loss

            # Masking procedure
            # Saving Current State
            state_dict.append(copy.deepcopy(model.state_dict()))
            best_states.append(best_state_dict)
            utils.checkdir(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/saves/model_state_dicts/")
            T.save(model, f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/saves/model_state_dicts/state_dict_{_ite}.pth.tar")
            # Masking
            prune_by_percentile(_ite+1, args.prune_frac)
            # Rewind to pruned version of initial state -> optional
            if "init" in args.rewinding and _ite != args.num_prune_cycles-1:
                rewinding(mask, initial_state_dict)
            elif "best" in args.rewinding:
                rewinding(mask, best_states[-1])
            # Saving Mask
            mask_list.append(copy.deepcopy(mask))
            utils.checkdir(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/dumps/masks/")
            with open(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/dumps/masks/mask_{comp1}.pkl", 'wb') as output_file:
                pickle.dump(mask, output_file)

            # Saving State of Optimizer and Scheduler
            utils.checkdir(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/saves/optimizer/")
            T.save(optimizer.state_dict(), f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/saves/optimizer/opt_{_ite}.pt")
            utils.checkdir(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/saves/scheduler/")
            T.save(scheduler.state_dict(), f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/saves/scheduler/sched_{_ite}.pt")

            # Saving relevant Data
            # Plotting training and validation Loss, Iteration Curve
            # NOTE training Loss is computed for every iteration
            # while validation Loss is computed only for every {test_freq_prune} iterations
            # Therefore validation Loss saved is constant during the iterations inbetween.
            plt.plot(np.arange(1, args.num_epochs_prune + 1), all_train_loss, c="blue", label="Training Loss")
            plt.plot(np.arange(1, args.num_epochs_prune + 1), all_val_loss, c="red", label="Validation Loss")
            plt.title(f"Training and Validation Loss Vs Iterations (WikiText2, Language Model)")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(color="gray")
            utils.checkdir(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/plots/")
            plt.savefig(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/plots/TrainingVsValidationLoss_{comp1}.png", dpi=1200)
            plt.close()
            utils.checkdir(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/saves/best_val_loss/")
            T.save(best_val_loss,f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/saves/best_val_loss/prune_cycle{_ite}.pt")

            # Dump Plot values
            utils.checkdir(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/dumps/train_loss/")
            all_train_loss.dump(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/dumps/train_loss/all_train_loss_{comp1}.dat")
            utils.checkdir(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/dumps/validation_loss/")
            all_val_loss.dump(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/dumps/validation_loss/all_val_loss_{comp1}.dat")
        else:
            print(f"Recycling Values from the Run that generated the File: {model_file}")

            # Load the Model to the desired Device
            model = T.load(model_file, map_location=T.device(device))
            state_dict.append(copy.deepcopy(model.state_dict()))  # Add the State Dictionary to the List of State Dicts

            # Load the best Model to the desired Device
            best_model_file = Path(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/saves/best_models/{_ite}_model.pth.tar")
            best_model = T.load(best_model_file, map_location=T.device(device))
            best_states.append(copy.deepcopy(best_model.state_dict()))  # Add the State Dictionary to the List of State Dicts

            # Load the Mask
            with open(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/dumps/masks/mask_{comp1}.pkl", 'rb') as input_file:
                mask_temp = pickle.load(input_file)
            for j in range(len(mask_temp)):
                mask[j] = mask_temp[j].to(device)
            # The following completes the Masking and loads the Mask to the desired Device
            if ("init" in args.rewinding) and _ite != args.num_prune_cycles - 1:
                rewinding(mask, initial_state_dict)
            elif ("best" in args.rewinding) and _ite != args.num_prune_cycles - 1:
                rewinding(mask, best_states[-1])
            elif ("dont" in args.rewinding) and _ite != args.num_prune_cycles - 1:
                j = 0
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        w_c = param.data.to(device)  # Current Weight
                        m_w = mask[j].to(device)     # Mask for this Weight
                        param.data = (w_c * m_w).to(device)
                        j += 1
            mask_list.append(copy.deepcopy(mask))

            # Load the Optimizer and Scheduler
            optimizer = T.optim.SGD(model.parameters(), lr=1.0)  # LR here does not matter, as it will be replaced through load state dict
            scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=max(10, args.test_freq_prune))
            optimizer.load_state_dict(T.load(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/saves/optimizer/opt_{_ite}.pt"))
            scheduler.load_state_dict(T.load(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/saves/scheduler/sched_{_ite}.pt"))

            # Load the best Validation Loss of the Pruningcycle
            best_val[_ite] = T.load(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/saves/best_val_loss/prune_cycle{_ite}.pt")


        # Resetting variables to 0
        best_val_loss  = np.inf
        all_train_loss = np.zeros(args.num_epochs_prune, float)
        all_val_loss   = np.zeros(args.num_epochs_prune, float)

        # Resetting Schedulers best such that the LR won't get regulated based on previous pruning cycles
        scheduler.best = float("inf")

    # Dumping Values for Plotting
    utils.checkdir(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/dumps/summary_plot_data/")
    comp.dump(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/dumps/summary_plot_data/compression.dat")
    best_val.dump(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/dumps/summary_plot_data/best_val.dat")

    # Plotting
    a = np.arange(args.num_prune_cycles)
    plt.plot(a, best_val, c="blue", label="Winning tickets")
    plt.title(f"Validation Loss Vs Unpruned Weights Percentage (WikiText2, Language Model)")
    plt.xlabel("Unpruned Weights Percentage")
    plt.ylabel("Validation Loss")
    plt.xticks(a, comp, rotation="vertical")
    plt.legend()
    plt.grid(color="gray")
    utils.checkdir(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/plots/")
    plt.savefig(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/plots/ValidationLossVsWeights.png", dpi=1200)
    plt.close()


# Function implementing the set difference for the masks
def set_difference() -> list:
    sym_dif_list = []
    for a,b in zip(mask_list[:-1],mask_list[1:]):
        sym_dif = copy.deepcopy(a)
        for j in range(len(a)):
            sym_dif[j] = sym_dif[j].to(bool)
            sym_dif[j][T.eq(a[j], b[j])] = False
        sym_dif_list.append(sym_dif)
    return sym_dif_list


# Function implementing reintroduction schemes
def reintroduction(mask_dif:  list, choice: str = args.choice, model_state: int = -1) -> None:
    """
    Input:
        mask_dif    -> 1 denotes the capacity of the network that shall be reintroduced
        model_state -> previous state of the model which may be considered in the reintroduction scheme
        choice      -> can be anything from {"old";"rng";"top"}, denotes the reintroduction scheme
    Output:
        None

    Reintroduces previously pruned capacity in the network.
    """
    if choice == "old":
        supplement = state_dict[model_state] #  Supplement is last reached State of corresponding sparsity
    elif choice == "rng":
        supplement = transformer_modell.TransformerModel(args.ntokens, args.emsize, args.nhead, args.d_hid, args.nlayers, args.dropout, args.decoder).state_dict()
    elif choice == "top":
        supplement = best_states[model_state]  # Supplement is best reached State of corresponding sparsity
    elif choice == "org":
        supplement = initial_state_dict  # Supplement is the original initialization
    else:
        supplement = None
        print(f"\nI do not know this choice of reintroduction scheme. Please be so kind and teach me.\n")
        pass
    i = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            param.data += (mask_dif[i].to(param.device) * supplement[name].to(param.device))
            i = i + 1


# Function defining the training of the model during the reintroduction procedure
def train_reintro(sym_dif_list: list, epoch: int, mask_num: int, variation: str = args.variation) -> float:
    total_loss   = 0.
    comp_loss    = 0.    # Used for comparison down below
    log_interval = 200
    start_time   = time.time()
    src_mask     = transformer_modell.generate_square_subsequent_mask(args.bptt).to(device)
    model.train()  # turn on train mode
    for batch_num, batch in enumerate(train_iter):
        optimizer.zero_grad()
        data_pts, targets = batch.text.to(device), batch.target.to(device)
        batch_size_local  = data_pts.size(0)
        if batch_size_local != args.bptt:  # only on last batch
            src_mask = src_mask[:batch_size_local, :batch_size_local]
        output = (model(data_pts, src_mask)).view(-1, args.ntokens)
        t_loss = criterion(output, targets.view(output.size(0)))
        t_loss.backward()
        if variation == "dynamic":
            # Manipulating the learningrate according to reintroduction time
            i = 0
            for mask_difference in sym_dif_list:
                factor = (1./0.95)**(i+1)  # The LR for capacity pruned in the prior pruning step
                i     += 1                 # Count pos in list; not using enumerate() as this way is more flexible. 
                j      = 0                 # Used to match position in mask with layer in the network
                for name, p in model.named_parameters():  # Could benefit from vektorization
                    if 'weight' in name:
                        # Change this Part to adjust the learningrate
                        p.grad.data[mask_difference[j]] = p.grad.data[mask_difference[j]]*factor
                        j += 1
            j=0
            for name, p in model.named_parameters():  # Could benefit from vektorization
                if 'weight' in name:
                    p.grad.data = p.grad.data * mask_list[-(mask_num+1)][j]
                    j+=1
        elif variation == "freezing":
            j = 0
            for name, p in model.named_parameters():  # Could benefit from vektorization
                if 'weight' in name:
                    p.grad.data = p.grad.data * sym_dif_list[-1][j]
                    j += 1
        elif variation == "identical":
            j = 0
            for name, p in model.named_parameters():  # Could benefit from vektorization
                if 'weight' in name:
                    p.grad.data = p.grad.data * mask_list[-(mask_num + 1)][j]
                    j += 1
        else:
            print("This Variation does not exist yet - Teach me!")
        # Clipping Gradients
        T.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += t_loss.item()
        if batch_num % log_interval == 0 and batch_num > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = (total_loss - comp_loss) / (log_interval*args.batch_size)
            comp_loss = total_loss
            print(f'| epoch {epoch:3d} | {batch_num:5d}/{len(train_iter):5d} batches | '
                  f'lr {optimizer.param_groups[0]["lr"]:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f}')
            start_time = time.time()
    return total_loss / (args.batch_size * len(train_iter))  # Loss per Datapoint, a little smaler


# Function defining the procedure of regaining lost capacity
def regaining_procedure(experiment: str = args.experiment, choice: str = args.choice, variation: str = args.variation) -> None:
    """
    while possible
        reintroduction procedure
        subsequent training scheme
    """
    global model
    global optimizer
    global scheduler

    # Information needed for Reintroduction
    s_d_mask_list = set_difference()
    s_d_mask_list.reverse()
    state_dict.reverse()
    best_states.reverse()

    # Compression Rate
    comp = np.zeros(len(s_d_mask_list), float)

    # Initialising storage for performance indicators
    best_val_loss  = np.inf
    best_val       = np.full(len(s_d_mask_list), np.inf)
    all_train_loss = np.zeros(args.num_epochs_reint, float)
    all_val_loss   = np.zeros(args.num_epochs_reint, float)

    for reint_step in range(len(s_d_mask_list)):  # Similar to _ite in pruning_procedure(); Number needed
        # Reintroduce the lifted mask
        reintroduction(s_d_mask_list[reint_step], choice, reint_step)

        # The following is similar to pruning_procedure()
        # Progressbar
        pbar = tqdm(range(args.num_epochs_reint))

        # List fraction of remaining Weights per layer
        print()
        comp1            = utils.print_nonzeros(model)
        comp[reint_step] = comp1

        model_file = Path(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/saves/model_reint_state_dicts/reint_state_dict_{reint_step}.pth.tar")
        if not model_file.exists():
            print(f"\n--- Reintroduction Level [{reint_step}/{len(s_d_mask_list)} of {experiment}.{seed}]: ---")
            # Training and Testing cycle
            for __iter in pbar:
                # Training
                print()
                train_loss = train_reintro(s_d_mask_list[:reint_step+1], __iter, reint_step, variation)
                # Testing
                if __iter % args.test_freq_reint == 0:
                    val_loss = evaluate()
                    # Save Weights if best (might be unneccessary)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        utils.checkdir(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/saves/best_model_reint/")
                        T.save(model, f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/saves/best_model_reint/{reint_step}_model.pth.tar")
                # Save training- and validation Loss
                all_val_loss[__iter] = val_loss
                all_train_loss[__iter] = train_loss
                # Print training- and validation Loss
                if __iter % args.print_freq_reint == 0:
                    pbar.set_description(f'Loss: {train_loss:.3f} Validation: {val_loss:.3f}')
                scheduler.step(val_loss)

            best_val[reint_step] = best_val_loss

            # Saving Current State
            # reint_state_dict.append(copy.deepcopy(model.state_dict()))
            utils.checkdir(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/saves/model_reint_state_dicts/")
            T.save(model, f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/saves/model_reint_state_dicts/reint_state_dict_{reint_step}.pth.tar")

            # Saving State of Optimizer and Scheduler
            utils.checkdir(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/saves/optimizer_reint/")
            T.save(optimizer.state_dict(), f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/saves/optimizer_reint/opt_{reint_step}.pt")
            utils.checkdir(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/saves/scheduler_reint/")
            T.save(scheduler.state_dict(), f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/saves/scheduler_reint/sched_{reint_step}.pt")

            # Saving relevant Data
            # Plotting training and validation Loss, Iteration Curve
            # NOTE training Loss is computed for every iteration
            # while validation Loss is computed only for every {test_freq_prune} iterations
            # Therefore validation Loss saved is constant during the iterations inbetween.
            plt.plot(np.arange(1, args.num_epochs_reint + 1), all_train_loss, c="blue", label="Training Loss")
            plt.plot(np.arange(1, args.num_epochs_reint + 1), all_val_loss, c="red", label="Validation Loss")
            plt.title(f"Training and Validation Loss Vs Iterations (WikiText2, Language Model)")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(color="gray")
            utils.checkdir(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/plots/")
            plt.savefig(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/plots/TrainingVsValidationLoss_{comp1}.png", dpi=1200)
            plt.close()
            utils.checkdir(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/saves/best_val_loss_reint/")
            T.save(best_val_loss, f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/saves/best_val_loss_reint/reint_step{reint_step}.pt")

            # Dump Plot values
            utils.checkdir(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/dumps/train_loss_reint/")
            all_train_loss.dump(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/dumps/train_loss_reint/all_train_loss_{comp1}.dat")
            utils.checkdir(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/dumps/validation_loss_reint/")
            all_val_loss.dump(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/dumps/validation_loss_reint/all_val_loss_{comp1}.dat")
        else:
            print(f"Recycling Values from the Run that generated the File: {model_file}")

            # Load the Model to the desired Device
            model = T.load(model_file)
            model.to(device)
            model.train()
            # Load the Optimizer and Scheduler
            optimizer = T.optim.SGD(model.parameters(), lr= 1.0)  # LR here does not matter, as it will be replaced through load state dict
            scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=max(10, args.test_freq_prune))
            optimizer.load_state_dict(T.load(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/saves/optimizer_reint/opt_{reint_step}.pt"))
            scheduler.load_state_dict(T.load(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/saves/scheduler_reint/sched_{reint_step}.pt"))

            # Load the best Validation Loss of the Pruningcycle
            best_val[reint_step] = T.load(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/saves/best_val_loss_reint/reint_step{reint_step}.pt")

        # Resetting variables to 0
        best_val_loss  = np.inf
        all_train_loss = np.zeros(args.num_epochs_reint, float)
        all_val_loss   = np.zeros(args.num_epochs_reint, float)

    # Dumping Values for Plotting
    utils.checkdir(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/dumps/reint_summary_plot_data/")
    comp.dump(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/dumps/reint_summary_plot_data/compression.dat")
    best_val.dump(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/dumps/reint_summary_plot_data/best_val.dat")

    # Plotting
    a = np.arange(len(s_d_mask_list))
    plt.plot(a, best_val, c="blue", label="Winning tickets")
    plt.title(f"Validation Loss Vs Unpruned Weights Percentage (WikiText2, Language Model)")
    plt.xlabel("Unpruned Weights Percentage")
    plt.ylabel("Validation Loss")
    plt.xticks(a, comp, rotation="vertical")
    plt.legend()
    plt.grid(color="gray")
    utils.checkdir(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/plots/")
    plt.savefig(f"{path_mdl_size}{path_prune_vr}reintroduction/{args.experiment}/{args.seed}/plots/ValidationLossVsWeights.png", dpi=1200)
    plt.close()


# Making Initial Mask
mask = make_mask()

# List of all masks generated during the algorithm
mask_list = [(copy.deepcopy(mask))]

# Set Seed
seed = args.seed
T.manual_seed(seed)
np.random.seed(seed)


# Main
def main(experiment: str = args.experiment, choice: str = args.choice, variation: str = args.variation) -> None:
    if args.baseline:
        comparison("baseline")
    elif args.miniature:
        mask_file = Path(f"{path_mdl_size}{path_prune_vr}pruning/{args.seed}/dumps/masks/mask_1.4.pkl")
        if mask_file.exists():
            with open(mask_file, 'rb') as input_file:
                mask_temp = pickle.load(input_file)
            for j in range(len(mask_temp)):
                mask[j] = mask_temp[j].to(device)
            comparison("miniature")
        else:
            print("Miniature Experiment failed, it looks like mask_file is missing. Please help.")
    else:
        print(f"Using device: {device}")
        starting_time = time.perf_counter()
        # Pruning Procedure
        pruning_procedure(experiment)
        time_pruning = time.perf_counter()
        print(f"Runtime of the pruning procedure {time_pruning-starting_time} [s]")

        # Reintroduction Procedure
        regaining_procedure(experiment, choice, variation)
        time_reintroduction = time.perf_counter()
        print(f"Runtime of the reintroduction procedure {time_reintroduction-time_pruning} [s]")

        # Show Timings
        print(f"Runtime overall {time_reintroduction - starting_time} [s]")
    pass


if __name__ == "__main__":
    main()
