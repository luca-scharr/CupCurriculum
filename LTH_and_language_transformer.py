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
device = T.device("cuda" if T.cuda.is_available() else "cpu")

# Use a Parser to specify Hyperparams etc.
parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, default="old_dynamic", help="The Name of the Experiment setting")
parser.add_argument("--seed", type=int, default=0, help="The Seed used for the Run")
# Set Hyperparams for Batches
parser.add_argument("--batch_size", type=int, default=100, help="The Batchsize used for Training")
parser.add_argument("--bptt", type=int, default=35, help="The Length of Backpropagation through Time")
# Set Hyperparams specifying the Model
parser.add_argument("--ntokens", type=int, default=33280, help="The Number of Tokens used by the Model")
parser.add_argument("--emsize", type=int, default=200, help="The Embedding Dimension used by the Model")
parser.add_argument("--d_hid", type=int, default=200, help="The Dimension of the FFN Model used in the Encoder")
parser.add_argument("--nlayers", type=int, default=2, help="The Number of Encoderlayers used in the Encoder")
parser.add_argument("--nhead", type=int, default=2, help="The Number of Heads used in the Multihead-Attention")
parser.add_argument("--dropout", type=float, default=0.2, help="The Dropout Probability used in the Model")
# Set Hyperparams defining the Pruning Procedure
# TODO: Think about adding rewind option and number of warmup steps
# Facebook Paper uses num_prune_cycles = 20 and prune_frac = 0.20 as well as 50,000 updates (overall?)
parser.add_argument("--num_prune_cycles", type=int, default=20, help="The Number of Pruning Cycles")  # 20
parser.add_argument("--num_epochs_prune", type=int, default=50, help="The Number of Epochs per Pruning Cycle")  # 50
parser.add_argument("--prune_frac", type=float, default=0.20, help="The Fraction of remaining Weights to be pruned in each Iteration")
parser.add_argument("--print_freq_prune", type=int, default=1, help="The Printing-Frequency of Train- and Test Loss during Pruning")
parser.add_argument("--test_freq_prune", type=int, default=1, help="The Testing Frequency during Pruning")
# Set Hyperparams defining the Reintroduction Procedure
parser.add_argument("--choice", type=str, default="old", choices=["old", "rng"], help="The Choice of Reintroductionscheme")
parser.add_argument("--variation", type=str, default="dynamic", choices=["dynamic", "freezing", "identical"], help="The Variation of subsequent Trainingscheme")
parser.add_argument("--num_epochs_reint", type=int, default=50, help="The Number of Epochs per Reintroduction")  # 50
parser.add_argument("--print_freq_reint", type=int, default=1, help="The Printing Frequency of Train- and Test Loss durinig Reintroduction")
parser.add_argument("--test_freq_reint", type=int, default=1, help="The Testing Frequency during Reintroduction")
# TODO: Think about adding LR, the Factor used in scheduler, etc.
parser.add_argument("-v", "--verbosity", action="count", default=1)
args = parser.parse_args()


# Generate Batches inside the Datasets. NOT SHUFFLED
train_iter, test_iter, val_iter = WikiText2.iters(batch_size=args.batch_size)  # shape [seq_len, batch_size]
# Finished preparing the Data

# Building the Model to be trained
init_file = Path(f"{os.getcwd()}/pruning/{args.seed}/saves/model_state_dicts/initial_state_dict.pth.tar")
if not init_file.exists():
    model = transformer_modell.TransformerModel(args.ntokens, args.emsize, args.nhead, args.d_hid, args.nlayers,
                                                args.dropout).to(device)

    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{os.getcwd()}/pruning/{args.seed}/saves/model_state_dicts/")
    T.save(model, f"{os.getcwd()}/pruning/{args.seed}/saves/model_state_dicts/initial_state_dict.pth.tar")
else:
    model = T.load(init_file)
    model.eval()
    model.to(device)
    initial_state_dict = copy.deepcopy(model.state_dict())
# Finished defining/ loading the Model


# List of Modelstates
state_dict       = [initial_state_dict]
# reint_state_dict = []

# Specify the objective Function
criterion = nn.CrossEntropyLoss()
lr        = 5.0  # learning rate
# Stated in Successfully applying ... (Aachen) Adafactor gives better results than Adam, they include warmup after reset
optimizer = T.optim.SGD(model.parameters(), lr=lr)
scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=max(10, args.test_freq_prune))
# lr = 5.0 and factor = 0.95 gives a maximum of 333 updates to lr before the update gets smaler than 1e-8
# Finished specifying the objective Function


# Function defining the Warm-Up Procedure used
def warmup(num_warmup: int = 5) -> None:
    # Progressbar
    bar = tqdm(range(num_warmup))
    for epoch in bar:
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
                cur_loss = (total_loss - comp_loss) / log_interval
                comp_loss = total_loss
                print(f'| epoch {epoch:3d} | {batch_num:5d}/{len(train_iter):5d} batches | '
                      f'lr {optimizer.param_groups[0]["lr"]:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                      f'loss {cur_loss:5.2f}')
                start_time = time.time()
    # Copying and Saving State after Warm-Up
    utils.checkdir(f"{os.getcwd()}/{args.experiment}/{args.seed}/saves/model_state_dicts/")
    T.save(model, f"{os.getcwd()}/{args.experiment}/{args.seed}/saves/model_state_dicts/warmup_state_dict.pth.tar")
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
            cur_loss = (total_loss - comp_loss) / log_interval
            comp_loss = total_loss
            print(f'| epoch {epoch:3d} | {batch_num:5d}/{len(train_iter):5d} batches | '
                  f'lr {optimizer.param_groups[0]["lr"]:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f}')
            start_time = time.time()
    return total_loss / len(train_iter)  # Loss per Datapoint, a little smaler


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
            total_loss += batch_size_local * criterion(output, targets.view(output.size(0))).item()
    return total_loss / (args.batch_size * len(val_iter))  # Approximatley the Loss per Datapoint, a little smaler


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
            dif = (abs(w_c) - abs(w_i))                  # Difference by wich pruned Weights are decided
            dif = T.where(m_w != 0.,dif,float("-inf"))   # Make sure that all pruned Weights remain pruned
            b,i = T.sort(dif.view(w_c.numel()))          # i gives the Information needed for pruning, b is irrelevant
            m_w.view(w_c.numel())[i[:round(w_c.numel() * (1-(1-percent)**pruning_cycle))]] = 0  # int better?
            # Apply new weight and mask
            param.data = (w_c * m_w).to(param.device)
            mask[j] = m_w.to(device)
            j += 1


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
            mask[i] = T.ones_like(param.data)  # Complete Mask containing ones
            i = i + 1
    return mask


# Function to apply the Mask to the Network and rewind
def original_initialization(mask_temp:  list, initial_state_dict:  dict) -> None:
    i = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            param.data = (mask_temp[i].to(param.device) * initial_state_dict[name].to(param.device))
            i = i + 1
        else:
            param.data = initial_state_dict[name].to(param.device)
    if args.verbosity >= 2:
        print("rewinding complete")


# Function defining the pruning procedure used
def pruning_procedure(rewind: bool = True, experiment: str = args.experiment) -> None:
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

        model_file = Path(f"{os.getcwd()}/pruning/{args.seed}/saves/model_state_dicts/state_dict_{_ite}.pth.tar")
        if not model_file.exists():
            print(f"\n--- Pruning Level [{_ite}/{args.num_prune_cycles} of {experiment}.{seed}]: ---")
            # Training and Testing cycle
            for iter_ in pbar:
                # Training
                print()
                #mask
                train_loss = train_prune(iter_)
                # Testing
                if iter_ % args.test_freq_prune == 0:
                    val_loss = evaluate()
                    # Save Weights if best (might be unneccessary)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        utils.checkdir(f"{os.getcwd()}/pruning/{args.seed}/saves/best_models/")
                        T.save(model, f"{os.getcwd()}/pruning/{args.seed}/saves/best_models/{_ite}_model.pth.tar")
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
            utils.checkdir(f"{os.getcwd()}/pruning/{args.seed}/saves/model_state_dicts/")
            T.save(model, f"{os.getcwd()}/pruning/{args.seed}/saves/model_state_dicts/state_dict_{_ite}.pth.tar")
            # Masking
            prune_by_percentile(_ite+1, args.prune_frac)
            # Rewind to pruned version of initial state -> optional
            if rewind and _ite != args.num_prune_cycles-1:
                original_initialization(mask, initial_state_dict)
            # Saving Mask
            mask_list.append(copy.deepcopy(mask))
            utils.checkdir(f"{os.getcwd()}/pruning/{args.seed}/dumps/masks/")
            with open(f"{os.getcwd()}/pruning/{args.seed}/dumps/masks/mask_{comp1}.pkl", 'wb') as output_file:
                pickle.dump(mask, output_file)

            # Saving State of Optimizer and Scheduler
            utils.checkdir(f"{os.getcwd()}/pruning/{args.seed}/saves/optimizer/")
            T.save(optimizer.state_dict(), f"{os.getcwd()}/pruning/{args.seed}/saves/optimizer/opt_{_ite}.pt")
            utils.checkdir(f"{os.getcwd()}/pruning/{args.seed}/saves/scheduler/")
            T.save(scheduler.state_dict(), f"{os.getcwd()}/pruning/{args.seed}/saves/scheduler/sched_{_ite}.pt")

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
            utils.checkdir(f"{os.getcwd()}/pruning/{args.seed}/plots/")
            plt.savefig(f"{os.getcwd()}/pruning/{args.seed}/plots/TrainingVsValidationLoss_{comp1}.png", dpi=1200)
            plt.close()
            utils.checkdir(f"{os.getcwd()}/pruning/{args.seed}/saves/best_val_loss/")
            T.save(best_val_loss,f"{os.getcwd()}/pruning/{args.seed}/saves/best_val_loss/prune_cycle{_ite}.pt")

            # Dump Plot values
            utils.checkdir(f"{os.getcwd()}/pruning/{args.seed}/dumps/train_loss/")
            all_train_loss.dump(f"{os.getcwd()}/pruning/{args.seed}/dumps/train_loss/all_train_loss_{comp1}.dat")
            utils.checkdir(f"{os.getcwd()}/pruning/{args.seed}/dumps/validation_loss/")
            all_val_loss.dump(f"{os.getcwd()}/pruning/{args.seed}/dumps/validation_loss/all_val_loss_{comp1}.dat")
        else:
            print(f"Recycling Values from the Run that generated the File: {model_file}")

            # Load the Model to the desired Device
            model = T.load(model_file)
            model.to(device)
            state_dict.append(copy.deepcopy(model.state_dict()))  # Add the State Dictionary to the List of State Dicts

            # Load the Mask
            with open(f"{os.getcwd()}/pruning/{args.seed}/dumps/masks/mask_{comp1}.pkl", 'rb') as input_file:
                mask = pickle.load(input_file)
            mask_list.append(copy.deepcopy(mask))
            # The following completes the Masking and loads the Mask to the desired Device
            if rewind and _ite != args.num_prune_cycles - 1:
                original_initialization(mask, initial_state_dict)
            elif (not rewind) and _ite != args.num_prune_cycles - 1:
                j = 0
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        w_c = param.data.to(device)  # Current Weight
                        m_w = mask[j].to(device)     # Mask for this Weight
                        param.data = (w_c * m_w).to(device)
                        j += 1

            # Load the Optimizer and Scheduler
            optimizer = T.optim.SGD(model.parameters(), lr=1.0)  # LR here does not matter, as it will be replaced through load state dict
            scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=max(10, args.test_freq_prune))
            optimizer.load_state_dict(T.load(f"{os.getcwd()}/pruning/{args.seed}/saves/optimizer/opt_{_ite}.pt"))
            scheduler.load_state_dict(T.load(f"{os.getcwd()}/pruning/{args.seed}/saves/scheduler/sched_{_ite}.pt"))

            # Load the best Validation Loss of the Pruningcycle
            best_val[_ite] = T.load(f"{os.getcwd()}/pruning/{args.seed}/saves/best_val_loss/prune_cycle{_ite}.pt")


        # Resetting variables to 0
        best_val_loss  = np.inf
        all_train_loss = np.zeros(args.num_epochs_prune, float)
        all_val_loss   = np.zeros(args.num_epochs_prune, float)

        # Resetting Schedulers best such that the LR won't get regulated based on previous pruning cycles
        scheduler.best = float("inf")

    # Dumping Values for Plotting
    utils.checkdir(f"{os.getcwd()}/pruning/{args.seed}/dumps/summary_plot_data/")
    comp.dump(f"{os.getcwd()}/pruning/{args.seed}/dumps/summary_plot_data/compression.dat")
    best_val.dump(f"{os.getcwd()}/pruning/{args.seed}/dumps/summary_plot_data/best_val.dat")

    # Plotting
    a = np.arange(args.num_prune_cycles)
    plt.plot(a, best_val, c="blue", label="Winning tickets")
    plt.title(f"Validation Loss Vs Unpruned Weights Percentage (WikiText2, Language Model)")
    plt.xlabel("Unpruned Weights Percentage")
    plt.ylabel("Validation Loss")
    plt.xticks(a, comp, rotation="vertical")
    plt.legend()
    plt.grid(color="gray")
    utils.checkdir(f"{os.getcwd()}/pruning/{args.seed}/plots/")
    plt.savefig(f"{os.getcwd()}/pruning/{args.seed}/plots/ValidationLossVsWeights.png", dpi=1200)
    plt.close()


# Function implementing symmetric difference for the masks
def symmetric_difference() -> list:
    sym_dif_list = []
    for i in range(len(mask_list) - 1):
        a = mask_list[i]
        b = mask_list[i + 1]
        sym_dif = copy.deepcopy(a)
        for j in range(len(a)):
            sym_dif[j] = sym_dif[j].to(bool)
            sym_dif[j][T.eq(a[j], b[j])] = False
        sym_dif_list.append(sym_dif)
    return sym_dif_list


# Function implementing reintroduction schemes
def reintroduction(mask_dif:  list, choice: str = args.choice, model_state: dict = None) -> None:
    """
    Input:
        mask_dif    -> 1 denotes the capacity of the network that shall be reintroduced
        model_state -> previous state of the model which may be considered in the reintroduction scheme
        choice      -> can be anything from {"old";"rng"}, denotes the reintroduction scheme
    Output:
        None

    Reintroduces previously pruned capacity in the network.
    """
    if choice == "old":
        supplement = model_state
    elif choice == "rng":
        supplement = transformer_modell.TransformerModel(args.ntokens, args.emsize, args.nhead, args.d_hid, args.nlayers, args.dropout)
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
                factor = (1./0.95)**(i+1)  # The LR prior to the current LR, up for discussion; 0.95 = factor in Scheduler
                i     += 1                 # Count pos in list; not using enumerate() as this way is more flexible. CHANGE?
                j      = 0                 # Used to match position in mask with layer in the network
                for name, p in model.named_parameters():  # Vektorisieren nachscahuen; Pytorch discussion seite
                    if 'weight' in name:
                        # Change this Part to adjust the learningrate
                        p.grad.data[mask_difference[j]] = p.grad.data[mask_difference[j]]*factor
                        j += 1
            j=0
            for name, p in model.named_parameters():  # Vektorisieren nachscahuen; Pytorch discussion seite
                if 'weight' in name:
                    p.grad.data = p.grad.data * mask_list[-(mask_num+1)][j]
                    j+=1
        elif variation == "freezing":
            j = 0
            for name, p in model.named_parameters():  # Vektorisieren nachscahuen; Pytorch discussion seite
                if 'weight' in name:
                    p.grad.data = p.grad.data * sym_dif_list[-1][j]
                    j += 1
        elif variation == "identical":
            j = 0
            for name, p in model.named_parameters():  # Vektorisieren nachscahuen; Pytorch discussion seite
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
            cur_loss = (total_loss - comp_loss) / log_interval
            comp_loss = total_loss
            print(f'| epoch {epoch:3d} | {batch_num:5d}/{len(train_iter):5d} batches | '
                  f'lr {optimizer.param_groups[0]["lr"]:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f}')
            start_time = time.time()
    return total_loss / len(train_iter)  # Loss per Datapoint, a little smaler


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
    s_d_mask_list = symmetric_difference()
    s_d_mask_list.reverse()
    state_dict.reverse()

    # Compression Rate
    comp = np.zeros(len(s_d_mask_list), float)

    # Initialising storage for performance indicators
    best_val_loss  = np.inf
    best_val       = np.full(len(s_d_mask_list), np.inf)
    all_train_loss = np.zeros(args.num_epochs_reint, float)
    all_val_loss   = np.zeros(args.num_epochs_reint, float)

    for reint_step in range(len(s_d_mask_list)):  # Similar to _ite in pruning_procedure(); Number needed
        # Reintroduce the lifted mask
        reintroduction(s_d_mask_list[reint_step], choice, state_dict[reint_step])

        # The following is similar to pruning_procedure()
        # Progressbar
        pbar = tqdm(range(args.num_epochs_reint))

        # List fraction of remaining Weights per layer
        print()
        comp1            = utils.print_nonzeros(model)
        comp[reint_step] = comp1

        model_file = Path(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/saves/model_reint_state_dicts/reint_state_dict_{reint_step}.pth.tar")
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
                        utils.checkdir(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/saves/best_model_reint/")
                        T.save(model, f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/saves/best_model_reint/{reint_step}_model.pth.tar")
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
            utils.checkdir(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/saves/model_reint_state_dicts/")
            T.save(model, f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/saves/model_reint_state_dicts/reint_state_dict_{reint_step}.pth.tar")

            # Saving State of Optimizer and Scheduler
            utils.checkdir(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/saves/optimizer_reint/")
            T.save(optimizer.state_dict(), f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/saves/optimizer_reint/opt_{reint_step}.pt")
            utils.checkdir(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/saves/scheduler_reint/")
            T.save(scheduler.state_dict(), f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/saves/scheduler_reint/sched_{reint_step}.pt")

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
            utils.checkdir(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/plots/")
            plt.savefig(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/plots/TrainingVsValidationLoss_{comp1}.png", dpi=1200)
            plt.close()
            utils.checkdir(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/saves/best_val_loss_reint/")
            T.save(best_val_loss, f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/saves/best_val_loss_reint/reint_step{reint_step}.pt")

            # Dump Plot values
            utils.checkdir(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/dumps/train_loss_reint/")
            all_train_loss.dump(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/dumps/train_loss_reint/all_train_loss_{comp1}.dat")
            utils.checkdir(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/dumps/validation_loss_reint/")
            all_val_loss.dump(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/dumps/validation_loss_reint/all_val_loss_{comp1}.dat")
        else:
            print(f"Recycling Values from the Run that generated the File: {model_file}")

            # Load the Model to the desired Device
            model = T.load(model_file)
            model.to(device)
            model.train()
            # reint_state_dict.append(copy.deepcopy(model.state_dict()))  # Add the State Dictionary to the List of Reintroduction State Dicts

            # Load the Optimizer and Scheduler
            optimizer = T.optim.SGD(model.parameters(), lr= 1.0)  # LR here does not matter, as it will be replaced through load state dict
            scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=max(10, args.test_freq_prune))
            optimizer.load_state_dict(T.load(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/saves/optimizer_reint/opt_{reint_step}.pt"))
            scheduler.load_state_dict(T.load(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/saves/scheduler_reint/sched_{reint_step}.pt"))

            # Load the best Validation Loss of the Pruningcycle
            best_val[reint_step] = T.load(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/saves/best_val_loss_reint/reint_step{reint_step}.pt")

        # Resetting variables to 0
        best_val_loss  = np.inf
        all_train_loss = np.zeros(args.num_epochs_reint, float)
        all_val_loss   = np.zeros(args.num_epochs_reint, float)

    # Dumping Values for Plotting
    utils.checkdir(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/dumps/reint_summary_plot_data/")
    comp.dump(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/dumps/reint_summary_plot_data/compression.dat")
    best_val.dump(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/dumps/reint_summary_plot_data/best_val.dat")

    # Plotting
    a = np.arange(len(s_d_mask_list))
    plt.plot(a, best_val, c="blue", label="Winning tickets")
    plt.title(f"Validation Loss Vs Unpruned Weights Percentage (WikiText2, Language Model)")
    plt.xlabel("Unpruned Weights Percentage")
    plt.ylabel("Validation Loss")
    plt.xticks(a, comp, rotation="vertical")
    plt.legend()
    plt.grid(color="gray")
    utils.checkdir(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/plots/")
    plt.savefig(f"{os.getcwd()}/reintroduction/{args.experiment}/{args.seed}/plots/ValidationLossVsWeights.png", dpi=1200)
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
def main(rewind: bool = True, experiment: str = args.experiment, choice: str = args.choice, variation: str = args.variation) -> None:
    print(f"Using device: {device}")
    starting_time = time.time()
    # Warm-Up Training? For rewinding as little as one epoch is enough
    # warmup()
    # warmup_state_dict = copy.deepcopy(model.state_dict())
    time_warmup = time.time()
    print(f"Runtime of the warmup {time_warmup-starting_time} [s]")

    # Pruning Procedure
    pruning_procedure(rewind, experiment)  # The Literature finds rewinding improving the performance when rewinded to warmup state
    time_pruning = time.time()
    print(f"Runtime of the pruning procedure {time_pruning-time_warmup} [s]")

    # Reintroduction Procedure
    regaining_procedure(experiment, choice, variation)
    time_reintroduction = time.time()
    print(f"Runtime of the reintroduction procedure {time_reintroduction-time_pruning} [s]")

    # Show and Save Timings
    print(f"Runtime overall {time_reintroduction - starting_time} [s]")
    # times = T.tensor([time_warmup-starting_time, time_pruning-time_warmup,
    #                   time_reintroduction-time_pruning, time_reintroduction-starting_time])
    # utils.checkdir(f"{os.getcwd()}/{args.experiment}/{args.seed}/saves/runtimes/")
    # T.save(times, f'{os.getcwd()}/{args.experiment}/{args.seed}/saves/runtimes/tensor.pt')


if __name__ == "__main__":
    main()
