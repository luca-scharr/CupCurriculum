# Importing Libraries
import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
import seaborn as sns
import torch.nn.init as init
import pickle
import math
from typing import Tuple

from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import copy
import time

# Custom Libraries
import utils

# Tensorboard initialization
writer = SummaryWriter()

# Plotting Style
sns.set_style('darkgrid')

# Setting the computing device
device = T.device("cuda" if T.cuda.is_available() else "cpu")


# Preparing the Data for usage
# Function preprocessing the Data
def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [T.tensor(vocab(tokenizer(item)), dtype=T.long) for item in raw_text_iter]
    return T.cat(tuple(filter(lambda t: t.numel() > 0, data)))


# Function preparing the generation of Batches
def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.
    Args:
        data: Tensor, shape [N]
        bsz: int, batch size
    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data    = data[:seq_len * bsz]
    data    = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


# Function returning Batches
def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int
    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data    = source[i:i + seq_len]
    target  = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target


# Build the vocabulary
train_iter = WikiText2(split='train')
tokenizer  = get_tokenizer('basic_english')
vocab      = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Create Datasets
train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data   = data_process(val_iter)
test_data  = data_process(test_iter)

# Set Hyperparams for Batches
batch_size      = 20
eval_batch_size = 10
bptt            = 35

# Generate Batches inside the Datasets. NOT SHUFFLED
train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
val_data   = batchify(val_data, eval_batch_size)
test_data  = batchify(test_data, eval_batch_size)
# Finished preparing the Data


# Building the Model to be trained
# Defining the Architecture
class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type          = 'Transformer'
        self.pos_encoder         = PositionalEncoding(d_model, dropout)
        encoder_layers           = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder             = nn.Embedding(ntoken, d_model)
        self.d_model             = d_model
        self.decoder             = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src    = self.encoder(src) * math.sqrt(self.d_model)  # Wordembeddings
        src    = self.pos_encoder(src)  # Positional Encoding
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return T.triu(T.ones(sz, sz) * float('-inf'), diagonal=1)


# Implementing Positional Encoding, i.e. where are the words in the Sentence
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        position       = T.arange(max_len).unsqueeze(1)
        div_term       = T.exp(T.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe             = T.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = T.sin(position * div_term)
        pe[:, 0, 1::2] = T.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


ntokens = len(vocab)  # size of vocabulary
emsize  = 200         # embedding dimension
d_hid   = 200         # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2           # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead   = 2           # number of heads in nn.MultiheadAttention
dropout = 0.2         # dropout probability
model   = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
# Finished defining the Model

# Copying and Saving Initial State
initial_state_dict = copy.deepcopy(model.state_dict())
utils.checkdir(f"{os.getcwd()}/saves/model_state_dicts/")
T.save(model, f"{os.getcwd()}/saves/model_state_dicts/initial_state_dict.pth.tar")

# List of Modelstates
state_dict       = [initial_state_dict]
reint_state_dict = []

# Specify Hyperparams of the Pruning
num_epochs_prune = 50     # Number of Epochs per pruning, 50 seems reasonable
num_prune_cycles = 28     # Number of Pruning Cycles, 28 is used in LTH paper
prune_percent    = 19.91  # Relative Percentage of Weights to be pruned in each Iteration, 19.91 is used in LTH paper
print_freq_prune = 1      # Printing Frequency for pruning of Train- and Test Loss
test_freq_prune  = 1      # Testing Frequency for pruning

# Specify Hyperparams of the reintroduction
num_epochs_reint = num_epochs_prune  # Number of Epochs per Reintialisation
print_freq_reint = print_freq_prune  # Printing Frequency for reinitialising of Train- and Test Loss
test_freq_reint  = test_freq_prune   # Testing Frequency for reinitialising

# Specify the objective Function
criterion = nn.CrossEntropyLoss()
lr        = 5.0  # learning rate
optimizer = T.optim.SGD(model.parameters(), lr=lr)
scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=max(5, test_freq_prune))
# lr = 5.0 and factor = 0.95 gives a maximum of 333 updates to lr before the update gets smaler than 1e-8
# Finished specifying the objective Function


# Function defining the training of the Model during the pruning procedure
def train_prune(epoch: int) -> float:
    epsilon      = 1e-6  # Possible that smaller is needed depending on the datatype used
    total_loss   = 0.
    comp_loss    = 0.    # Used for comparison down below
    log_interval = 200
    start_time   = time.time()
    src_mask     = generate_square_subsequent_mask(bptt).to(device)
    num_batches  = len(train_data) // bptt
    model.train()  # turn on train mode
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        optimizer.zero_grad()
        data_pts, targets = get_batch(train_data, i)
        batch_size_local  = data_pts.size(0)
        if batch_size_local != bptt:  # only on last batch
            src_mask = src_mask[:batch_size_local, :batch_size_local]
        output = model(data_pts, src_mask)
        t_loss = criterion(output.view(-1, ntokens), targets)
        t_loss.backward()
        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor      = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < epsilon, 0, grad_tensor)
                p.grad.data = T.from_numpy(grad_tensor).to(device)
        # Clipping Gradients
        T.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += t_loss.item()
        if batch % log_interval == 0 and batch > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss     = (total_loss - comp_loss) / log_interval
            # ppl          = math.exp(cur_loss)
            comp_loss = total_loss
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f}')  # | ppl {ppl:8.2f}')
            start_time = time.time()
    return total_loss / (len(train_data) - 1)


# Function defining the evaluation of the Model
def evaluate(eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with T.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data_pts, targets = get_batch(eval_data, i)
            batch_size_local  = data_pts.size(0)
            data_pts, targets = data_pts.to(device), targets.to(device)
            if batch_size_local != bptt:
                src_mask = src_mask[:batch_size_local, :batch_size_local]
            output      = model(data_pts, src_mask)
            total_loss += batch_size_local * criterion(output.view(-1, ntokens), targets).item()
    return total_loss / (len(eval_data) - 1)


# Function pruning each layer by percentile
def prune_by_percentile(percent: float) -> None:
    # Calculate percentile value
    i = 0
    for name, param in model.named_parameters():
        # Using terminology of "Deconstructing Lottery Tickets"
        # Not pruning bias term
        if 'weight' in name:
            w_c = param.data.cpu().numpy()                       # Current Weight
            w_i = (initial_state_dict[name]).data.cpu().numpy()  # Initial Weight
            percentile_value = np.percentile(abs(w_c[np.nonzero(w_c)]) - abs(w_i[np.nonzero(w_c)]), percent)
            # Convert Tensors to numpy and calculate
            weight_dev = param.device  # Get device
            new_mask = np.where((abs(w_c) - abs(w_i)) > percentile_value, mask[i], 0)
            # Apply new weight and mask
            param.data = T.from_numpy(w_c * new_mask).to(weight_dev)
            mask[i] = new_mask
            i += 1


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
            tensor = param.data.cpu().numpy()
            mask[i] = np.ones_like(tensor)  # Complete Mask containing ones
            i = i + 1
    return mask


# Function to apply the Mask to the Network and rewind
def original_initialization(mask_temp:  list, initial_state_dict:  dict) -> None:
    i = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = T.from_numpy(mask_temp[i] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            i = i + 1
        else:
            param.data = initial_state_dict[name]


# Function defining the pruning procedure used
def pruning_procedure(rewind: bool = False, experiment: int = 0) -> None:
    # Compression Rate
    comp = np.zeros(num_prune_cycles, float)

    # Initialising storage for performance indicators
    best_val_loss  = np.inf
    best_val       = np.full(num_prune_cycles, np.inf)
    all_train_loss = np.zeros(num_epochs_prune, float)
    all_val_loss   = np.zeros(num_epochs_prune, float)

    # Pruning cycle
    for _ite in range(num_prune_cycles):
        # Progressbar
        pbar = tqdm(range(num_epochs_prune))

        # List fraction of remaining Weights per layer
        print()
        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1

        print(f"\n--- Pruning Level [{experiment}.{seed}:{_ite}/{num_prune_cycles}]: ---")
        # Training and Testing cycle
        for iter_ in pbar:
            # Training
            train_loss = train_prune(iter_)
            # Testing
            if iter_ % test_freq_prune == 0:
                val_loss = evaluate(val_data)
                # Save Weights if best (might be unneccessary)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    utils.checkdir(f"{os.getcwd()}/saves/prune/")
                    T.save(model, f"{os.getcwd()}/saves/prune/{_ite}_model.pth.tar")
            # Save training- and validation Loss
            all_val_loss[iter_]   = val_loss
            all_train_loss[iter_] = train_loss
            # Print training- and validation Loss
            if iter_ % print_freq_prune == 0:
                pbar.set_description(f'Train Epoch: {iter_}/{num_epochs_prune} training Loss: {train_loss:.6f} validation Loss: {val_loss:.2f}% Best validation Loss: {best_val_loss:.2f}%')
            scheduler.step(val_loss)

        writer.add_scalar('val_loss/test', best_val_loss, comp1)
        best_val[_ite] = best_val_loss

        # Masking procedure
        if not _ite == num_prune_cycles-1:
            # Saving Current State
            state_dict.append(copy.deepcopy(model.state_dict()))
            utils.checkdir(f"{os.getcwd()}/saves/model_state_dicts/")
            T.save(model, f"{os.getcwd()}/saves/model_state_dicts/state_dict_{_ite}.pth.tar")
            # Masking
            prune_by_percentile(prune_percent)
            # Rewind to pruned version of initial state -> optional
            if rewind:
                original_initialization(mask, initial_state_dict)
            # Saving Mask
            mask_list.append(mask)
            utils.checkdir(f"{os.getcwd()}/dumps/prune/")
            with open(f"{os.getcwd()}/dumps/prune/mask_{comp1}.pkl", 'wb') as fp:
                pickle.dump(mask, fp)

        # Saving relevant Data
        # Plotting training and validation Loss, Iteration Curve
        # NOTE training Loss is computed for every iteration
        # while validation Loss is computed only for every {test_freq_prune} iterations
        # Therefore validation Loss saved is constant during the iterations inbetween.
        plt.plot(np.arange(1, num_epochs_prune + 1), all_train_loss, c="blue", label="Training Loss")
        plt.plot(np.arange(1, num_epochs_prune + 1), all_val_loss, c="red", label="Validation Loss")
        plt.title(f"Training and Validation Loss Vs Iterations (WikiText2, Language Model)")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(color="gray")
        utils.checkdir(f"{os.getcwd()}/plots/prune/")
        plt.savefig(f"{os.getcwd()}/plots/prune/TrainingVsValidationLoss_{comp1}.png", dpi=1200)
        plt.close()

        # Dump Plot values
        utils.checkdir(f"{os.getcwd()}/dumps/prune/")
        all_train_loss.dump(f"{os.getcwd()}/dumps/prune/all_train_loss_{comp1}.dat")
        all_val_loss.dump(f"{os.getcwd()}/dumps/prune/all_val_loss_{comp1}.dat")

        # Resetting variables to 0
        best_val_loss  = np.inf
        all_train_loss = np.zeros(num_epochs_prune, float)
        all_val_loss   = np.zeros(num_epochs_prune, float)

    # Dumping Values for Plotting
    utils.checkdir(f"{os.getcwd()}/dumps/prune/")
    comp.dump(f"{os.getcwd()}/dumps/prune/compression.dat")
    best_val.dump(f"{os.getcwd()}/dumps/prune/best_val.dat")

    # Plotting
    a = np.arange(num_prune_cycles)
    plt.plot(a, best_val, c="blue", label="Winning tickets")
    plt.title(f"Validation Loss Vs Unpruned Weights Percentage (WikiText2, Language Model)")
    plt.xlabel("Unpruned Weights Percentage")
    plt.ylabel("Validation Loss")
    plt.xticks(a, comp, rotation="vertical")
    plt.legend()
    plt.grid(color="gray")
    utils.checkdir(f"{os.getcwd()}/plots/prune/")
    plt.savefig(f"{os.getcwd()}/plots/prune/ValidationLossVsWeights.png", dpi=1200)
    plt.close()


# Function implementing symmetric difference for the masks
def symmetric_difference() -> list:
    sym_dif_list = []
    for i in range(len(mask_list) - 1):
        a = mask_list[i]
        b = mask_list[i + 1]
        sym_dif = copy.deepcopy(a)
        for j in range(len(a)):
            sym_dif[j].to(bool)
            sym_dif[j][T.eq(a[j], b[j])] = False
        sym_dif_list.append(sym_dif)
    return sym_dif_list


# Function implementing reintroduction schemes
def reintroduction(mask_dif:  list, choice: str = "old", model_state: dict = None) -> None:
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
        supplement = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)
    else:
        supplement = None
        print(f"\nI do not know this choice of reintroduction scheme. Please be so kind and teach me.\n")
        pass
    i = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data += T.from_numpy(mask_dif[i] * supplement[name].cpu().numpy()).to(weight_dev)
            i = i + 1


# Function defining the training of the model during the reintroduction procedure
def train_reintro(sym_dif_list: list, epoch: int) -> float:
    total_loss   = 0.
    comp_loss    = 0.    # Used for comparison down below
    log_interval = 200
    start_time   = time.time()
    src_mask     = generate_square_subsequent_mask(bptt).to(device)
    num_batches  = len(train_data) // bptt
    model.train()  # turn on train mode
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        optimizer.zero_grad()
        data_pts, targets = get_batch(train_data, i)
        batch_size_local  = data_pts.size(0)
        if batch_size_local != bptt:  # only on last batch
            src_mask = src_mask[:batch_size_local, :batch_size_local]
        output = model(data_pts, src_mask)
        t_loss = criterion(output.view(-1, ntokens), targets)
        t_loss.backward()
        # Manipulating the learningrate according to reintroduction time
        i = 0
        for mask_difference in sym_dif_list:
            factor = (1./0.95)**(i+1)  # The LR prior to the current LR, still up for discussion
            i     += 1                 # Count pos in list; not using enumerate() as this way is more flexible. CHANGE?
            j      = 0                 # Used to match position in mask with layer in the network
            for name, p in model.named_parameters():
                if 'weight' in name:
                    grad_tensor = p.grad.data.cpu().numpy()
                    # Change this Part to adjust the learningrate
                    grad_tensor[mask_difference[j]] = grad_tensor[mask_difference[j]]*factor
                    p.grad.data = T.from_numpy(grad_tensor).to(device)
                    j += 1
        # Clipping Gradients
        T.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += t_loss.item()
        if batch % log_interval == 0 and batch > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = (total_loss - comp_loss) / log_interval
            # ppl          = math.exp(cur_loss)
            comp_loss = total_loss
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f}')  # | ppl {ppl:8.2f}')
            start_time = time.time()
    return total_loss / (len(train_data) - 1)


# Function defining the procedure of regaining lost capacity
def regaining_procedure(experiment: int = 0) -> None:
    """
    while possible
        reintroduction procedure
        subsequent training scheme
    """
    # Information needed for Reintroduction
    s_d_mask_list = symmetric_difference()
    s_d_mask_list.reverse()
    state_dict.reverse()

    # Compression Rate
    comp = np.zeros(len(s_d_mask_list), float)

    # Initialising storage for performance indicators
    best_val_loss  = np.inf
    best_val       = np.full(len(s_d_mask_list), np.inf)
    all_train_loss = np.zeros(num_epochs_reint, float)
    all_val_loss   = np.zeros(num_epochs_reint, float)

    for reint_step in range(len(s_d_mask_list)):  # Similar to _ite in pruning_procedure(); Number needed
        # Reintroduce the lifted mask
        reintroduction(s_d_mask_list[reint_step], "old", state_dict[reint_step])

        # The following is similar to pruning_procedure()
        # Progressbar
        pbar = tqdm(range(num_epochs_reint))

        # List fraction of remaining Weights per layer
        print()
        comp1 = utils.print_nonzeros(model)
        comp[reint_step] = comp1

        print(f"\n--- Reintroduction Level [{experiment}.{seed}:{reint_step}/{len(s_d_mask_list)}]: ---")
        # Training and Testing cycle
        for __iter in pbar:
            # Training
            train_loss = train_reintro(s_d_mask_list[:reint_step+1], __iter)
            # Testing
            if __iter % test_freq_reint == 0:
                val_loss = evaluate(val_data)
                # Save Weights if best (might be unneccessary)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    utils.checkdir(f"{os.getcwd()}/saves/reint/")
                    T.save(model, f"{os.getcwd()}/saves/reint/{reint_step}_model.pth.tar")
            # Save training- and validation Loss
            all_val_loss[__iter] = val_loss
            all_train_loss[__iter] = train_loss
            # Print training- and validation Loss
            if __iter % print_freq_reint == 0:
                pbar.set_description(
                    f'Train Epoch: {__iter}/{num_epochs_reint} training Loss: {train_loss:.6f} validation Loss: {val_loss:.2f}% Best validation Loss: {best_val_loss:.2f}%')
            # TODO: Scheduler will regulate the lr down (no increase). This might be (very) bad. Solution?
            scheduler.step(val_loss)

        writer.add_scalar('val_loss/test', best_val_loss, comp1)
        best_val[reint_step] = best_val_loss

        # Saving Current State
        reint_state_dict.append(copy.deepcopy(model.state_dict()))
        utils.checkdir(f"{os.getcwd()}/saves/model_reint_state_dicts/")
        T.save(model, f"{os.getcwd()}/saves/model_reint_state_dicts/reint_state_dict_{reint_step}.pth.tar")

        # Saving relevant Data
        # Plotting training and validation Loss, Iteration Curve
        # NOTE training Loss is computed for every iteration
        # while validation Loss is computed only for every {test_freq_prune} iterations
        # Therefore validation Loss saved is constant during the iterations inbetween.
        plt.plot(np.arange(1, num_epochs_reint + 1), all_train_loss, c="blue", label="Training Loss")
        plt.plot(np.arange(1, num_epochs_reint + 1), all_val_loss, c="red", label="Validation Loss")
        plt.title(f"Training and Validation Loss Vs Iterations (WikiText2, Language Model)")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(color="gray")
        utils.checkdir(f"{os.getcwd()}/plots/reint/")
        plt.savefig(f"{os.getcwd()}/plots/reint/TrainingVsValidationLoss_{comp1}.png", dpi=1200)
        plt.close()

        # Dump Plot values
        utils.checkdir(f"{os.getcwd()}/dumps/reint/")
        all_train_loss.dump(f"{os.getcwd()}/dumps/reint/all_train_loss_{comp1}.dat")
        all_val_loss.dump(f"{os.getcwd()}/dumps/reint/all_val_loss_{comp1}.dat")

        # Resetting variables to 0
        best_val_loss  = np.inf
        all_train_loss = np.zeros(num_epochs_reint, float)
        all_val_loss   = np.zeros(num_epochs_reint, float)

    # Dumping Values for Plotting
    utils.checkdir(f"{os.getcwd()}/dumps/reint/")
    comp.dump(f"{os.getcwd()}/dumps/reint/compression.dat")
    best_val.dump(f"{os.getcwd()}/dumps/reint/best_val.dat")

    # Plotting
    a = np.arange(len(s_d_mask_list))
    plt.plot(a, best_val, c="blue", label="Winning tickets")
    plt.title(f"Validation Loss Vs Unpruned Weights Percentage (WikiText2, Language Model)")
    plt.xlabel("Unpruned Weights Percentage")
    plt.ylabel("Validation Loss")
    plt.xticks(a, comp, rotation="vertical")
    plt.legend()
    plt.grid(color="gray")
    utils.checkdir(f"{os.getcwd()}/plots/reint/")
    plt.savefig(f"{os.getcwd()}/plots/reint/ValidationLossVsWeights.png", dpi=1200)
    plt.close()


# Making Initial Mask
mask      = make_mask()

# List of all masks generated during the algorithm
mask_list = [mask]

# Set Seed
seed = 0
T.manual_seed(seed)
np.random.seed(seed)


# Main
def main(rewind: bool = False, experiment: int = 0, verbose: bool = False) -> None:
    if verbose:
        # Print named params of the model
        for name, param in model.named_parameters():
            print(name, param.size())
    # Warm up training? For rewinding as little as one epoch is enough

    # Pruning procedure
    pruning_procedure(rewind, experiment)  # The Literature finds rewinding improving the performance when rewinded to warmup state


if __name__ == "__main__":
    main()
