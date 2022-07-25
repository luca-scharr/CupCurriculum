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

# Specify the objective Function
criterion = nn.CrossEntropyLoss()
lr        = 5.0  # learning rate
optimizer = T.optim.SGD(model.parameters(), lr=lr)
scheduler = T.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
# Finished specifying the objective Function


# TODO: Output überlegen
# Function training the Model
def train(epoch: int) -> float:
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
        T.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        total_loss += t_loss.item()
        if batch % log_interval == 0 and batch > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss     = (total_loss - comp_loss) / log_interval
            ppl          = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            comp_loss  = total_loss
            start_time = time.time()
    return total_loss / (len(train_data) - 1)


# Function evaluating the Model
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


# TODO: Change to other pruning criterion, initial_state_dict might be helpful
# Prune by Percentile module
def prune_by_percentile(percent: int) -> None:
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
            # Sure? Looks like I want np.where(abs(tensor) < percentile_value, mask[i], 0) instead
            new_mask = np.where((abs(w_c) - abs(w_i)) < percentile_value, mask[i], 0)
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
            mask[i] = np.ones_like(tensor)  # Complete Mask of ones
            i = i + 1
    return mask


# Function to apply the Mask to the Network
def original_initialization(mask_temp:  list, initial_state_dict:  dict) -> None:
    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = T.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        else:
            param.data = initial_state_dict[name]


# Specify Hyperparams of the Pruning
num_epochs       = 2  # Number of Epochs
num_prune_cycles = 5   # Number of Pruning Cycles
prune_percent    = 10  # Relative Percentage of Weights to be pruned in each Iteration
print_freq       = 1   # Printing Frequency of Train- and Test Loss
test_freq        = 1   # Testing Frequency

# Making Initial Mask
mask = make_mask()

# Set Seed
seed = 0
T.manual_seed(seed)
np.random.seed(seed)


# Main
def main(experiment: int = 0, verbose: bool = False) -> None:
    if verbose:
        # Print named params of the model
        for name, param in model.named_parameters():
            print(name, param.size())

    # Pruning
    # Compression Rate
    comp = np.zeros(num_prune_cycles, float)

    # Keeping track of performance
    best_val_loss  = np.inf
    best_val       = np.full(num_prune_cycles, np.inf)
    all_train_loss = np.zeros(num_epochs, float)
    all_val_loss   = np.zeros(num_epochs, float)

    for _ite in range(num_prune_cycles):
        # Progressbar
        pbar = tqdm(range(num_epochs))

        # Masking
        if not _ite == 0:
            prune_by_percentile(prune_percent)
            original_initialization(mask, initial_state_dict)

        # List fraction of remaining Weights per layer
        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1

        print(f"\n--- Pruning Level [{experiment}:{_ite}/{num_prune_cycles}]: ---")
        # Pruning cycle
        for iter_ in pbar:
            # Training
            train_loss = train(iter_)
            # Testing
            if iter_ % test_freq == 0:
                val_loss = evaluate(val_data)
                # Save Weights if best (maybe exchange and save all?)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    utils.checkdir(f"{os.getcwd()}/saves/")
                    T.save(model, f"{os.getcwd()}/saves/{_ite}_model.pth.tar")
            # Save training- and validation Loss
            all_val_loss[iter_]   = val_loss
            all_train_loss[iter_] = train_loss
            # Print training- and validation Loss
            if iter_ % print_freq == 0:
                pbar.set_description(f'Train Epoch: {iter_}/{num_epochs} training Loss: {train_loss:.6f} validation Loss: {val_loss:.2f}% Best valuation Loss: {best_val_loss:.2f}%')

        writer.add_scalar('val_loss/test', best_val_loss, comp1)
        best_val[_ite] = best_val_loss

        # Saving relevant Data
        # Plotting training and validation Loss, Iteration Curve
        # NOTE training Loss is computed for every iteration
        # while validation Loss is computed only for every {test_freq} iterations
        # Therefore validation Loss saved is constant during the iterations inbetween.
        plt.plot(np.arange(1, num_epochs + 1), all_train_loss, c="blue", label="Training Loss")
        plt.plot(np.arange(1, num_epochs + 1), all_val_loss, c="red", label="Validation Loss")
        plt.title(f"Training and Validation Loss Vs Iterations (WikiText2, Language Model)")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(color="gray")
        utils.checkdir(f"{os.getcwd()}/plots/lt/")
        plt.savefig(f"{os.getcwd()}/plots/lt/TrainingVsValidationLoss_{comp1}.png", dpi=1200)
        plt.close()

        # Dump Plot values
        utils.checkdir(f"{os.getcwd()}/dumps/lt/")
        all_train_loss.dump(f"{os.getcwd()}/dumps/lt/all_train_loss_{comp1}.dat")
        all_val_loss.dump(f"{os.getcwd()}/dumps/lt/all_val_loss_{comp1}.dat")

        # Dumping mask
        utils.checkdir(f"{os.getcwd()}/dumps/lt/")
        with open(f"{os.getcwd()}/dumps/lt/mask_{comp1}.pkl", 'wb') as fp:
            pickle.dump(mask, fp)

        # Resetting variables to 0
        best_val_loss  = np.inf
        all_train_loss = np.zeros(num_epochs, float)
        all_val_loss   = np.zeros(num_epochs, float)

    # Dumping Values for Plotting
    utils.checkdir(f"{os.getcwd()}/dumps/lt/")
    comp.dump(f"{os.getcwd()}/dumps/lt/compression.dat")
    best_val.dump(f"{os.getcwd()}/dumps/lt/best_val.dat")

    # Plotting
    a = np.arange(num_prune_cycles)
    plt.plot(a, best_val, c="blue", label="Winning tickets")
    plt.title(f"Validation Loss Vs Unpruned Weights Percentage (WikiText2, Language Model)")
    plt.xlabel("Unpruned Weights Percentage")
    plt.ylabel("Validation Loss")
    plt.xticks(a, comp, rotation="vertical")
    plt.legend()
    plt.grid(color="gray")
    utils.checkdir(f"{os.getcwd()}/plots/lt/")
    plt.savefig(f"{os.getcwd()}/plots/lt/ValidationLossVsWeights.png", dpi=1200)
    plt.close()


if __name__ == "__main__":
    main()
