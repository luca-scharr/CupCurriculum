# Importing Libraries
import os
import sys
import argparse
from typing import Tuple
from tensorboardX import SummaryWriter
import copy
import time
import math
import numpy as np
import pickle

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

# Tensorboard initialization
writer = SummaryWriter()

# Plotting Style
sns.set_style('darkgrid')

# Setting the computing device
device = T.device("cuda" if T.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Use a Parser to specify Hyperparams etc.
parser = argparse.ArgumentParser()
# TODO: Think about adding the seed or experiment number
# TODO: Change the {utils.checkdir(f"{os.getcwd()}/some/path/")} expressions to something like
# TODO: {utils.checkdir(f"{os.getcwd()}/some/path/Seed{seed}/")} or {utils.checkdir(f"{os.getcwd()}/some/path/Experiment{experiment}/")}
# TODO: Obviously change the save commands as well
# Set Hyperparams for Batches
parser.add_argument("--batch_size", type=int, default=20, help="The Batchsize used for Training")
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
# Facebook Paper uses num_prune_cycles = 20 and prune_percent = 20. as well as 50,000 updates (overall?)
parser.add_argument("--num_prune_cycles", type=int, default=2, help="The Number of Pruning Cycles")  # 20
parser.add_argument("--num_epochs_prune", type=int, default=50, help="The Number of Epochs per Pruning Cycle")  # 50
parser.add_argument("--prune_percent", type=float, default=20., help="The Percentage of remaining Weights to be pruned in each Iteration")
parser.add_argument("--print_freq_prune", type=int, default=1, help="The Printing-Frequency of Train- and Test Loss during Pruning")
parser.add_argument("--test_freq_prune", type=int, default=1, help="The Testing Frequency during Pruning")
# Set Hyperparams defining the Reintroduction Procedure
# TODO: Think about adding choice option (selecting reintroduction scheme)
parser.add_argument("--num_epochs_reint", type=int, default=50, help="The Number of Epochs per Reintialisation")  # 50
parser.add_argument("--print_freq_reint", type=int, default=1, help="The Printing Frequency of Train- and Test Loss durinig Reinitialisation")
parser.add_argument("--test_freq_reint", type=int, default=1, help="The Testing Frequency during Reinitialisation")
# TODO: Think about adding LR, the Factor used in scheduler, etc.
parser.add_argument("-v", "--verbosity", action="count", default=0)
args = parser.parse_args()


# Generate Batches inside the Datasets. NOT SHUFFLED
train_iter, test_iter, val_iter = WikiText2.iters(batch_size=args.batch_size)  # shape [seq_len, batch_size]
"""train_iter.to(device)
test_iter.to(device)
val_iter.to(device)"""
# Finished preparing the Data


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


# Building the Model to be trained
model = TransformerModel(args.ntokens, args.emsize, args.nhead, args.d_hid, args.nlayers, args.dropout).to(device)
# Finished defining the Model

# Copying and Saving Initial State
initial_state_dict = copy.deepcopy(model.state_dict())  # TODO: Is it better to use this or warmup_state_dict?
utils.checkdir(f"{os.getcwd()}/saves/model_state_dicts/")
T.save(model, f"{os.getcwd()}/saves/model_state_dicts/initial_state_dict.pth.tar")

# List of Modelstates
state_dict       = [initial_state_dict]
reint_state_dict = []

# Specify the objective Function
criterion = nn.CrossEntropyLoss()
lr        = 5.0  # learning rate
# Stated in Successfully applying ... (Aachen) Adafactor gives better results than Adam, they include warmup after reset
optimizer = T.optim.SGD(model.parameters(), lr=lr)
scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=max(5, args.test_freq_prune))
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
        src_mask     = generate_square_subsequent_mask(args.bptt).to(device)
        model.train()  # turn on train mode
        for batch_num, batch in enumerate(train_iter):
            optimizer.zero_grad()
            data_pts, targets = batch.text, batch.target
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
                      f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                      f'loss {cur_loss:5.2f}')
                start_time = time.time()
    # Copying and Saving State after Warm-Up
    utils.checkdir(f"{os.getcwd()}/saves/model_state_dicts/")
    T.save(model, f"{os.getcwd()}/saves/model_state_dicts/warmup_state_dict.pth.tar")
    pass


# Function defining the training of the Model during the pruning procedure
def train_prune(epoch: int) -> float:
    epsilon      = 1e-6  # Possible that smaller is needed depending on the datatype used
    total_loss   = 0.
    comp_loss    = 0.    # Used for comparison down below
    log_interval = 200
    start_time   = time.time()
    src_mask     = generate_square_subsequent_mask(args.bptt).to(device)
    model.train()  # turn on train mode
    for batch_num, batch in enumerate(train_iter):
        optimizer.zero_grad()
        data_pts, targets = batch.text, batch.target
        batch_size_local  = data_pts.size(0)
        if batch_size_local != args.bptt:  # only on last batch
            src_mask = src_mask[:batch_size_local, :batch_size_local]
        output = (model(data_pts, src_mask)).view(-1, args.ntokens)
        t_loss = criterion(output, targets.view(output.size(0)))
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
        if batch_num % log_interval == 0 and batch_num > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = (total_loss - comp_loss) / log_interval
            comp_loss = total_loss
            print(f'| epoch {epoch:3d} | {batch_num:5d}/{len(train_iter):5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f}')
            start_time = time.time()
    return total_loss / (args.batch_size * len(train_iter))  # Approximatley the Loss per Datapoint, a little smaler


# Function defining the evaluation of the Model
def evaluate() -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(args.bptt).to(device)
    with T.no_grad():
        for batch in val_iter:
            data_pts, targets = batch.text, batch.target
            batch_size_local  = data_pts.size(0)
            if batch_size_local != args.bptt:
                src_mask = src_mask[:batch_size_local, :batch_size_local]
            output = (model(data_pts, src_mask)).view(-1, args.ntokens)
            total_loss += batch_size_local * criterion(output, targets.view(output.size(0))).item()
    return total_loss / (args.batch_size * len(val_iter))  # Approximatley the Loss per Datapoint, a little smaler


# Function pruning each layer by percentile
def prune_by_percentile(percent: float) -> None:
    # Calculate percentile value
    i = 0
    for name, param in model.named_parameters():
        # Using terminology of "Deconstructing Lottery Tickets"
        # Not pruning bias term
        if 'weight' in name:
            w_c = param.data.cpu().numpy()                       # Current Weight
            w_i = (initial_state_dict[name]).data.cpu().numpy()  # Initial Weight #TODO Exchange for warmup state dict
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
    comp = np.zeros(args.num_prune_cycles, float)

    # Initialising storage for performance indicators
    best_val_loss  = np.inf
    best_val       = np.full(args.num_prune_cycles, np.inf)
    all_train_loss = np.zeros(args.num_epochs_prune, float)
    all_val_loss   = np.zeros(args.num_epochs_prune, float)

    # TODO: Warmup at every pruning iteration?

    # Pruning cycle
    for _ite in range(args.num_prune_cycles):
        # Progressbar
        pbar = tqdm(range(args.num_epochs_prune))

        # List fraction of remaining Weights per layer
        print()
        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1

        print(f"\n--- Pruning Level [{experiment}.{seed}:{_ite}/{args.num_prune_cycles}]: ---")
        # Training and Testing cycle
        for iter_ in pbar:
            # Training
            train_loss = train_prune(iter_)
            # Testing
            if iter_ % args.test_freq_prune == 0:
                val_loss = evaluate()
                # Save Weights if best (might be unneccessary)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    utils.checkdir(f"{os.getcwd()}/saves/prune/")
                    T.save(model, f"{os.getcwd()}/saves/prune/{_ite}_model.pth.tar")
            # Save training- and validation Loss
            all_val_loss[iter_]   = val_loss
            all_train_loss[iter_] = train_loss
            # Print training- and validation Loss
            if iter_ % args.print_freq_prune == 0:
                pbar.set_description(f'Train Epoch: {iter_}/{args.num_epochs_prune} Training Loss: {train_loss:.6f} Validation Loss: {val_loss:.2f}% Best Validation Loss: {best_val_loss:.2f}%')
            scheduler.step(val_loss)

        writer.add_scalar('val_loss/test', best_val_loss, comp1)
        best_val[_ite] = best_val_loss

        # Masking procedure
        if not _ite == args.num_prune_cycles-1:
            # Saving Current State
            state_dict.append(copy.deepcopy(model.state_dict()))
            utils.checkdir(f"{os.getcwd()}/saves/model_state_dicts/")
            T.save(model, f"{os.getcwd()}/saves/model_state_dicts/state_dict_{_ite}.pth.tar")
            # Masking
            prune_by_percentile(args.prune_percent)
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
        plt.plot(np.arange(1, args.num_epochs_prune + 1), all_train_loss, c="blue", label="Training Loss")
        plt.plot(np.arange(1, args.num_epochs_prune + 1), all_val_loss, c="red", label="Validation Loss")
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
        all_train_loss = np.zeros(args.num_epochs_prune, float)
        all_val_loss   = np.zeros(args.num_epochs_prune, float)

    # Dumping Values for Plotting
    utils.checkdir(f"{os.getcwd()}/dumps/prune/")
    comp.dump(f"{os.getcwd()}/dumps/prune/compression.dat")
    best_val.dump(f"{os.getcwd()}/dumps/prune/best_val.dat")

    # Plotting
    a = np.arange(args.num_prune_cycles)
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
            sym_dif[j].astype(bool)
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
        supplement = TransformerModel(args.ntokens, args.emsize, args.nhead, args.d_hid, args.nlayers, args.dropout)
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
    src_mask     = generate_square_subsequent_mask(args.bptt).to(device)
    model.train()  # turn on train mode
    for batch_num, batch in enumerate(train_iter):
        optimizer.zero_grad()
        data_pts, targets = batch.text, batch.target
        batch_size_local  = data_pts.size(0)
        if batch_size_local != args.bptt:  # only on last batch
            src_mask = src_mask[:batch_size_local, :batch_size_local]
        output = (model(data_pts, src_mask)).view(-1, args.ntokens)
        t_loss = criterion(output, targets.view(output.size(0)))
        t_loss.backward()
        # Manipulating the learningrate according to reintroduction time
        i = 0
        for mask_difference in sym_dif_list:
            factor = (1./0.95)**(i+1)  # The LR prior to the current LR, up for discussion; 0.95 = factor in Scheduler
            i     += 1                 # Count pos in list; not using enumerate() as this way is more flexible. CHANGE?
            j      = 0                 # Used to match position in mask with layer in the network
            for name, p in model.named_parameters():  # Vektorisieren nachscahuen; Pytorch discussion seite
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
        if batch_num % log_interval == 0 and batch_num > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = (total_loss - comp_loss) / log_interval
            # ppl          = math.exp(cur_loss)
            comp_loss = total_loss
            print(f'| epoch {epoch:3d} | {batch_num:5d}/{len(train_iter):5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f}')
            start_time = time.time()
    return total_loss / (args.batch_size * len(train_iter))  # Approximatley the Loss per Datapoint, a little smaler


# Function defining the procedure of regaining lost capacity
def regaining_procedure(experiment: int = 0, choice: str = "old") -> None:
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
        comp1 = utils.print_nonzeros(model)
        comp[reint_step] = comp1

        print(f"\n--- Reintroduction Level [{experiment}.{seed}:{reint_step}/{len(s_d_mask_list)}]: ---")
        # Training and Testing cycle
        for __iter in pbar:
            # Training
            train_loss = train_reintro(s_d_mask_list[:reint_step+1], __iter)
            # Testing
            if __iter % args.test_freq_reint == 0:
                val_loss = evaluate()
                # Save Weights if best (might be unneccessary)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    utils.checkdir(f"{os.getcwd()}/saves/reint/")
                    T.save(model, f"{os.getcwd()}/saves/reint/{reint_step}_model.pth.tar")
            # Save training- and validation Loss
            all_val_loss[__iter] = val_loss
            all_train_loss[__iter] = train_loss
            # Print training- and validation Loss
            if __iter % args.print_freq_reint == 0:
                pbar.set_description(
                    f'Train Epoch: {__iter}/{args.num_epochs_reint} training Loss: {train_loss:.6f} validation Loss: {val_loss:.2f}% Best validation Loss: {best_val_loss:.2f}%')
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
        plt.plot(np.arange(1, args.num_epochs_reint + 1), all_train_loss, c="blue", label="Training Loss")
        plt.plot(np.arange(1, args.num_epochs_reint + 1), all_val_loss, c="red", label="Validation Loss")
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
        all_train_loss = np.zeros(args.num_epochs_reint, float)
        all_val_loss   = np.zeros(args.num_epochs_reint, float)

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
mask = make_mask()

# List of all masks generated during the algorithm
mask_list = [mask]

# Set Seed
seed = 0
T.manual_seed(seed)
np.random.seed(seed)


# Main
def main(rewind: bool = False, experiment: int = 0, choice: str = "old") -> None:
    starting_time = time.time()
    # Warm-Up Training? For rewinding as little as one epoch is enough
    warmup()
    warmup_state_dict = copy.deepcopy(model.state_dict())  # TODO: Is it better to use this or initial_state_dict?
    time_warmup = time.time()
    print(f"Runtime of the warmup {time_warmup-starting_time} [s]")

    # Pruning Procedure
    pruning_procedure(rewind, experiment)  # The Literature finds rewinding improving the performance when rewinded to warmup state
    time_pruning = time.time()
    print(f"Runtime of the pruning procedure {time_pruning-time_warmup} [s]")

    # Reintroduction Procedure
    regaining_procedure(experiment, choice)
    time_reintroduction = time.time()
    print(f"Runtime of the reintroduction procedure {time_reintroduction-time_pruning} [s]")

    # Show and Save Timings
    print(f"Runtime overall {time_reintroduction - starting_time} [s]")
    utils.checkdir(f"{os.getcwd()}/saves/runtimes/")
    times = T.tensor([time_warmup-starting_time, time_pruning-time_warmup,
                      time_reintroduction-time_pruning, time_reintroduction-starting_time])
    T.save(times, '{os.getcwd()}/saves/runtimes/tensor.pt')


if __name__ == "__main__":
    main()
