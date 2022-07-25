import time
import math
import os
import torch as T
import torch.nn as nn
import data
import transformer_modell
import datasets
from torch.utils.data import DataLoader

device = T.device('cpu')

# Load the dataset
wmt16_dataset_de_en = datasets.load_dataset('wmt16', 'de-en')

# Generate the dataloader for training and test splits
train_dataloader = DataLoader(wmt16_dataset_de_en['train'], batch_size=8, shuffle=True)
test_dataloader = DataLoader(wmt16_dataset_de_en['test'], batch_size=8, shuffle=True)

# Set Params of Network
d_model            = 512
nhead              = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward    = 2048
dropout            = 0.1

# Generate instance of the transformer
model = transformer_modell.TransformerModel(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                            dim_feedforward, dropout).to(device)

# Setting the training parameters
criterion = nn.CrossEntropyLoss() #BLEU?
lr        = 5.0
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train(model: nn.Module) -> None:#Still work to do!!!
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    # not sure yet, could be for getting one word at the time
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        batch_size = data.size(0)
        if batch_size != bptt:  # only on last batch
            src_mask = src_mask[:batch_size, :batch_size]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            batch_size = data.size(0)
            if batch_size != bptt:
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += batch_size * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)