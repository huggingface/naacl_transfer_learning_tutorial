# Copyright (c) 2019-present, Thomas Wolf.
# All rights reserved. This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.
from collections import namedtuple
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import PiecewiseLinear, ProgressBar
from pytorch_pretrained_bert import OpenAIGPTTokenizer, cached_path
from model import TransformerWithLMHead

Config = namedtuple('Config',
    field_names="embed_dim, hidden_dim, num_max_positions, num_embeddings, num_heads, num_layers," 
                "dropout, initializer_range, batch_size, lr, max_norm, n_epochs, n_warmup, device"
                "gradient_accumulation_steps",
    defaults   =[ 256     , 1024      , 256              , 50000         , 8        , 6         ,
                 0.1    , 0.02             , 32         , 1e-3, 1     , 10      , 1000    , "cuda",
                 8])  # You need a 

# Load a pre-defined BPE tokenizer, create config and model
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
args = Config(num_embeddings=len(tokenizer), device="cuda" if torch.cuda.is_available() else "cpu")
model = TransformerWithLMHead(args)
model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Download and tokenize wikitext-2 training dataset
dataset_file = cached_path("https://s3.amazonaws.com/datasets.huggingface.co/wikitext-2/train.txt")
with open(dataset_file, "r", encoding="utf-8") as f:
    dataset = f.readlines()
dataset = list(tokenizer.encode(line.strip()) for line in tqdm(dataset) if len(line.strip()))
dataset = torch.tensor([index for line in dataset for index in line], dtype=torch.long)

# Organize the dataset in consecutive blocs of num_max_positions tokens for the transformer
num_sequences = (dataset.size(0) // args.num_max_positions) * args.num_max_positions
dataset = dataset.narrow(0, 0, num_sequences).view(-1, args.num_max_positions)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Define training function
def update(engine, batch):
    model.train()
    batch = batch.transpose(0, 1).contiguous().to(args.device)  # to shape [seq length, batch]
    logits, loss = model(batch, labels=batch)
    loss = loss / args.gradient_accumulation_steps
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
    if engine.state.iteration % args.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()
trainer = Engine(update)

# Add progressbar with loss
RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
ProgressBar(persist=True).attach(trainer, metric_names=['loss'])

# Learning rate schedule: linearly warm-up to lr and then decrease the learning rate to zero
scheduler = PiecewiseLinear(optimizer, "lr",
                 [(0, 0.0), (args.n_warmup, args.lr), (args.n_epochs * len(dataloader), 0.0)])
trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

# Train
trainer.run(dataloader, max_epochs=args.n_epochs)
