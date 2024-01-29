from pykeen.pipeline import pipeline
from pykeen.losses import NSSALoss
from pykeen.datasets import Nations
from pykeen.training import SLCWATrainingLoop
from torch.optim import Adam

from uvxy import UVXY

from os.path import exists, basename, splitext
from argparse import ArgumentParser
from json import load
from datetime import datetime

parser = ArgumentParser()
parser.add_argument("config_path")
args = parser.parse_args()

config_path = args.config_path
config_name = splitext(basename(config_path))[0]

config = load(open(config_path, "r")) if exists(config_path) else {}

print(f"Found configuration {config_name}: {config}")

def with_default(name, default):
    return config[name] if name in config else default

dim = with_default("dim", 64)
p = with_default("p", 1)
constraints = with_default("constraints", "uvxy")
trainable_weights = with_default("trainable_weights", False)
ds = with_default("ds", Nations)
epochs = with_default("epochs", 500)
patience = with_default("patience", 3)
margin = with_default("margin", 3)
lr = with_default("lr", 1e-2)
negs = with_default("negs", 5)
batch_size = with_default("batch_size", 256)

ts = int(datetime.now().timestamp())

result = pipeline(
    model=UVXY,
    model_kwargs=dict(
        embedding_dim=dim,
        p=p,
        constraints=constraints,
        with_attention_weights=trainable_weights
    ),

    dataset=ds,

    loss=NSSALoss,
    loss_kwargs=dict(margin=margin),
    training_loop=SLCWATrainingLoop,
    negative_sampler_kwargs=dict(num_negs_per_pos=negs),
    optimizer=Adam,
    optimizer_kwargs=dict(lr=lr),

    training_kwargs=dict(
        num_epochs=epochs,
        batch_size=batch_size,
        checkpoint_directory="checkpoints",
        checkpoint_name= f"{config_name}.pt",
        checkpoint_frequency=50,
    ),

    stopper="early",
    stopper_kwargs=dict(patience=patience,relative_delta=0.005,frequency=10),
)

result_path = f"results/{config_name}/{ts}"
result.save_to_directory(result_path)

print(f"Evaluation results saved to folder: {result_path}")