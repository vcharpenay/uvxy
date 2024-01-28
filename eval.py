from pykeen.pipeline import pipeline
from pykeen.hpo import hpo_pipeline
from pykeen.losses import MarginRankingLoss, BCEWithLogitsLoss, BCEAfterSigmoidLoss, NSSALoss, SoftplusLoss, CrossEntropyLoss
from pykeen.models import TransE, DistMult, RotatE, TuckER
from pykeen.datasets import FB15k237, WN18RR, Nations
from pykeen.training import SLCWATrainingLoop, LCWATrainingLoop
from torch.optim import Adam

from uvxy import UVXY

from os.path import exists
from argparse import ArgumentParser
from json import load
from datetime import datetime

parser = ArgumentParser()
parser.add_argument("config_name")
args = parser.parse_args()

config_name = args.config_name
config_path = config_name + ".json"

config = load(open(config_path, "r")) if exists(config_path) else {}

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

result.save_to_directory(f"results/{config_name}/{ts}")
