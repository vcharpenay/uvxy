from math import floor, ceil
from pathlib import Path
from typing import Any, Dict, Iterator, Iterable, List, Mapping, Optional, TextIO, Union
from abc import abstractmethod
from class_resolver import HintOrType, OptionalKwargs
from pandas import read_csv
from pykeen.lr_schedulers import LRScheduler
from pykeen.models import Model
from pykeen.stoppers import Stopper
from pykeen.trackers import ResultTracker
from pykeen.training import TrainingLoop, SLCWATrainingLoop
from pykeen.training.callbacks import TrainingCallbackHint, TrainingCallbackKwargsHint
from pykeen.triples import CoreTriplesFactory, TriplesFactory
from pykeen.sampling import NegativeSampler, negative_sampler_resolver
from pykeen.typing import EntityMapping, InductiveMode, MappedTriples, RelationMapping
from torch.utils.data import IterableDataset
from torch import FloatTensor, LongTensor, tensor, cat, stack, randint
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

class PathsFactory:

    mapped_paths: LongTensor

    def __init__(
        self,
        mapped_paths
    ) -> None:
        self.mapped_paths = mapped_paths

    @classmethod
    def from_file(cls, filename, entities_to_ids, relations_to_ids):
        df = read_csv(filename, sep="\t", names=("h", "r1", "r2", "t"))

        hs = entities_to_ids(df["h"])
        r1s = relations_to_ids(df["r1"])
        r2s = relations_to_ids(df["r2"])
        ts = entities_to_ids(df["t"])

        ps = tensor([
            [h, r1, r2, t]
            for h, r1, r2, t in zip(hs, r1s, r2s, ts)
        ])

        return PathsFactory(ps)
    
    @classmethod
    def from_triples(cls, mapped_triples: LongTensor, nb_paths = None):
        if not nb_paths:
            nb_paths = mapped_triples.size(0)

        h = mapped_triples[:,0]
        r = mapped_triples[:,1]
        t = mapped_triples[:,2]

        paths = []

        for h, r1, e in mapped_triples:
            joins = mapped_triples[mapped_triples[:, 0] == e]
            nb_joins = joins.size(0)

            if nb_joins > 0:
                i = randint(0, nb_joins, (1,)).item()
                e, r2, t = joins[i]

                paths.append([h, r1, r2, t])

            if len(paths) > nb_paths: break

        return PathsFactory(tensor(paths))
    
class PathSampler:

    @abstractmethod
    def sample(self, triples_batch: LongTensor) -> LongTensor:
        raise NotImplementedError
    
class PredefinedPathSampler(PathSampler):

    def __init__(self, mapped_paths: LongTensor) -> None:
        self.mapped_paths = mapped_paths
        self.nb_paths = mapped_paths.size(0)

    def sample(self, triples_batch: LongTensor) -> MappedTriples:
        batch_size = triples_batch.size(0)
        indices = randint(0, self.nb_paths, (batch_size,))
        # TODO draw paths having head, rel and/or tail in common with batch
        return self.mapped_paths[indices]

class PathDataset(IterableDataset):

    def __init__(
        self,
        mapped_triples: MappedTriples,
        batch_size: int = 1,
        drop_last: bool = True,
        num_entities: Optional[int] = None,
        num_relations: Optional[int] = None,
        negative_sampler: HintOrType[NegativeSampler] = None,
        negative_sampler_kwargs: OptionalKwargs = None,
        path_sampler: HintOrType[PathSampler] = None
    ) -> None:
        self.mapped_triples = mapped_triples
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.negative_sampler = negative_sampler_resolver.make(
            negative_sampler,
            pos_kwargs=negative_sampler_kwargs,
            mapped_triples=self.mapped_triples,
            num_entities=num_entities,
            num_relations=num_relations,
        )

        # TODO put in resolver
        pf = PathsFactory.from_triples(mapped_triples)
        self.path_sampler = path_sampler(mapped_paths=pf.mapped_paths)

        self.nb_batches = self._get_nb_batches(mapped_triples.size(0), batch_size, drop_last)

    def __getitem__(self, indices: List[int]) -> Any:
        positive_triples = self.mapped_triples[indices]
        negative_triples, _ = self.negative_sampler.sample(positive_batch=positive_triples)

        if self.path_sampler:
            positive_paths = self.path_sampler.sample(positive_triples)
            negative_paths, _ = self.negative_sampler.sample(positive_batch=positive_paths)
        else:
            positive_paths = None
            negative_paths = None

        return positive_triples, negative_triples, positive_paths, negative_paths

    def iter_triple_ids(self) -> Iterable[List[int]]:
        yield from BatchSampler(
            sampler=RandomSampler(data_source=range(len(self.mapped_triples))),
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )

    def __iter__(self) -> Iterator:
        for triple_ids in self.iter_triple_ids():
            yield self[triple_ids]
    
    def __len__(self) -> int:
        return self.nb_batches
    
    def _get_nb_batches(self, nb_triples, batch_size, drop_last):
        nb = nb_triples / batch_size
        
        if nb % batch_size == 0: return int(nb)
        else: return floor(nb) if drop_last else ceil(nb)

class PathTrainingLoop(SLCWATrainingLoop):

    def __init__(
        self,
        path_sampler: PathSampler = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.path_sampler = path_sampler

    def _create_training_data_loader(
        self,
        triples_factory: CoreTriplesFactory,
        *,
        sampler: str,
        batch_size: int,
        drop_last: bool,
        **kwargs
    ) -> DataLoader:
        # shouldn't be taken into account by the DataLoader
        kwargs.pop("shuffle")
        
        return DataLoader(
            dataset=PathDataset(
                mapped_triples=triples_factory.mapped_triples,
                num_entities=triples_factory.num_entities,
                num_relations=triples_factory.num_relations,
                batch_size=batch_size,
                drop_last=drop_last,
                negative_sampler=self.negative_sampler,
                negative_sampler_kwargs=self.negative_sampler_kwargs,
                path_sampler=self.path_sampler
            ),
            # PyTorch's automatic batching is disabled (why?)
            batch_size=None,
            batch_sampler=None,
            **kwargs,
        )

    def _process_batch(
        self,
        batch: Any,
        start: int,
        stop: int,
        label_smoothing: float = 0,
        slice_size: int = None
    ) -> FloatTensor:
        mode = self.mode
        model = self.model
        loss = self.loss

        pos_triple_batch, neg_triple_batch, pos_path_batch, neg_path_batch = batch

        # for subbatching
        pos_triple_batch = pos_triple_batch[start:stop].to(device=model.device)
        neg_triple_batch = neg_triple_batch[start:stop]

        # linearize all negs for scoring (expects [batch_size, 3]) 
        neg_triple_shape = neg_triple_batch.shape[:-1]
        neg_triple_batch = neg_triple_batch.view(-1, 3)
        # should have already be done by NegativeSampler
        neg_triple_batch = neg_triple_batch.to(model.device)

        # compute scores for triples
        pos_scores = model.score_hrt(pos_triple_batch, mode=mode)
        neg_scores = model.score_hrt(neg_triple_batch, mode=mode).view(*neg_triple_shape)

        # compute sccores for paths (2p)
        if pos_path_batch != None:
            pos_path_batch = pos_path_batch[start:stop]
            neg_path_batch = neg_path_batch[start:stop]
            
            pos_path_scores = model.score_hrt(pos_path_batch, mode=mode)
            pos_scores = cat((pos_scores, pos_path_scores))

            # linearize negs for scoring (same as for triples)
            neg_path_shape = neg_path_batch.shape[:-1]
            neg_path_batch = neg_path_batch.view(-1, 4)
            neg_path_batch = neg_path_batch.to(model.device)
            
            neg_path_scores = model.score_hrt(neg_path_batch, mode=mode).view(*neg_path_shape)
            neg_scores = cat((neg_scores, neg_path_scores))

        return (
            loss.process_slcwa_scores(
                positive_scores=pos_scores,
                negative_scores=neg_scores,
                label_smoothing=label_smoothing,
                # note: filtering of false negative not supported
                num_entities=model._get_entity_len(mode=mode),
            )
            + model.collect_regularization_term()
        )