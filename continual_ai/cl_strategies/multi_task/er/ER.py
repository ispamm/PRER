import logging
import numpy as np

from typing import Union

import torch

import torch.nn.functional as F

from continual_ai.cl_strategies import NaiveMethod, Container
from continual_ai.iterators import Sampler
from continual_ai.utils import ExperimentConfig


class EmbeddingRegularization(NaiveMethod):
    """
    @article{POMPONI2020,
    title = "Efficient continual learning in neural networks with embedding regularization",
    journal = "Neurocomputing",
    year = "2020",
    issn = "0925-2312",
    doi = "https://doi.org/10.1016/j.neucom.2020.01.093",
    url = "http://www.sciencedirect.com/science/article/pii/S092523122030151X",
    author = "Jary Pomponi and Simone Scardapane and Vincenzo Lomonaco and Aurelio Uncini",
    keywords = "Continual learning, Catastrophic forgetting, Embedding, Regularization, Trainable activation functions",
    }
    """
    def __init__(self, config: ExperimentConfig, logger: logging.Logger=None,
                 random_state: Union[np.random.RandomState, int] = None, **kwargs):

        NaiveMethod.__init__(self)
        config = config.cl_technique_config

        self.memorized_task_size = config.get('task_memory_size', 300)
        self.sample_size = min(config.get('sample_size', 100), self.memorized_task_size)
        self.importance = config.get('penalty_importance', 1)
        self.distance = config.get('distance', 'cosine')
        # self.supervised = config.get('supervised', True)
        self.normalize = config.get('normalize', False)
        self.batch_size = config.get('batch_size', 25)

        if random_state is None or isinstance(random_state, int):
            self.RandomState = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.RandomState = random_state
        else:
            raise ValueError("random_state can be None, Int or numpy RandomState object, an {} was give"
                             .format(type(random_state)))

        if logger is not None:
            logger.info('ER parameters:')
            logger.info(F'\tMemorized task size: {self.memorized_task_size}')
            logger.info(F'\tSample size: {self.sample_size}')
            logger.info(F'\tPenalty importance: {self.importance}')
            logger.info(F'\tDistance: {self.distance}')
            logger.info(F'\tNormalize: {self.normalize}')

        self.task_memory = []

    def on_task_ends(self, container: Container, *args, **kwargs):

        task = container.current_task

        task.train()

        _, images, _ = task.sample(size=self.memorized_task_size)

        container.encoder.eval()

        embs = container.encoder(images)
        if self.normalize:
            embs = F.normalize(embs, p=2, dim=1)

        m = list(zip(images.detach(), embs.detach()))

        self.task_memory.append(m)

    def before_gradient_calculation(self, container: Container, *args, **kwargs):

        if len(self.task_memory) > 0:

            to_back = []
            loss = 0

            for t in self.task_memory:

                b = Sampler(t, dimension=self.sample_size, replace=False, return_indexes=False)()
                image, embeddings = zip(*b)

                image = torch.stack(image)
                embeddings = torch.stack(embeddings)

                new_embedding = container.encoder(image)

                if self.normalize:
                    new_embedding = F.normalize(new_embedding, p=2, dim=1)

                if self.distance == 'euclidean':
                    dist = (embeddings - new_embedding).norm(p=None, dim=1)
                elif self.distance == 'cosine':
                    cosine = torch.nn.functional.cosine_similarity(embeddings, new_embedding, dim=1)
                    dist = 1 - cosine
                else:
                    assert False

                loss += dist.mean() * self.importance


            container.current_loss += loss
