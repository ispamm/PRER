from typing import Union

import logging

import itertools
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from continual_ai.cl_strategies import NaiveMethod, Container
from .utils import qp
from continual_ai.utils import ExperimentConfig


class GradientEpisodicMemory(NaiveMethod):
    """
    @inproceedings{lopez2017gradient,
      title={Gradient episodic memory for continual learning},
      author={Lopez-Paz, David and Ranzato, Marc'Aurelio},
      booktitle={Advances in Neural Information Processing Systems},
      pages={6467--6476},
      year={2017}
    }
    """

    def __init__(self, config: ExperimentConfig, logger: logging.Logger = None,
                 random_state: Union[np.random.RandomState, int] = None,
                 **kwargs):

        super().__init__()

        gem_config = config.cl_technique_config
        self.margin = gem_config.get('margin', 0.5)
        self.task_memory_size = gem_config.get('task_memory_size', 500)

        if random_state is None or isinstance(random_state, int):
            self.RandomState = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.RandomState = random_state
        else:
            raise ValueError("random_state can be None, Int or numpy RandomState object, an {} was give"
                             .format(type(random_state)))

        if logger is not None:
            logger.info('GEM parameters:')
            logger.info(F'\tMargin: {self.margin}')
            logger.info(F'\tTask memory size: {self.task_memory_size}')

        self.task_memory = []
        self.loss_f = nn.CrossEntropyLoss(reduction='mean')

    def on_task_ends(self, container: Container, *args, **kwargs):

        task = container.current_task

        task.train()
        _, images, labels = task.sample(size=self.task_memory_size)

        self.task_memory.append((images.detach(), labels.detach()))

    def after_back_propagation(self, container: Container, *args, **kwargs):

        if len(self.task_memory) > 0:
            named_parameters = dict(itertools.chain(container.encoder.named_parameters(),))
                                                    # container.solver.named_parameters()))
            current_gradients = {}

            for n, p in named_parameters.items():
                if p.requires_grad and p.grad is not None:
                    current_gradients[n] = deepcopy(p.grad.data.view(-1).cpu())

            tasks_gradients = {}

            for i, t in enumerate(self.task_memory):

                container.encoder.train()
                container.solver.train()

                container.encoder.zero_grad()
                container.solver.zero_grad()


                image, label = t

                emb = container.encoder(image)
                o = container.solver(emb, task=i)

                loss = self.loss_f(o, label)

                loss.backward()

                gradients = {}
                for n, p in named_parameters.items():
                    if p.requires_grad and p.grad is not None:
                        gradients[n] = p.grad.data.view(-1).cpu()

                tasks_gradients[i] = deepcopy(gradients)

            container.encoder.zero_grad()
            container.solver.zero_grad()
            done = False

            for n, cg in current_gradients.items():
                tg = []
                for t, tgs in tasks_gradients.items():
                    tg.append(tgs[n])

                tg = torch.stack(tg, 1).cpu()
                a = torch.mm(cg.unsqueeze(0), tg)

                if (a < 0).sum() != 0:
                    done = True
                    cg_np = cg.unsqueeze(1).cpu().contiguous().numpy().astype(np.double)
                    tg = tg.numpy().transpose().astype(np.double)

                    try:
                        v = qp(tg, cg_np, self.margin)

                        cg_np += np.expand_dims(np.dot(v, tg), 1)

                        del tg

                        p = named_parameters[n]
                        p.grad.data.copy_(torch.from_numpy(cg_np).view(p.size()))

                    except Exception as e:
                        print(e)

            if not done:
                for n, p in named_parameters.items():
                    if p.requires_grad and p.grad is not None:
                        p.grad.copy_(current_gradients[n].view(p.grad.data.size()).cpu())
