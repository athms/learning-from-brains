#!/usr/bin/env python3

from torchviz import make_dot


def plot_model_graph(
    model,
    dataloader,
    path: str=None
    ) -> None:
    batch = next(iter(dataloader))
    outputs = model(batch=batch)['outputs']
    dot = make_dot(
        outputs,
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=True
    )
    if path is None:
        path = f'{model.name}_graph'
    dot.render(path)