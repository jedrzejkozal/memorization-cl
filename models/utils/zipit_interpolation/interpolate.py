import os
import torch
import torch.nn as nn

from tqdm.auto import tqdm
from copy import deepcopy

from .utils import inject_pair, prepare_experiment_config, get_config_from_name, flatten_nested_dict, prepare_graph, get_merging_fn, get_metric_fns
from .model_merger import ModelMerge
from .evaluate import evaluate_model


def interpolate1(source_network, premutation_nework, train_loader, device, alpha=0.5, permuation_epochs=1, batchnorm_epochs=1) -> nn.Module:
    config = {
        'dataset': {
            'name': 'cifar50',
            'shuffle_train': True
        },
        'model': {
            'name': 'resnet18x1',  # 'resnet20x1',
            'dir': './checkpoints/',
            'bases': [source_network, premutation_nework],
            'output_dim': 100,
        },
        'merging_fn': 'match_tensors_zipit',
        'eval_type': 'logits',  # 'clip',
        'merging_metrics': ['covariance', 'mean'],
        'device': device,
        'data': {
            'train': {'full': train_loader},
            'test': {
                'full': train_loader,
                'class_names': list(range(100)),
            },
        },
        'models': {
            'bases': [source_network, premutation_nework],
            'new': deepcopy(source_network)  # new_model  # this will be the merged model
        }
    }
    config['graph'] = prepare_graph(config['model'])
    config['merging_fn'] = get_merging_fn(config['merging_fn'])
    config['metric_fns'] = get_metric_fns(config['merging_metrics'])

    node_config = {
        'stop_node': 21,
        'params': {'a': .01, 'b': .1}  # {'a': .0001, 'b': .075}
    }

    base_models = [reset_bn_stats(base_model, train_loader) for base_model in config['models']['bases']]

    Grapher = config['graph']
    graphs = [Grapher(deepcopy(base_model)).graphify() for base_model in base_models]

    Merge = ModelMerge(*graphs, device=device)

    Merge.transform(
        deepcopy(config['models']['new']),
        train_loader,
        transform_fn=config['merging_fn'],
        metric_classes=config['metric_fns'],
        stop_at=node_config['stop_node'],
        **node_config['params']
    )

    reset_bn_stats(Merge, train_loader)

    return config['models']['new']


def interpolate(device, dataloader):
    config_name = 'cifar50_resnet20'
    experiment_configs = [
        # best experiment
        {'stop_node': 21, 'params': {'a': .0001, 'b': .075}},
        # {'stop_node': 21, 'params':{'a': 1., 'b': 1.}},
        # {'stop_node': None, 'params':{'a': 0.01, 'b': 1.0}, 'dataset': {'train_fraction': .0001, 'no_transform': False}},
        # Alpha Ablations
        # {'stop_node': None, 'params': {'a': .0, 'b': 1.}},
    ]

    data = {
        'train': {'full': dataloader},
        'test': {
            'full': dataloader,
            'class_names': list(range(100)),
        },
    }

    raw_config = get_config_from_name(config_name, device=device)
    model_dir = raw_config['model']['dir']
    model_name = raw_config['model']['name']
    # run_pairs = find_runable_pairs(model_dir, model_name, skip_pair_idxs=skip_pair_idxs)
    run_pairs = [('50_50', '50_51')]
    print(raw_config['merging_fn'])

    with torch.no_grad():
        for node_config in experiment_configs:
            raw_config['dataset'].update(node_config.get('dataset', {}))
            run_node_experiment(
                node_config=node_config,
                experiment_config=raw_config,
                pairs=run_pairs,
                device=device,
                raw_config=raw_config,
                data=data
            )


def run_node_experiment(node_config, experiment_config, pairs, device, raw_config, data):
    for pair in tqdm(pairs, desc='Evaluating Pairs...'):
        experiment_config = inject_pair(experiment_config, pair)
        config = prepare_experiment_config(raw_config, data)
        train_loader = config['data']['train']['full']
        base_models = [reset_bn_stats(base_model, train_loader) for base_model in config['models']['bases']]
        config['node'] = node_config

        Grapher = config['graph']
        graphs = [Grapher(deepcopy(base_model)).graphify() for base_model in base_models]
        print(f'Graphs: {graphs}')

        Merge = ModelMerge(*graphs, device=device)

        Merge.transform(
            deepcopy(config['models']['new']),
            train_loader,
            transform_fn=config['merging_fn'],
            metric_classes=config['metric_fns'],
            stop_at=node_config['stop_node'],
            **node_config['params']
        )

        reset_bn_stats(Merge, train_loader)

        results = evaluate_model(experiment_config['eval_type'], Merge, config)
        # for idx, split in enumerate(pair):
        #     results[f'Split {CONCEPT_TASKS[idx]}'] = split
        results['Time'] = Merge.compute_transform_time
        results['Merging Fn'] = config['merging_fn'].__name__
        results['Model Name'] = config['model']['name']
        results.update(flatten_nested_dict(node_config, sep=' '))
        print(results)

    print(f'Results of {node_config}: {results}')
    return results


# use the train loader with data augmentation as this gives better results
# taken from https://github.com/KellerJordan/REPAIR
def reset_bn_stats(model, loader, reset=True):
    """Reset batch norm stats if nn.BatchNorm2d present in the model."""
    device = get_device(model)
    has_bn = False
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) == nn.BatchNorm2d:
            if reset:
                m.momentum = None  # use simple average
                m.reset_running_stats()
            has_bn = True

    if not has_bn:
        return model

    # run a single train epoch with augmentations to recalc stats
    model.train()
    with torch.no_grad():
        for images, _ in tqdm(loader, desc='Resetting batch norm'):
            _ = model(images.to(device))
    return model


def get_device(model):
    """Get the device of the model."""
    return next(iter(model.parameters())).device


@torch.no_grad()
def evaluate(network, test_loaders, device):
    status = network.training
    network.eval()
    network.to(device)
    accs = []
    for test_loader in test_loaders:
        correct, total = 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = network(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

        accs.append(correct / total * 100)

    network.train(status)
    return accs
