import os
import torch
from copy import deepcopy
from inspect import getmembers, isfunction
from tqdm.auto import tqdm

from collections.abc import MutableMapping
from .metric_calculators import get_metric_fns


def get_merging_fn(name):
    """ Get alignment function from name. """
    # import matching_functions as matching_functions
    from . import matching_functions
    matching_fns = dict([(k, v) for (k, v) in getmembers(matching_functions, isfunction) if 'match_tensors' in k])
    return matching_fns[name]


def inject_pair(config, pair, ignore_bases=False):
    model_name = config['model']['name']
    config['dataset']['class_splits'] = [split_str_to_ints(split) for split in pair]
    if not ignore_bases:
        config['model']['bases'] = [os.path.join(config['model']['dir'], split, f'{model_name}_v0.pt') for split in pair]
    return config


def split_str_to_ints(split):
    return [int(i) for i in split.split('_')]


def prepare_experiment_config(config, data):
    """ Load all functions/classes/models requested in config to experiment config dict. """
    # data = prepare_data(config['dataset'], device=config['device'])
    if config['eval_type'] == 'logits':
        config['model']['output_dim'] = len(data['test']['class_names'])
    else:
        config['model']['output_dim'] = 512
    new_config = {
        'graph': prepare_graph(config['model']),
        'data': data,
        'models': prepare_models(config['model'], device=config['device']),
        'merging_fn': get_merging_fn(config['merging_fn']),
        'metric_fns': get_metric_fns(config['merging_metrics']),
    }
    # Add outstanding elements
    for key in config:
        if key not in new_config:
            new_config[key] = config[key]
    return new_config


def get_config_from_name(name, device=None):
    """ Load config based on its name. """
    # out = deepcopy(getattr(__import__('configs.' + name), name).config)
    out = {
        'dataset': {
            'name': 'cifar50',
            'shuffle_train': True
        },
        'model': {
            'name': 'resnet20x1',  # TODO it was resnet20x8
            'dir': './checkpoints/',
            'bases': []
        },
        'merging_fn': 'match_tensors_zipit',
        'eval_type': 'logits',  # 'clip',
        'merging_metrics': ['covariance', 'mean'],
    }
    if device is None and 'device' not in out:
        out['device'] = 'cuda'
    elif device is not None:
        out['device'] = device
    return out


def prepare_graph(config):
    """ Get graph class of experiment models in config. """
    if config['name'].startswith('resnet'):
        model_name = config['name'].split('x')[0]
        from .graphs import resnet_graph as graph_module
        graph = getattr(graph_module, model_name)
    else:
        raise NotImplementedError(config['name'])
    return graph


def prepare_models(config, device='cuda'):
    """ Load all pretrained models in config. """
    if config['name'].startswith('resnet'):
        return prepare_resnets(config, device)
    else:
        raise NotImplementedError(config['name'])


def prepare_resnets(config, device):
    """ Load all pretrained resnet models in config. """
    bases = []
    name = config['name']

    if 'x' in name:
        width = int(name.split('x')[-1])
        name = name.split('x')[0]
    else:
        width = 1

    if 'resnet20' in name:
        # from backbones.resnet import resnet18 as wrapper_w  # TODO check what is thier implementation of resnet
        from models.resnets import resnet20 as wrapper_w
        def wrapper(num_classes): return wrapper_w(width, num_classes)
    elif 'resnet50' in name:
        from torchvision.models import resnet50 as wrapper
    elif 'resnet18' in name:
        from torchvision.models import resnet18 as wrapper
    else:
        raise NotImplementedError(config['name'])

    output_dim = config['output_dim']
    for base_path in tqdm(config['bases'], desc="Preparing Models"):
        base_sd = torch.load(base_path, map_location=torch.device(device))

        # Remove module for dataparallel
        for k in list(base_sd.keys()):
            if k.startswith('module.'):
                base_sd[k.replace('module.', '')] = base_sd[k]
                del base_sd[k]

        base_model = wrapper(num_classes=output_dim).to(device)
        base_model.load_state_dict(base_sd)
        bases.append(base_model)
    new_model = wrapper(num_classes=output_dim).to(device)
    return {
        'bases': bases,
        'new': new_model  # this will be the merged model
    }


def flatten_nested_dict(d, parent_key='', sep='_'):
    """Flatten a nested dictionary. {a: {b: 1}} -> {a_b: 1}"""
    # https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
