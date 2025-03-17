import string
import torch

from torch.cuda.amp import autocast
from tqdm.auto import tqdm

CONCEPT_TASKS = list(string.ascii_uppercase)


def evaluate_model(eval_type, model, config, **opt_kwargs):
    """ Evaluate methods on arbitrary experiment kinds. """
    if opt_kwargs.get("opt_dataloader", None) is not None:
        loader = opt_kwargs["opt_dataloader"]
        num_classes = opt_kwargs["opt_classes"]
    else:
        loader = config['data']['test']['full']
        num_classes = len(config['data']['test']['class_names'])

    if eval_type == 'logits':
        acc_overall, acc_avg, perclass_acc = evaluate_logits_alltasks(
            model, loader,
            splits=config['dataset']['class_splits'],
            num_classes=num_classes
        )
    else:
        raise ValueError(f'Invalid eval_type: {eval_type}! Must be one of [logits, clip].')

    results = {'Joint': acc_overall, 'Per Task Avg': acc_avg}
    for task_idx, task_acc in enumerate(perclass_acc):
        results[f'Task {CONCEPT_TASKS[task_idx]}'] = task_acc

    return results


def evaluate_logits_alltasks(model, loader, splits, num_classes):
    model.eval()
    correct = 0
    total = 0

    splits = [list(split) for split in splits]

    totals = [0] * num_classes
    corrects = [0] * num_classes

    device = get_device(model)

    all_splits = torch.tensor(sum(splits, [])).to(device)

    task_map = {}
    for i, split in enumerate(splits):
        for _cls in split:
            task_map[_cls] = i

    task_map = [task_map[_cls] if _cls in task_map else -1 for _cls in range(num_classes)]
    task_map = torch.tensor(task_map).to(device)

    splits = torch.tensor(splits).to(device)

    with torch.no_grad(), autocast():
        for inputs, labels in tqdm(loader, 'Evaluating multihead head model'):
            inputs, labels = inputs.to(device), labels.to(device)

            class_selector = torch.isin(labels, all_splits)
            inputs, labels = inputs[class_selector, :, :, :], labels[class_selector]

            batch_size = inputs.shape[0]
            if batch_size == 0:
                continue

            task_idx = task_map[labels]
            outputs = model(inputs)

            if isinstance(outputs, list):
                # Filter out predictions on classes not trained
                for i in range(len(outputs)):
                    exclude_labels = torch.tensor([l for l in all_splits.cpu().numpy() if l not in splits[i].cpu().numpy()], device=all_splits.device).to(torch.long)
                    outputs[i][:, exclude_labels] = -torch.inf
                outputs = torch.stack(outputs, dim=1)
                outputs2 = outputs.softmax(dim=-1).to(outputs.dtype).max(dim=-2)[0]
                outputs2[:, all_splits] += 2
                outputs = outputs[range(batch_size), task_idx, :]
            else:
                outputs2 = outputs.clone()
                for i in range(splits.shape[0]):
                    outputs2[:, splits[i]] = torch.softmax(outputs2[:, splits[i]], dim=-1).to(outputs.dtype) + 2
            outputs2 = outputs2.argmax(dim=-1)

            task_splits = splits[task_idx, :]
            outputs = outputs.gather(dim=-1, index=task_splits).argmax(dim=-1)
            outputs = task_splits.gather(dim=-1, index=outputs[:, None])[:, 0]

            for gt, p, p2 in zip(labels, outputs, outputs2):
                totals[gt] += 1

                if gt == p:
                    corrects[gt] += 1
                if gt == p2:
                    correct += 1

                total += 1

    split_accs = [0] * len(splits)

    for i, split in enumerate(splits):
        split_total = 0
        for _cls in split:
            split_accs[i] += corrects[_cls]
            split_total += totals[_cls]
        split_accs[i] /= max(split_total, 1e-4)

    return correct / total, sum(split_accs) / len(split_accs), split_accs


def get_device(model):
    """Get the device of the model."""
    return next(iter(model.parameters())).device
