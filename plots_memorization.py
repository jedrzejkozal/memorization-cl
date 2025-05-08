import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tables import *
import os
os.environ['QT_XCB_GL_INTEGRATION'] = 'none'


def main():
    runs_dict = {
        'er': ['63d06dda0258480e9275b18a61c3e7a0', '54a8c4be08874adcb7cfa59290322f8a', 'd2f3694cb4324520b684ea1445e279a3', '4287d76782b14e19815c6807c6d6c7be', 'd6e45200b3de4fcb9b80d65976c90751'],
        'full': ['439874c1c8844572ab49f9aee1f1774d', '551998af45a44ebda533ea870ffb7983', '2f60ebcbc4ed499faaedc7005905f4b4', 'f737868dd28f4d058c6d294fd5d80e68', '10bfbbf7ee954e1b8495bea16a358e1c'],
        'lwf': ['93e1907e382541b9a64ca40c626fcd5f', 'b045885a28174f39a6b761d92c2a26ab', 'ac269737061b4a159aba4a203c718966', '1bf5ddd62ca54847b10b37fd1089aca7', 'f02f61daa8f34fb7810099dbb8c74823'],
    }
    run_experiments = {
        'er': '976651693442159172',
        'full': '0',
        'lwf': '0',
    }

    algorithms = list(runs_dict.keys())
    n_tasks = 7

    mlruns_path = '///home/jkozal/Documents/PWr/memorization-cl/memorization-cl/mlruns/'

    results = {a: [] for a in algorithms}
    results_std = {a: [] for a in algorithms}
    results_longtail = {a: [] for a in algorithms}
    results_longtail_std = {a: [] for a in algorithms}
    for algorithm_name in algorithms:
        run_ids = runs_dict[algorithm_name]
        experiment_id = run_experiments[algorithm_name]
        for task_id in range(n_tasks):
            task_accs = []
            longtail_accs = []
            for run_id in run_ids:
                acc, acc_long = get_task_acc(run_id, experiment_id, task_id)
                task_accs.append(acc)
                longtail_accs.append(acc_long)

            task_acc, task_std = reduction(task_accs)
            results[algorithm_name].append(task_acc)
            results_std[algorithm_name].append(task_std)

            long_acc, long_std = reduction(longtail_accs)
            results_longtail[algorithm_name].append(long_acc)
            results_longtail_std[algorithm_name].append(long_std)
    print(results)
    # exit()

    color_range = plt.cm.tab10.colors

    plt.subplot(1, 3, 1)
    for task_idx, color in zip(range(7), color_range):
        selected_task_acc = results['er'][task_idx]
        selected_lt_acc = results_longtail['er'][task_idx]

        task_x = list(range(task_idx, 10))
        plt.plot(task_x, selected_task_acc, label=f'Task {task_idx}', c=color)
        plt.plot(task_x, selected_lt_acc, '--', c=color)
        plt.xlabel('Task')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.subplot(1, 3, 2)
    for task_idx, color in zip(range(7), color_range):
        selected_task_acc = results['full'][task_idx]
        selected_lt_acc = results_longtail['full'][task_idx]

        task_x = list(range(task_idx, 10))
        plt.plot(task_x, selected_task_acc, label=f'Task {task_idx}', c=color)
        plt.plot(task_x, selected_lt_acc, '--', c=color)
        plt.xlabel('Task')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.subplot(1, 3, 3)
    for task_idx, color in zip(range(7), color_range):
        selected_task_acc = results['lwf'][task_idx]
        selected_lt_acc = results_longtail['lwf'][task_idx]

        task_x = list(range(task_idx, 10))
        plt.plot(task_x, selected_task_acc, label=f'Task {task_idx}', c=color)
        plt.plot(task_x, selected_lt_acc, '--', c=color)
        plt.xlabel('Task')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.show()


def get_task_acc(run_id, experiment_id, task_id):
    acc = read_full_metric(experiment_id, run_id, f'acc_class_il_task_{task_id}')
    acc_long = read_full_metric(experiment_id, run_id, f'acc_longtail_task_{task_id}')
    return acc, acc_long


def read_full_metric(experiment_id, run_id, metric_name):
    run_path = pathlib.Path(f'mlruns/{experiment_id}/{run_id}/metrics/')
    filepath = run_path / metric_name
    metric_values = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            acc_str = line.split()[-2]
            acc = float(acc_str)
            metric_values.append(acc)
    return metric_values


def reduction(metrics):
    metrics = np.array(metrics)
    avrg = metrics.mean(axis=0)
    std = metrics.std(axis=0)
    return avrg, std


if __name__ == '__main__':
    main()
