import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tables import *
import os
os.environ['QT_XCB_GL_INTEGRATION'] = 'none'


def main():
    runs_dict = {
        0.25: {
            'er': ['63d06dda0258480e9275b18a61c3e7a0', '54a8c4be08874adcb7cfa59290322f8a', 'd2f3694cb4324520b684ea1445e279a3', '4287d76782b14e19815c6807c6d6c7be', 'd6e45200b3de4fcb9b80d65976c90751'],
            'full': ['439874c1c8844572ab49f9aee1f1774d', '551998af45a44ebda533ea870ffb7983', '2f60ebcbc4ed499faaedc7005905f4b4', 'f737868dd28f4d058c6d294fd5d80e68', '10bfbbf7ee954e1b8495bea16a358e1c'],
            'lwf': ['93e1907e382541b9a64ca40c626fcd5f', 'b045885a28174f39a6b761d92c2a26ab', 'ac269737061b4a159aba4a203c718966', '1bf5ddd62ca54847b10b37fd1089aca7', 'f02f61daa8f34fb7810099dbb8c74823'],
        },
        0.5: {
            'er': ['60a84e62de034bb0bb0dbb328908bc52', 'f02defd31e8b4cf99bc0754ba9a4129e', '3f4612b62ffb4a11902d1610a7ebcab0', '0841c9778a8148c4939950a998df7974', 'f4094541eff141cea7c041d40af830b9'],
            'full': ['b979bc134d79483a91b0b5cd2800209d', 'c775e35167df4bde939ce37b63e0676e', '8423b4441068402bbd3d4679bc0d6f50', 'edc14089837447f0b671d64556c2bcee', '27f7a908454449a9aa92e56d9f387ed3'],
            'lwf': ['48a491ffc74e4c06a90d60084d8d4f5a', 'd9714d333aa34fc680d2f20a14883ecf', '6bdc100a882b4980a16dc8ac7d2ccff2', '21d17fc1e1184361a4931ccde98354e3', '27abd0ae1d8d42448ab589dfb11ad44c'],
        },
        0.75: {
            'er': ['433322efe59442abbdaa90633c2cba22', '1e34f9c884e24f1db8d5a8fb50c5e59f', 'faf286229edf4d739c314e210dbe5a32', '3e8f2552651347f487dc1c9066d7667f', 'ff8615a42c9b4283bb6f0b9d4fc45cf1'],
            'full': ['65594a03a3634cb59d4a7ff97dc7a39f', 'b22827c12489416798e606669199868e', '738a429144ec46abab52e3214752f958', 'bb640bf93a37449cb4c38520143f9346', 'ab4d2654775d4e3a9d7aa44055d2aedb'],
            'lwf': ['7ca23ca5b07a4410a385861752d53a4b', '01a6c67df52e4a7eabf52500c9c2d368', 'ad28a296b0304b38b9128f04f3af19b0', '4da4f17d815e4924913283bf47aecdb6', '53ece4d6a2cd4e91815e5447c841c805']
        },
        0.9: {
            'er': ['915ea3dcea9e400880c2a287fe5c51e0', 'f00c477580f348ebb85671bc5b963689', 'fce8e803dabe4bbcafebdc8ae0c7ed63', 'f1c30205f74343eba45301698eaac2c5', '8cb3e41ca49043eca27fc33863c840b7'],
            'full': ['a382b6d73c434c1380b806eb39c161b8', '36750086f8a84fb99f6827ea6416994f', 'f5ad0eaa75d042648de9d3457426f95a', '0aaf985f37ce4cf0bc7ae7bc729fe52b', '8ce629295fdc4a30a7b70a85cb85dda9'],
            'lwf': ['e702c66202f04165ab80503f9f784bc5', 'e065e93d788345099c63c87e40b118da', 'a37240d03cc84ed093fa03945cc412ae', '4ddde3026ade4ca38495329959549375', '5df2d7e897c34a9491e06f758eda0570']
        },
    }
    run_experiments = {
        'er': '976651693442159172',
        'full': '0',
        'lwf': '0',
    }

    plot_results(runs_dict[0.25], run_experiments)
    run_experiments['er'] = '0'
    plot_results(runs_dict[0.5], run_experiments)
    plot_results(runs_dict[0.75], run_experiments)
    plot_results(runs_dict[0.9], run_experiments)
    plt.show()


def plot_results(runs_dict, run_experiments):
    algorithms = list(runs_dict.keys())
    n_tasks = 7
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
    plt.figure(figsize=(18, 4))

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
