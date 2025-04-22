import pathlib
import numpy as np
import mlflow
import tabulate


def main():
    standard_benchmarks()


def standard_benchmarks():
    runs_standard_benchmarks = {
        'cifar10': {
        },
        'cifar100': {
            # 'Joint': ['340276f083b64cc49dccd2d69c9ad66a', 'ef826a5b9e334f4b92cc3fa695847b8a', '663bcd4b333248a5b559a01e376585dd', 'b008c7833f404dfca446753cefdcdf1e', '4f53e6af47174940b04fa70d618af51a'],
            # 'Finetuning': ['17b3122b6a5f420b9661e004af2a0588', '93980b3382fd438fb84635aead2e89df', '7b026e572ccd47938ce600c8b76248a8', 'f7f8eba20dbb454d80df3c4f2fa00c8d', '5d0ab7d56d3746d08d5d7c98cf3232ff'],
            # 'oEWC': ['191280d66c724ba390016d49b4544ce8', 'befd7e8a18604801a486d69f45d8e0ac', 'b0c5e9759ad64040ac330ddf1be4382a', '0d2b0c75b79e4a0997cc1b0168300965', '7e39d2481c5f4c3ea57e97df906006aa'],
            # 'SI': ['47d5e185871d4f8184f80ae580807906', '75d48d891fe64254b4bc5e252c5aa7c1', 'db800f218e5848b8b7cb17978bd85f84', 'fc99b0f9ff0f41329966ff78b507a148', '0f10cd9068fe4654ac90a25fd3a2d97a'],
            # 'iCARL': ['4dd5033dfda04f6a98b646774b1d8229', 'b3639bf984824afd88e9aec812290dce', '41737907295f43cfa6696761f2455fad', 'e9e2562273844d2f9ac8a8c8ccb507f4', '84de7552699e4beda7196121027622c5'],
            # 'ER': ['46fda71735f64c8da98d050960dd3cd9', '7835e0307ca143689620fced42f62c9d', '2239aa1376dd4467852af4347adf4811', 'fe4136cbe534423ab257fcf0e8914300', 'be01e5f0683f410885922ef516eb5591'],
            'reservoir': None,
            'reservoir balanced': ['bebddf41427b405ab6f62ffb47859ad2', '891d5218513c4806afea65e972a2a3cc', 'e2836045f5d7423aad78e9525ef1a6ad', '6f0e770354cf496cb82a0b258d4d36b0', '0710078d0c37486c9caf39e8ee5186c5'],
            'rainbow memory': None,
            'CLIB': None,
            'GSS': None,
            'GCR': None,
            'goldilocks': None,
            'bottom-k memscores': ['20ee555f00694b06ada8cef55625c42a', 'f6055777a13c4e9499c3e1c8a272d9bb', '9e1dffae9cdd4b2ebe6a9dc3ba1cdf08', '140893946fc746e2960fc3935900d911', '975375c2d35d496583b66c3a784a7930'],
            'middle-k memsocres': ['2204230c9dfa41dcb367e31b0e32c28e', '1be9cdd652424198a54f9216e460688e', 'a09a00b7f3fe41d189867a164613550e', 'e5c0e64a68b74d6e9dd6d9d06dfda5b6', '4ceb856389574ef88a02c1bf32674e1b'],
            'top-k memscores': ['a5be730e0fc64327b0d9e39eb02d3393', '823af178c35d4c40aa3bcecbf932dcf2', '03a6b3da5bdb4c4aae142b17f30a3523', '51b7604757d84f47a6957ca7f0340848', 'cb51df5347474927b9bf98165334b6b4'],
        },
        'tiny-imagenet': {
        }
    }

    # assert runs_standard_benchmarks['cifar10'].keys() == runs_standard_benchmarks['cifar100'].keys() == runs_standard_benchmarks['tiny-imagenet'].keys()
    algorithms = list(runs_standard_benchmarks['cifar100'].keys())

    mlruns_path = '///home/jkozal/Documents/PWr/memorization-cl/memorization-cl/mlruns/'
    client = mlflow.tracking.MlflowClient(mlruns_path)

    dataset_experiments = {
        # 'cifar10': '',
        'cifar100': '976651693442159172',
        # 'tiny-imagenet': ''
    }

    dataset_list = ['cifar100',]  # ['cifar10', 'cifar100', 'tiny-imagenet']
    dataset_n_tasks = [10]  # [5, 10, 20]

    table = list()
    for algorithm_name in algorithms:
        row = list()
        row.append(algorithm_name)
        for dataset, n_tasks in zip(dataset_list, dataset_n_tasks):
            run_ids = runs_standard_benchmarks[dataset][algorithm_name]
            experiment_id = dataset_experiments[dataset]
            metrics = calc_average_metrics(run_ids, client, experiment_id, n_tasks=n_tasks, digits=2)
            row.extend(metrics[:-1])
        table.append(row)

    tab_latex = tabulate.tabulate(table, tablefmt="latex", headers=['method', 'acc', 'FM',])
    tab_latex = tab_latex.replace('\\textbackslash{}', '\\')
    tab_latex = tab_latex.replace('\\{', '{')
    tab_latex = tab_latex.replace('\\}', '}')
    print(tab_latex)
    print("\n\n")


def calc_average_metrics(dataset_run_ids, client, experiment_id, n_tasks=20, digits=3):
    if dataset_run_ids == None:
        return '-', '-', '-'

    acc_all = []
    fm_all = []
    last_task_acc_all = []
    for run_id in dataset_run_ids:
        acc = get_metrics(run_id, client)
        acc_all.append(acc)
        fm = calc_forgetting_measure(run_id, client, experiment_id=experiment_id, num_tasks=n_tasks)  # TODO fix logging num_tasks in experiments
        fm_all.append(fm)
        last_task_acc = get_last_task_acc(run_id, client, experiment_id=experiment_id, num_tasks=n_tasks)
        last_task_acc_all.append(last_task_acc)

    avrg_acc, acc_std = rounded_reduction(acc_all, digits=digits)
    acc = f'{avrg_acc}±{acc_std}'
    avrg_fm, fm_std = rounded_reduction(fm_all, digits=digits)
    forgetting = f'{avrg_fm}±{fm_std}'
    avrg_last_acc, last_acc_std = rounded_reduction(last_task_acc_all, digits=digits)
    last_acc = f'{avrg_last_acc}±{last_acc_std}'
    return acc, forgetting, last_acc


def get_metrics(run_id, client):
    run = client.get_run(run_id)
    run_metrics = run.data.metrics
    acc = run_metrics['mean_acc_class_il']
    return acc


def rounded_reduction(metrics, digits=4):
    metrics = np.array(metrics)
    avrg = metrics.mean()
    avrg = round(avrg, digits)
    std = metrics.std()
    std = round(std, digits)
    return avrg, std


def calc_forgetting_measure(run_id, client, experiment_id, num_tasks=None):
    run_path = pathlib.Path(f'mlruns/{experiment_id}/{run_id}/metrics/')
    if num_tasks is None:
        run = client.get_run(run_id)
        num_tasks = run.data.params['n_experiences']
        num_tasks = int(num_tasks)

    fm = 0.0

    for task_id in range(num_tasks):
        filepath = run_path / f'acc_class_il_task_{task_id}'
        task_accs = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                acc_str = line.split()[-2]
                acc = float(acc_str)
                task_accs.append(acc)

        fm += abs(task_accs[-1] - max(task_accs))

    fm = fm / num_tasks
    return fm


def get_last_task_acc(run_id, client, experiment_id, num_tasks=None):
    run_path = pathlib.Path(f'mlruns/{experiment_id}/{run_id}/metrics/')
    if num_tasks is None:
        run = client.get_run(run_id)
        num_tasks = run.data.params['n_experiences']
        num_tasks = int(num_tasks)

    filepath = run_path / f'acc_class_il_task_{num_tasks-1}'
    with open(filepath, 'r') as f:
        for line in f.readlines():
            acc_str = line.split()[-2]
            last_acc = float(acc_str)

    return last_acc


if __name__ == '__main__':
    main()
