import pathlib
import numpy as np
import mlflow
import tabulate


def main():
    standard_benchmarks()
    # standard_methods()


def standard_benchmarks():
    runs_standard_benchmarks = {
        'cifar10': {
            'reservoir': ['9869e957ffab48ed9148a0cba50361a1', 'dd0daf585b7847da89d26ae71caca579', '8c806ffb3fbe44fa9ad0cf3f709e7845', '6add5478768e4b059d5e364ec4fdf6a2', 'c03c8993f0b148d2a3f62144952ed06e'],
            'reservoir balanced': ['c99bab55574646bea5eabe2de592828c', '77ce71763f9f4962b99457876c700ab1', '97b3f3836bce44e1ad3d82e50cdf646b', '2f693d34665d4675a3df4db5c3cc7c43', '60a1794a3f4e41f584109f749b84efcd'],
            'rainbow memory': ['3ac8ea471e6d4bfbbbaa18046d3b9419', '9a2586087bdd4172afc0362d069b1d39', '18d0a6ada98d458c9575991c26e74ed4', 'e95bb971354544f19de7a5ae369658ea', 'c15c31ad707240828aac0cffbdc4686c'],
            'CLIB': None,
            'GSS': None,
            'GCR': None,
            'PBCS': ['9705d7a5fa1046149ed2822d27f06f28', 'd66552597d9142908fc93f94ca3225e6'],
            'BCSR': ['d10b86c74914407a9c49c1d758bc98b9', 'f1f41e0970ab41e59b95be5f4291afb1', '9eea09d684114e0991ad8018ba8a62b5', '4997659f513243a3a0e27281cc904705', '6b6c65b709a546e9bf8c404228ea6e98'],
            'bottom-k memscores': ['c4fbde6ec2e745a1ae7b4698a13dead4', 'c6fe21fa552c4e5898bf883c10b0d306', '6bc9c2989221444aa53324b4ac2c5bee', '38d1aa29e1984d78b6847544e7391b1c', '54c64dc0d42d425b952f63bfdb008baf'],
            'middle-k memsocres': ['02956d57737142118196cb808c44a054', 'a0f294e605894d629e13623673701904', 'e8486e951df541149e3365c43eece0bd', 'd6536dbbd47a472ab4bd2b4dd6e61096', '9471fae61b4a436eaa26937459f0600e'],
            'top-k memscores': ['5c49010705db4a2cba48e39129853bc8', 'ab428c0e241743b7a4a8bf1be45f5cee', 'aed3396ef8254f3d99f37deea093b063', 'ade66bfcfc5e4aaa8427cf00eb5262e7', '4edc25bc81fb48999d7c090af444130a'],
        },
        'cifar100': {
            # 'Joint': ['340276f083b64cc49dccd2d69c9ad66a', 'ef826a5b9e334f4b92cc3fa695847b8a', '663bcd4b333248a5b559a01e376585dd', 'b008c7833f404dfca446753cefdcdf1e', '4f53e6af47174940b04fa70d618af51a'],
            # 'Finetuning': ['17b3122b6a5f420b9661e004af2a0588', '93980b3382fd438fb84635aead2e89df', '7b026e572ccd47938ce600c8b76248a8', 'f7f8eba20dbb454d80df3c4f2fa00c8d', '5d0ab7d56d3746d08d5d7c98cf3232ff'],
            # 'oEWC': ['191280d66c724ba390016d49b4544ce8', 'befd7e8a18604801a486d69f45d8e0ac', 'b0c5e9759ad64040ac330ddf1be4382a', '0d2b0c75b79e4a0997cc1b0168300965', '7e39d2481c5f4c3ea57e97df906006aa'],
            # 'SI': ['47d5e185871d4f8184f80ae580807906', '75d48d891fe64254b4bc5e252c5aa7c1', 'db800f218e5848b8b7cb17978bd85f84', 'fc99b0f9ff0f41329966ff78b507a148', '0f10cd9068fe4654ac90a25fd3a2d97a'],
            # 'iCARL': ['4dd5033dfda04f6a98b646774b1d8229', 'b3639bf984824afd88e9aec812290dce', '41737907295f43cfa6696761f2455fad', 'e9e2562273844d2f9ac8a8c8ccb507f4', '84de7552699e4beda7196121027622c5'],
            # 'ER': ['46fda71735f64c8da98d050960dd3cd9', '7835e0307ca143689620fced42f62c9d', '2239aa1376dd4467852af4347adf4811', 'fe4136cbe534423ab257fcf0e8914300', 'be01e5f0683f410885922ef516eb5591'],
            'reservoir': ['63d06dda0258480e9275b18a61c3e7a0', '54a8c4be08874adcb7cfa59290322f8a', 'd2f3694cb4324520b684ea1445e279a3', '4287d76782b14e19815c6807c6d6c7be', 'd6e45200b3de4fcb9b80d65976c90751'],
            'reservoir balanced': ['bebddf41427b405ab6f62ffb47859ad2', '891d5218513c4806afea65e972a2a3cc', 'e2836045f5d7423aad78e9525ef1a6ad', '6f0e770354cf496cb82a0b258d4d36b0', '0710078d0c37486c9caf39e8ee5186c5'],
            'rainbow memory': ['e3dceca426d7480d903e102723880c17', '93f27b255025492193c43c7f03d69b92', '19840cdf7f554318954ff8df9cc6cd00', '438fb077921c4514a3fb731889efb6f0', 'f6977e5dce1a425fa7d5503ec2d730e2'],
            'CLIB': None,
            'GSS': None,
            'GCR': None,
            'PBCS': ['c396778f803d40078a21bca1b75b7991', '38eedda0fc6242dc99c710cfcacb2cf0', '75ceaf111b7a458e9ed1551332da62fd', 'b17fde4cbd744feb9871e03d8c3cea7f', '41067024d5954abd9b6fe9d4096dc022'],
            'BCSR': ['b338a354a0ef4563a078b90f10c6817e', '71804fb02b404a3a9080a6db28ef0481', 'eacd7a53856a4f8cb6cc1da69488292b', '27204094a76b4d61a9cf7cc2acb3d5c5', '5c64641876234bf4bc073b5e9bb6493f'],
            'bottom-k memscores': ['20ee555f00694b06ada8cef55625c42a', 'f6055777a13c4e9499c3e1c8a272d9bb', '9e1dffae9cdd4b2ebe6a9dc3ba1cdf08', '140893946fc746e2960fc3935900d911', '975375c2d35d496583b66c3a784a7930'],
            'middle-k memsocres': ['2204230c9dfa41dcb367e31b0e32c28e', '1be9cdd652424198a54f9216e460688e', 'a09a00b7f3fe41d189867a164613550e', 'e5c0e64a68b74d6e9dd6d9d06dfda5b6', '4ceb856389574ef88a02c1bf32674e1b'],
            'top-k memscores': ['a5be730e0fc64327b0d9e39eb02d3393', '823af178c35d4c40aa3bcecbf932dcf2', '03a6b3da5bdb4c4aae142b17f30a3523', '51b7604757d84f47a6957ca7f0340848', 'cb51df5347474927b9bf98165334b6b4'],
        },
        'tiny-imagenet': {
            'reservoir': ['84e655117d4847f3b974dee022ab659a', 'f06969fe7b7e4a849aa18ff102217c3c', 'e76f03d6a49f4c5e8df4f47c081fcac0', '8ca59c8ed7ce4779924a3a5790915eda', 'bd6158fd9fb84938b373749ff3a189e1'],
            'reservoir balanced': ['bafa5b7f16bd45fea0c7db4a1554f7d6', 'a5d541e1f405438b8eab4e0e8d5bc4ff', '881cef773361478d9541d8a319b3f26b', '34a413fad6854f8ea5eb9cdc1037b9da', '5d9f3f2216a440cdb6a6e98f555dfe57'],
            'rainbow memory': ['bd96c380ef2d4171913ac7ac37a96684', '8eab1078f38a48dfbc0e5027f8c45402', '5b5f5cea4d824534afe63e39f5d842df', '25b0c6e30c664d0d9bca5dbba2989d5f', 'fef4eeeca5e449609a8b26bd25a416da'],
            'CLIB': None,
            'GSS': None,
            'GCR': None,
            'PBCS': ['0ca36c65bf274a01bea5fb8d9d3831d9', '21d4f4269d8a4ccd90971564c8aea453'],
            'BCSR': ['ac4a154643d6443d82980852a24355c7', 'c076b493479d48688cc7b7d4b02c043c', 'cdea5779181a4b33887693bca8b1b816', 'c3f49ab29cb442d5bda0be05e11ad50f', 'c9cafc6c15ee40d2b7fba9264b2f6406'],
            'bottom-k memscores': ['8002808d66f84a27b0ad9358a65f28f7', '36470106ec524a87b3e0b79c0e155123', 'b0b31cec3fe045d6a894f8fed686d3da', '2afa12d886ac4152b0c9e4ef9d7423ff', '217cbd7d2ca64066bd3aa44f3d4d7f5a'],
            'middle-k memsocres': ['ce62375f56f640f3b54604ee05a7a822', 'd80a0a58ecca4f6ab532ccfbd3ec2626', '05a411228036473580bfa7018b2a4e2f', '11380ab8acaf465e8e75ec1f715f0319', '0867c505763043f484af3714838fb173'],
            'top-k memscores': ['642ec6bc7d984227ac3f16e54f8f5a1b', '8ba4fc428ffb486aacdd022ab0846d0e', '8aa7e387e47942d090c688ad78ccc6f1', '40b404a8276e4e108d0b300ca7fa5b28', 'bb1aa9161f45473cb4142a34ba31641e'],
        }
    }

    # assert runs_standard_benchmarks['cifar10'].keys() == runs_standard_benchmarks['cifar100'].keys() == runs_standard_benchmarks['tiny-imagenet'].keys()
    algorithms = list(runs_standard_benchmarks['cifar100'].keys())

    # mlruns_path = '///home/jkozal/Documents/PWr/memorization-cl/memorization-cl/mlruns/'
    mlruns_path = '///home/jedrzejkozal/Documents/memorization-cl/mlruns/'
    client = mlflow.tracking.MlflowClient(mlruns_path)

    dataset_experiments = {
        'cifar10': '899231754584608899',
        'cifar100': '976651693442159172',
        'tiny-imagenet': '112104018620214336'
    }

    dataset_list = ['cifar10', 'cifar100', 'tiny-imagenet']
    dataset_n_tasks = [5, 10, 20]  # [5, 10, 20]

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

    tab_latex = tabulate.tabulate(table, tablefmt="latex", headers=['method', 'acc', 'FM', 'acc', 'FM', 'acc', 'FM',])
    tab_latex = tab_latex.replace('\\textbackslash{}', '\\')
    tab_latex = tab_latex.replace('\\{', '{')
    tab_latex = tab_latex.replace('\\}', '}')
    print(tab_latex)
    print("\n\n")


def standard_methods():
    runs_standard_benchmarks = {
        'cifar100': {
            'er-ace': ['9994e279b2f24745869612aeaffab008', '57f90e0404b34bfc95e37f4ce47004c0', '0ca3305c1b63448d9a63b82038845597', 'af78e6556ae74498a997af78fb153d27', '4a43a60bf1d640a484c2751fcb4517c8'],
            'er-ace+bottom-k': ['6060bc368d8f4af6b35efe9ca6603385', '821a745afdb94c158a427b35a1da9df9', 'fad87309e0ec4495b693937fe0cb0203', '7adeba65ee304460be449b4b0887cd45', '00f490a1a4d543e09345457ae9c0e6e9'],
            'er-ace+mid-k': ['e2ba945c6b4a4e1dac737cd0e72b390e', 'e5f2f15edf28410daba24fa84b708be8', 'c3a18e260079419a90c1a18c309423e2', '911b14e66816447eaa31e552b9b736d5', '6ae6e41af81b43e9a00f0dcbe9c1781c'],
            'er-ace+top-k': ['e9a91f850cb940b3a2621510fe4ac9bb', '2331b808fca24197a077d3593b0814ff', '158d347d47814b778db3636e9e342e04', 'e1638921ddba4eb79dc149907ab2f4dd', '1ab63151959e4d108f53d91f892a2e4d'],
            'der++': ['01db0d89336549eb91aabb936e80f2af', '138714f6703e4d88ad3587796c4acb9e', 'a8f0f30242554ca5a5af0111ccc54471', '8e240a7b89aa48bcbd8c38063691440d', '7475416ff3294a5bb345ef25141d80f2'],
            # ['4e38a6508bda40f5bf744e28ada86d14', 'b01701d1c8a8437c9fa70fefea252225', 'c9c82233001145df856dc500db2903e6', '76fd7ab45890419eb6ead3ea9eb103b6'],
            'derpp+bottom-k': ['86e3fc75b0b3499fa0c016af79d5f75f', '31230b66885d43d081970d6f518231e7', '3b7ecf3b2e95439db6c10eff7d60f118', '4911a6d90c474181b69613e58a6950e4', '55a4cf25823745c6b938e33ae8aabd2e'],
            # ['a3e85fe4f5a54f17ba5d8f964563d969', 'fcf9f88d9e2c40269e32a25c6e632bc0', '8c4c85b22fe243e48ec8d0d8fa45e50f'],
            'derpp+mid-k': ['a330a0b78f304ee5bbbe08216e9beae8', '451cf69e852446e1899c3447e7669823', '31dc426298ca4587b5a1b0b8f4332d81', '5a0cab56e4cc4362bfd42725c181fe26', 'b45a3e237d2e4bf5958fd01deb520614'],
            # ['e44fc9cef8604273ad0c04847553de6b', '82ce32208bc443958bf5aefdfc2b2ea5', '0af44abf8a554dca94b7642a1c2a6ec6'],
            'derpp+top-k': ['74cb46824ded45539cb92a9379741f64', 'a23115a46c9d48fda4e8eff480ed14bc', '04289448afd24345b48a9376adc2dd72', 'd9e92ebaa6024c7dbde0040e1b07a26e', '0e1ea958d6fd4e9281f563925d3642b6'],
        },
        'tiny-imagenet': {
            'er-ace': ['7e5454c30d8d484e9fd35f0cf0267034', '7d8fd3f886e44619abfabce160f2803f', '0fe6c4c4b70142b98d85c0769bfe310d', '6aa6a7cb644d425eaf224162a4ea20af', '64124d9bd23c4ad5bbe45d8b061702f2'],
            'er-ace+bottom-k': ['eba603563ebf41d18df817240a8c38fe', '2bf59f10e4b64fdfaa70112660018d8a', '8e8dbaf5a1b1429d9b282add39dfb865', 'a4d45b7328904d279e57bb0700628310', 'bc07623766fe40dca45046d4252e4369'],
            'er-ace+mid-k': ['1b9725a311874d50812abb35c9898de8', '2f1f476eabe94574b282f7332c2184ef', '6451181337244d87899db70638935775', '7b92e8615ddb4017970d5eebdc244bb2', '746687b33e4240dd84a83a2420cf85c2'],
            'er-ace+top-k': ['70496ef7e80f464c8e76704000b7a4f7', '10ed8f24624e479b802699e6241c7a22', '6bfa36f4d5474be29f8ede5dfaccd063', '1b5aaeb31c504f8699cc3dc7393e9728', '6103d32f6e79450b927c541ea9039096'],
            'der++': ['396be4ffa24b47e4a4d6705589c9e84b', '16409d33c37f4b51a72d9fd16d71bcf0', 'eee0c4e69c494f38b0fd93c5da8eb8f1', '6fdce285864442d4b33f232bb13edb3e', '2b070a15b6454f0ba4d967c17456b328'],
            # ['e89aa480ffcb4e9da94961923d6d05de', '6ec5e12998b14d4f8b370dc4eeb43e86'],
            'derpp+bottom-k': ['b538cf327a8c4e26ba14da0a58735a3d', '9cef74a637db4384a0fd2d54d2e3c88b', '140208e8e7e04390a38a06b387dea2bf', '7732dd08d1ee49868b65772b50829257', '585cf2f653814da2bc05f6f2629fdb50'],
            # ['f84cf2b4fde147638a2b39550fab4fe5', ],
            'derpp+mid-k': ['9b90db8f3ac14ae482e70797546d1da5', 'de46efc59b1848bebe281a4e9a786c89', 'ec8c6d4e47b040378dfb2b430bf54d3e', '71a1fb1470214d5683ac117810403436', 'b81b23014ccf459e862e4b23f47e46a7'],
            # ['005e1f84627b4c8a83ce07cde9131c52', ],
            'derpp+top-k': ['9772885e85b64cf09736e814d370c68d', 'b1ca34bde01d4b008f55d344bdfcd033', 'eb3c37fdf3a24e85aa4603234ae01c79', 'dce1bfb98ffb4fe4bd4ba04f2419b9de', '038ec58ea8d44abbbaafaa8142e4e043'],
        }
    }

    # assert runs_standard_benchmarks['cifar10'].keys() == runs_standard_benchmarks['cifar100'].keys() == runs_standard_benchmarks['tiny-imagenet'].keys()
    algorithms = list(runs_standard_benchmarks['cifar100'].keys())

    mlruns_path = '///home/jkozal/Documents/PWr/memorization-cl/memorization-cl/mlruns/'
    # mlruns_path = '///home/jedrzejkozal/Documents/memorization-cl/mlruns/'
    client = mlflow.tracking.MlflowClient(mlruns_path)

    dataset_experiments = {
        'cifar10': '899231754584608899',
        'cifar100': '976651693442159172',
        'tiny-imagenet': '112104018620214336'
    }

    dataset_list = ['cifar100', 'tiny-imagenet']
    dataset_n_tasks = [10, 20]

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

    tab_latex = tabulate.tabulate(table, tablefmt="latex", headers=['method', 'acc', 'FM', 'acc', 'FM', 'acc', 'FM',])
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
