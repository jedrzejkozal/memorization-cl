import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tables import *
import os
os.environ['QT_XCB_GL_INTEGRATION'] = 'none'


def main():
    runs_buffer_sizes = {
        500: {
            'reservoir balanced': ['bebddf41427b405ab6f62ffb47859ad2', '891d5218513c4806afea65e972a2a3cc', 'e2836045f5d7423aad78e9525ef1a6ad', '6f0e770354cf496cb82a0b258d4d36b0', '0710078d0c37486c9caf39e8ee5186c5'],
            'bottom-k memscores': ['20ee555f00694b06ada8cef55625c42a', 'f6055777a13c4e9499c3e1c8a272d9bb', '9e1dffae9cdd4b2ebe6a9dc3ba1cdf08', '140893946fc746e2960fc3935900d911', '975375c2d35d496583b66c3a784a7930'],
            'middle-k memsocres': ['2204230c9dfa41dcb367e31b0e32c28e', '1be9cdd652424198a54f9216e460688e', 'a09a00b7f3fe41d189867a164613550e', 'e5c0e64a68b74d6e9dd6d9d06dfda5b6', '4ceb856389574ef88a02c1bf32674e1b'],
            'top-k memscores': ['a5be730e0fc64327b0d9e39eb02d3393', '823af178c35d4c40aa3bcecbf932dcf2', '03a6b3da5bdb4c4aae142b17f30a3523', '51b7604757d84f47a6957ca7f0340848', 'cb51df5347474927b9bf98165334b6b4'],
        },
        2000: {
            'reservoir balanced': ['2ebcd75900134e0b99f265af37f2a021', 'efaf135020b64f2c9b5d2e12ce544fb8', '47e987cc1cef4661871a70afef793fd2', 'b4491156bc454fb5bb2605aa3a86cb6c', 'c5560f46876946bca4a5d6bc941d19ca'],
            'bottom-k memscores': ['d61362a4b91045929f4825bbc75994d1', '8126eeb49fff4c59967ea60f18d414fe', 'd109874ab4d7438bbea637b1541b7732', 'bec77f4c0f5e4ef1b9e521190fe89446', '5a92baf38f4341a38d6a4c51d3a0a89d'],
            'middle-k memsocres': ['9aa1c559d1444dc683e14ec096ed2d66', '513b330ed86445f0815234cae199f4bb', '780cb412acd64673b9e1396b06f17252', '472db3215be545a68a95908878ff815b', '305797cf7054429aad11b65caef95491'],
            'top-k memscores': ['2484ee1c0baa447687d60b3337ce1f2e', '34eeb10ddbe8464a86ed9a7ade1ce789', 'ddd2fe4f131246dfac7124787222eaf9', 'd4500120d4c2455e8bffd3455853dcef', '0240f16a24bd43749af6453956901001'],
            'high bottom-k': ['d94b77931c234bcdbe9655ff2b4f0f77', 'c883dfd5b55e4aa792beb9356494b2d8', '7227e24ef6e044cd998c39c996bf5664', '17decb3351134d67aa18455d39fa0d91', '46f04156b58644bfb57b52c6cf02f049'],
            'high middle-k': ['abd3935aaaf24055a64004531bca72e3', '5b260ffdf4d14c0693a873d665c64eed', '251ac1efe81d49e38516df4329025c1e', 'd9bd72625ddb4e1a998b43b94571b933', '078c7b91a1914c0384540206853b4992'],
        },
        5000: {
            'reservoir balanced': ['eb52c27be53b4cc0872b61e048b5f0bf', '9873e31e659f47de83258b228145940b', '87abbf11a6e34de6b1801c3657572e07', '8643540f07134afdae3b9d49d7f30c50', '79a6057c032a41029d08bdb9549ec08a'],
            'bottom-k memscores': ['9d92d89f1f8c40e3b38e3c11fa5cbcee', '15710c8aa83747a2848890daa4ea3b87', '954e73f7c46446a19bc92c6bc684c633', '08137f26aae54769b6146d490392b6b4', '4f03c73ec8624f4d8ed4d6e9cc2504fa'],
            'middle-k memsocres': ['49be1c196c3e495b875c181cd42dedca', '2226ab327d7448bc9077487db290f66a', '30604bc8fd904223ae07ebb67b8374ae', 'a24bd062d8ac49f4b9edda767ce1f592', '8c6bc99d78f04a2796e1a843471119d1'],
            'top-k memscores': ['68f7fa4cf0ab4a0b9396ed954099f17b', '0198702c4cad476f8e9848fe03d0c913', '5170316ceae647238006e184e3bf6638', '3b1cdfa9a28a494d9d174ca279a96a67', 'eb9e91a5ea6d4132a03f2bddd9a1501e'],
            'high bottom-k': ['4e857742f6c14f65a2febb8aebcb832f', 'bddeb4735a764e1089a57ba9858e2982', '0aa2d2fdc7e84d78ace5329452bd5845', '1927c0782ae44f1ca3ed2d8be946235f', '6ab1259a4fef4bbbb173637746e964b2'],
            'high middle-k': ['858a8ffb8f594165b72009b37e6252b6', '7b24848876d7451aaac2d037efe81884', '46b0036b682944278d559aa323d3067e', '9e0d7dc2b2fa4635809725998760ef10', '1f3c089aba5240ada45c483f51e3262c'],
        },
        10000: {
            'reservoir balanced': ['e90131c96f5a416a8424a48cc6fac82d', 'acd55c0ad8dc4033945c4fce417befeb', 'ff70ada31e5b4d078bd0b3335039ac15', '7c9c06a4e33c4e1cb8a37d1b97e65b00', '39ddc0bedaa34c4e91ebf4fb13dd71f8'],
            'bottom-k memscores': ['01c68c82b98940318d72f7b411da3ec1', '883f9b464d554cb4a13fa42253498ea1', 'd6bb284059594c8f8204501cc12fe92a', '4b60c507a79547b097f9dbc5614427b6', '7e43c8a887c4409abae22bbe0fe06dab'],
            'middle-k memsocres': ['252891186c26471895cda347ac1d9579', 'f5a1c414b2624e73bc39c8fc6230496c', 'b2f0f8cdc20149b283a285e7e990783d', 'd484e3e1eb5a4e52856c5931b51de453', 'ae8b0773b14541d3ababbe4982696757'],
            'top-k memscores': ['cd4ffd0d56d24f5e944103a2780f922a', '7e9b81b0e4c3465ca6e923675d1a7d95', 'c89aad1bff6e4c0386b57e5a64fe7332', '95fd6cde7d3f4109bd5ef6eb1d9eb357', 'f50480ca33bc440fb4ba9762804a8908'],
            'high bottom-k': ['fd7812bd6ca74a3b96d46eac83d251db', '718c87f155e64438bb57a8e7366b5c54', '6ce162a81ec5414a8cab34bc2be58f23', 'f2e296c20c3c4f0980b614aa4a73b06e', '7484240d2b0740bf82c1fbf8d0539b26'],
            'high middle-k': ['e49de119a42042ad995c0742741abbf0', 'cf2cf39f57404b969b20eb106500fc64', 'fadb2648084549a5b26578ec68fc0786', '4290a4f8b7e24cdaa7aab3c2969c88fb', '9b0fc96162734dc1b6a249c697480294'],

        },
        20000: {
            'reservoir balanced': ['b532ceae3488484da552408733c5685d', '10f476ca42494c70ac192fd5d054a6ea', '6d11b2afe31041468e782f01017e7f4f', '272b36bd58a6493b9379edeabfa4e79f', 'a1db1f032f014de099939a8bf81ec46f'],
            'bottom-k memscores': ['cf56e9bdf26e4be483bee61ebf50630b', '917897ef51ed41b0a3158e4bd32e5cb8', '478f5e26340347839839491b3032ba5c', '2dd1326f849f46c898564c6eb0f0ad40', '0153c98db32346deb0d5667fe673bb8d'],
            'middle-k memsocres': ['ae9157560c214c8b9f30bb580361a736', '3e0877fd789444c099011bf03381b2ff', '2ddd208a643245069fb321b078a7a603', '6afe7e9ff5504a9fbe920e65c23e5307', '23fa9a2967ab4b83b7f28f98166e3504'],
            'top-k memscores': ['6dc27dd193eb4db9b96d02846eaf18ae', 'b68989d45ba9406b9a2a8b5a337a0389', 'f83eec22e63b45d197ba842a879537d4', 'b15c13064fe24b2b891e0c19d4027ce2', '0c41f34c20ee4d3a865fcf37a7b8cfc4'],
            'high bottom-k': ['e4d961a2e4f84e70b85a36ab6454c0b4', 'b84f7bf17c274e9e91fdaca768e91d96', 'cddff21ea2dc439b802e02cbaa6ffa54', '0a7230d39cbe4b76b95d2f3481315c6e', 'de8530e093564bfb973aee446c1970a1'],
            'high middle-k': ['6b80582b5b2d4a578244303b69348c9b', '6621e1e4934d4ad6917be58519830138', '2571cfc3c00f481eaf7322c69155cba5', '0687ba965a874767ae30c7a09cfe57dd', '4d2254a446934095a6c150415e50a345'],
        },
        30000: {
            'reservoir balanced': ['7038f66a9c9741e2be8688392042e1aa', 'bbe59f90cee34818bfe03d988d1c6639', '356921d87aa44811b7054ed3583f66f8', '60392b4215da4b6482c95c752ec59a0a', '97a81cc6e2354cdeb50eeae5e8de1139'],
            'bottom-k memscores': ['aa907596d8f940a1807d03a8b6ee526c', '6ca28dda92f0403ab4e08fbb988edf7e', 'ae0f62ae701542e2b46f690b36efa98e', 'd0dfa16826b24ed4ab8595a0d01b95b4', '307ae84a5aab4624b5bfa1434c22bb60'],
            'middle-k memsocres': ['02b8c348216b43dc8b763a40fbd7d9f0', '57972e482dce4a4a85ea53ee7cd34778', '398444c04876475a931263b884f01b08', 'e9a40c7e6d714202a50cde3e2632ddb5', '2468b8c468614b61a8f27d5faef3e849'],
            'top-k memscores': ['46c506bd4da041398fa744331441ae3b', 'b41e6421b61a47319fa80b93638145c8', '0b50e9fcb42f49b4825979b14dc286e1', 'fbf6aac0e68d4352bb489de07bad9952', '64baa531033449299bdf40e6487ded5b'],

        },
        40000: {
            'reservoir balanced': ['7c32f0290d3b4af3ba579239e3787c4b', '435bbf64f32b4cfe8fcd5a72610522aa', 'f45a78998666480d944fb6463120de14', '799a64658c2f400e85a4a016c718b8c8', '44351eb1cbb645fe8f069c367a6318c3'],
            'bottom-k memscores': ['6e7ec05ffc244f59892109809475a10e', 'f1c7fef71ca840d6924343e5c7ef3307', '86e1f4f5ad5a4b989da85c8779303725', 'd753a4758f1f49e5ac8232add61e9b80', 'e549596d2875413ea31983a222ede2c6'],
            'middle-k memsocres': ['0959975a636041609334a0694e1338e0', '624c58e5b18c4c63b9fd0729ba712f7b', '6102e0a5f6e34b5282f044d6d72f9a6a', '8cf7a9492f5d444795a8c7636beacdbf', '1c42cf0ab2f64381a5cb638c9c4b14b2'],
            'top-k memscores': ['9faccb39090d4bd48afecf7c0c358223', 'ce9dfcc46ca247d18f43a0e36198037a', '5a4496a049404c57be3ded62ef0db379', '7740fcef6f7c4c2981cd1529b8d1e004', '919521a8bad34f54a1b893ff7999c77c'],
        }
    }

    buffer_sizes = list(runs_buffer_sizes.keys())
    algorithms = list(runs_buffer_sizes[2000].keys())
    print(mcolors.TABLEAU_COLORS)
    print(type(mcolors.TABLEAU_COLORS))
    color_pallete = list(mcolors.TABLEAU_COLORS.keys())
    algortihm_colors = {alg_name: color_pallete[i] for i, alg_name in enumerate(algorithms)}

    mlruns_path = '///home/jkozal/Documents/PWr/memorization-cl/memorization-cl/mlruns/'
    client = mlflow.tracking.MlflowClient(mlruns_path)

    experiment_id = '976651693442159172'

    results = {a: [] for a in algorithms}
    results_std = {a: [] for a in algorithms}
    for algorithm_name in algorithms:
        for buffer_size in buffer_sizes:
            if algorithm_name not in runs_buffer_sizes[buffer_size]:
                continue
            run_ids = runs_buffer_sizes[buffer_size][algorithm_name]
            acc, acc_std = calc_average_metrics(run_ids, client, digits=8)
            results[algorithm_name].append(acc)
            results_std[algorithm_name].append(acc_std)

    print(results)

    # plt.subplot(1, 2, 1)
    algorithms = ['reservoir balanced', 'bottom-k memscores', 'middle-k memsocres', 'top-k memscores']
    for algorithm_name in algorithms:
        accs = results[algorithm_name]
        acc_std = results_std[algorithm_name]
        series_buffer_sizes = [buf_size for buf_size in buffer_sizes if algorithm_name in runs_buffer_sizes[buf_size]]
        plt.plot(series_buffer_sizes, accs, label=algorithm_name, linewidth=1.0, color=algortihm_colors[algorithm_name])
        plt.errorbar(series_buffer_sizes, accs, yerr=acc_std, linewidth=1.0, color=algortihm_colors[algorithm_name], capsize=3)

    plt.axhline(y=68.362, color='black', linestyle='--', linewidth=1.0,)  # label='y=68.362')
    plt.legend()
    plt.xlabel('buffer size')
    plt.ylabel('test accuracy')

    # plt.subplot(1, 2, 2)
    # algorithms = ['bottom-k memscores', 'middle-k memsocres', 'high bottom-k', 'high middle-k']
    # for algorithm_name in algorithms:
    #     accs = results[algorithm_name]
    #     if len(accs) > 4:
    #         accs = accs[1:5]
    #     # series_buffer_sizes = [buf_size for buf_size in buffer_sizes if algorithm_name in runs_buffer_sizes[buf_size]]
    #     series_buffer_sizes = [2000, 5000, 10000, 20000]
    #     plt.plot(series_buffer_sizes, accs, label=algorithm_name, linewidth=1.0, color=algortihm_colors[algorithm_name])
    # plt.legend()
    # plt.xlabel('buffer size')
    # plt.ylabel('test accuracy')

    plt.show()


def calc_average_metrics(dataset_run_ids, client, digits=3):
    acc_all = []
    for run_id in dataset_run_ids:
        acc = get_metrics(run_id, client)
        acc_all.append(acc)
    avrg_acc, acc_std = rounded_reduction(acc_all, digits=digits)
    return avrg_acc, acc_std


if __name__ == '__main__':
    main()
