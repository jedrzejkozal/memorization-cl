#! /bin/bash

# python train_splits.py --dataset_name cifar100 --weights_dir=trained_weights/cifar100_resnet18_subset_01/ --dataset_size=0.1 --num_workers=4 --use_multiprocessing
# python train_splits.py --dataset_name cifar100 --weights_dir=trained_weights/cifar100_resnet18_subset_02/ --dataset_size=0.2 --num_workers=4 --use_multiprocessing
# python train_splits.py --dataset_name cifar100 --weights_dir=trained_weights/cifar100_resnet18_subset_05/ --dataset_size=0.5 --num_workers=4 --use_multiprocessing

# python eval_splits.py --dataset_name="cifar100" --weights_dir=trained_weights/cifar100_resnet18_subset_01/ --dataset_size=0.1 --device="cuda:0" --num_workers=4 --out_filename="memorsation_scores_cifar100_01.npy"
# python eval_splits.py --dataset_name="cifar100" --weights_dir=trained_weights/cifar100_resnet18_subset_02/ --dataset_size=0.2 --device="cuda:0" --num_workers=4 --out_filename="memorsation_scores_cifar100_02.npy"
# python eval_splits.py --dataset_name="cifar100" --weights_dir=trained_weights/cifar100_resnet18_subset_05/ --dataset_size=0.5 --device="cuda:0" --num_workers=4 --out_filename="memorsation_scores_cifar100_05.npy"


# missing CIFAR10 runs for dataset size 1.0

python train_splits.py --dataset_name="cifar10" --weights_dir=trained_weights/cifar10_subset_10/ --dataset_size=1.0 --device="cuda:0" --num_workers=4 --use_multiprocessing

python eval_splits.py --dataset_name="cifar10" --weights_dir=trained_weights/cifar10_subset_10/ --dataset_size=1.0 --device="cuda:0" --num_workers=4 --out_filename="memorsation_scores_cifar10_10.npy"

git pull
git pull
git add -A
git pull
git commit -m "add memorization results for cifar10 1.0 subset"
git push

# cifar100 different architectures rerun

python train_splits.py --dataset_name="cifar100" --model_name=resnet34 --weights_dir=trained_weights/cifar100_resnet34_rerun/ --weight_decay=1e-3 --batch_size=32 --n_epochs=50 --lr=0.03 --num_workers=4 --use_multiprocessing
python train_splits.py --dataset_name="cifar100" --model_name=resnet50 --weights_dir=trained_weights/cifar100_resnet50_rerun/ --weight_decay=1e-3 --batch_size=32 --n_epochs=50 --lr=0.03 --num_workers=4 --use_multiprocessing

python eval_splits.py --dataset_name="cifar100" --model_name=resnet34 --weights_dir=trained_weights/cifar100_resnet34_rerun/ --device="cuda:0" --num_workers=4 --out_filename="memorsation_scores_cifar100_resnet34_rerun.npy"
python eval_splits.py --dataset_name="cifar100" --model_name=resnet50 --weights_dir=trained_weights/cifar100_resnet50_rerun/ --device="cuda:0" --num_workers=4 --out_filename="memorsation_scores_cifar100_resnet50_rerun.npy"

git pull
git pull
git add -A
git pull
git commit -m "add memorization results for cifar100 resnet34 and 50 reruns"
git push

# cifar100 - different weight decay

python train_splits.py --dataset_name cifar100 --weights_dir=trained_weights/cifar100_resnet18_weight_decay_1e03/ --weight_decay=1e-3 --lr=0.1 --num_workers=4 --use_multiprocessing
python train_splits.py --dataset_name cifar100 --weights_dir=trained_weights/cifar100_resnet18_weight_decay_1e04/ --weight_decay=1e-4 --lr=0.1 --num_workers=4 --use_multiprocessing
python train_splits.py --dataset_name cifar100 --weights_dir=trained_weights/cifar100_resnet18_weight_decay_1e05/ --weight_decay=1e-5 --lr=0.1 --num_workers=4 --use_multiprocessing
python train_splits.py --dataset_name cifar100 --weights_dir=trained_weights/cifar100_resnet18_weight_decay_1e06/ --weight_decay=1e-6 --lr=0.1 --num_workers=4 --use_multiprocessing
python train_splits.py --dataset_name cifar100 --weights_dir=trained_weights/cifar100_resnet18_weight_decay_0/ --weight_decay=0.0 --lr=0.1 --num_workers=4 --use_multiprocessing

python eval_splits.py --dataset_name="cifar100" --weights_dir=trained_weights/cifar100_resnet18_weight_decay_1e03/ --device="cuda:0" --num_workers=4 --out_filename="memorsation_scores_cifar100_weight_decay_1e03.npy"
python eval_splits.py --dataset_name="cifar100" --weights_dir=trained_weights/cifar100_resnet18_weight_decay_1e04/ --device="cuda:0" --num_workers=4 --out_filename="memorsation_scores_cifar100_weight_decay_1e04.npy"
python eval_splits.py --dataset_name="cifar100" --weights_dir=trained_weights/cifar100_resnet18_weight_decay_1e05/ --device="cuda:0" --num_workers=4 --out_filename="memorsation_scores_cifar100_weight_decay_1e05.npy"
python eval_splits.py --dataset_name="cifar100" --weights_dir=trained_weights/cifar100_resnet18_weight_decay_1e06/ --device="cuda:0" --num_workers=4 --out_filename="memorsation_scores_cifar100_weight_decay_1e06.npy"
python eval_splits.py --dataset_name="cifar100" --weights_dir=trained_weights/cifar100_resnet18_weight_decay_0/ --device="cuda:0" --num_workers=4 --out_filename="memorsation_scores_cifar100_weight_decay_0.npy"

git pull
git pull
git add -A
git pull
git commit -m "add memorization results for cifar100 different weight decay"
git push
