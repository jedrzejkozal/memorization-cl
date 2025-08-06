#! /bin/bash

python train_splits.py --dataset_name cifar100 --weights_dir=trained_weights/cifar100_resnet18_subset_01/ --dataset_size=0.1 --num_workers=4 --use_multiprocessing
python train_splits.py --dataset_name cifar100 --weights_dir=trained_weights/cifar100_resnet18_subset_02/ --dataset_size=0.2 --num_workers=4 --use_multiprocessing
python train_splits.py --dataset_name cifar100 --weights_dir=trained_weights/cifar100_resnet18_subset_05/ --dataset_size=0.5 --num_workers=4 --use_multiprocessing

python eval_splits.py --dataset_name="cifar100" --weights_dir=trained_weights/cifar100_resnet18_subset_01/ --dataset_size=0.1 --device="cuda:0" --num_workers=4 --out_filename="memorsation_scores_cifar100_01.npy"
python eval_splits.py --dataset_name="cifar100" --weights_dir=trained_weights/cifar100_resnet18_subset_02/ --dataset_size=0.2 --device="cuda:0" --num_workers=4 --out_filename="memorsation_scores_cifar100_02.npy"
python eval_splits.py --dataset_name="cifar100" --weights_dir=trained_weights/cifar100_resnet18_subset_05/ --dataset_size=0.5 --device="cuda:0" --num_workers=4 --out_filename="memorsation_scores_cifar100_05.npy"

git pull
git pull
git add -A
git pull
git commit -m "add memorization results for cifar100 subsets"
git push
