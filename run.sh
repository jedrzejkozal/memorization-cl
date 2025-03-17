#! /bin/bash

# echo "adaptive sam"
# python main.py --model="clewipp" --dataset="seq-cifar100" --ignore_other_metrics --seed=42 --lr=0.1 --n_epochs=50 --batch_size 32 --optim_wd=0.0 --optim_mom=0.0 --buffer_size=500 --interpolation_alpha=0.3 --n_tasks=10 --sam_adaptive

echo " "
echo "rho=0.08"
python main.py --model="clewipp" --dataset="seq-cifar100" --ignore_other_metrics --seed=42 --lr=0.1 --n_epochs=50 --batch_size 32 --optim_wd=0.0 --optim_mom=0.0 --buffer_size=500 --interpolation_alpha=0.3 --n_tasks=10 --sam_rho=0.08
echo " "
echo "rho=0.1"
python main.py --model="clewipp" --dataset="seq-cifar100" --ignore_other_metrics --seed=42 --lr=0.1 --n_epochs=50 --batch_size 32 --optim_wd=0.0 --optim_mom=0.0 --buffer_size=500 --interpolation_alpha=0.3 --n_tasks=10 --sam_rho=0.1
echo " "
echo "rho=0.2"
python main.py --model="clewipp" --dataset="seq-cifar100" --ignore_other_metrics --seed=42 --lr=0.1 --n_epochs=50 --batch_size 32 --optim_wd=0.0 --optim_mom=0.0 --buffer_size=500 --interpolation_alpha=0.3 --n_tasks=10 --sam_rho=0.2
echo " "
echo "rho=0.5"
python main.py --model="clewipp" --dataset="seq-cifar100" --ignore_other_metrics --seed=42 --lr=0.1 --n_epochs=50 --batch_size 32 --optim_wd=0.0 --optim_mom=0.0 --buffer_size=500 --interpolation_alpha=0.3 --n_tasks=10 --sam_rho=0.5
echo " "
echo "rho=1.0"
python main.py --model="clewipp" --dataset="seq-cifar100" --ignore_other_metrics --seed=42 --lr=0.1 --n_epochs=50 --batch_size 32 --optim_wd=0.0 --optim_mom=0.0 --buffer_size=500 --interpolation_alpha=0.3 --n_tasks=10 --sam_rho=1.0