#! /bin/bash


# experiments with standard benchmarks
## cifar100
for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar100" --lr=0.1 --model="maer" --buffer_size=500 --buffer_policy="min" --n_tasks=10 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR100" --run_name="maer min seed $SEED" --device="cuda:0"; done

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar100" --lr=0.1 --model="maer" --buffer_size=500 --buffer_policy="middle" --n_tasks=10 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR100" --run_name="maer middle seed $SEED" --device="cuda:0"; done

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar100" --lr=0.1 --model="maer" --buffer_size=500 --buffer_policy="max" --n_tasks=10 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR100" --run_name="maer max seed $SEED" --device="cuda:0"; done

## tinyimagenet
for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-tinyimg" --lr=0.1 --model="maer" --buffer_size=500 --buffer_policy="min" --n_tasks=20 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="TinyImageNet" --run_name="maer min seed $SEED" --device="cuda:0"; done

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-tinyimg" --lr=0.1 --model="maer" --buffer_size=500 --buffer_policy="middle" --n_tasks=20 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="TinyImageNet" --run_name="maer middle seed $SEED" --device="cuda:0"; done

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-tinyimg" --lr=0.1 --model="maer" --buffer_size=500 --buffer_policy="max" --n_tasks=20 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="TinyImageNet" --run_name="maer max seed $SEED" --device="cuda:0"; done

## cifar10
for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar10" --lr=0.1 --model="maer" --buffer_size=500 --buffer_policy="min" --n_tasks=5 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR10" --run_name="maer min seed $SEED" --device="cuda:0"; done

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar10" --lr=0.1 --model="maer" --buffer_size=500 --buffer_policy="middle" --n_tasks=5 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR10" --run_name="maer middle seed $SEED" --device="cuda:0"; done

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar10" --lr=0.1 --model="maer" --buffer_size=500 --buffer_policy="max" --n_tasks=5 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR10" --run_name="maer max seed $SEED" --device="cuda:0"; done


# experiment with different buffer sizes
for BUFFER_SIZE in 500 2000 5000 10000 20000 30000 40000; do for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar100" --lr=0.1 --model="maer" --buffer_size=$BUFFER_SIZE --n_tasks=10 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR100" --run_name="maer min balanced seed $SEED buffer size $BUFFER_SIZE" --device="cuda:0" --buffer_policy="min";  done; done

for BUFFER_SIZE in 500 2000 5000 10000 20000 30000 40000; do for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar100" --lr=0.1 --model="maer" --buffer_size=$BUFFER_SIZE --n_tasks=10 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR100" --run_name="maer middle balanced seed $SEED buffer size $BUFFER_SIZE" --device="cuda:0" --buffer_policy="middle";  done; done

for BUFFER_SIZE in 500 2000 5000 10000 20000 30000 40000; do for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar100" --lr=0.1 --model="maer" --buffer_size=$BUFFER_SIZE --n_tasks=10 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR100" --run_name="maer max balanced seed $SEED buffer size $BUFFER_SIZE" --device="cuda:0" --buffer_policy="max";  done; done


# strategy based on mixing high memscores with low or middle
for BUFFER_SIZE in 500 2000 5000 10000 20000 30000 40000; do for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar100" --lr=0.1 --model="maer_high" --buffer_size=$BUFFER_SIZE --n_tasks=10 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR100" --run_name="maer high min balanced seed $SEED buffer size $BUFFER_SIZE" --device="cuda:0" --buffer_policy="min";  done; done

for BUFFER_SIZE in 500 2000 5000 10000 20000 30000 40000; do for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar100" --lr=0.1 --model="maer_high" --buffer_size=$BUFFER_SIZE --n_tasks=10 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR100" --run_name="maer high middle balanced seed $SEED buffer size $BUFFER_SIZE" --device="cuda:0" --buffer_policy="middle";  done; done


# maer with other forms of rehersal
for SEED in 0 1 2 3 4; do for BUFFER_POLICY in "min" "middle" "max"; do echo "buffer policy $BUFFER_POLICY seed $SEED"; python main.py --dataset="seq-cifar100" --lr=0.1 --model="maer_derpp" --buffer_size=500 --buffer_policy="$BUFFER_POLICY" --n_tasks=10 --ignore_other_metrics --seed=$SEED --device="cuda:0" --n_epochs=50 --experiment_name="CIFAR100" --run_name="maer_derpp $BUFFER_POLICY seed $SEED" --alpha=0.1 --beta=0.5;   done; done

for SEED in 0 1 2 3 4; do for BUFFER_POLICY in "min" "middle" "max"; do echo "buffer policy $BUFFER_POLICY seed $SEED"; python main.py --dataset="seq-cifar100" --lr=0.1 --model="maer_er_ace" --buffer_size=500 --buffer_policy="$BUFFER_POLICY" --n_tasks=10 --ignore_other_metrics --seed=$SEED --device="cuda:0" --n_epochs=50 --experiment_name="CIFAR100" --run_name="maer_er_ace $BUFFER_POLICY seed $SEED";   done; done

for SEED in 0 1 2 3 4; do for BUFFER_POLICY in "min" "middle" "max"; do echo "buffer policy $BUFFER_POLICY seed $SEED"; python main.py --dataset="seq-tinyimg" --lr=0.1 --model="maer_derpp" --buffer_size=500 --buffer_policy="$BUFFER_POLICY" --n_tasks=20 --ignore_other_metrics --seed=$SEED --device="cuda:0" --n_epochs=50 --experiment_name="TinyImageNet" --run_name="maer_derpp $BUFFER_POLICY seed $SEED" --alpha=0.2 --beta=0.5;   done; done

for SEED in 0 1 2 3 4; do for BUFFER_POLICY in "min" "middle" "max"; do echo "buffer policy $BUFFER_POLICY seed $SEED"; python main.py --dataset="seq-tinyimg" --lr=0.1 --model="maer_er_ace" --buffer_size=500 --buffer_policy="$BUFFER_POLICY" --n_tasks=20 --ignore_other_metrics --seed=$SEED --device="cuda:0" --n_epochs=50 --experiment_name="TinyImageNet" --run_name="maer_er_ace $BUFFER_POLICY seed $SEED";   done; done


# reproduction of baselines
for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar100" --lr=0.1 --model="er" --buffer_size=500 --buffer_policy="reservoir" --n_tasks=10 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR100" --run_name="er seed $SEED" --device="cuda:0"; done

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar100" --lr=0.1 --model="er" --buffer_size=500 --buffer_policy="balanced_reservoir" --n_tasks=10 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR100" --run_name="er balanced seed $SEED" --device="cuda:0"; done

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-tinyimg" --lr=0.1 --model="er" --buffer_size=500 --buffer_policy="reservoir" --n_tasks=20 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="TinyImageNet" --run_name="er reservoir seed $SEED" --device="cuda:0"; done

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-tinyimg" --lr=0.1 --model="er" --buffer_size=500 --buffer_policy="balanced_reservoir" --n_tasks=20 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="TinyImageNet" --run_name="er balanced seed $SEED" --device="cuda:0"; done

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar10" --lr=0.1 --model="er" --buffer_size=500 --buffer_policy="reservoir" --n_tasks=5 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR10" --run_name="er reservoir seed $SEED" --device="cuda:0"; done

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar10" --lr=0.1 --model="er" --buffer_size=500 --buffer_policy="balanced_reservoir" --n_tasks=5 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR10" --run_name="er balanced seed $SEED" --device="cuda:0"; done

for BUFFER_SIZE in 500 2000 5000 10000 20000 30000 40000; do for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar100" --lr=0.1 --model="er" --buffer_size=$BUFFER_SIZE --buffer_policy="balanced_reservoir" --n_tasks=10 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR100" --run_name="er balanced seed $SEED buffer size $BUFFER_SIZE" --device="cuda:0";  done; done

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar100" --lr=0.1 --model="bcsr" --buffer_size=500 --buffer_policy="balanced_reservoir" --n_tasks=10 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR100" --run_name="bcsr reservoir seed $SEED" --device="cuda:0"; done

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-tinyimg" --lr=0.1 --model="bcsr" --buffer_size=500 --buffer_policy="balanced_reservoir" --n_tasks=20 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="TinyImageNet" --run_name="bcsr reservoir seed $SEED" --device="cuda:0"; done

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar10" --lr=0.1 --model="bcsr" --buffer_size=500 --buffer_policy="balanced_reservoir" --n_tasks=5 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR10" --run_name="bcsr reservoir seed $SEED" --device="cuda:0"; done

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar100" --lr=0.1 --model="derpp" --buffer_size=500 --n_tasks=10 --ignore_other_metrics --seed=$SEED --device="cuda:0" --n_epochs=50 --experiment_name="CIFAR100" --run_name="derpp seed $SEED" --alpha=0.2 --beta=0.5;   done

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-tinyimg" --lr=0.1 --model="derpp" --buffer_size=500 --n_tasks=20 --ignore_other_metrics --seed=$SEED --device="cuda:0" --n_epochs=50 --experiment_name="TinyImageNet" --run_name="derpp seed $SEED" --alpha=0.2 --beta=0.5;   done



for SEED in 0 1 2 3 4; do echo "rainbow memory seed $SEED"; python main.py --dataset="seq-cifar100" --lr=0.1 --model="rainbow_memory" --buffer_size=500 --n_tasks=10 --ignore_other_metrics --seed=$SEED --device="cuda:0" --n_epochs=50 --experiment_name="CIFAR100" --run_name="rainbow memory seed $SEED"; done;    

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar10" --lr=0.1 --model="rainbow_memory" --buffer_size=500 --n_tasks=5 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR10" --run_name="rainbow memory seed $SEED" --device="cuda:0"; done

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-tinyimg" --lr=0.1 --model="rainbow_memory" --buffer_size=500 --n_tasks=20 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="TinyImageNet" --run_name="rainbow memory seed $SEED" --device="cuda:0"; done




for SEED in 0 1 2 3 4; do echo "rainbow memory seed $SEED"; python main.py --dataset="seq-cifar100" --lr=0.1 --model="pbcs" --buffer_size=500 --n_tasks=10 --ignore_other_metrics --seed=$SEED --device="cuda:0" --n_epochs=50 --experiment_name="CIFAR100" --run_name="pbcs seed $SEED"; done;

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-cifar10" --lr=0.1 --model="pbcs" --buffer_size=500 --n_tasks=5 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="CIFAR10" --run_name="pbcs seed $SEED" --device="cuda:0"; done

for SEED in 0 1 2 3 4; do echo "seed $SEED"; python main.py --dataset="seq-tinyimg" --lr=0.1 --model="pbcs" --buffer_size=500 --n_tasks=20 --ignore_other_metrics --seed=$SEED --n_epochs=50 --experiment_name="TinyImageNet" --run_name="pbcs seed $SEED" --device="cuda:0"; done
