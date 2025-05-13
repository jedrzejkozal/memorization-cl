# What is the role of memorization in Continual Learning?

This repo contains code with experiments for paper "What is the role of memorization in Continual Learning?"



## Environment

We provide the specification of envioronment used in the form of conda .yml file.

Before runing any experiments please run following commands to create and activate the environement:

```
conda env create -f env.yml -y
conda activate interpolation
```


Most of the experiments were done with nvidia Driver Version: 535.230.02 and CUDA Version: 12.2.

## Repo organization

Our repo is cased on Continual Learning mammoth library [link](https://github.com/aimagelab/mammoth)



## scripts to run and reproduce results from the paper

TODO add bash scripts

## Preview results from the paper

We use MLFLow ver 2.2.2 to manage our experiments. Repo contains MLFlow logs with our experiments results and detailed hyperparameters. 

To access the GUI with experiments please make sure that your have MLFlow in proper version installed an run:

```
mlflow ui
```

And then go to [http://127.0.0.1:5000/#/](http://127.0.0.1:5000/#/) in your brower to see all the results from the experiments we runned and exact hyperparameters used in each run.



## Citation policy

TBD