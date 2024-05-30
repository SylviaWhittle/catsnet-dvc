# catsnet-dvc

This repo is my first look into [Data Version Control](https://dvc.org/doc). It is an attempt at improving reproducibility and rigour for my deep learning experiments and hopefully will enable me to track my experiments in a properly documented fashion going forward in my PhD.

The goal of this mini-project is to segment cats.

The data is private to the University of Sheffield and only staff members can access it. I will likely upload open datasets in the future.

For UoS staff who want to try this out:
- Clone this repo
- Create a new conda environment (works on 3.10 for me)
- Install the required packages `pip install -r requirements.txt`
- Download the data automatically using DVC: `dvc pull`. This will take a few minutes as Google Drive's servers are slow.

- You can then have a look at the experiments via `dvc exp show` and `dvc exp list`
- You can have a look at the processing pipeline via `dvc dag` and `cat .dvc.yaml`
- Run an experiment simply by running `dvc exp run`. It's likely that it will say that there are no updates and so it won't run anything - this is intended! It will only run an experiment if the result will change, so try editing a parameter like batch size or epochs: `dvc exp run --set-param "train.batch_size=10"`
- Multiple experiments can be queued via `dvc exp run --queue --set-param "train.batch_size=8,16,24"` for example. (Not done this yet but it's in the docs).

- Also use the Visual Studio Code DVCLive extension to see the cool graphs and tables comparing experiments!

## Branching

If you want to upload your own experiment, try creating a branch with a descriptive name and running one, change the code if you like too, DVC seems to be able to handle that. Then message me to add you as a contributor and you can push your experiment.

## DVC Studio

I'll be looking into using DVC Studio soon to host the experiments. Here is a link to some info on [sharing experiments](https://dvc.org/doc/user-guide/experiment-management/sharing-experiments).

