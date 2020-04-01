import os.path
import sys
import experiments


args = sys.argv
if len(args) > 1:
    experiment_fname = args[1]
    print(experiment_fname)
    if os.path.exists(experiment_fname):
        experiments.load_and_plot(experiment_fname)
    else:
        print(experiments.run_experiment(experiment_fname))
