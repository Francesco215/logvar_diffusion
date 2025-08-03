
import os
import subprocess
import click

@click.command()
@click.option('--exp', type=str, default=None)
def main(exp):
    experiments = {
        'baseline': 'python toy_example_new.py train --new=False --sm=False --outdir=out/baseline --viz=False',
        'new_G_not_trained': 'python toy_example_new.py train --new=True --sm=False --outdir=out/new_G_not_trained --viz=False',
        'new_G_trained': 'python toy_example_new.py train --new=True --sm=False --outdir=out/new_G_trained --viz=False',
        'score_matching': 'python toy_example_new.py train --new=True --sm=True --outdir=out/score_matching --viz=False',
    }

    if exp is not None:
        if exp not in experiments:
            print(f"Experiment {exp} not found.")
            return
        print(f"Running experiment: {exp}")
        subprocess.run(experiments[exp], shell=True)
    else:
        for name, command in experiments.items():
            print(f"Running experiment: {name}")
            subprocess.run(command, shell=True)

if __name__ == '__main__':
    main()
