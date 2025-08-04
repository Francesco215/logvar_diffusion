import os
import subprocess
import click

@click.command()
@click.option('--exp', type=str, default=None, help='Run a specific experiment by name.')
@click.option('--action', type=click.Choice(['train', 'plot', 'all']), default='all', help='Action to perform.')
def main(exp, action):
    # --- Training Commands ---
    training_experiments = {
        'baseline': 'python toy_example_new.py train --new=False --sm=False --outdir=out/baseline --viz=False',
        'new_method': 'python toy_example_new.py train --new=True --sm=False --outdir=out/new_method --viz=False',
        'score_matching': 'python toy_example_new.py train --new=False --sm=True --outdir=out/score_matching --viz=False',
    }

    # --- Plotting Commands ---
    # These generate plots for each experiment, using the baseline model as a guide for comparison.
    plotting_experiments = {
        'baseline': 'python toy_example_new.py plot --new=False --net=out/baseline/iter_4096.pt --gnet=out/baseline/iter_1024.pt --save=ablation_images/baseline.png',
        'new_method': 'python toy_example_new.py plot --new=True --net=out/new_method/iter_4096.pt --gnet=out/new_method/iter_1024.pt --save=ablation_images/g_not_trained.png',
        'score_matching': 'python toy_example_new.py plot --new=False  --net=out/score_matching/iter_4096.pt --gnet=out/score_matching/iter_1024.pt --save=ablation_images/score-matching.png',
    }

    # Determine which experiments to run
    experiments_to_run = {exp: training_experiments.get(exp) for exp in [exp]} if exp else training_experiments

    if exp is not None and exp not in training_experiments:
        print(f"Experiment '{exp}' not found.")
        return
        
    # Execute the requested actions
    for name, command in experiments_to_run.items():
        if action in ['train', 'all']:
            print(f"--- Running Training: {name} ---")
            subprocess.run(command, shell=True, check=True)
            
        if action in ['plot', 'all']:
            plot_command = plotting_experiments.get(name)
            if plot_command:
                print(f"--- Generating Plot: {name} ---")
                subprocess.run(plot_command, shell=True, check=True)

    print("\n✨ Ablation run complete.")

if __name__ == '__main__':
    main()