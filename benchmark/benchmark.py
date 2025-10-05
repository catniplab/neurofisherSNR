import timeit
import datetime
import platform
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neurofisherSNR.latent_dynamics import generate_gp_trajectory
from neurofisherSNR.observation import gen_poisson_observations
from neurofisherSNR.snr import SNR_bound_instantaneous, SNR_bound_instantaneous_vectorized

def run_benchmark():
    """
    Benchmark script to compare the performance of different SNR computation methods.
    """
    # Define the parameter grid for the benchmark
    param_grid = {
        'n_timepoints': [100, 500, 1000, 2000],
        'd_neurons': [10, 50, 100, 200],
        'd_latent': [5],
        'tgt_snr': [10]
    }

    results = []

    # Iterate over the parameter grid
    for n_timepoints in param_grid['n_timepoints']:
        for d_neurons in param_grid['d_neurons']:
            for d_latent in param_grid['d_latent']:
                for tgt_snr in param_grid['tgt_snr']:
                    print(f"Running benchmark for: n_timepoints={n_timepoints}, d_neurons={d_neurons}")

                    # Generate synthetic data
                    time_range = np.linspace(0, 10, n_timepoints)
                    latent_trajectory = generate_gp_trajectory(time_range=time_range, d_latent=d_latent, lengthscale=0.5)

                    _, C, b, _, _ = gen_poisson_observations(
                        x=latent_trajectory,
                        d_neurons=d_neurons,
                        tgt_snr=tgt_snr
                    )
                    CT = C.T

                    # Time the original function
                    t_original = timeit.timeit(
                        lambda: SNR_bound_instantaneous(latent_trajectory, CT, b),
                        number=10
                    )

                    # Time the vectorized function
                    t_vectorized = timeit.timeit(
                        lambda: SNR_bound_instantaneous_vectorized(latent_trajectory, CT, b),
                        number=10
                    )

                    # Verify that the results are close
                    snr_original = SNR_bound_instantaneous(latent_trajectory, CT, b)
                    snr_vectorized = SNR_bound_instantaneous_vectorized(latent_trajectory, CT, b)

                    results.append({
                        'n_timepoints': n_timepoints,
                        'd_neurons': d_neurons,
                        'time_original': t_original,
                        'time_vectorized': t_vectorized,
                        'snr_original': snr_original,
                        'snr_vectorized': snr_vectorized,
                        'snr_diff': abs(snr_original - snr_vectorized)
                    })

    # Convert results to a pandas DataFrame
    df = pd.DataFrame(results)

    # Print summary
    print("\n--- Benchmark Results ---")
    print(df)
    print(f"\nAverage difference in SNR values: {df['snr_diff'].mean()}")

    # Log results to a file
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, "benchmark_results.log")
    with open(log_file_path, "w") as f:
        f.write("--- Benchmark Run ---\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("--- System Information ---\n")
        f.write(f"System: {platform.system()} {platform.release()}\n")
        f.write(f"Processor: {platform.processor()}\n")
        f.write(f"Python Version: {platform.python_version()}\n\n")
        f.write("--- Benchmark Results ---\n")
        f.write(df.to_string())
        f.write("\n")

    print(f"\nBenchmark results logged to {log_file_path}")

    # Plotting results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot for n_timepoints
    for d_neurons_val in param_grid['d_neurons']:
        subset = df[df['d_neurons'] == d_neurons_val]
        axes[0].plot(subset['n_timepoints'], subset['time_original'], 'o-', label=f'Original (d_neurons={d_neurons_val})')
        axes[0].plot(subset['n_timepoints'], subset['time_vectorized'], 's--', label=f'Vectorized (d_neurons={d_neurons_val})')
    axes[0].set_xlabel('Number of Timepoints')
    axes[0].set_ylabel('Execution Time (s)')
    axes[0].set_title('Performance vs. Timepoints')
    axes[0].legend()
    axes[0].grid(True)

    # Plot for d_neurons
    for n_timepoints_val in param_grid['n_timepoints']:
        subset = df[df['n_timepoints'] == n_timepoints_val]
        axes[1].plot(subset['d_neurons'], subset['time_original'], 'o-', label=f'Original (n_timepoints={n_timepoints_val})')
        axes[1].plot(subset['d_neurons'], subset['time_vectorized'], 's--', label=f'Vectorized (n_timepoints={n_timepoints_val})')
    axes[1].set_xlabel('Number of Neurons')
    axes[1].set_ylabel('Execution Time (s)')
    axes[1].set_title('Performance vs. Neurons')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('figs/benchmark_performance.png')
    print("\nBenchmark plot saved to figs/benchmark_performance.png")

if __name__ == '__main__':
    run_benchmark()