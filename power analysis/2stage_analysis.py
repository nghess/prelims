import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Parameters to vary
n_animals_range = np.arange(4, 21, 2)  # Range of animals to test
n_sessions_per_animal = 10  # Multiple sessions per animal
n_neurons_per_session = 25  # Neurons recorded in each session

# Effect size parameters - adjust for each region
# CA1 parameters
ca1_baseline_si = 1.5  # Baseline spatial info (bits)
ca1_si_change = 0.4    # Drop in spatial info after cue flip 
ca1_remapping_prop = 0.7  # Proportion of cells that remap

# OB parameters (for comparison)
ob_baseline_si = 0.6   # Lower baseline spatial tuning
ob_si_change = 0.1     # Smaller change with cue flip
ob_remapping_prop = 0.25  # Fewer cells show cue-dependent changes

# Variance components
animal_var = 0.15   # Between-animal variance
session_var = 0.10  # Between-session variance (within animal)
neuron_var = 0.25   # Between-neuron variance (within session)
residual_var = 0.30  # Within-neuron variance (trial-to-trial)

def simulate_two_stage_analysis(n_animals, region='CA1', n_iterations=1000):
    """
    Simulate data and analyze using a two-stage approach:
    1. First, aggregate data at the neuron level
    2. Then, aggregate at the animal level
    3. Finally, perform a t-test at the animal level
    
    This approach is much more robust than mixed-effects models
    and aligns with common practices in neurophysiology.
    """
    # Set parameters based on region
    if region == 'CA1':
        baseline_si = ca1_baseline_si
        si_change = ca1_si_change
        remapping_prop = ca1_remapping_prop
    else:  # OB
        baseline_si = ob_baseline_si
        si_change = ob_si_change
        remapping_prop = ob_remapping_prop
    
    # Track power
    significant_results = 0
    
    for iter in range(n_iterations):
        # Animal-level means (will store one value per animal)
        animal_si_changes = []
        
        # Generate data for each animal
        for animal in range(n_animals):
            animal_effect = np.random.normal(0, np.sqrt(animal_var))
            neuron_si_changes = []  # Will store changes for each neuron
            
            # Multiple sessions for this animal
            for session in range(n_sessions_per_animal):
                session_effect = np.random.normal(0, np.sqrt(session_var))
                
                # Multiple neurons per session
                for neuron in range(n_neurons_per_session):
                    # Neuron's baseline properties
                    neuron_effect = np.random.normal(0, np.sqrt(neuron_var))
                    remaps = np.random.random() < remapping_prop
                    
                    # Calculate neuron's SI in original and flipped conditions
                    si_orig = baseline_si + animal_effect + session_effect + neuron_effect
                    si_orig += np.random.normal(0, np.sqrt(residual_var))  # Trial noise
                    
                    # Effect depends on whether this neuron remaps
                    effect = si_change if remaps else 0
                    
                    # SI in flipped condition
                    si_flip = si_orig - effect + np.random.normal(0, np.sqrt(residual_var))
                    
                    # Store the change for this neuron
                    neuron_si_changes.append(si_orig - si_flip)
            
            # Stage 1: Average across neurons for this animal
            animal_si_changes.append(np.mean(neuron_si_changes))
        
        # Stage 2: Test if the animal-level changes are significant
        # Use one-sample t-test (are changes different from zero?)
        t_stat, p_value = stats.ttest_1samp(animal_si_changes, 0)
        
        # Count significant results (p < 0.05)
        if p_value < 0.05:
            significant_results += 1
    
    # Calculate power
    power = significant_results / n_iterations
    return power

# Run power analysis for CA1
results_ca1 = []
for n in n_animals_range:
    print(f"Analyzing CA1 with {n} animals...")
    power = simulate_two_stage_analysis(n, region='CA1')
    results_ca1.append(power)
    print(f"  Power: {power:.3f}")

# Run power analysis for OB
results_ob = []
for n in n_animals_range:
    print(f"Analyzing OB with {n} animals...")
    power = simulate_two_stage_analysis(n, region='OB')
    results_ob.append(power)
    print(f"  Power: {power:.3f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(n_animals_range, results_ca1, 'b-o', linewidth=2, label='CA1')
plt.plot(n_animals_range, results_ob, 'g-s', linewidth=2, label='OB')
plt.axhline(y=0.8, color='r', linestyle='--', label='Target Power (0.8)')

plt.xlabel('Number of Animals', fontsize=12)
plt.ylabel('Statistical Power', fontsize=12)
plt.title('Power Analysis for Place Cell Remapping', fontsize=14)
plt.xticks(n_animals_range)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Print recommended sample sizes
ca1_threshold = next((i for i, x in enumerate(results_ca1) if x >= 0.8), len(results_ca1) - 1)
ob_threshold = next((i for i, x in enumerate(results_ob) if x >= 0.8), len(results_ob) - 1)

print(f"\nRecommended minimum number of animals:")
print(f"  For CA1 hypothesis: {n_animals_range[ca1_threshold]}")
print(f"  For OB hypothesis: {n_animals_range[ob_threshold]}")
print(f"  Overall recommendation: {max(n_animals_range[ca1_threshold], n_animals_range[ob_threshold])}")