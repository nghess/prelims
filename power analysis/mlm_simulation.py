import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# Parameters to vary
n_animals = np.arange(4, 21, 2)  # Range of animals to test
n_sessions_per_animal = 2  # Reduced number of sessions for simplicity
n_neurons_per_session = 20  # Neurons recorded in each session

# Simplify: Just analyze one region at a time (CA1 for example)
# CA1 parameters
baseline_si = 1.5  # Baseline spatial info (bits)
si_change = 0.4    # Drop in spatial info after cue flip
remapping_prop = 0.7  # Proportion of cells that remap

# Variance components - adjusted for better convergence
animal_var = 0.15   # Between-animal variance 
session_var = 0.10  # Between-session variance (within animal)
neuron_var = 0.25   # Between-neuron variance (within session)
residual_var = 0.30  # Within-neuron variance (observation noise)

# Simulation function for a single region (e.g., CA1)
def simulate_experiment(n_animals, n_iterations=200):
    power_results = {'Condition_Effect': 0}
    successful_models = 0
    
    for i in range(n_iterations):
        # Create dataset
        data = []
        for animal in range(n_animals):
            animal_effect = np.random.normal(0, np.sqrt(animal_var))
            
            for session in range(n_sessions_per_animal):
                session_effect = np.random.normal(0, np.sqrt(session_var))
                
                # Neurons for this session
                for neuron in range(n_neurons_per_session):
                    neuron_effect = np.random.normal(0, np.sqrt(neuron_var))
                    remaps = np.random.random() < remapping_prop
                    
                    # Original condition
                    si_orig = baseline_si + animal_effect + session_effect + neuron_effect + np.random.normal(0, np.sqrt(residual_var))
                    
                    # Flipped condition
                    if remaps:
                        change = si_change
                    else:
                        change = 0
                    si_flip = si_orig - change + np.random.normal(0, np.sqrt(residual_var))
                    
                    data.append({'animal': f"A{animal}", 'session': f"S{animal}_{session}", 
                                 'neuron': f"N{animal}_{session}_{neuron}", 
                                 'condition': 'original', 'SI': si_orig})
                    data.append({'animal': f"A{animal}", 'session': f"S{animal}_{session}", 
                                 'neuron': f"N{animal}_{session}_{neuron}", 
                                 'condition': 'flipped', 'SI': si_flip})
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        try:
            # Simpler model with fewer random effects and string-based grouping
            # This is more stable than the complex nested model
            model = MixedLM.from_formula(
                "SI ~ condition", 
                groups="animal",
                re_formula="1", 
                vc_formula={'session': '0 + C(session)'},
                data=df
            )
            
            # Use robust starting values and algorithm
            result = model.fit(reml=True, method='bfgs')
            
            # Check significance of condition effect
            p_value = result.pvalues.get('condition[T.original]', 
                      result.pvalues.get('condition[T.flipped]', 1.0))
            power_results['Condition_Effect'] += p_value < 0.05
            successful_models += 1
            
        except:
            # If model fails, skip this iteration
            continue
    
    # Calculate power based on successful models
    if successful_models > 0:
        for key in power_results:
            power_results[key] = power_results[key] / successful_models
        power_results['successful_rate'] = successful_models / n_iterations
    else:
        for key in power_results:
            power_results[key] = np.nan
        power_results['successful_rate'] = 0
    
    return power_results

# Alternative approach: use simpler statistical test for power analysis
def simulate_experiment_simple(n_animals, n_iterations=500):
    power_results = {'Condition_Effect': 0}
    
    for i in range(n_iterations):
        # Create dataset
        animal_means = []
        
        for animal in range(n_animals):
            animal_effect = np.random.normal(0, np.sqrt(animal_var))
            
            # For each animal, calculate mean SI change
            si_changes = []
            for session in range(n_sessions_per_animal):
                session_effect = np.random.normal(0, np.sqrt(session_var))
                
                for neuron in range(n_neurons_per_session):
                    remaps = np.random.random() < remapping_prop
                    
                    if remaps:
                        change = si_change
                    else:
                        change = 0
                    
                    # Add noise to the change
                    observed_change = change + np.random.normal(0, np.sqrt(2*residual_var))
                    si_changes.append(observed_change)
            
            # Average change for this animal
            animal_means.append(np.mean(si_changes))
        
        # Simple paired t-test on animal means (more stable)
        _, p_value = stats.ttest_1samp(animal_means, 0)
        power_results['Condition_Effect'] += p_value < 0.05
    
    # Calculate power
    for key in power_results:
        power_results[key] = power_results[key] / n_iterations
    
    return power_results

# Run simulations for different sample sizes using the simpler approach
print("Running simulations...")
power_by_n = {}

for n in n_animals:
    print(f"Testing {n} animals...")
    # Try both methods - use the simple one as fallback
    mixed_model_result = simulate_experiment(n)
    
    if np.isnan(mixed_model_result['Condition_Effect']) or mixed_model_result['successful_rate'] < 0.5:
        print(f"  Mixed model had low success rate ({mixed_model_result.get('successful_rate', 0):.2f}), using simpler approach")
        power_by_n[n] = simulate_experiment_simple(n)
    else:
        print(f"  Mixed model successful ({mixed_model_result['successful_rate']:.2f})")
        power_by_n[n] = mixed_model_result

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(n_animals, [power_by_n[n]['Condition_Effect'] for n in n_animals], 
         label='CA1 cue flip effect', marker='o', linewidth=2)
plt.axhline(y=0.8, color='r', linestyle='--', label='Power = 0.8')
plt.xlabel('Number of animals', fontsize=12)
plt.ylabel('Statistical Power', fontsize=12)
plt.title('Power Analysis for Place Cell Remapping Study (CA1)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(n_animals)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.tight_layout()