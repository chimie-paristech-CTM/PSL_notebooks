import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from baybe import Campaign
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from collections.abc import Iterable
from baybe.targets.base import Target
from baybe.acquisition.acqfs import AcquisitionFunction
from baybe.surrogates.base import Surrogate
from baybe.objectives.base import Objective



def create_campaign(
    acquisition_function: AcquisitionFunction,
    surrogate_model: Surrogate,
    searchspace: SearchSpace,
    objective: Objective,
    two_phase=True
) -> Campaign:

    """
    Build a BayBE Campaign using a two-phase recommender:
      - RandomRecommender for the first few points
      - BotorchRecommender with your GP (or other) surrogate & acquisition function thereafter
    """

    # Create two phase recommender system (Random -> BO)
    if two_phase:
        recommender = TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(
                surrogate_model=surrogate_model, acquisition_function=acquisition_function
                # , optimizer_options={"num_restarts": 10, "raw_samples": 128}
            ), switch_after=4
        )
    else:
        recommender=BotorchRecommender(
            surrogate_model=surrogate_model, acquisition_function=acquisition_function
        )
    
    # Create campaign with according variables
    campaign = Campaign(
    searchspace=searchspace,
    recommender=recommender,
    objective=objective,
    )

    return campaign


def run_campaign(campaign: Campaign, searschspace: SearchSpace, optimal_reference=None, runs=10, init_random=True) -> Campaign:


    # Initiate global maximum
    global_max = float('-inf')

    # Potential reference values as optimum (used for campaign generation --> usually UNKOWN)
    if optimal_reference is None:
        optimal_reference = {
            "Solvent":           ["DMAc"],
            "Base":              ["Cesium pivalate"],
            "Ligand":            ["BrettPhos"],
            "Temperature":       [120, 90],
            "Concentration":     [0.10],
            "Catalyst Loading":  list(np.arange(0.7,1.3,0.05))
            }

    if init_random:
        # Start first random recommendations (size of 4)
        recommendation = campaign.recommend(batch_size=4)

        # Add some fake measurements to drive optimization algorithm
        add_fake_measurements_arti(recommendation, searschspace, campaign.targets, good_reference_values=optimal_reference,
                            good_intervals={"yield":(85, 95)}, bad_intervals={"yield":(50, 65)})

        # Print
        print("Started new campaign with artificial experimentation data...")

        # Acquire current max yield as measure to drive algorithm (for adding fake measurements)
        current_max = np.max(recommendation['yield'])

        # Add created recommendations to current campaign
        campaign.add_measurements(recommendation)

        # Current maximum
        global_max = current_max

    # Continue the campaign by adding some additional experiments.
    for i in range(0, runs):
        if i % 2 == 0:
            print(f'\nround {i}/{runs-1}...')
            print(f"Current maximum yield so far: {global_max:.3f}")
            print()
        
        recommendation = campaign.recommend(batch_size=2)

        # Add new measurements with respect to maximum already found and reference optimal
        maximum_bounds = {"yield":(85, 95)}
        bad_yields = {"yield":(50, 65)}
        add_fake_measurements_arti(recommendation, searschspace, campaign.targets, good_reference_values=optimal_reference,
                            good_intervals=maximum_bounds, bad_intervals=bad_yields)
        
        # Acquire current max yield as measure to drive algorithm
        current_max = np.max(recommendation['yield'])

        # Update global max if necessary
        global_max = max(global_max, current_max)

        # Feed them back into the campaign so the surrogate updates
        campaign.add_measurements(recommendation)


    return campaign


def plot_ucb_comparison(ucb_results: pd.DataFrame):
    """
    Plot GP posterior means and ±1σ bands for different UCB beta values.
    
    Parameters
    ----------
    ucb_results : pd.DataFrame
        Must contain columns ['yield_mean', 'yield_std', 'Beta'].
        Each group of rows with the same Beta is plotted as one curve.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for beta, group in ucb_results.groupby('Beta'):
        # reset index so x = 0,1,2,… per campaign run
        grp = group.reset_index(drop=True)
        x    = grp.index
        mean = grp['yield_mean']
        std  = grp['yield_std']
        
        ax.plot(x, mean, label=f'β = {beta:.2f}')
        ax.fill_between(x,
                        mean - std,
                        mean + std,
                        alpha=0.2)
    
    ax.set_xlabel('Experiment Index')
    ax.set_ylabel('Yield')
    ax.set_title('GP Posterior Means ±1σ for Various UCB β Values')
    ax.legend(title='UCB β')
    plt.tight_layout()
    plt.show()



def plot_EI_PI_comparison(ucb_results: pd.DataFrame):
    """
    Plot GP posterior means and ±1σ bands for different PI-EI functions.
    
    Parameters
    ----------
    ucb_results : pd.DataFrame
        Must contain columns ['yield_mean', 'yield_std', 'acqs'].
        Each group of rows with the same Acquisition function is plotted as one curve.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for func, group in ucb_results.groupby('acqs'):
        # reset index so x = 0,1,2,… per campaign run
        grp = group.reset_index(drop=True)
        x    = grp.index
        mean = grp['yield_mean']
        std  = grp['yield_std']
        
        ax.plot(x, mean, label=f'{func}')
        ax.fill_between(x,
                        mean - std,
                        mean + std,
                        alpha=0.2)
    
    ax.set_xlabel('Experiment Index')
    ax.set_ylabel('Yield')
    ax.set_title('GP Posterior Means ±1σ for Various UCB β Values')
    ax.legend()
    plt.tight_layout()
    plt.show()


def add_fake_measurements_arti(
    data: pd.DataFrame,
    searchspace: SearchSpace,
    targets: Iterable[Target],
    good_reference_values: dict[str, list] | None = None,
    good_intervals: dict[str, tuple[float, float]] | None = None,
    bad_intervals: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:

    # Consider hybrid search space parameters
    discrete_params = [param.name for param in searchspace.discrete.parameters]
    continuous_params = [param.name for param in searchspace.continuous.parameters]

    for target in targets:
        # 1) start with uniformly bad values
        lo_bad, hi_bad = bad_intervals[target.name]
        data[target.name] = np.random.uniform(lo_bad, hi_bad, size=len(data))

        # 2) build discrete match masks
        discrete_masks = []
        for param in discrete_params:
            if param in good_reference_values:
                opt = good_reference_values[param]

                # infer uniform step-size from your search‐space definition
                vals = sorted(
                    next(p for p in searchspace.discrete.parameters if p.name==param).values
                )                
                step = vals[1] - vals[0]

                # ---- Option A: Gaussian kernel ----
                sigma = (vals[-1] - vals[0]) / 6.0
                w = np.exp(-0.5 * ((data[param] - opt) / sigma)**2)

                discrete_masks.append(pd.Series(w, index=data.index))

        # 3) build continuous proximity weights
        cont_weights = []
        for param in continuous_params:
            if param in good_reference_values:
                # assume first entry in list is the optimum
                opt = good_reference_values[param][0]
                # get parameter bounds
                lb, ub = next(p for p in searchspace.continuous.parameters if p.name==param).bounds.to_tuple()
                # use range/4 as Gaussian sigma
                sigma = (ub - lb) / 6.0
                # compute gaussian weight in [0,1]
                w = np.exp(-0.5 * ((data[param] - opt) / sigma)**2)
                cont_weights.append(pd.Series(w, index=data.index))

        # 4) aggregate scores
        all_scores = []
        all_scores.extend(discrete_masks)
        all_scores.extend(cont_weights)

        if not all_scores:
            continue  # nothing to do if no reference parameters

        score_df = pd.concat(all_scores, axis=1)
        # fractional score from 0 (worst) to 1 (best)
        frac = score_df.mean(axis=1)

        # 5) interpolate between bad and good intervals row-wise
        lo_good, hi_good = good_intervals[target.name]
        lo_row = lo_bad + (lo_good - lo_bad) * frac
        hi_row = hi_bad + (hi_good - hi_bad) * frac

        # 6) draw one sample per row in its own [lo_row, hi_row]
        rand = np.random.rand(len(frac))
        sampled = lo_row + (hi_row - lo_row) * rand

        # 7) overwrite those rows with any non-zero frac
        mask_positive = frac > 0
        data.loc[mask_positive, target.name] = sampled[mask_positive]

    return data



def plot_transfer_learning_campgaigns(current_campaign: Campaign):

    # 1) Get your posterior stats (includes any Batch + Flow data you added)
    results = current_campaign.posterior_stats()

    # 2) Split into two halves
    n = len(results)
    mid = n // 2

    # 3) Extract arrays
    x    = results.index
    mean = results['yield_mean']
    std  = results['yield_std']

    # 4) Plot
    plt.figure(figsize=(10,7))

    # First half (Batch) in red
    plt.plot(x[:mid], mean[:mid],   color='red',   label='Batch runs')
    plt.fill_between(x[:mid],
                    mean[:mid] - std[:mid],
                    mean[:mid] + std[:mid],
                    color='red', alpha=0.3)

    # Second half (Flow) in blue
    plt.plot(x[mid:], mean[mid:],   color='blue',  label='Flow runs')
    plt.fill_between(x[mid:],
                    mean[mid:] - std[mid:],
                    mean[mid:] + std[mid:],
                    color='blue', alpha=0.3)

    plt.xlabel('Experiment Index')
    plt.ylabel('Yield')
    plt.title('GP Posterior Means ±1σ: Batch vs. Flow')
    plt.legend()
    plt.show()


def plot_campaign_comparison(
    tl0_results: pd.DataFrame,
    tl1_results: pd.DataFrame,
    tl2_results: pd.DataFrame
):
    """
    Plot GP posterior means and ±1σ bands for three campaigns (no-TL, TL1, TL2).

    Parameters
    ----------
    tl0_results : pd.DataFrame
        Results for the baseline campaign (no transfer learning).
        Must contain columns ['yield_mean', 'yield_std'].
    tl1_results : pd.DataFrame
        Results for the TL1 campaign (e.g., TaskParameter-based TL).
        Must contain columns ['yield_mean', 'yield_std'].
    tl2_results : pd.DataFrame
        Results for the TL2 campaign (e.g., SubstanceParameter-based TL).
        Must contain columns ['yield_mean', 'yield_std'].

    Notes
    -----
    Each DataFrame is assumed to represent sequential experiments in that campaign,
    so the index (0, 1, 2, …) corresponds to the experiment number. If your DataFrame
    already has an 'iteration' column instead of a plain index, you can reset_index() first.
    """
    # Tag each DataFrame with its campaign label
    tl0 = tl0_results.copy()
    tl0['campaign'] = 'No TL (tl0)'
    tl1 = tl1_results.copy()
    tl1['campaign'] = 'Task‐TL (tl1)'
    tl2 = tl2_results.copy()
    tl2['campaign'] = 'Desc‐TL (tl2)'
    
    # Concatenate into a single DataFrame
    all_results = pd.concat([tl0, tl1, tl2], ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by campaign and plot each
    for camp_label, group in all_results.groupby('campaign'):
        grp = group.reset_index(drop=True)
        x    = grp.index
        mean = grp['yield_mean']
        std  = grp['yield_std']
        
        ax.plot(x, mean, label=camp_label)
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.2
        )
    
    ax.set_xlabel('Experiment Index')
    ax.set_ylabel('Yield')
    ax.set_title('Comparison of GP Posterior Means ±1σ\nAcross tl0, tl1, and tl2 Campaigns')
    ax.legend()
    plt.tight_layout()
    plt.show()