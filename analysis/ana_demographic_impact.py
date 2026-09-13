import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
import arviz as az
import pickle
from pprint import pprint

def move_1_2_options_desc():
    return [
    "(a) Fire at Chinese vessels within close proximity of CSG with any weapon mission commander deems appropriate",
    "(b) Hold fire unless fired upon",
    "(c) Hold fire without approval from President",
    "(d) Place automated defenses on auto-fire. AI decides target/AI decides firing.",
    "(e) Place automated defenses on manual firing control. AI decides target/human decides firing.",
    "(f) Human decides target/human decides firing",
    "(g) Hold fire at all costs"
]

def move_2_2_options_desc():
    # Replace this placeholder with the actual code you provide.
    # Example: return a list of column names (strings) for move2 options.
    return [
    # "(a) Military Action", # Skip
    "(1) Preserve Status Quo/Deter",
    "(2) Invade/Attack",
    "(3) Defend",
    "(b) Activate Civilian Reserve/Draft",
    "(c) Surge Domestic Defense Production",
    "(d) Diplomacy",
    "(e) Economic Punishment",
    "(f) Economic Incentives",
    "(g) Clandestine/Special Operations",
    "(h) Information Operations",
    "(i) Conduct Foreign Intelligence",
    "(j) Conduct Domestic Intelligence",
    "(k) Cyber Operations"
]

# --- Data Processing Functions ---

def process_full_survey(df):
    # Select the desired columns. (Column names are assumed to be strings.)
    selected_columns = [
        "Unnamed: 4",  # Group ID
        "Q19",      # Player ID
        # "Q18",   # Self identified Group ID (commented out)
        "Q2",       # Age
        "Q4",       # Gender 
        "Q22",      # Nationality
        "Q23",      # Background
        "Q24",      # Professionality Level
        "Q15",      # Ai Expertise
        "Q16",      # China Military Expertise
        "Q17",      # US Military Expertise
        "Q8",       # p(China invades Taiwan) pre-game
        "Q8.1",     # p(China invades Taiwan) post-game
        "Q9",       # Long-term goals of China pre-game
        "Q9.1",     # Long-term goals of China post-game
        "Q10",      # Short-term goals of China pre-game
        "Q10.1",    # Short-term goals of China post-game
        "Q11",      # Worry About LAWS pre-game
        "Q11.1",    # Worry About LAWS post-game
        "Q12",      # LAWS Support pre-game
        "Q12.1",    # LAWS Support post-game
        "Q14",   # Support Death Penalty (commented out)
        "Unnamed: 33", # AI Acc
        "Unnamed: 34", # AI Team Training
        "Unnamed: 35", # China Status
        "Unnamed: 36", # Move 1 RoE
        "Q20",      # Did you trust AI in Simulation?
        "Q25",      # Credibel Simulation?
    ]
    df_subset = df[selected_columns].copy()

    # Rename columns according to the mapping.
    column_rename = {
        "Unnamed: 4": "GID",
        "Q19": "PID",
        "Q2": "Age",
        "Q4": "Gender",
        "Q22": "Nat",
        "Q23": "Background",
        "Q24": "Professionality",
        "Q15": "AIExp",
        "Q16": "ChinaMilExp",
        "Q17": "USMilExp",
        "Q8": "ChinaInavdes_pre",
        "Q8.1": "ChinaInavdes_post",
        "Q9": "ChinaGoalLong_pre",
        "Q9.1": "ChinaGoalLong_post",
        "Q10": "ChinaGoalShort_pre",
        "Q10.1": "ChinaGoalShort_post",
        "Q11": "LAWSWorry_pre",
        "Q11.1": "LAWSWorry_post",
        "Q12": "LAWSSupport_pre",
        "Q12.1": "LAWSSupport_post",
        "Q14": "DeathPenaltySupport",
        "Unnamed: 33": "AIAcc",
        "Unnamed: 34": "AITrain",
        "Unnamed: 35": "ChinaStatus",
        "Unnamed: 36": "Move1",
        "Q20": "AITrust",
        "Q25": "SimTrust",
    }
    df_subset.rename(columns=column_rename, inplace=True)
    
    # Drop any rows with missing values.
    df_subset.dropna(inplace=True)
    
    return df_subset


def create_demo_results(df_subset):
    # Get unique values from selected columns.
    # Exclusive options (Need Dirichlet Treatment, subdimensions distributed on simplex)
    unique_gids = df_subset["GID"].unique()
    unique_age = df_subset["Age"].unique()
    unique_gender = df_subset["Gender"].unique()
    unique_exp = df_subset["Professionality"].unique()
    unique_deathpenalty = df_subset["DeathPenaltySupport"].unique()
    unique_aiexpertise = df_subset["AIExp"].unique()
    unique_chinamilexpertise = df_subset["ChinaMilExp"].unique()
    unique_usmilexpertise = df_subset["USMilExp"].unique()

    # Comma separated options (independent)
    us = df_subset["Background"].unique()
    us = [u.split(sep=",") for u in us]
    us_flat = []
    for xs in us:
        for x in xs:
            us_flat.append(x)
    unique_back = np.unique(us_flat)
    
    analysis_target_labs = [
        ("Age", unique_age),
        ("Gender", unique_gender),
        ("Professionality", unique_exp),
        ("Background", unique_back),
        ("DeathPenaltySupport", unique_deathpenalty),
        ("AIExp", unique_aiexpertise),
        ("ChinaMilExp", unique_chinamilexpertise),
        ("USMilExp", unique_usmilexpertise),
    ]
    
    demo_results = []
    # Process each group (skip players with "UNK" in GID).
    for tar_id in unique_gids:
        if "UNK" in str(tar_id):
            continue
        
        df_filtered = df_subset[df_subset["GID"] == tar_id]
        n_players = len(df_filtered)
        row_info = {"GID": tar_id, "n": n_players}

        # For each target label, count occurrences and compute fractions.
        for label, uniques in analysis_target_labs:
            demo_counter = {str(key): 0 for key in uniques}

            if label not in ["Background"]:
                for player in df_filtered[label]:
                    demo_counter[player] = demo_counter.get(player, 0) + 1
            
            # I expect exceptions for each class
            elif label == "Background":
                for player in df_filtered[label]:
                    for background in player.split(sep=","):
                        demo_counter[background] = demo_counter.get(background, 0) + 1

            row_info[label] = demo_counter
            row_info[label + "_frac"] = {k: v / n_players for k, v in demo_counter.items()}

        demo_results.append(row_info)
    
    return pd.DataFrame(demo_results), analysis_target_labs


def create_simulation_results(df_simulation):
    # Get move options from the external functions.
    move1_options = move_1_2_options_desc()             
    move2_options = move_2_2_options_desc()
    
    sim_results = []
    unique_game_ids = df_simulation["game_ID"].unique()
    for gid in unique_game_ids:
        # Select the row corresponding to the current game_ID.
        row = df_simulation[df_simulation["game_ID"] == gid].iloc[0]
        # Extract move1 and move2 values.
        vals_move1 = [row[col] for col in move1_options]
        vals_move2 = [row[col] for col in move2_options]
        sim_results.append({"game_ID": gid, "move1": vals_move1, "move2": vals_move2})
    
    return pd.DataFrame(sim_results)


def extract_2d_array(df1, df2, move_index, gender_key, gender_frac_col, move):
    """
    For each row in df1, find the matching row in df2 (where GID equals df1's game_ID).
    Then, extract x_value from the dictionary in column gender_frac_col using gender_key,
    and y_value from the move1 list (using move_index).
    """
    result = []
    for i, row1 in df1.iterrows():
        game_id = row1["game_ID"]
        # Find the corresponding row in df2.
        df2_row = df2[df2["GID"] == game_id]
        if df2_row.empty:
            raise ValueError(f"Game ID {game_id} not found in df2")
        # Assuming only one match.
        gender_frac_dict = df2_row.iloc[0][gender_frac_col]
        x_value = gender_frac_dict[gender_key]
        move1_vector = row1[f"move{move}"]
        y_value = move1_vector[move_index]
        result.append([x_value, y_value])
    
    return np.array(result)


def plot_data_logistic_regression(result, output_filename="move1_f_male.png"):
    """
    Fits a logistic regression model to the (x, y) data, computes bootstrap
    95% confidence intervals, and plots the data, fitted curve, and CI.
    """
    df_model = pd.DataFrame({
        "x": result[:, 0].astype(float),
        "y": result[:, 1].astype(int)
    })
    model = smf.logit("y ~ x", data=df_model).fit(disp=False)
    
    # Create range of x values for prediction.
    x_min, x_max = df_model["x"].min(), df_model["x"].max()
    x_range = np.linspace(x_min, x_max, 100)
    df_pred = pd.DataFrame({"x": x_range})
    predicted_probabilities = model.predict(df_pred)
    
    # --- Bootstrap resampling for 95% CI ---
    n_bootstrap = 1000
    bootstrap_preds_list = []
    n = len(df_model)
    for i in range(n_bootstrap):
        sample_indices = np.random.choice(n, n, replace=True)
        df_boot = df_model.iloc[sample_indices]
        try:
            model_boot = smf.logit("y ~ x", data=df_boot).fit(disp=False)
            bootstrap_preds_list.append(model_boot.predict(df_pred))
        except np.linalg.LinAlgError:
            # Skip this iteration if the design matrix is singular.
            continue
    bootstrap_preds = np.array(bootstrap_preds_list)
    lower_band = np.percentile(bootstrap_preds, 2.5, axis=0)
    upper_band = np.percentile(bootstrap_preds, 97.5, axis=0)
    
    # Print 95% confidence intervals for parameters.
    params = model.params
    conf = model.conf_int(alpha=0.05)
    print("95% Confidence Intervals for logistic regression parameters:")
    s = ""
    SIGNIFICANT = False
    for param in params.index:
        lower, upper = conf.loc[param]
        s += f"{param}: [{lower:0.2f}, {upper:0.2f}] \n"
        if param == "x":
            print(param, lower, upper)
            if lower > 0. or upper < 0.:
                SIGNIFICANT = True
                print(SIGNIFICANT)
                print(param, lower, upper)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(df_model["x"], df_model["y"], label="Data", color="black", zorder=3)
    plt.plot(x_range, predicted_probabilities, label="Logistic Fit", lw=2)
    plt.fill_between(x_range, lower_band, upper_band, color="gray", alpha=0.3, label=f"95% CI")
    plt.title("Logistic Regression with 95% Bootstrap CI")
    plt.xlabel("Fraction of Demo [ ]")
    plt.ylabel("Probability of choosing Action [ ]")
    plt.legend(loc="center right")
    plt.text(
        0.4, 0.5, s,  # (x, y) in axis coordinates (0 to 1)
        transform=plt.gca().transAxes,  # use axis coordinates
        fontsize=12,
        verticalalignment='center',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
    )

    plt.savefig(output_filename)
    # plt.show()
    return SIGNIFICANT


def HBM_data_helper(data_raw):
    X_data = []
    Y_data = None

    for key in data_raw:
        if Y_data is None:
            Y_data = data_raw[key].T[1]
        X_data.append(data_raw[key].T[0])
    X_data = np.array(X_data).T
    return X_data, Y_data


def calculate_residuals(posterior_samples, X_data, Y_data):

    X_tensor = torch.tensor(X_data, dtype=torch.float)
    Y_tensor = torch.tensor(Y_data, dtype=torch.float)
    N, D = X_tensor.shape

    # Compute posterior predictive probabilities for y.
    # We use the posterior samples of the latent theta, beta, and intercept to compute the predicted probability.
    theta_samples = posterior_samples["theta"]
    beta_samples = posterior_samples["beta"]
    intercept_samples = posterior_samples["intercept"]

    # ALR transformation on theta for each sample and observation.
    alr_samples = torch.log(theta_samples[:, :, :D-1] / theta_samples[:, :, D-1].unsqueeze(-1))

    # Compute the linear predictor for each sample and observation.
    logit_p_samples = intercept_samples.unsqueeze(1) + (alr_samples * beta_samples.unsqueeze(1)).sum(dim=2)
    p_samples = torch.sigmoid(logit_p_samples)

    p_mean = p_samples.mean(dim=0).numpy()
    residuals = Y_tensor.numpy() - p_mean

    return residuals


def create_HBM(X_data, Y_data, output_filename):


    # Add a small constant to X_data to avoid zeros (Dirichlet requires strictly positive values)
    epsilon = 1e-6
    X_data_adj = X_data + epsilon
    X_data_adj = X_data_adj / X_data_adj.sum(axis=1, keepdims=True)

    # Convert data to torch tensors
    X_tensor = torch.tensor(X_data_adj, dtype=torch.float)
    Y_tensor = torch.tensor(Y_data, dtype=torch.float)  # Bernoulli expects float values

    N, D = X_tensor.shape  # N observations, D bins

    def model(X, Y):
        # Global parameters:
        precision = pyro.sample("precision", dist.Gamma(2.0, 0.1))
        beta = pyro.sample("beta", dist.Normal(torch.zeros(D - 1), 10.0 * torch.ones(D - 1)))
        intercept = pyro.sample("intercept", dist.Normal(0.0, 10.0))
        
        with pyro.plate("data", N):
            # Latent true composition for each observation.
            theta = pyro.sample("theta", dist.Dirichlet(torch.ones(D)))
            
            # Observation model for the compositional data draw from a Dirichlet: 
            pyro.sample("X_obs", dist.Dirichlet(theta * precision), obs=X)
            
            # Apply an additive log-ratio (ALR) transformation with last component as denominator:
            alr = torch.log(theta[:, :D-1] / theta[:, D-1].unsqueeze(1))
            
            # Logistic regression:
            logit_p = intercept + (alr * beta).sum(dim=1)
            p = torch.sigmoid(logit_p)
            
            # Likelihood for the binary outcome.
            pyro.sample("y_obs", dist.Bernoulli(p), obs=Y)

    pyro.set_rng_seed(42)
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
    mcmc.run(X_tensor, Y_tensor)
    posterior_samples = mcmc.get_samples()

    print("Posterior summary for global parameters:")
    # For precision and intercept, print the mean and 95% credible intervals.
    for param in ["precision", "intercept"]:
        samples = posterior_samples[param].detach().numpy()
        lower, upper = np.percentile(samples, [2.5, 97.5])
        mean_val = np.mean(samples)
        print(f"\nParameter: {param}")
        print(f"Mean: {mean_val:.3f}, 95% CI: [{lower:.3f}, {upper:.3f}]")

    # For beta coefficients, print the 95% CI for each bin effect.
    beta_samples = posterior_samples["beta"].detach().numpy()  # shape: (num_samples, D-1)
    print("\nBeta coefficients (per ALR-transformed bin):")
    for i in range(beta_samples.shape[1]):
        lower, upper = np.percentile(beta_samples[:, i], [2.5, 97.5])
        mean_val = np.mean(beta_samples[:, i])
        significance = "Significant" if lower > 0 or upper < 0 else "Not Significant"
        print(f"Beta[{i}]: Mean = {mean_val:.3f}, 95% CI: [{lower:.3f}, {upper:.3f}] -> {significance}")

    residuals = calculate_residuals(posterior_samples, X_data, Y_data)

    # Plot residuals vs. observation index.
    print("Mean residuals: ", residuals.mean())
    plt.figure(figsize=(8, 4))
    plt.scatter(np.arange(N), residuals)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Observation index")
    plt.ylabel("Residual (Observed - Predicted Probability)")
    plt.title("Residuals of the Logistic Regression Model")
    plt.text(
        0.8, 0.5, f"Mean = {residuals.mean():0.3f}",
        transform=plt.gca().transAxes,  # use axis coordinates
        fontsize=12,
        verticalalignment='center',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
    )
    plt.tight_layout()
    # plt.show()
    plt.savefig(output_filename + "_residuals.png")

    # Convert Pyro MCMC results into an ArviZ InferenceData object and plot traceplots.
    idata = az.from_pyro(mcmc)
    az.plot_trace(idata, var_names=["beta", "intercept", "precision"])
    # plt.show()
    plt.savefig(output_filename + "_arviz.png")

    return posterior_samples

def evaluate_new_points(x, posterior_samples):
    # Add epsilon and normalize to ensure all entries are positive and sum to 1.
    epsilon = 1e-6
    X_new_adj = x + epsilon
    X_new_adj = X_new_adj / X_new_adj.sum()

    # Convert X_new_adj to a torch tensor.
    X_new_tensor = torch.tensor(X_new_adj, dtype=torch.float)

    # In this example, we'll assume that the observed X_new is a good approximation for the latent composition theta.
    # (Alternatively, you might infer a latent theta from X_new using your model's likelihood.)
    theta_new = X_new_tensor  # shape: (D,)

    # Apply the ALR transformation using the last bin as the reference.
    D = X_new_tensor.shape[0]
    alr_new = torch.log(theta_new[:D-1] / theta_new[D-1])

    # Use the posterior samples to compute predictions.
    num_samples = posterior_samples["intercept"].shape[0]

    # Compute the linear predictor for each posterior sample.
    logit_new = posterior_samples["intercept"] + (posterior_samples["beta"] * alr_new).sum(dim=1)
    p_new_samples = torch.sigmoid(logit_new)
    predicted_probability = p_new_samples.mean().item()
    
#     print("Predicted probability that the new group picks the action:", predicted_probability)
    p_new_array = p_new_samples.detach().numpy()
    lower, upper = np.percentile(p_new_array, [2.5, 97.5])
#     print(f"95% Credible Interval for p(action picked): [{lower:.3f}, {upper:.3f}]")

#     hpd_interval = az.hdi(p_new_array, hdi_prob=0.95)
#     print(f"95% HPD interval for p(action picked): [{hpd_interval[0]:.3f}, {hpd_interval[1]:.3f}]")

    
#     plt.hist(p_new_samples.detach().numpy(), bins=20, edgecolor='k')
#     plt.xlabel("Predictive probability")
#     plt.ylabel("Frequency")
#     plt.title("Posterior Predictive Distribution for New Group with Intervention")
#     plt.show()
    
    return predicted_probability, [lower, upper]





if __name__ == "__main__":
    # Read the survey CSV (skip the first two rows to start at row 3).
    df_survey = pd.read_csv("data/full_player_surveydata.csv").drop(index=[0])
    
    # Process the survey data.
    df_subset = process_full_survey(df_survey)
    
    # Create demo results.
    df_demo_result, analysis_target_labs = create_demo_results(df_subset)

    # Read the simulation data.
    df_simulation = pd.read_csv("data/ganz_data_full_updateAug24.csv")
    df_sim_result = create_simulation_results(df_simulation)

    for frac_col in [entry[0] +"_frac" for entry in analysis_target_labs]:
        frac_dicts = df_demo_result[frac_col]
        df_frac = pd.DataFrame(list(frac_dicts))
        
        # Determine the number of categories to plot.
        n_categories = len(df_frac.columns)
        
        # Create a figure with subplots; here we use one row and as many columns as there are categories.
        fig, axes = plt.subplots(1, n_categories, figsize=(6 * n_categories, 4), squeeze=False)
        
        # Loop through each column in the DataFrame and plot the histogram.
        for i, col in enumerate(df_frac.columns):
            ax = axes[0, i]
            nonzero_vals = df_frac[col][df_frac[col] != 0]
            ax.hist(nonzero_vals, bins=np.linspace(0, 1, 6), edgecolor="black")
            ax.set_title(col)
            ax.set_xlabel("Fraction")
            ax.set_ylabel("Frequency")
            ax.text(
                0.8, 0.5, f"Mean = {nonzero_vals.mean():0.2f}",  # (x, y) in axis coordinates (0 to 1)
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='center',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
            )
        
        # Adjust the layout and save the figure.
        plt.tight_layout()
        plt.savefig(f"demo_{frac_col}.png")
        # plt.close()
    
    significants = []
    sample_cache = {}
    for c, unique_c_vals in analysis_target_labs:
        sample_cache[c] = {}
        for move in [1, 2]:
            sample_cache[c][move] = {}
            if move == 1:
                action_inds = list(range(len(move_1_2_options_desc())))
                action_labs = move_1_2_options_desc()
            else:
                action_inds = list(range(len(move_2_2_options_desc())))
                action_labs = move_2_2_options_desc()
            for action_n in action_inds:
                res = {}
                for c_val in unique_c_vals:
                    try:
                        result_array = extract_2d_array(
                            df_sim_result, df_demo_result, move_index=action_n, gender_key=c_val, gender_frac_col=c + "_frac", move=move,
                        )

                        res[c_val] = result_array
                        # sig = plot_data_logistic_regression(result_array, output_filename=f"move{move}_action{action_n}_{c}_{c_val}.png")
                        # if sig:
                        #     significants.append([move, action_n, c, c_val, action_labs[action_n]])
                    except np.linalg.LinAlgError:
                        pass
                
                
                X_data, Y_data = HBM_data_helper(res)
                samples = create_HBM(X_data, Y_data, output_filename=f"hbm_move{move}_action{action_n}_{c}")
                sample_cache[c][move][action_n] = [samples, res] 
                # with open('hbm_sample_data.pkl', 'wb') as outp:
                #     pickle.dump(sample_cache, outp)

                n_unique_c_vals = unique_c_vals.shape[0]
                X_new = [
                    X_data.mean(axis=0),
                    np.array([1./n_unique_c_vals] * n_unique_c_vals),
                ]
                for i in range(n_unique_c_vals):
                    tmp = [0.0] * n_unique_c_vals
                    tmp[i] = 1.0
                    X_new.append(np.array(tmp))
                
                print("\n---------------\n")
                print(action_labs[action_n])
                pprint(res.keys())
                print("Baseline mean probability ", Y_data.mean())
                print("Baseline mean fraction", X_data.mean(axis=0))
                for x in X_new:
                    pprint(x)
                    pprint(evaluate_new_points(x, samples)[0])
    with open('hbm_sample_data.pkl', 'wb') as outp:
        pickle.dump(sample_cache, outp)


    print(significants)

# Todo:
# - how to treat independent attributes and categoricals
# - add concatenation of low-yield bins
# - add HBM model + model storage to save compute time
# - create big table for inidividual bins? (at least for estimate)


# "(b) Hold fire unless fired upon",
# [1, 1, 'Professionality', 'Entry level professional (up to 5 years experience)'], 
# Higher fraction, higher probability
# [1, 1, 'Background', np.str_('Military')], 
# Higher fraction, higher probability

# "(c) Hold fire without approval from President",
# [1, 2, 'DeathPenaltySupport', 'Strongly oppose'], 
# Higher fraction, higher probability
# [1, 2, 'Gender', 'Male']
# Higher fraction, lower probability
# [1, 2, 'Gender', 'Female']
# Higher fraction, higher probability

# "(e) Place automated defenses on manual firing control. AI decides target/human decides firing.",
# [1, 4, 'AIExp', 'Artificial intelligence policy expert'], 
# Higher fraction, lower probability
# [1, 4, 'ChinaMilExp', 'Policy and technical expert'], 
# Higher fraction, lower probability
# [1, 4, 'ChinaMilExp', 'Routine understanding'], 
# Higher fraction, higher probability
# [1, 4, 'USMilExp', 'Routine understanding'], 
# higher fraction, higher probability
# [1, 4, 'USMilExp', 'Policy and technical expert']]
# higher fraction, lower probability

# '(1) Preserve Status Quo/Deter'
# [2, 0, 'DeathPenaltySupport', 'Oppose'], 
# higher fraction, higher probability

# '(i) Conduct Foreign Intelligence'
# [2, 10, 'Age', '45-54 years old'], 
# higher fraction, lower probability
# [2, 10, 'Background', np.str_('Other')], 
# higher fraction, lower probability

# '(g) Clandestine/Special Operations'
# [2, 8, 'Background', np.str_('Private Industry')], 
# higher fraction, higher probability
# [2, 8, 'DeathPenaltySupport', 'Favor'],
# higher fraction, higher probability 
# [2, 8, 'DeathPenaltySupport', 'Strongly oppose'], 
# higher fraction, lower probability

# '(e) Economic Punishment'
# [2, 6, 'AIExp', 'Routine understanding of concepts behind artificial intelligence'], 
# higher fraction, lower probability

# '(j) Conduct Domestic Intelligence'
# [2, 11, 'AIExp', 'Routine understanding of concepts behind artificial intelligence']
# higher fraction, higher probability