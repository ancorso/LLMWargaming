 
using CSV
using DataFrames
using Plots
using GLM 
# Needed?
using StatsBase
using Statistics
include("../src/game.jl")
include("../src/config.jl")

# Skip second row with comments/question texts
df_survey = CSV.read("data/full_player_surveydata.csv", DataFrame; skipto=3)

function process_full_survey(df)

    selected_columns = [
        :"Column5", # Group ID
        :"Q19", # Player ID
        # :"Q18", # Self identified Group ID
        :"Q2", # Age
        :"Q4", # Gender 
        :"Q22", # Nationality
        :"Q23", # Background (Comma separated, need to turn into list)
        :"Q24", # Professionality Level
        :"Q15", # Ai Expertise
        :"Q16", # China Military Expertise
        :"Q17", # US Military Expertise
        :"Q8", # p(China invades Taiwan) pre-game
        :"Q8_1", # p(China invades Taiwan) post-game
        :"Q9", # Long-term goals of China pre-game
        :"Q9_1", # Long-term goals of China post-game
        :"Q10", # Short-term goals of China pre-game
        :"Q10_1", # Short-term goals of China post-game
        :"Q11", # Worry About LAWS pre-game
        :"Q11_1", # Worry About LAWS post-game
        :"Q12", # LAWS Support pre-game
        :"Q12_1", # LAWS Support post-game
        # :"Q14", # Support Death Penalty
        :"Column34", # AI Acc
        :"Column35", # AI Team Training
        :"Column36", # China Status
        :"Column37", # Move 1 RoE
        :"Q20", # Did you trust AI in Simulation?
        :"Q25", # Credibel SImulation?
    ]
    df_subset = df_survey[:, selected_columns] # should be df[:, selected_columns]?

    column_rename = [
        ["Column5", "GID"],
        ["Q19", "PID"],
        ["Q2", "Age"],
        ["Q4", "Gender"],
        ["Q22", "Nat"],
        ["Q23", "Background"],
        ["Q24", "Professionality"],
        ["Q15", "AIExp"],
        ["Q16", "ChinaMilExp"],
        ["Q17", "USMilExp"],
        ["Q8", "ChinaInavdes_pre"],
        ["Q8_1", "ChinaInavdes_post"],
        ["Q9", "ChinaGoalLong_pre"],
        ["Q9_1", "ChinaGoalLong_post"],
        ["Q10", "ChinaGoalShort_pre"],
        ["Q10_1", "ChinaGoalShort_post"],
        ["Q11", "LAWSWorry_pre"],
        ["Q11_1", "LAWSWorry_post"],
        ["Q12", "LAWSSupport_pre"],
        ["Q12_1", "LAWSSupport_post"],
        ["Column34", "AIAcc"],
        ["Column35", "AITrain"],
        ["Column36", "ChinaStatus"],
        ["Column37", "Move1"],
        ["Q20", "AITrust"],
        ["Q25", "SimTrust"],
    ]

    println(names(df_subset))
    for pair in column_rename
        old_col = Symbol(pair[1])
        new_col = Symbol(pair[2])
        rename!(df_subset, old_col => new_col)
    end

    # dropmissing!(df_subset, :"GID")
    # dropmissing!(df_subset, :"Age")
    # dropmissing!(df_subset, :"Gender")
    dropmissing!(df_subset)

    return df_subset
end

function create_demo_results(df_subset)



    # Get unique values for GID and TAR_DEMO
    unique_gids = unique(df_subset[!, "GID"])
    # Todo: Adjust for TAR_DEMO
    unique_age = unique(df_subset[!, "Age"])
    unique_gender = unique(df_subset[!, "Gender"])
    unique_nat = unique(df_subset[!, "Nat"])
    unique_back = unique(df_subset[!, "Background"])
    unique_exp = unique(df_subset[!, "Professionality"])

    analysis_target_labs = [
        ["Age", unique_age], 
        ["Gender", unique_gender], 
        # ["Nat", unique_nat], 
        # ["Background", unique_back], 
        ["Professionality", unique_exp],
    ]

    demo_results = []
    for tar_id in unique_gids
        # Skip UNK players
        if occursin("UNK", tar_id)
            continue
        end
        gid_filter = df_subset."GID" .== tar_id
        df_filtered = df_subset[gid_filter, :]
        n_players = sum(gid_filter)

        row_info = Dict(:GID => tar_id, :n => n_players)
        for (c, uniques) in analysis_target_labs
            demo_counter = Dict(key => 0 for key in uniques)
            for row in eachrow(df_filtered)
                demo_counter[row[c]] += 1
            end
            
            row_info[Symbol(c)] = demo_counter

            row_info[Symbol(c * "_frac")] = Dict(k => v / n_players for (k, v) in demo_counter)
        end
        
        push!(demo_results, row_info)
    end

    return demo_result_df = DataFrame(demo_results)
end

function create_simulation_results(df_simulation)
    move1_options = move_1_2_options_desc()             
    move2_options = move_2_2_options_desc()[2:end]
    result_df = DataFrame(game_ID = String[], move1 = Vector{Any}[], move2 = Vector{Any}[])

    for gid in unique(df_simulation[!, "game_ID"])
        row = df_simulation[df_simulation."game_ID" .== gid, :]
        
        vals_move1 = collect(row[1, move1_options])
        vals_move2 = collect(row[1, move2_options])

        push!(result_df, (game_ID = gid, move1 = vals_move1, move2 = vals_move2))
    end

    return result_df
end

df_subset = process_full_survey(df_survey)
df_demo_result = create_demo_results(df_subset)
df_sim_result = create_simulation_results(df_real_aug24)



age_frac_dicts = df_demo_result[!, "Gender_frac"]
df_age = DataFrame(age_frac_dicts)

plot_list = []
for col in names(df_age)
    nonzero_vals = [x for x in df_age[!, col] if x != 0]
    p = histogram(nonzero_vals, bins=range(0., 1.0, length=5), title=col, xlabel="Fraction", ylabel="Frequency")
 
    push!(plot_list, p)
end
plot(plot_list..., layout = (length(plot_list), 1), size=(600, 200 * length(plot_list)))
savefig("demo_gender.png")

age_frac_dicts = df_demo_result[!, "Age_frac"]
df_age = DataFrame(age_frac_dicts)

plot_list = []
for col in names(df_age)
    nonzero_vals = [x for x in df_age[!, col] if x != 0]
    p = histogram(nonzero_vals, bins=range(0., 1.0, length=5), title=col, xlabel="Fraction", ylabel="Frequency")
 
    push!(plot_list, p)
end
plot(plot_list..., layout = (length(plot_list), 1), size=(600, 200 * length(plot_list)))
savefig("demo_age.png")

age_frac_dicts = df_demo_result[!, "Professionality_frac"]
df_age = DataFrame(age_frac_dicts)

plot_list = []
for col in names(df_age)
    nonzero_vals = [x for x in df_age[!, col] if x != 0]
    p = histogram(nonzero_vals, bins=range(0., 1.0, length=5), title=col, xlabel="Fraction", ylabel="Frequency")
 
    push!(plot_list, p)
end
plot(plot_list..., layout = (length(plot_list), 1), size=(600, 200 * length(plot_list)))
savefig("demo_prof.png")

df_subset
df_sim_result
df_demo_result


function extract_2d_array(df1::DataFrame, df2::DataFrame, move_index::Int, gender_key::String, gender_frac_col::Symbol)
    # Preallocate a 2D array to store the results
    result = Array{Any}(undef, nrow(df1), 2)
    
    for i in 1:nrow(df1)
        # Get the game_ID from df1
        game_id = df1[i, :game_ID]
        
        # Find the corresponding row in df2 where GID matches game_id
        row_idx = findfirst(row -> row.GID == game_id, eachrow(df2))
        if isnothing(row_idx)
            error("Game ID $game_id not found in df2")
        end
        
        # Extract the dictionary from the specified column in df2
        gender_frac_dict = df2[row_idx, gender_frac_col]
        # Retrieve the x value using the provided gender_key
        x_value = gender_frac_dict[gender_key]
        
        # Get the move1 vector from df1 and select the element at move_index (y value)
        move1_vector = df1[i, :move1]
        y_value = move1_vector[move_index]
        
        # Store the pair in the result array
        result[i, :] = [x_value, y_value]
    end
    
    return result
end

function plot_data_logistic_regression(result)

    # Convert the first column to Float64 and the second to integer (if needed)
    df_model = DataFrame(x = Float64.(result[:, 1]), y = Int.(result[:, 2]))

    # Fit the logistic regression model (y ~ x)
    model = glm(@formula(y ~ x), df_model, Binomial(), LogitLink())

    # Create a range of x values for prediction
    x_range = range(minimum(df_model.x), maximum(df_model.x), length=100)
    x_range_vec = collect(x_range)  # ensure it's a vector
    df_pred = DataFrame(x = x_range_vec)

    # Get predictions from the fitted model
    predicted_probabilities = predict(model, df_pred)

    # --- Bootstrap resampling to compute 95% uncertainty bands ---
    function bootstrap_predictions(df_model, x_range_vec; n_bootstrap=1000)
        n = nrow(df_model)
        # Preallocate an array to store predictions for each bootstrap sample
        bootstrap_preds = zeros(n_bootstrap, length(x_range_vec))
        
        for i in 1:n_bootstrap
            # Sample indices with replacement
            sample_indices = sample(1:n, n; replace=true)
            df_boot = df_model[sample_indices, :]
            
            # Fit logistic regression on the bootstrap sample
            model_boot = glm(@formula(y ~ x), df_boot, Binomial(), LogitLink())
            
            # Predict probabilities for the x_range
            df_pred_boot = DataFrame(x = x_range_vec)
            bootstrap_preds[i, :] = predict(model_boot, df_pred_boot)
        end
        
        return bootstrap_preds
    end

    # Run bootstrap with, e.g., 1000 iterations
    n_bootstrap = 1000
    bootstrap_preds = bootstrap_predictions(df_model, x_range_vec; n_bootstrap=n_bootstrap)

    # Compute the 2.5th and 97.5th percentiles at each x value
    lower_band = [quantile(bootstrap_preds[:, j], 0.025) for j in 1:length(x_range_vec)]
    upper_band = [quantile(bootstrap_preds[:, j], 0.975) for j in 1:length(x_range_vec)]

    # --- Plotting the results ---
    # Plot the raw data points
    plotty = scatter(df_model.x, df_model.y, label="Data", 
            title="Logistic Regression with 95% Bootstrap CI",
            xlabel="Fraction of Demo [ ]", 
            ylabel="Probability of choosing Action [ ]", 
            legend=:topright)

    # Overlay the fitted logistic regression curve
    plot!(x_range_vec, predicted_probabilities, label="Logistic Fit", lw=2)

    # Add a shaded region for the 95% confidence interval from the bootstrap
    plot!(x_range_vec, lower_band, fillrange=upper_band, fillalpha=0.3, 
          label="95% CI", lw=0)

    # --- Compute and print 95% confidence intervals for logistic regression parameters ---
    coef_est = coef(model)
    se = stderror(model)
    ci_lower = coef_est .- 1.96 .* se
    ci_upper = coef_est .+ 1.96 .* se

    println("95% Confidence Intervals for logistic regression parameters:")
    for (name, est, lower, upper) in zip(coefnames(model), coef_est, ci_lower, ci_upper)
        println("$name: [$lower, $upper]")
    end

    # # --- Compute and print odds ratios and their 95% confidence intervals ---
    # println("\nOdds Ratios and their 95% Confidence Intervals:")
    # for (name, beta, lower, upper) in zip(coefnames(model), coef_est, ci_lower, ci_upper)
    #     odds_ratio = exp(beta)
    #     or_lower = exp(lower)
    #     or_upper = exp(upper)
    #     println("$name: odds ratio = $odds_ratio, 95% CI = [$or_lower, $or_upper]")
    # end

    return plotty
end

move_1_2_options_shortdesc()[4]
move_2_2_options_desc()[2:end]

result = extract_2d_array(df_sim_result, df_demo_result, 6, "Male", Symbol("Gender_frac"))
plot_data_logistic_regression(result)
savefig("move1_f_male.png")

result = extract_2d_array(df_sim_result, df_demo_result, 6, "Female", Symbol("Gender_frac"))
plot_data_logistic_regression(result)
savefig("move1_f_female.png")

result = extract_2d_array(df_sim_result, df_demo_result, 6, "25-34 years old", Symbol("Age_frac"))
plot_data_logistic_regression(result)
savefig("move1_f_25-34years.png")

result = extract_2d_array(df_sim_result, df_demo_result, 6, "35-44 years old", Symbol("Age_frac"))
plot_data_logistic_regression(result)
savefig("move1_f_35-44years.png")

result = extract_2d_array(df_sim_result, df_demo_result, 6, "45-54 years old", Symbol("Age_frac"))
plot_data_logistic_regression(result)
savefig("move1_f_45-54years.png")


# unique_exp = unique(df_subset[!, "Professionality"])

# result = extract_2d_array(df_sim_result, df_demo_result, 6, "Mid-level professional (5-15 years experience)", Symbol("Professionality_frac"))
# plot_data_logistic_regression(result)
# savefig("move1_f_midlevel.png")

# result = extract_2d_array(df_sim_result, df_demo_result, 6, "Senior professional (15+ years experience)", Symbol("Professionality_frac"))
# plot_data_logistic_regression(result)
# savefig("move1_f_senior.png")

# result = extract_2d_array(df_sim_result, df_demo_result, 6, "Entry level professional (up to 5 years experience)", Symbol("Professionality_frac"))
# plot_data_logistic_regression(result)
# savefig("move1_f_entry.png")



# result = extract_2d_array(df_sim_result, df_demo_result, 5, "Mid-level professional (5-15 years experience)", Symbol("Professionality_frac"))
# plot_data_logistic_regression(result)
# savefig("move1_e_midlevel.png")

# result = extract_2d_array(df_sim_result, df_demo_result, 5, "Senior professional (15+ years experience)", Symbol("Professionality_frac"))
# plot_data_logistic_regression(result)
# savefig("move1_e_senior.png")

# result = extract_2d_array(df_sim_result, df_demo_result, 5, "Entry level professional (up to 5 years experience)", Symbol("Professionality_frac"))
# plot_data_logistic_regression(result)
# savefig("move1_e_entry.png")





# result = extract_2d_array(df_sim_result, df_demo_result, 2, "Male", Symbol("Gender_frac"))
# plot_data_logistic_regression(result)
# savefig("move1_b_male.png")

# result = extract_2d_array(df_sim_result, df_demo_result, 2, "Female", Symbol("Gender_frac"))
# plot_data_logistic_regression(result)
# savefig("move1_b_female.png")




# result = extract_2d_array(df_sim_result, df_demo_result, 4, "Male", Symbol("Gender_frac"))
# plot_data_logistic_regression(result)
# savefig("move1_d_male.png")

# result = extract_2d_array(df_sim_result, df_demo_result, 4, "Female", Symbol("Gender_frac"))
# plot_data_logistic_regression(result)
# savefig("move1_d_female.png")





# result = extract_2d_array(df_sim_result, df_demo_result, 1, "Male", Symbol("Gender_frac"))
# plot_data_logistic_regression(result)
# savefig("move1_a_male.png")

# result = extract_2d_array(df_sim_result, df_demo_result, 1, "Female", Symbol("Gender_frac"))
# plot_data_logistic_regression(result)
# savefig("move1_a_female.png")
