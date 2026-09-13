
using CSV
using DataFrames

# Skip second row with comments/question texts
df = CSV.read("data/full_player_surveydata.csv", DataFrame; skipto=3)
# print(df[:, ["Q19"]])
# print(names(df))

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
    :"Q20", # Did you trust AI in Simulation?
    :"Q25", # Credibel SImulation?
]
df_subset = df[:, selected_columns]

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
    ["Q20", "AITrust"],
    ["Q25", "SimTrust"],
]

println(names(df_subset))
for pair in column_rename
    old_col = Symbol(pair[1])
    new_col = Symbol(pair[2])
    rename!(df_subset, old_col => new_col)
end

# Optionally, you can display the new DataFrame.
# println(df_subset)
println(names(df_subset))


analysis_pairs = [
    ["ChinaInavdes_pre", "ChinaInavdes_post"],
    ["LAWSWorry_pre", "LAWSWorry_post"],
    ["LAWSSupport_pre", "LAWSSupport_post"],
]
target_treatments = ["ChinaStatus", "AIAcc"]



# Helper function to compute conditional probabilities P(t2 | t1) for a given DataFrame.
function compute_conditional_probs_for_df(df, tar1, tar2, tar1_uvals, tar2_uvals)
    probs = []  # This will be a vector of vectors: one per t1 value.
    
    for t1 in tar1_uvals
        denom = sum(df[!, tar1] .== t1)
        t1_probs = []  # Will store probabilities for each t2 given the current t1.
        
        for t2 in tar2_uvals
            num = sum((df[!, tar1] .== t1) .& (df[!, tar2] .== t2))
            p_current = denom > 0 ? num / denom : 0.0
            push!(t1_probs, p_current)
        end
        
        push!(probs, t1_probs)
    end
    
    return probs
end

# Main function to perform the conditional analysis.
function do_corr_analysis(df_0, analysis_pair, target_treatment)
    tar1 = analysis_pair[1]
    tar2 = analysis_pair[2]

    # Remove rows with missing values in tar1, tar2, and target_treatment columns.
    df_0 = dropmissing(df_0[:, [tar1, tar2, target_treatment]])

    # Extract unique values.
    tar1_uvals = unique(df_0[!, tar1])
    tar2_uvals = unique(df_0[!, tar2])
    treatment_uvals = unique(df_0[!, target_treatment])
    
    println("Unique values:")
    println("tar1: ", tar1_uvals)
    println("tar2: ", tar2_uvals)
    println("Treatment: ", treatment_uvals)
 
    # Create filtered DataFrames based on the first two unique treatments.
    if length(treatment_uvals) < 2
        error("Not enough treatment groups in the data!")
    end

    df_01 = df_0[df_0[!, target_treatment] .== treatment_uvals[1], :]
    df_02 = df_0[df_0[!, target_treatment] .== treatment_uvals[2], :]
    df_list = [df_0, df_01, df_02]

    # Compute the nested list of conditional probabilities for each DataFrame.
    conditional_probs = []
    for df in df_list
        probs = compute_conditional_probs_for_df(df, tar1, tar2, tar1_uvals, tar2_uvals)
        push!(conditional_probs, probs)
    end

    return conditional_probs, tar1_uvals, tar2_uvals, treatment_uvals
end

# Function to build a DataFrame from a probability matrix.
# This DataFrame will have a column for each unique tar2 value and a first column for tar1 values.
function build_conditional_probs_df(prob_matrix, tar1_uvals, tar2_uvals)
    # Create a DataFrame with a column for tar1 values.
    df_cond = DataFrame(tar1 = tar1_uvals)
    
    # For each unique tar2 value, add a column to the DataFrame.
    for (j, t2_val) in enumerate(tar2_uvals)
        # Extract the j-th element from each row in the probability matrix, rounding to 3 decimal places.
        col_data = [ round(row[j], digits=3) for row in prob_matrix ]
        df_cond[!, string(t2_val)] = col_data
    end
    
    return df_cond
end


for (pair_i, pair) in enumerate(analysis_pairs)
    for (treat_i, treat) in enumerate(target_treatments)
        conditional_probs, tar1_uvals, tar2_uvals, treatment_uvals = do_corr_analysis(df_subset, pair, treat)

        for i in [1, 2, 3, 4]
            if i == 1
                println(treat .* " Joint Both Treatments")
            elseif i < 4
                println(treat .* " " .* treatment_uvals[i - 1])
            else
                println(treat .* " " .* treatment_uvals[1] .* " - " .* treatment_uvals[2])
            end
            println("p( " .* pair[2] .* " | " .* pair[1] .* " )")
            println(pair[1] .* " / " .* pair[2])

            if i < 4
                df_labeled = build_conditional_probs_df(conditional_probs[i], tar1_uvals, tar2_uvals)
            else
                df_labeled = build_conditional_probs_df(conditional_probs[2] - conditional_probs[3], tar1_uvals, tar2_uvals)
            end
            println(df_labeled)
            println()
        end
    end
end

# conditional_probs, tar1_uvals, tar2_uvals, treatment_uvals = do_corr_analysis(df_subset, analysis_pairs[1], target_treatments[1])

# # For example, use the probabilities computed for the full DataFrame (first element in conditional_probs):
# full_probs = conditional_probs[1]
# #
# # Now convert these probabilities into a labeled DataFrame:
# df_labeled = build_conditional_probs_df(full_probs, tar1_uvals, tar2_uvals)
# #
# # println("Conditional probabilities (P(tar2 | tar1)) for the full DataFrame:")
# println(df_labeled)