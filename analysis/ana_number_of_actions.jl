
using Random
using CSV
using DataFrames
include("../src/game.jl")

Random.seed!(1234)

ai_column_name = "AI Accuracy"
ai_accuracies = ["70-85%", "95-99%"]

train_column_name = "AI System Training"
train_quality = ["basic", "significant"]

china_column_name = "China Status"
china_treatments = ["revisionist", "status_quo"]

df = CSV.read("results/data2023-11-15_2.csv", DataFrame)
# df = CSV.read("results/v1_results.csv", DataFrame)

df_real = CSV.read("data/ganz_data_full.csv", DataFrame)
df_dialogno = CSV.read("results/sensitivity_studies/data_dialogno.csv", DataFrame)
df_dialog1 = CSV.read("results/sensitivity_studies/data_dialog1.csv", DataFrame)
df_dialog3 = CSV.read("results/sensitivity_studies/data_dialog3.csv", DataFrame)
df_dialog6 = CSV.read("results/sensitivity_studies/data_dialog6.csv", DataFrame)
df_nochief = CSV.read("results/sensitivity_studies/data_nochiefs.csv", DataFrame)
df_playeruniform = CSV.read("results/sensitivity_studies/data_playeruniform.csv", DataFrame)


function get_number_of_answers(df)
    options = [move_1_2_options_desc()..., move_2_2_options_desc()...]

    res = df[!, options]
    res_val = [[values(e)...] for e in eachrow(res)]
    action_counts_all = [sum(r) for r in res_val]
    action_counts = mean(action_counts_all)

    # do bootstrap
    n_b = 20000
    n_dat = length(action_counts_all)

    boot_res = []
    for _ in 1:n_b
        boot_count = StatsBase.sample(action_counts_all, n_dat, replace=true)
        boot = mean(boot_count)
        append!(boot_res, [boot])
    end

    lower = sort([d for d in boot_res])[round(Int, n_b * 0.025)]
    upper = sort([d for d in boot_res])[round(Int, n_b * 0.975)]
    errors = ((action_counts - lower), (upper - action_counts))

    return action_counts, errors
end

# get_number_of_answers(df_dialogno)
# get_number_of_answers(df_dialog1)
# get_number_of_answers(df_dialog3)
# get_number_of_answers(df_dialog6)

get_number_of_answers(df_dialog3)
get_number_of_answers(df_real)

# Result --> need more data, but will be hard without more human data
# (10.61, (0.73, 0.73))
# (9.20, (1.0, 1.07))