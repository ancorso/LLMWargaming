
using Random
using CSV
using DataFrames
include("../src/game.jl")

SEED = 42

ai_column_name = "AI Accuracy"
ai_accuracies = ["70-85%", "95-99%"]

train_column_name = "AI System Training"
train_quality = ["basic", "significant"]

china_column_name = "China Status"
china_treatments = ["revisionist", "status_quo"]

df = CSV.read("results/data2023-11-15_2.csv", DataFrame)
# df = CSV.read("results/v1_results.csv", DataFrame)

df_real = CSV.read("data/ganz_data_full.csv", DataFrame)
df_real_feb24 = CSV.read("data/ganz_data_full_updateFeb24.csv", DataFrame)
df_dialogno = CSV.read("results/sensitivity_studies/data_dialogno.csv", DataFrame)
df_dialog1 = CSV.read("results/sensitivity_studies/data_dialog1.csv", DataFrame)
df_dialog2 = CSV.read("results/sensitivity_studies_new/data_dialog2.csv", DataFrame)
df_dialog3 = CSV.read("results/sensitivity_studies/data_dialog3.csv", DataFrame)
df_dialog4 = CSV.read("results/sensitivity_studies_new/data_dialog4.csv", DataFrame)
df_dialog5 = CSV.read("results/sensitivity_studies_new/data_dialog5.csv", DataFrame)
df_dialog6 = CSV.read("results/sensitivity_studies/data_dialog6.csv", DataFrame)
df_dialog7 = CSV.read("results/sensitivity_studies_new/data_dialog7.csv", DataFrame)

df4_dialogno = CSV.read("results/gpt4_turbo/data_dialogno.csv", DataFrame)
df4_dialog1 = CSV.read("results/gpt4_turbo/data_dialog1.csv", DataFrame)
df4_dialog2 = CSV.read("results/gpt4_turbo/data_dialog2.csv", DataFrame)
df4_dialog3 = CSV.read("results/gpt4_turbo/data_dialog3_2.csv", DataFrame)
# df4_dialog3_new = CSV.read("results/gpt4_turbo/data_dialog3.csv", DataFrame)
df4_dialog4 = CSV.read("results/gpt4_turbo/data_dialog4.csv", DataFrame)
df4_dialog5 = CSV.read("results/gpt4_turbo/data_dialog5.csv", DataFrame)
df4_dialog6 = CSV.read("results/gpt4_turbo/data_dialog6.csv", DataFrame)
df4_dialog7 = CSV.read("results/gpt4_turbo/data_dialog7.csv", DataFrame)


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

get_number_of_answers(df4_dialog3)


get_number_of_answers(df_real)

get_number_of_answers(df_dialogno)
get_number_of_answers(df_dialog1)
get_number_of_answers(df_dialog2)
get_number_of_answers(df_dialog3)
get_number_of_answers(df_dialog4)
get_number_of_answers(df_dialog5)
get_number_of_answers(df_dialog6)
get_number_of_answers(df_dialog7)


get_number_of_answers(df4_dialogno)
get_number_of_answers(df4_dialog1)
get_number_of_answers(df4_dialog2)
get_number_of_answers(df4_dialog3)
get_number_of_answers(df4_dialog4)
get_number_of_answers(df4_dialog5)
get_number_of_answers(df4_dialog6)
get_number_of_answers(df4_dialog7)


function make_plot_both()
    Random.seed!(SEED)
    n_actions = [
        get_number_of_answers(df_dialogno),
        get_number_of_answers(df_dialog1),
        get_number_of_answers(df_dialog2),
        get_number_of_answers(df_dialog3),
        get_number_of_answers(df_dialog4),
        get_number_of_answers(df_dialog5),
        get_number_of_answers(df_dialog6),
        get_number_of_answers(df_dialog7),
    ]
    n_actions_h = get_number_of_answers(df_real)
    n_actions4 = [
        get_number_of_answers(df4_dialogno),
        get_number_of_answers(df4_dialog1),
        get_number_of_answers(df4_dialog2),
        get_number_of_answers(df4_dialog3),
        get_number_of_answers(df4_dialog4),
        get_number_of_answers(df4_dialog5),
        get_number_of_answers(df4_dialog6),
        get_number_of_answers(df4_dialog7),
    ]

    s = plot(
        xlabel="Length of Simulated Dialog [a.u.]",
        ylabel="Number of Chosen Actions [ ]",
        title="",
        legend=:bottomright,
        # ylims=[0.11, 0.32],
    )
    plot!([0., 7.], fill(n_actions_h[1], 2), ribbon=n_actions_h[2], label="Human Players (95% Conf.)", fillalpha=0.2, color=2, lw=4.)
    scatter!(
        [0., 1., 2., 3., 4., 5., 6., 7.] .- 0.05,
        [a[1] for a in n_actions],
        yerror=[a[2] for a in n_actions],
        marker=true,
        label="GPT-3.5 (95% Conf.)",
        dpi=300,
        color=1,
    )
    scatter!(
        [0., 1., 2., 3., 4., 5., 6., 7.] .+ 0.05,
        [a[1] for a in n_actions4],
        yerror=[a[2] for a in n_actions4],
        marker=true,
        label="GPT-4 (95% Conf.)",
        dpi=300,
        color=0,
    )   

    return s
end

make_plot_both()
# savefig("both_moves_nactions_v_dialoglength.png")

# Result --> need more data, but will be hard without more human data
# (10.61, (0.73, 0.73))
# (9.20, (1.0, 1.07))