
using Random
using Plots
using CSV
using DataFrames
using StatsBase
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

function get_frac_aggr(df)
    options = [move_1_2_options_desc()..., move_2_2_options_desc()...]

    res = df[!, options]
    res_val = [[values(e)...] for e in eachrow(res)]

    move_1_aggro = [1, 0, 0, 1, 1, 1, 0]
    move_2_aggro = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]
    mask_aggro = [move_1_aggro..., move_2_aggro...]
    norm = length(mask_aggro)

    move_1_paci = [0, 1, 1, 0, 0, 0, 1]
    move_2_paci = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
    mask_paci = [move_1_paci..., move_2_paci...]

    fac_aggro_all = [sum(r .* mask_aggro) for r in res_val]
    fac_paci_all = [sum(r .* mask_paci) for r in res_val]

    aggro = (mean(fac_aggro_all) - mean(fac_paci_all)) / norm

    # do bootstrap
    n_b = 20000
    n_dat = length(fac_aggro_all)

    boot_res = []
    for _ in 1:n_b
        boot_aggro = StatsBase.sample(fac_aggro_all, n_dat, replace=true)
        boot_paci = StatsBase.sample(fac_paci_all, n_dat, replace=true)

        boot = (mean(boot_aggro) - mean(boot_paci)) / norm

        append!(boot_res, [boot])
    end

    lower = sort([d for d in boot_res])[round(Int, n_b * 0.025)]
    upper = sort([d for d in boot_res])[round(Int, n_b * 0.975)]
    errors = ((aggro - lower), (upper - aggro))

    return aggro, errors
end

function make_plot()
    aggro = [
        get_frac_aggr(df_dialogno),
        get_frac_aggr(df_dialog1),
        get_frac_aggr(df_dialog3),
        get_frac_aggr(df_dialog6),
    ]
    aggro_h = get_frac_aggr(df_real)

    s = plot(
        xlabel="Length of Simulated Dialog [a.u.]",
        ylabel="Aggresivness [a.u.]",
        title="",
        legend=:bottomright,
    )
    scatter!(
        [0., 1., 3., 6.],
        [a[1] for a in aggro],
        yerror=[a[2] for a in aggro],
        marker=true,
        label="LLM Data (95% Conf.)",
        dpi=300,
    )
    plot!([0., 1., 3., 6.], fill(aggro_h[1], 4), ribbon=aggro_h[2], label="Human Players (95% Conf.)", fillalpha=0.2)

    return s
end

make_plot()
# savefig("both_moves_aggresivness_v_dialoglength.png")
