
using Random
using Plots
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

    fac_aggro = mean([sum(r .* mask_aggro) for r in res_val])
    fac_paci = mean([sum(r .* mask_paci)for r in res_val])

    return (fac_aggro - fac_paci)/norm
end

aggro = [
    get_frac_aggr(df_dialogno),
    get_frac_aggr(df_dialog1),
    get_frac_aggr(df_dialog3),
    get_frac_aggr(df_dialog6),
]
aggro_h = get_frac_aggr(df_real)

plot(
    xlabel="Length of Simulated Dialog [a.u.]",
    ylabel="Aggresivness [a.u.]",
    title="",
    legend=:bottomright,
)
plot!(
    [0., 1., 3., 6.],
    aggro,
    marker=true,
    label="LLM Data",
    dpi=300,
)
hline!([aggro_h], linestyle=:dash, linecolor="#100B00", label="Human Players")
savefig("both_moves_aggresivness_v_dialoglength.png")
