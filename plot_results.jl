using Random
using StatsPlots
using Plots.Measures
# using StatsBase
using CSV
using DataFrames
include("src/game.jl")

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

function compare_treatments_move1(df, column_name, treatments)
    short_options = move_1_2_options_shortdesc()
    options = move_1_2_options_desc()

    group = repeat(treatments, inner=length(options))

    y = [sum(df[df[!, column_name] .== t, o]) for t in treatments for o in options]
    names = repeat(short_options, outer=2)

    return names, y, group
end

function compare_treatments_move2(df, column_name, treatments)
    options = move_2_2_options_desc()

    group = repeat(treatments, inner=length(options))

    y = [sum(df[df[!, column_name] .== t, o]) for t in treatments for o in options]
    names = repeat(options, outer=2)

    return names, y, group
end

n, y, g = compare_treatments_move1(df, ai_column_name, ai_accuracies)
groupedbar(n, y, group=g, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 1", dpi=300)
# savefig("move1.png")

n, y, g = compare_treatments_move1(df, train_column_name, train_quality)
groupedbar(n, y, group=g, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 1", dpi=300)
# savefig("move1.png")

n, y, g = compare_treatments_move2(df, china_column_name, china_treatments)
groupedbar(n, y, group=g, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 2", dpi=300)
# savefig("move2.png")

# Humans

n, y, g = compare_treatments_move1(df_real, ai_column_name, ai_accuracies)
groupedbar(n, y, group=g, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 1", dpi=300)
# savefig("move1.png")

n, y, g = compare_treatments_move2(df_real, china_column_name, china_treatments)
groupedbar(n, y, group=g, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 2", dpi=300)
# savefig("move2.png")
