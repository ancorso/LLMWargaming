using Random
using StatsPlots
using Plots.Measures
using CSV
using DataFrames
include("src/game.jl")
include("src/config.jl")


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

n, y, g = compare_treatments_move1(df_gpt4_dialog3_fix, ai_column_name, ai_accuracies)
groupedbar(n, y, group=g, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 1", dpi=300)
# savefig("move1.png")

n, y, g = compare_treatments_move1(df_gpt4_dialog3_fix, train_column_name, train_quality)
groupedbar(n, y, group=g, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 1", dpi=300)
# savefig("move1.png")

n, y, g = compare_treatments_move2(df_gpt4_dialog3_fix, china_column_name, china_treatments)
groupedbar(n, y, group=g, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 2", dpi=300)
# savefig("move2.png")

# Humans

n, y, g = compare_treatments_move1(df_real, ai_column_name, ai_accuracies)
groupedbar(n, y, group=g, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 1", dpi=300)
# savefig("move1.png")

n, y, g = compare_treatments_move2(df_real, china_column_name, china_treatments)
groupedbar(n, y, group=g, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 2", dpi=300)
# savefig("move2.png")
