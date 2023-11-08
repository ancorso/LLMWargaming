
using StatsPlots
using Plots.Measures
df = CSV.read("results/sample_results.csv", DataFrame)


function compare_treatments_move1(df, column_name, treatments)
    short_options = move_1_2_options_shortdesc()
    options = move_1_2_options_desc()

    group = repeat(treatments, inner=length(options))

    y = [sum(df[df[!, column_name] .== t, o]) for t in treatments for o in options]
    names = repeat(short_options, outer=2)
    groupedbar(names, y, group=group, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 1")
end

function compare_treatments_move2(df, column_name, treatments)
    options = move_2_2_options_desc()

    group = repeat(treatments, inner=length(options))

    y = [sum(df[df[!, column_name] .== t, o]) for t in treatments for o in options]
    names = repeat(options, outer=2)
    groupedbar(names, y, group=group, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 2")
end


AI_column_name = "AI Accuracy"
AI_Accuracies = ["70-85%", "95-99%"]

china_column_name = "China Status"
china_treatments = ["revisionist", "status_quo"]

compare_treatments_move1(df, AI_column_name, AI_Accuracies)
compare_treatments_move2(df, china_column_name, china_treatments)