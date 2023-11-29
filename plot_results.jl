
using StatsPlots
using Plots.Measures
using StatsBase
using CSV
using DataFrames
include("src/game.jl")


df = CSV.read("results/data2023-11-15_2.csv", DataFrame)
# df = CSV.read("results/v1_results.csv", DataFrame)

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

ai_column_name = "AI Accuracy"
ai_accuracies = ["70-85%", "95-99%"]

train_column_name = "AI System Training"
train_quality = ["basic", "significant"]

china_column_name = "China Status"
china_treatments = ["revisionist", "status_quo"]

n, y, g = compare_treatments_move1(df, ai_column_name, ai_accuracies)
groupedbar(n, y, group=g, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 1", dpi=300)
# savefig("move1.png")

n, y, g = compare_treatments_move1(df, train_column_name, train_quality)
groupedbar(n, y, group=g, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 1", dpi=300)
# savefig("move1.png")

n, y, g = compare_treatments_move2(df, china_column_name, china_treatments)
groupedbar(n, y, group=g, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 2", dpi=300)
# savefig("move2.png")



df_real = CSV.read("data/ganz_data_full.csv", DataFrame)
df_dialogno = CSV.read("results/sensitivity_studies/data_dialogno.csv", DataFrame)
df_dialog1 = CSV.read("results/sensitivity_studies/data_dialog1.csv", DataFrame)
df_dialog3 = CSV.read("results/sensitivity_studies/data_dialog3.csv", DataFrame)
df_dialog6 = CSV.read("results/sensitivity_studies/data_dialog6.csv", DataFrame)
df_nochief = CSV.read("results/sensitivity_studies/data_nochiefs.csv", DataFrame)
df_playeruniform = CSV.read("results/sensitivity_studies/data_playeruniform.csv", DataFrame)


# Get dataset of vectors for a config, one vector = 1 game
# Calculate total effect via bootstrap resampling (avoids norming issues for number of experiments)


function create_boot_diff(df_0, df_1, tit, column_name, treatments; move=1)
    if move == 1
        short_options = move_1_2_options_shortdesc()
        options = move_1_2_options_desc()
    elseif move == 2
        options = move_2_2_options_desc()
        short_options = options
    end
    n_b = 10000
    fig = def_plot(short_options, "Move $(move): " * tit)

    delta = 0.
    for t in treatments
        println(t)
        res_0 = df_0[df_0[!, column_name] .== t, options]
        res_1 = df_1[df_1[!, column_name] .== t, options]
        mus, errors = calc_comp(res_0, res_1, n_b)

        println(mus)
        println(errors)

        scatter!(
            collect(1:length(options)) .+ delta,
            mus,
            yerror=errors,
            label="AI Acc. $(t)"
        )
        delta += 0.1
    end

    fig
end

function create_boot_diff(df_0, df_1, tit; move=1)
    if move == 1
        short_options = move_1_2_options_shortdesc()
        options = move_1_2_options_desc()
    elseif move == 2
        options = move_2_2_options_desc()
        short_options = options
    end
    
    n_b = 10000
    fig = def_plot(short_options, "Move $(move): " * tit)

    res_0 = df_0[!, options]
    res_1 = df_1[!, options]
    mus, errors = calc_comp(res_0, res_1, n_b)
    println(mus)
    println(errors)

    scatter!(
        collect(1:length(options)),
        mus,
        yerror=errors,
        label="Diff"
    )

    fig
end


function calc_comp(r0, r1, n_b)
    res_0_val = [[values(e)...] for e in eachrow(r0)]
    res_1_val = [[values(e)...] for e in eachrow(r1)]
    n_dat = length(eachrow(r0))

    mus = 1.0 * mean(res_0_val) - 1.0 * mean(res_1_val)

    boot_res = []
    for b in range(1, n_b)
        boot_0 = StatsBase.sample(res_0_val, n_dat, replace=true)
        boot_1 = StatsBase.sample(res_1_val, n_dat, replace=true)

        append!(boot_res, [1.0 * mean(boot_0) - 1.0 * mean(boot_1)])
    end

    lower = [sort([d[i] for d in boot_res])[round(Int, n_b * 0.025)] for i in 1:length(mus)]
    upper = [sort([d[i] for d in boot_res])[round(Int, n_b * 0.975)] for i in 1:length(mus)]
    errors = [((mus - lower)[i], (upper - mus)[i]) for i in 1:length(mus)]

    return mus, errors
end

function def_plot(opts, tit)
    return plot(
        xticks=(1:length(opts), opts),
        xrot=60,
        bottom_margin=15mm,
        ylabel="Total Effect [ ]",
        ylims=[-0.72, 0.72],
        title=tit,
    )
end




# Dialog length sensitivity
create_boot_diff(df_dialog3, df_dialogno, "Dialog3 - no Dialog")
create_boot_diff(df_dialog3, df_dialog1, "Dialog3 - Dialog1")
create_boot_diff(df_dialog3, df_dialog3, "Dialog3 - Dialog3")
create_boot_diff(df_dialog3, df_dialog6, "Dialog3 - Dialog6")
create_boot_diff(df_dialogno, df_dialog6, "no Dialog - Dialog6")

create_boot_diff(df_dialog3, df_dialogno, "Dialog3 - no Dialog", ai_column_name, ai_accuracies)
create_boot_diff(df_dialog3, df_dialog1, "Dialog3 - Dialog1", ai_column_name, ai_accuracies)
create_boot_diff(df_dialog3, df_dialog3, "Dialog3 - Dialog3", ai_column_name, ai_accuracies)
create_boot_diff(df_dialog3, df_dialog6, "Dialog3 - Dialog6", ai_column_name, ai_accuracies)
create_boot_diff(df_dialogno, df_dialog6, "no Dialog - Dialog6", ai_column_name, ai_accuracies)

# No chiefs (emphasize human background more)
create_boot_diff(df_dialog3, df_nochief, "Dialog3 - no Chiefs")
create_boot_diff(df_dialog3, df_nochief, "Dialog3 - no Chiefs", ai_column_name, ai_accuracies)

# Uniform player background
create_boot_diff(df_dialog3, df_playeruniform, "Dialog3 - uniform Players")
create_boot_diff(df_dialog3, df_playeruniform, "Dialog3 - uniform Players", ai_column_name, ai_accuracies)

# Compare to human data
create_boot_diff(df_dialog3, df_real, "Dialog3 - Human Data")
create_boot_diff(df_dialog3, df_real, "Dialog3 - Human Data", ai_column_name, ai_accuracies)



# Move 2
create_boot_diff(df_dialogno, df_dialog6, "no Dialog - Dialog6"; move=2)
create_boot_diff(df_dialogno, df_dialog6, "no Dialog - Dialog6", china_column_name, china_treatments; move=2)

create_boot_diff(df_dialog3, df_nochief, "Dialog3 - no Chiefs"; move=2)
create_boot_diff(df_dialog3, df_nochief, "Dialog3 - no Chiefs", china_column_name, china_treatments; move=2)

create_boot_diff(df_dialog3, df_playeruniform, "Dialog3 - uniform Players"; move=2)
create_boot_diff(df_dialog3, df_playeruniform, "Dialog3 - uniform Players", china_column_name, china_treatments; move=2)


create_boot_diff(df_dialog3, df_real, "Dialog3 - Human Data"; move=2)
create_boot_diff(df_dialog3, df_real, "Dialog3 - Human Data", china_column_name, china_treatments; move=2)
