
using Random
using Plots
using StatsBase
using CSV
using DataFrames
using PrettyTables
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
df_dialog3_new = CSV.read("results/sensitivity_studies_new/data_dialog3.csv", DataFrame)
df_dialog4 = CSV.read("results/sensitivity_studies_new/data_dialog4.csv", DataFrame)
df_dialog5 = CSV.read("results/sensitivity_studies_new/data_dialog5.csv", DataFrame)
df_dialog6 = CSV.read("results/sensitivity_studies/data_dialog6.csv", DataFrame)
df_nochief = CSV.read("results/sensitivity_studies/data_nochiefs.csv", DataFrame)
df_playeruniform = CSV.read("results/sensitivity_studies/data_playeruniform.csv", DataFrame)
df_pacifism = CSV.read("results/sensitivity_studies_new/data_pacifism.csv", DataFrame)
df_sociopath = CSV.read("results/sensitivity_studies_new/data_sociopath.csv", DataFrame)
df_more_disagreement = CSV.read("results/sensitivity_studies_new/data_moredisagree.csv", DataFrame)

df_gpt4_dialog1 = CSV.read("results/gpt4_turbo/data_dialog1.csv", DataFrame)
df_gpt4_dialog3 = CSV.read("results/gpt4_turbo/data_dialog3.csv", DataFrame)
df_gpt4_dialog6 = CSV.read("results/gpt4_turbo/data_dialog6.csv", DataFrame)
df_gpt4_dialogno = CSV.read("results/gpt4_turbo/data_dialogno.csv", DataFrame)
df_gpt4_pacifism = CSV.read("results/gpt4_turbo/data_pacifism.csv", DataFrame)
df_gpt4_sociopath = CSV.read("results/gpt4_turbo/data_sociopaths.csv", DataFrame)


# post fix
df_gpt4_dialog0_fix = CSV.read("results/post_fix/data_gpt4_dialog0.csv", DataFrame)
df_gpt4_dialog3_fix = CSV.read("results/post_fix/data_gpt4_dialog3.csv", DataFrame)
df_gpt35_dialog0_fix = CSV.read("results/post_fix/data_gpt35_dialog0.csv", DataFrame)
df_gpt35_dialog3_fix = CSV.read("results/post_fix/data_gpt35_dialog3.csv", DataFrame)


function create_boot_diff(df_0, df_1, tit, column_name, treatments; move=1, return_data=false)
    if move == 1
        short_options = move_1_2_options_shortdesc()
        options = move_1_2_options_desc()
        leg = ""
    elseif move == 2
        options = move_2_2_options_desc()
        short_options = options
        leg = ""
    end
    n_b = 10000
    fig = def_plot(short_options, "Move $(move): " * tit)

    delta = 0.
    for (t_ind, t) in enumerate(treatments)
        println(t)
        res_0 = df_0[df_0[!, column_name] .== t, options]
        res_1 = df_1[df_1[!, column_name] .== t, options]
        mus, errors = calc_comp(res_0, res_1, n_b)

        println(mus)
        println(errors)

        scatter!(
            mus,
            collect(1:length(options)) .+ delta,
            xerror=errors,
            label= leg * " $(t)",
            color=t_ind,
            dpi=300,
        )
        delta += 0.1
    end

    if return_data
        return short_options, mus, errors
    else
        return fig
    end
end

function create_boot_diff(df, tit, column_name, treatments; move=1, return_data=false)
    if move == 1
        short_options = move_1_2_options_shortdesc()
        options = move_1_2_options_desc()
        leg = ""
    elseif move == 2
        options = move_2_2_options_desc()
        short_options = options
        leg = ""
    end
    n_b = 10000
    fig = def_plot(short_options, "Move $(move): " * tit)

    res_0 = df[df[!, column_name] .== treatments[1], options]
    res_1 = df[df[!, column_name] .== treatments[2], options]
    mus, errors = calc_comp(res_0, res_1, n_b)
    println(mus)
    println(errors)

    scatter!(
        mus,
        collect(1:length(options)),
        xerror=errors,
        label= leg * "$(treatments[1]) - $(treatments[2])",
        color=1,
        dpi=300,
    )


    if return_data
        return short_options, mus, errors
    else
        return fig
    end
end

function create_boot_diff(df_0, df_1, tit; move=1, return_data=false)
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
        mus,
        collect(1:length(options)),
        xerror=errors,
        label="Diff",
        color=1,
        dpi=300,
    )

    if return_data
        return short_options, mus, errors
    else
        return fig
    end
end


function calc_comp(r0, r1, n_b)
    Random.seed!(SEED)

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
    s = plot(
        yticks=(1:length(opts), opts),
        # xrot=60,
        # bottom_margin=15mm,
        # left_margin=12mm,
        xlabel="Total Effect on Counts [ ]",
        xlims=[-0.82, 0.82],
        title=tit,
        legend=:bottomright,
    )
    vline!([0], linestyle=:dash, linecolor="#100B00", label=nothing)
    return s
end


# Dialog length sensitivity
create_boot_diff(df_dialog3, df_dialogno, "Dialog3 - no Dialog")
# savefig("move1_dialog3_v_nodialog.png")
create_boot_diff(df_dialog3, df_dialog1, "Dialog3 - Dialog1")
# savefig("move1_dialog3_v_dialog1.png")
create_boot_diff(df_dialog3, df_dialog3, "Dialog3 - Dialog3")
create_boot_diff(df_dialog3, df_dialog6, "Dialog3 - Dialog6")
# savefig("move1_dialog3_v_dialog6.png")
create_boot_diff(df_dialogno, df_dialog6, "no Dialog - Dialog6")
# savefig("move1_nodialog_v_dialog6.png")
create_boot_diff(df_dialog1, df_dialog6, "Dialog1 - Dialog6")
# savefig("move1_dialog1_v_dialog6.png")

create_boot_diff(df_dialog3, df_dialogno, "Dialog3 - no Dialog", ai_column_name, ai_accuracies)
# savefig("move1_dialog3_v_nodialog_aiacc.png")
create_boot_diff(df_dialog3, df_dialog1, "Dialog3 - Dialog1", ai_column_name, ai_accuracies)
# savefig("move1_dialog3_v_dialog1_aiacc.png")
create_boot_diff(df_dialog3, df_dialog3, "Dialog3 - Dialog3", ai_column_name, ai_accuracies)
create_boot_diff(df_dialog3, df_dialog6, "Dialog3 - Dialog6", ai_column_name, ai_accuracies)
# savefig("move1_dialog3_v_dialog6_aiacc.png")
create_boot_diff(df_dialogno, df_dialog6, "no Dialog - Dialog6", ai_column_name, ai_accuracies)
# savefig("move1_nodialog_v_dialog6_aiacc.png")

# No chiefs (emphasize human background more)
create_boot_diff(df_dialog3, df_nochief, "Dialog3 - no Chiefs")
# savefig("move1_dialog3_v_nochiefs.png")
create_boot_diff(df_dialog3, df_nochief, "Dialog3 - no Chiefs", ai_column_name, ai_accuracies)
# savefig("move1_dialog3_v_nochiefs_aiacc.png")

# Uniform player background
create_boot_diff(df_dialog3, df_playeruniform, "Dialog3 - uniform Players")
# savefig("move1_dialog3_v_uniformplayers.png")
create_boot_diff(df_dialog3, df_playeruniform, "Dialog3 - uniform Players", ai_column_name, ai_accuracies)
# savefig("move1_dialog3_v_uniformplayers_aiacc.png")

# Compare to human data
create_boot_diff(df_dialog6, df_real, "Dialog6 - Human Data")
# savefig("move1_dialog6_v_humans.png")
create_boot_diff(df_dialog6, df_real, "Dialog6 - Human Data", ai_column_name, ai_accuracies)
# savefig("move1_dialog6_v_humans_aiacc.png")

create_boot_diff(df_dialog3, df_real, "Dialog3 - Human Data")
# savefig("move1_dialog3_v_humans.png")
create_boot_diff(df_dialog3, df_real, "Dialog3 - Human Data", ai_column_name, ai_accuracies)
# savefig("move1_dialog3_v_humans_aiacc.png")

create_boot_diff(df_dialogno, df_real, "no Dialog - Human Data")
# savefig("move1_nodialog_v_humans.png")
create_boot_diff(df_dialogno, df_real, "no Dialog - Human Data", ai_column_name, ai_accuracies)
# savefig("move1_nodialog_v_humans_aiacc.png")


# Move 2
create_boot_diff(df_dialogno, df_dialog6, "no Dialog - Dialog6"; move=2)
# savefig("move2_nodialog_v_dialog6.png")
create_boot_diff(df_dialogno, df_dialog6, "no Dialog - Dialog6", china_column_name, china_treatments; move=2)
# savefig("move2_nodialog_v_dialog6_china.png")

create_boot_diff(df_dialog3, df_nochief, "Dialog3 - no Chiefs"; move=2)
# savefig("move2_dialog3_v_nochiefs.png")
create_boot_diff(df_dialog3, df_nochief, "Dialog3 - no Chiefs", china_column_name, china_treatments; move=2)
# savefig("move2_dialog3_v_nochiefs_china.png")


create_boot_diff(df_dialog3, df_playeruniform, "Dialog3 - uniform Players"; move=2)
# savefig("move2_dialog3_v_uniformplayers.png")
create_boot_diff(df_dialog3, df_playeruniform, "Dialog3 - uniform Players", china_column_name, china_treatments; move=2)
# savefig("move2_dialog3_v_uniformplayers_china.png")


create_boot_diff(df_dialog6, df_real, "Dialog6 - Human Data"; move=2)
# savefig("move2_dialog6_v_humans.png")
create_boot_diff(df_dialog6, df_real, "Dialog6 - Human Data", china_column_name, china_treatments; move=2)
# savefig("move2_dialog6_v_humans_china.png")

create_boot_diff(df_dialog1, df_real, "Dialog1 - Human Data"; move=2)
# savefig("move2_dialog1_v_humans.png")
create_boot_diff(df_dialog1, df_real, "Dialog1 - Human Data", china_column_name, china_treatments; move=2)
# savefig("move2_dialog1_v_humans_china.png")



create_boot_diff(df_dialog3, df_real, "Dialog3 - Human Data"; move=2)
# savefig("move2_dialog3_v_humans.png")
create_boot_diff(df_dialog3, df_real, "Dialog3 - Human Data", china_column_name, china_treatments; move=2)
# savefig("move2_dialog3_v_humans_china.png")


create_boot_diff(df_dialogno, df_real, "no Dialoge - Human Data"; move=2)
# savefig("move2_nodialog_v_humans.png")
create_boot_diff(df_dialogno, df_real, "no Dialog - Human Data", china_column_name, china_treatments; move=2)
# savefig("move2_nodialog_v_humans_china.png")





######

create_boot_diff(df_dialog3, df_dialog3_new, "Dialoge3 - Dialoge3"; move=1)
create_boot_diff(df_dialog3, df_dialog3_new, "Dialoge3 - Dialoge3"; move=2)

create_boot_diff(df_dialog3, df_more_disagreement, "Dialoge3 - More Discussion"; move=1)
create_boot_diff(df_dialog3, df_more_disagreement, "Dialoge3 - More Discussion"; move=2)

create_boot_diff(df_dialog3, df_pacifism, "Dialoge3 - Pacifism"; move=1)
create_boot_diff(df_dialog3, df_sociopath, "Dialoge3 - Sociopath"; move=1)
create_boot_diff(df_dialog3, df_pacifism, "Dialoge3 - Pacifism"; move=2)
create_boot_diff(df_dialog3, df_sociopath, "Dialoge3 - Sociopath"; move=2)

create_boot_diff(df_sociopath, df_pacifism, "Sociopath - Pacifism"; move=1)
create_boot_diff(df_sociopath, df_pacifism, "Sociopath - Pacifism"; move=2)


#####

create_boot_diff(df_dialog3, "Dialog3", ai_column_name, ai_accuracies)
create_boot_diff(df_dialog3, "Dialog3", train_column_name, train_quality)
create_boot_diff(df_dialog3, "Dialog3", china_column_name, china_treatments; move=2)

create_boot_diff(df_real, "Human Data", ai_column_name, ai_accuracies)
create_boot_diff(df_real, "Human Data", train_column_name, train_quality)
create_boot_diff(df_real, "Human Data", china_column_name, china_treatments; move=2)

create_boot_diff(df_real_feb24, "Human Data", ai_column_name, ai_accuracies)
create_boot_diff(df_real_feb24, "Human Data", train_column_name, train_quality)
create_boot_diff(df_real_feb24, "Human Data", china_column_name, china_treatments; move=2)



######

# Make a table
o1, m1, s1 = create_boot_diff(df_dialog3, df_real, "Dialog3 - Human Data"; return_data=true)
o2, m2, s2 = create_boot_diff(df_dialog3, df_real, "Dialog3 - Human Data"; move=2, return_data=true)

o = vcat(o2)
m = vcat(m2)
s = vcat(s2)
m = map(x->round(x, digits=3), m)
sp = [err[2] for err in s]
sm = [err[1] for err in s]
sp = map(x->round(x, digits=3), sp)
sm = map(x->round(x, digits=3), sm)

data = transpose(hcat(m, sp, sm))
header = o

# \newcommand{\adderrors}[3][2]{(#2 + #3)^#1}
pretty_table(data, backend=Val(:latex), header=header)




# dialog 3
create_boot_diff(df_dialog3, df_gpt4_dialog3, "GPT3.5 - GPT4")
create_boot_diff(df_dialog3, df_gpt4_dialog3, "GPT3.5 - GPT4"; move=2)
create_boot_diff(df_dialog3, df_gpt4_dialog3, "GPT3.5 - GPT4", ai_column_name, ai_accuracies)
create_boot_diff(df_dialog3, df_gpt4_dialog3, "GPT3.5 - GPT4", china_column_name, china_treatments; move=2)


create_boot_diff(df_gpt4_dialog3, df_gpt4_dialog1, "Dialog3 - Dialog1")
create_boot_diff(df_gpt4_dialog3, df_gpt4_dialog1, "Dialog3 - Dialog1"; move=2)

create_boot_diff(df_gpt4_dialog3, df_gpt4_dialog6, "Dialog3 - Dialog6")
create_boot_diff(df_gpt4_dialog3, df_gpt4_dialog6, "Dialog3 - Dialog6"; move=2)

create_boot_diff(df_gpt4_dialog3, df_gpt4_dialogno, "Dialog3 - no Dialog")
create_boot_diff(df_gpt4_dialog3, df_gpt4_dialogno, "Dialog3 - no Dialog"; move=2)

create_boot_diff(df_gpt4_pacifism, df_gpt4_sociopath, "Pacifism - Sociopath")
create_boot_diff(df_gpt4_pacifism, df_gpt4_sociopath, "Pacifism - Sociopath"; move=2)


create_boot_diff(df_dialog3, df_real, "Dialog3 - Human Data")
create_boot_diff(df_gpt4_dialog3, df_real, "Dialog3 - Human Data")
create_boot_diff(df_dialog3, df_real, "Dialog3 - Human Data"; move=2)
create_boot_diff(df_gpt4_dialog3, df_real, "Dialog3 - Human Data"; move=2)

create_boot_diff(df_dialog3, df_real_feb24, "Dialog3 - Human Data")
create_boot_diff(df_gpt4_dialog3, df_real_feb24, "Dialog3 - Human Data")
create_boot_diff(df_dialog3, df_real_feb24, "Dialog3 - Human Data"; move=2)
create_boot_diff(df_gpt4_dialog3, df_real_feb24, "Dialog3 - Human Data"; move=2)

create_boot_diff(df_gpt4_dialog3, "GPT4 Dialog3", ai_column_name, ai_accuracies)
create_boot_diff(df_gpt4_dialog3, "GPT4 Dialog3", train_column_name, train_quality)
create_boot_diff(df_gpt4_dialog3, "GPT4 Dialog3", china_column_name, china_treatments; move=2)



############

create_boot_diff(df_gpt4_dialog3, df_gpt4_dialog3_fix, "GPT4 - GPT4 Fix")
create_boot_diff(df_gpt4_dialog3, df_gpt4_dialog3_fix, "GPT4 - GPT4 Fix"; move=2)

create_boot_diff(df_dialog3, df_gpt35_dialog3_fix, "GPT3 - GPT3 Fix")
create_boot_diff(df_dialog3, df_gpt35_dialog3_fix, "GPT3 - GPT3 Fix"; move=2)

create_boot_diff(df_gpt35_dialog3_fix, df_gpt4_dialog3_fix, "GPT3.5 Fix - GPT4 Fix")
create_boot_diff(df_gpt35_dialog3_fix, df_gpt4_dialog3_fix, "GPT3.5 Fix - GPT4 Fix"; move=2)

create_boot_diff(df_gpt35_dialog3_fix, df_real_feb24, "GPT3.5 Fix - Human Data")
create_boot_diff(df_gpt35_dialog3_fix, df_real_feb24, "GPT3.5 Fix - Human Data"; move=2)

create_boot_diff(df_gpt4_dialog3_fix, df_real_feb24, "GPT4 Fix - Human Data")
create_boot_diff(df_gpt4_dialog3_fix, df_real_feb24, "GPT4 Fix - Human Data"; move=2)

create_boot_diff(df_gpt35_dialog0_fix, df_gpt35_dialog3_fix, "GPT3 0 - GPT3 Fix")
create_boot_diff(df_gpt35_dialog0_fix, df_gpt35_dialog3_fix, "GPT3 0- GPT3 Fix"; move=2)


# Make a table
o1, m1, s1 = create_boot_diff(df_gpt4_dialog3, df_gpt4_dialog3_fix, "GPT3.5 - GPT3.5 Fix"; return_data=true)
o2, m2, s2 = create_boot_diff(df_gpt4_dialog3, df_gpt4_dialog3_fix, "GPT3.5 - GPT3.5 Fix"; move=2, return_data=true)

o = vcat(o2)
m = vcat(m2)
s = vcat(s2)
m = map(x->round(x, digits=3), m)
sp = [err[2] for err in s]
sm = [err[1] for err in s]
sp = map(x->round(x, digits=3), sp)

data = transpose(hcat(m, sp, sm))
header = [oi[1:3] for oi in o]

# \newcommand{\adderrors}[3][2]{(#2 + #3)^#1}
pretty_table(data, header=header)


function def_plot_selected(opts, tit)
    s = plot(
        # yticks=(1:length(opts), opts),
        # ylims=[0.5, length(opts) + 1],
        # yticks=([1, 2, 3, 4, 5, 7, 8, 9], opts),
        yticks=([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13], opts),
        ylims=[0.5, length(opts) + 2 - 0.1],
        # xrot=60,
        # bottom_margin=15mm,
        # left_margin=12mm,
        xlabel="Difference in Action Counts [ ]",
        xlims=[-0.9, 0.8],
        title=tit,
        legend=:bottomleft,
    )
    # vline!([0], lw=2, linestyle=:dash, linecolor="#100B00", label=nothing)
    plot!([0., 0], [0., 13.9], lw=1.5, linestyle=:dash, linecolor="#100B00", label=nothing)
    # vspan!([[0., 0.], [0.1, 8.]], lw=2, linestyle=:dash, linecolor="#100B00", label=nothing)
    # vspan!([[0., 0.], [0.1, 8.]], label=nothing)
    return s
end

function create_boot_diff_selected(df_0, df_1, df_2; return_data=false)

    # old [2, 5, 6]
    # [a, b, c, d, e, f, g]
    short_options = move_1_2_options_shortdesc()[[1, 2, 4, 5]]
    options = move_1_2_options_desc()[[1, 2, 4, 5]]

    # short_options = [s[1:4] * "'" * s[5:length(s)] * "'" for s in short_options]

    # [2, 5, 6, 9, 12, 13, 14]
    # old [5, 6, 9, 13, 14]
    # [a, a1, a2, a3, b, c, d, e, f, g, h, i, j, k]
    append!(options, move_2_2_options_desc()[[2, 4, 5, 6, 9, 12, 13, 14]])
    append!(short_options, move_2_2_options_desc()[[2, 4, 5, 6, 9, 12, 13, 14]])

    # short_options[4] = "MOVE 2: " * short_options[4]
    short_options = [s[1:4] * "'" * s[5:length(s)] * "'" for s in short_options]
    
    n_b = 10000
    fig = def_plot_selected(reverse(short_options), "")
    # fig = def_plot_selected(reverse(options), "")

    res_0 = df_0[!, options]
    res_1 = df_1[!, options]
    res_2 = df_2[!, options]
    mus, errors = calc_comp(res_0, res_1, n_b)
    mus2, errors2 = calc_comp(res_2, res_1, n_b)

    reverse!(mus)
    reverse!(errors)
    reverse!(mus2)
    reverse!(errors2)

    scatter!(
        mus,
        collect(1:length(options)) .- 0.07 .+ [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
        xerror=errors,
        label="GPT-3.5 (95% Conf.)",
        color=1,
        dpi=300,
    )
    scatter!(
        mus2,
        collect(1:length(options)) .+ 0.07 .+ [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
        xerror=errors2,
        label="GPT-4 (95% Conf.)",
        color=0,
        dpi=300,
    )
    # annotate!(0., 11., text("mytext", :red, :right, 3))
    annotate!(0.025, length(options) + 1. + 0.7, ("More Counts Than Humans", :left, 8))
    annotate!(-0.365, length(options) + 1. + 0.7, ("Less Counts", :left, 8))
    # annotate!(0.025, length(options) + 1. + 0.7, "Same Counts as Humans", :left)
    # hline!([6.], lw=3, linecolor="#100B00", label=nothing)
    hline!([9. + 1. / 3.], lw=3, linecolor="#100B00", label=nothing)

    # annotate!(-2., length(options) + 2. - 0.25, text("Move 1", :left, :bold, 10))
    # annotate!(-2., 5.75, text("Move 2", :left, :bold, 10))
    # annotate!(-2.0, length(options) + 2. - 0.25, "Move 1: Use AI Weapon?", :left)
    # annotate!(-2., 5.75, "Move 2: China Status", :left)
    annotate!(-2.0, length(options) + 2. - 0.25, ("Move 1: Use AI Weapon?", :left, 11))
    annotate!(-2., 8.75, ("Move 2: China Posture", :left, 11))

    if return_data
        return short_options, mus, errors
    else
        return fig
    end
end

create_boot_diff_selected(df_dialog3, df_real, df_gpt4_dialog3)
create_boot_diff_selected(df_dialog3, df_real_feb24, df_gpt4_dialog3)
# savefig("selected_gpt3_gpt4_v_humans_Feb24.png")

# Check selection
create_boot_diff_selected(df_gpt35_dialog3_fix, df_real_feb24, df_gpt4_dialog3_fix)
# savefig("selected_gpt3_gpt4_v_humans_Feb24_fix.png")
# savefig("selected_gpt3_gpt4_v_humans_Feb24_fix.pdf")