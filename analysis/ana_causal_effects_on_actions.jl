
using Random
using Plots
using StatsBase
using CSV
using DataFrames
using PrettyTables
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