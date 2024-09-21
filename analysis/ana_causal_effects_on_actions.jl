
using Random
using Plots
using StatsBase
using CSV
using DataFrames
using PrettyTables
include("../src/game.jl")
include("../src/config.jl")

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


#####

# Treatment Analysis (Take Difference Between Treatments, Not Data Types)
create_boot_diff(df_real_aug24, "Human Data", ai_column_name, ai_accuracies)
create_boot_diff(df_real_aug24, "Human Data", train_column_name, train_quality)
create_boot_diff(df_real_aug24, "Human Data", china_column_name, china_treatments; move=2)

create_boot_diff(df_gpt35_dialog3_fix, "GPT-3.5 (Dialog 3)", ai_column_name, ai_accuracies)
create_boot_diff(df_gpt35_dialog3_fix, "GPT-3.5 (Dialog 3)", train_column_name, train_quality)
create_boot_diff(df_gpt35_dialog3_fix, "GPT-3.5 (Dialog 3)", china_column_name, china_treatments; move=2)

create_boot_diff(df_gpt4_dialog3_fix, "GPT-4 (Dialog 3)", ai_column_name, ai_accuracies)
create_boot_diff(df_gpt4_dialog3_fix, "GPT-4 (Dialog 3)", train_column_name, train_quality)
create_boot_diff(df_gpt4_dialog3_fix, "GPT-4 (Dialog 3)", china_column_name, china_treatments; move=2)

create_boot_diff(df_gpt4o_dialog3_fix, "GPT-4o (Dialog 3)", ai_column_name, ai_accuracies)
create_boot_diff(df_gpt4o_dialog3_fix, "GPT-4o (Dialog 3)", train_column_name, train_quality)
create_boot_diff(df_gpt4o_dialog3_fix, "GPT-4o (Dialog 3)", china_column_name, china_treatments; move=2)

# Comparing LLMs Directly
create_boot_diff(df_gpt35_dialog3_fix, df_gpt4_dialog3_fix, "GPT3.5 Fix - GPT4 Fix")
create_boot_diff(df_gpt35_dialog3_fix, df_gpt4_dialog3_fix, "GPT3.5 Fix - GPT4 Fix"; move=2)

# Comparing LLMs to Humans
create_boot_diff(df_gpt35_dialog3_fix, df_real_aug24, "GPT3.5 Fix - Human Data")
create_boot_diff(df_gpt35_dialog3_fix, df_real_aug24, "GPT3.5 Fix - Human Data"; move=2)

create_boot_diff(df_gpt4_dialog3_fix, df_real_aug24, "GPT4 Fix - Human Data")
create_boot_diff(df_gpt4_dialog3_fix, df_real_aug24, "GPT4 Fix - Human Data"; move=2)

create_boot_diff(df_gpt4o_dialog3_fix, df_real_aug24, "GPT4 Fix - Human Data")
create_boot_diff(df_gpt4o_dialog3_fix, df_real_aug24, "GPT4 Fix - Human Data"; move=2)

# Old sensitivity studies
create_boot_diff(df_gpt35_dialog0_fix, df_gpt35_dialog3_fix, "GPT3 0 - GPT3 Fix")
create_boot_diff(df_gpt35_dialog0_fix, df_gpt35_dialog3_fix, "GPT3 0- GPT3 Fix"; move=2)

create_boot_diff(df_dialog3, df_more_disagreement, "Dialoge3 - More Discussion"; move=1)
create_boot_diff(df_dialog3, df_more_disagreement, "Dialoge3 - More Discussion"; move=2)

create_boot_diff(df_sociopath, df_pacifism, "Sociopath - Pacifism"; move=1)
create_boot_diff(df_sociopath, df_pacifism, "Sociopath - Pacifism"; move=2)

create_boot_diff(df_gpt4_dialog3, df_gpt4_dialogno, "Dialog3 - no Dialog")
create_boot_diff(df_gpt4_dialog3, df_gpt4_dialogno, "Dialog3 - no Dialog"; move=2)

create_boot_diff(df_gpt4_pacifism, df_gpt4_sociopath, "Pacifism - Sociopath")
create_boot_diff(df_gpt4_pacifism, df_gpt4_sociopath, "Pacifism - Sociopath"; move=2)

# Impact of Instructions
create_boot_diff(df_gpt4_dialog3, df_gpt4_dialog3_fix, "GPT4 - GPT4 Fix")
create_boot_diff(df_gpt4_dialog3, df_gpt4_dialog3_fix, "GPT4 - GPT4 Fix"; move=2)

create_boot_diff(df_dialog3, df_gpt35_dialog3_fix, "GPT3 - GPT3 Fix")
create_boot_diff(df_dialog3, df_gpt35_dialog3_fix, "GPT3 - GPT3 Fix"; move=2)

create_boot_diff(df_gpt4o_dialog3_noinstr, df_gpt4o_dialog3_fix, "GPT4o No Inst - GPT4o")
create_boot_diff(df_gpt4o_dialog3_noinstr, df_gpt4o_dialog3_fix, "GPT4o No Inst - GPT4o"; move=2)

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
        # legend=:topright,
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
    short_options[1] = "(a) 'Fire at vessels'"

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
    annotate!(-0.425, length(options) + 1. + 0.7, ("Fewer Counts", :left, 8))
    # annotate!(0.025, length(options) + 1. + 0.7, "Same Counts as Humans", :left)
    # hline!([6.], lw=3, linecolor="#100B00", label=nothing)
    hline!([9. + 1. / 3.], lw=3, linecolor="#100B00", label=nothing)

    # annotate!(-2., length(options) + 2. - 0.25, text("Move 1", :left, :bold, 10))
    # annotate!(-2., 5.75, text("Move 2", :left, :bold, 10))
    # annotate!(-2.0, length(options) + 2. - 0.25, "Move 1: Use AI Weapon?", :left)
    # annotate!(-2., 5.75, "Move 2: China Status", :left)
    annotate!(-2.15, length(options) + 2. - 0.25, ("Move 1: Use New AI Weapon?", :left, 11))
    annotate!(-2.05, 8.75, ("Move 2: Opponent Posture", :left, 11))

    if return_data
        return short_options, mus, errors
    else
        return fig
    end
end

function create_boot_diff_selected(df_0, df_1, df_2, df_3; return_data=false)

    # old [2, 5, 6]
    # [a, b, c, d, e, f, g]
    short_options = move_1_2_options_shortdesc()[[1, 2, 4, 5]]
    options = move_1_2_options_desc()[[1, 2, 4, 5]]
    # short_options[1] = "(a) 'Fire at vessels'"

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
    res_3 = df_3[!, options]
    mus, errors = calc_comp(res_1, res_0, n_b)
    mus2, errors2 = calc_comp(res_2, res_0, n_b)
    mus3, errors3 = calc_comp(res_3, res_0, n_b)

    reverse!(mus)
    reverse!(errors)
    reverse!(mus2)
    reverse!(errors2)
    reverse!(mus3)
    reverse!(errors3)

    scatter!(
        mus,
        collect(1:length(options)) .- 0.105 .+ [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
        xerror=errors,
        label="GPT-3.5 (95% Conf.)",
        color=cols[1],
        dpi=300,
    )
    scatter!(
        mus2,
        collect(1:length(options)) .+ 0.0 .+ [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
        xerror=errors2,
        label="GPT-4 (95% Conf.)",
        color=cols[2],
        dpi=300,
    )
    scatter!(
        mus3,
        collect(1:length(options)) .+ 0.105 .+ [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
        xerror=errors3,
        label="GPT-4o (95% Conf.)",
        color=cols[3],
        dpi=300,
    )
    # annotate!(0., 11., text("mytext", :red, :right, 3))
    annotate!(0.025, length(options) + 1. + 0.7, ("More Counts Than Humans", :left, 8))
    annotate!(-0.425, length(options) + 1. + 0.7, ("Fewer Counts", :left, 8))
    # annotate!(0.025, length(options) + 1. + 0.7, "Same Counts as Humans", :left)
    # hline!([6.], lw=3, linecolor="#100B00", label=nothing)
    hline!([9. + 1. / 3.], lw=3, linecolor="#100B00", label=nothing)

    # annotate!(-2., length(options) + 2. - 0.25, text("Move 1", :left, :bold, 10))
    # annotate!(-2., 5.75, text("Move 2", :left, :bold, 10))
    # annotate!(-2.0, length(options) + 2. - 0.25, "Move 1: Use AI Weapon?", :left)
    # annotate!(-2., 5.75, "Move 2: China Status", :left)
    annotate!(-2.15, length(options) + 2. - 0.25, ("Move 1: Use New AI Weapon?", :left, 11))
    annotate!(-2.05, 8.75, ("Move 2: Opponent Posture", :left, 11))

    if return_data
        return short_options, mus, errors
    else
        return fig
    end
end

# Check selection
# create_boot_diff_selected(df_gpt35_dialog3_fix, df_real_feb24, df_gpt4_dialog3_fix)
# # savefig("selected_gpt3_gpt4_v_humans_Feb24_fix.png")
# # savefig("selected_gpt3_gpt4_v_humans_Feb24_fix_nochina.png")
# # savefig("selected_gpt3_gpt4_v_humans_Feb24_fix.pdf")
# create_boot_diff_selected(df_gpt35_dialog3_fix, df_real_aug24, df_gpt4_dialog3_fix)
# # savefig("selected_gpt3_gpt4_v_humans_Aug24_fix_nochina.png")
# # savefig("selected_gpt3_gpt4_v_humans_Aug24_fix_nochina.pdf")

create_boot_diff_selected(df_real_aug24, df_gpt35_dialog3_fix, df_gpt4_dialog3_fix, df_gpt4o_dialog3_fix)
