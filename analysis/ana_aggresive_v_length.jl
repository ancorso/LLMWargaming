
using Random
using Plots
using CSV
using DataFrames
using StatsBase
using Measures

include("../src/game.jl")
include("../src/config.jl")


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


function make_plot_both()
    Random.seed!(SEED)
    aggro_h = get_frac_aggr(df_real_aug24)
    aggro_35 = [
        get_frac_aggr(df_gpt35_dialogno_fix),
        get_frac_aggr(df_gpt35_dialog1_fix),
        get_frac_aggr(df_gpt35_dialog2_fix),
        get_frac_aggr(df_gpt35_dialog3_fix),
        get_frac_aggr(df_gpt35_dialog4_fix),
        get_frac_aggr(df_gpt35_dialog5_fix),
        get_frac_aggr(df_gpt35_dialog6_fix),
    ]
    aggro_4 = [
        get_frac_aggr(df_gpt4_dialogno_fix),
        get_frac_aggr(df_gpt4_dialog1_fix),
        get_frac_aggr(df_gpt4_dialog2_fix),
        get_frac_aggr(df_gpt4_dialog3_fix),
        get_frac_aggr(df_gpt4_dialog4_fix),
        get_frac_aggr(df_gpt4_dialog5_fix),
        get_frac_aggr(df_gpt4_dialog6_fix),
    ]
    aggro_4o = [
        get_frac_aggr(df_gpt4o_dialogno_fix),
        get_frac_aggr(df_gpt4o_dialog1_fix),
        get_frac_aggr(df_gpt4o_dialog2_fix),
        get_frac_aggr(df_gpt4o_dialog3_fix),
        get_frac_aggr(df_gpt4o_dialog4_fix),
        get_frac_aggr(df_gpt4o_dialog5_fix),
        get_frac_aggr(df_gpt4o_dialog6_fix),
    ]

    n_actions_h = get_number_of_answers(df_real_aug24)
    n_actions_35 = [
        get_number_of_answers(df_gpt35_dialogno_fix),
        get_number_of_answers(df_gpt35_dialog1_fix),
        get_number_of_answers(df_gpt35_dialog2_fix),
        get_number_of_answers(df_gpt35_dialog3_fix),
        get_number_of_answers(df_gpt35_dialog4_fix),
        get_number_of_answers(df_gpt35_dialog5_fix),
        get_number_of_answers(df_gpt35_dialog6_fix),
    ]
    n_actions_4 = [
        get_number_of_answers(df_gpt4_dialogno_fix),
        get_number_of_answers(df_gpt4_dialog1_fix),
        get_number_of_answers(df_gpt4_dialog2_fix),
        get_number_of_answers(df_gpt4_dialog3_fix),
        get_number_of_answers(df_gpt4_dialog4_fix),
        get_number_of_answers(df_gpt4_dialog5_fix),
        get_number_of_answers(df_gpt4_dialog6_fix),
    ]
    n_actions_4o = [
        get_number_of_answers(df_gpt4o_dialogno_fix),
        get_number_of_answers(df_gpt4o_dialog1_fix),
        get_number_of_answers(df_gpt4o_dialog2_fix),
        get_number_of_answers(df_gpt4o_dialog3_fix),
        get_number_of_answers(df_gpt4o_dialog4_fix),
        get_number_of_answers(df_gpt4o_dialog5_fix),
        get_number_of_answers(df_gpt4o_dialog6_fix),
    ]

    s_aggro = plot(
        xlabel="",
        ylabel="Aggresivness [a.u.]",
        title="",
        legend=:bottomright,
        ylims=[0.12, 0.349],
        # xaxis=false,
    )
    plot!(
        [0., 6.3],
        fill(aggro_h[1], 2),
        ribbon=aggro_h[2],
        label="Human Players (95% Conf.)",
        fillalpha=0.2,
        color=cols[0],
        lw=4.,
    )
    scatter!(
        [0., 1., 2., 3., 4., 5., 6.] .- 0.08,
        [a[1] for a in aggro_35],
        yerror=[a[2] for a in aggro_35],
        marker=true,
        label="GPT-3.5 (95% Conf.)",
        dpi=300,
        color=cols[1],
    )
    scatter!(
        [0., 1., 2., 3., 4., 5., 6.] .+ 0.0,
        [a[1] for a in aggro_4],
        yerror=[a[2] for a in aggro_4],
        marker=true,
        label="GPT-4 (95% Conf.)",
        dpi=300,
        color=cols[2],
    )
    scatter!(
        [0., 1., 2., 3., 4., 5., 6.] .+ 0.08,
        [a[1] for a in aggro_4o],
        yerror=[a[2] for a in aggro_4o],
        marker=true,
        label="GPT-4o (95% Conf.)",
        dpi=300,
        color=cols[3],
    )

    s_nums = plot(
        xlabel="Length of Simulated Dialog [~350 words]",
        ylabel="#Chosen Actions [ ]",
        title="",
        legend=:bottomright,
        # ylims=[0.11, 0.32],
        ylims=[7., 12.2],
    )
    plot!(
        [0., 6.3], 
        fill(n_actions_h[1], 2), 
        ribbon=n_actions_h[2], 
        primary=false,
        fillalpha=0.2, 
        color=cols[0], 
        lw=4.,
    )
    scatter!(
        [0., 1., 2., 3., 4., 5., 6.] .- 0.08,
        [a[1] for a in n_actions_35],
        yerror=[a[2] for a in n_actions_35],
        marker=true,
        primary=false,
        dpi=300,
        color=cols[1],
    )
    scatter!(
        [0., 1., 2., 3., 4., 5., 6.] .+ 0.0,
        [a[1] for a in n_actions_4],
        yerror=[a[2] for a in n_actions_4],
        marker=true,
        primary=false,
        dpi=300,
        color=cols[2],
    )
    scatter!(
        [0., 1., 2., 3., 4., 5., 6.] .+ 0.08,
        [a[1] for a in n_actions_4o],
        yerror=[a[2] for a in n_actions_4o],
        marker=true,
        primary=false,
        dpi=300,
        color=cols[3],
    )      

    p = plot(
        s_aggro,
        s_nums,
        size=(600, 500),
        layout=Plots.grid(2, 1, heights=[0.5, 0.5]),
        bottom_margin=[-8.4mm 0mm],
    )

    return p
end

make_plot_both()
# savefig("both_moves_aggresivness_nactions_v_dialoglength_Aug24_fix.png")
# savefig("both_moves_aggresivness_nactions_v_dialoglength_Aug24_fix.pdf")
# savefig("both_moves_aggresivness_nactions_v_dialoglength_Feb24_fix.png")
# savefig("both_moves_aggresivness_nactions_v_dialoglength_Feb24_fix.pdf")