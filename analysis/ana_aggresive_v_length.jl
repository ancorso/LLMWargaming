
using Random
using Plots
using CSV
using DataFrames
using StatsBase
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
df_dialog7 = CSV.read("results/sensitivity_studies_new/data_dialog7.csv", DataFrame)

df4_dialogno = CSV.read("results/gpt4_turbo/data_dialogno.csv", DataFrame)
df4_dialog1 = CSV.read("results/gpt4_turbo/data_dialog1.csv", DataFrame)
df4_dialog2 = CSV.read("results/gpt4_turbo/data_dialog2.csv", DataFrame)
df4_dialog3 = CSV.read("results/gpt4_turbo/data_dialog3_2.csv", DataFrame)
# df4_dialog3_new = CSV.read("results/gpt4_turbo/data_dialog3.csv", DataFrame)
df4_dialog4 = CSV.read("results/gpt4_turbo/data_dialog4.csv", DataFrame)
df4_dialog5 = CSV.read("results/gpt4_turbo/data_dialog5.csv", DataFrame)
df4_dialog6 = CSV.read("results/gpt4_turbo/data_dialog6.csv", DataFrame)
df4_dialog7 = CSV.read("results/gpt4_turbo/data_dialog7.csv", DataFrame)

# post fix
df_gpt4_dialogno_fix = CSV.read("results/post_fix/data_gpt4_dialog0.csv", DataFrame)
df_gpt4_dialog1_fix = CSV.read("results/post_fix/data_gpt4_dialog1.csv", DataFrame)
df_gpt4_dialog2_fix = CSV.read("results/post_fix/data_gpt4_dialog2.csv", DataFrame)
df_gpt4_dialog3_fix = CSV.read("results/post_fix/data_gpt4_dialog3.csv", DataFrame)
df_gpt4_dialog4_fix = CSV.read("results/post_fix/data_gpt4_dialog4.csv", DataFrame)
df_gpt4_dialog5_fix = CSV.read("results/post_fix/data_gpt4_dialog5.csv", DataFrame)
df_gpt4_dialog6_fix = CSV.read("results/post_fix/data_gpt4_dialog6.csv", DataFrame)

df_gpt35_dialogno_fix = CSV.read("results/post_fix/data_gpt35_dialog0.csv", DataFrame)
df_gpt35_dialog1_fix = CSV.read("results/post_fix/data_gpt35_dialog1.csv", DataFrame)
df_gpt35_dialog2_fix = CSV.read("results/post_fix/data_gpt35_dialog2.csv", DataFrame)
df_gpt35_dialog3_fix = CSV.read("results/post_fix/data_gpt35_dialog3.csv", DataFrame)
df_gpt35_dialog4_fix = CSV.read("results/post_fix/data_gpt35_dialog4.csv", DataFrame)
df_gpt35_dialog5_fix = CSV.read("results/post_fix/data_gpt35_dialog5.csv", DataFrame)
df_gpt35_dialog6_fix = CSV.read("results/post_fix/data_gpt35_dialog6.csv", DataFrame)


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

function make_plot()
    Random.seed!(SEED)
    aggro = [
        get_frac_aggr(df_dialogno),
        get_frac_aggr(df_dialog1),
        get_frac_aggr(df_dialog2),
        get_frac_aggr(df_dialog3),
        get_frac_aggr(df_dialog4),
        get_frac_aggr(df_dialog5),
        get_frac_aggr(df_dialog6),
        get_frac_aggr(df_dialog7),
    ]
    aggro_h = get_frac_aggr(df_real)

    s = plot(
        xlabel="Length of Simulated Dialog [a.u.]",
        ylabel="Aggresivness [a.u.]",
        title="",
        legend=:bottomright,
        ylims=[0.12, 0.32],
    )
    scatter!(
        [0., 1., 2., 3., 4., 5., 6., 7.],
        [a[1] for a in aggro],
        yerror=[a[2] for a in aggro],
        marker=true,
        label="GPT-3.5 Data (95% Conf.)",
        dpi=300,
    )
    plot!([0., 7.], fill(aggro_h[1], 2), ribbon=aggro_h[2], label="Human Players (95% Conf.)", fillalpha=0.2)

    return s
end

function make_plot_4()
    Random.seed!(SEED)
    aggro = [
        get_frac_aggr(df4_dialogno),
        get_frac_aggr(df4_dialog1),
        get_frac_aggr(df4_dialog2),
        get_frac_aggr(df4_dialog3),
        get_frac_aggr(df4_dialog4),
        get_frac_aggr(df4_dialog5),
        get_frac_aggr(df4_dialog6),
        get_frac_aggr(df4_dialog7),
    ]
    aggro_h = get_frac_aggr(df_real)

    s = plot(
        xlabel="Length of Simulated Dialog [a.u.]",
        ylabel="Aggresivness [a.u.]",
        title="",
        legend=:bottomright,
        ylims=[0.12, 0.32],
    )
    scatter!(
        [0., 1., 2., 3., 4., 5., 6., 7.],
        [a[1] for a in aggro],
        yerror=[a[2] for a in aggro],
        marker=true,
        label="GPT-4 (95% Conf.)",
        dpi=300,
    )
    plot!([0., 7.], fill(aggro_h[1], 2), ribbon=aggro_h[2], label="Human Players (95% Conf.)", fillalpha=0.2)

    return s
end

function make_plot_both()
    Random.seed!(SEED)
    aggro = [
        get_frac_aggr(df_dialogno),
        get_frac_aggr(df_dialog1),
        get_frac_aggr(df_dialog2),
        get_frac_aggr(df_dialog3),
        get_frac_aggr(df_dialog4),
        get_frac_aggr(df_dialog5),
        get_frac_aggr(df_dialog6),
        get_frac_aggr(df_dialog7),
    ]
    aggro_h = get_frac_aggr(df_real)
    aggro4 = [
        get_frac_aggr(df4_dialogno),
        get_frac_aggr(df4_dialog1),
        get_frac_aggr(df4_dialog2),
        get_frac_aggr(df4_dialog3),
        get_frac_aggr(df4_dialog4),
        get_frac_aggr(df4_dialog5),
        get_frac_aggr(df4_dialog6),
        get_frac_aggr(df4_dialog7),
    ]

    s = plot(
        xlabel="Length of Simulated Dialog [a.u.]",
        ylabel="Aggresivness [a.u.]",
        title="",
        legend=:bottomright,
        ylims=[0.11, 0.32],
    )
    plot!([0., 7.], fill(aggro_h[1], 2), ribbon=aggro_h[2], label="Human Players (95% Conf.)", fillalpha=0.2, color=2, lw=4.)
    scatter!(
        [0., 1., 2., 3., 4., 5., 6., 7.] .- 0.05,
        [a[1] for a in aggro],
        yerror=[a[2] for a in aggro],
        marker=true,
        label="GPT-3.5 (95% Conf.)",
        dpi=300,
        color=1,
    )
    scatter!(
        [0., 1., 2., 3., 4., 5., 6., 7.] .+ 0.05,
        [a[1] for a in aggro4],
        yerror=[a[2] for a in aggro4],
        marker=true,
        label="GPT-4 (95% Conf.)",
        dpi=300,
        color=0,
    )

    return s
end

make_plot()
# savefig("both_moves_aggresivness_v_dialoglength.png")
make_plot_4()

make_plot_both()
# savefig("both_moves_aggresivness_v_dialoglength.png")


using Measures

function make_plot_both_both()
    Random.seed!(SEED)
    aggro = [
        get_frac_aggr(df_gpt35_dialogno_fix),
        get_frac_aggr(df_gpt35_dialog1_fix),
        get_frac_aggr(df_gpt35_dialog2_fix),
        get_frac_aggr(df_gpt35_dialog3_fix),
        get_frac_aggr(df_gpt35_dialog4_fix),
        get_frac_aggr(df_gpt35_dialog5_fix),
        get_frac_aggr(df_gpt35_dialog6_fix),
        # get_frac_aggr(df_dialog7),
    ]
    # aggro_h = get_frac_aggr(df_real)
    aggro_h = get_frac_aggr(df_real_feb24)
    aggro4 = [
        get_frac_aggr(df_gpt4_dialogno_fix),
        get_frac_aggr(df_gpt4_dialog1_fix),
        get_frac_aggr(df_gpt4_dialog2_fix),
        get_frac_aggr(df_gpt4_dialog3_fix),
        get_frac_aggr(df_gpt4_dialog4_fix),
        get_frac_aggr(df_gpt4_dialog5_fix),
        get_frac_aggr(df_gpt4_dialog6_fix),
        # get_frac_aggr(df4_dialog7),
    ]
    n_actions = [
        get_number_of_answers(df_gpt35_dialogno_fix),
        get_number_of_answers(df_gpt35_dialog1_fix),
        get_number_of_answers(df_gpt35_dialog2_fix),
        get_number_of_answers(df_gpt35_dialog3_fix),
        get_number_of_answers(df_gpt35_dialog4_fix),
        get_number_of_answers(df_gpt35_dialog5_fix),
        get_number_of_answers(df_gpt35_dialog6_fix),
        # get_number_of_answers(df_dialog7),
    ]
    # n_actions_h = get_number_of_answers(df_real)
    n_actions_h = get_number_of_answers(df_real_feb24)
    n_actions4 = [
        get_number_of_answers(df_gpt4_dialogno_fix),
        get_number_of_answers(df_gpt4_dialog1_fix),
        get_number_of_answers(df_gpt4_dialog2_fix),
        get_number_of_answers(df_gpt4_dialog3_fix),
        get_number_of_answers(df_gpt4_dialog4_fix),
        get_number_of_answers(df_gpt4_dialog5_fix),
        get_number_of_answers(df_gpt4_dialog6_fix),
        # get_number_of_answers(df4_dialog7),
    ]

    s_aggro = plot(
        xlabel="",
        ylabel="Aggresivness [a.u.]",
        title="",
        legend=:bottomright,
        ylims=[0.12, 0.34],
        # xaxis=false,
    )
    plot!([0., 6.3], fill(aggro_h[1], 2), ribbon=aggro_h[2], label="Human Players (95% Conf.)", fillalpha=0.2, color=2, lw=4.)
    scatter!(
        [0., 1., 2., 3., 4., 5., 6.] .- 0.05,
        [a[1] for a in aggro],
        yerror=[a[2] for a in aggro],
        marker=true,
        label="GPT-3.5 (95% Conf.)",
        dpi=300,
        color=1,
    )
    scatter!(
        [0., 1., 2., 3., 4., 5., 6.] .+ 0.05,
        [a[1] for a in aggro4],
        yerror=[a[2] for a in aggro4],
        marker=true,
        label="GPT-4 (95% Conf.)",
        dpi=300,
        color=0,
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
        # label="Human Players (95% Conf.)", 
        primary=false,
        fillalpha=0.2, 
        color=2, 
        lw=4.,
    )
    scatter!(
        [0., 1., 2., 3., 4., 5., 6.] .- 0.05,
        [a[1] for a in n_actions],
        yerror=[a[2] for a in n_actions],
        marker=true,
        # label="GPT-3.5 (95% Conf.)",
        primary=false,
        dpi=300,
        color=1,
    )
    scatter!(
        [0., 1., 2., 3., 4., 5., 6.] .+ 0.05,
        [a[1] for a in n_actions4],
        yerror=[a[2] for a in n_actions4],
        marker=true,
        # label="GPT-4 (95% Conf.)",
        primary=false,
        dpi=300,
        color=0,
    )   

    p = plot(
        s_aggro,
        s_nums,
        size=(600, 500),
        layout=Plots.grid(2, 1, heights=[0.5, 0.5]),
        bottom_margin=[-8.4mm 0mm],
    )
    # size!(200, 200)

    return p
end

make_plot_both_both()
# savefig("both_moves_aggresivness_nactions_v_dialoglength_Feb24_fix.png")
# savefig("both_moves_aggresivness_nactions_v_dialoglength_Feb24_fix.pdf")

