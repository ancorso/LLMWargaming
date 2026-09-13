using Random
using StatsPlots
using Plots.Measures
using CSV
using DataFrames
using PlotlyJS
include("../src/game.jl")
include("../src/config.jl")


function compare_treatments_move1(df, column_name, treatments)
    short_options = move_1_2_options_shortdesc()
    options = move_1_2_options_desc()

    group = repeat(treatments, inner=length(options))

    y = [sum(df[df[!, column_name] .== t, o]) for t in treatments for o in options]
    names = repeat(short_options, outer=2)

    return names, y, group
end

function compare_treatments_move2(df, column_name, treatments)
    options = move_2_2_options_desc()[2:end] 

    group = repeat(treatments, inner=length(options))

    y = [sum(df[df[!, column_name] .== t, o]) for t in treatments for o in options]
    names = repeat(options, outer=2)

    return names, y, group
end

function full_comp_treatments(df)
    # Set default font sizes for all plots
    # default(
    #     titlefontsize = 20,
    #     guidefontsize = 16,
    #     tickfontsize = 16,
    #     legendfontsize = 16
    # )
    println("Ai Acc")
    n_aiacc, y_aiacc, g_aiacc = compare_treatments_move1(df, ai_column_name, ai_accuracies)
    println(n_aiacc)
    println(y_aiacc)
    println(g_aiacc)
    println()
    p1 = groupedbar(
        n_aiacc, y_aiacc, group=g_aiacc,
        ylabel="Counts [ ]", title="Move One",
        dpi=300, xaxis=false
    )

    println("Ai Train")
    n_aitrain, y_aitrain, g_aitrain = compare_treatments_move1(df, train_column_name, train_quality)
    println(n_aitrain)
    println(y_aitrain)
    println(g_aitrain)
    println()
    p2 = groupedbar(
        n_aitrain, y_aitrain, group=g_aitrain,
        xrot=40, ylabel="Counts [ ]",
        dpi=300, bottom_margin=15mm
    )

    println("China")
    n_china, y_china, g_china = compare_treatments_move2(df, china_column_name, china_treatments)
    println(n_china)
    println(y_china)
    println(g_china)
    println()
    p3 = groupedbar(
        n_china, y_china, group=g_china,
        xrot=40, ylabel="Counts [ ]", title="Move Two",
        dpi=300, bottom_margin=15mm, 
        ylim=(0, 30)
    )

    # Adjust the overall plot size and spacing if necessary
    p = Plots.plot(
        p1, p2, p3,
        layout = @layout[           # Use the @layout macro
            a{0.31h};                # First plot occupies 25% of the height
            b{0.31h};                # Second plot occupies 25% of the height
            c{0.38h}                  # Third plot occupies 50% of the height
        ],
        size=(1200, 1000 ),  # Increase width to accommodate labels
        spacing=10mm,
        left_margin=15mm,
    )
    return p
end
full_comp_treatments(df_real_aug24)
# Plots.savefig("chosen_actions_treatment_overview.png")
# Plots.savefig("chosen_actions_treatment_overview.pdf")

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

n, y, g = compare_treatments_move1(df_real_aug24, ai_column_name, ai_accuracies)
groupedbar(n, y, group=g, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 1", dpi=300)
# savefig("move1.png")

n, y, g = compare_treatments_move1(df_real_aug24, train_column_name, train_quality)
groupedbar(n, y, group=g, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 1", dpi=300)

n, y, g = compare_treatments_move2(df_real_aug24, china_column_name, china_treatments)
groupedbar(n, y, group=g, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 2", dpi=300)
# savefig("move2.png")




function calc_treatment_counts(df, column_name, treatments; move=1)
    if move == 1
        n, y, g = compare_treatments_move1(df, column_name, treatments)
        n_possible_actions = length(move_1_2_options_desc())
    else
        n, y, g = compare_treatments_move2(df, column_name, treatments)
        # FIXME if superscript action si counted
        n_possible_actions = length(move_2_2_options_desc()) - 1
    end
    
    # Calculate total number of actions for each treatment
    n_actions_t0 = sum([y[i] for i in 1:length(g) if g[i] == treatments[1]])
    n_actions_t1 = sum([y[i] for i in 1:length(g) if g[i] == treatments[2]])

    # Calculate how many games were played for that treatment (Calculate second in case of invalid entries)
    n_t0 = sum([1 for i in 1:length(df[!, column_name]) if df[!, column_name][i] == treatments[1]])
    n_t1 = sum([1 for i in 1:length(df[!, column_name]) if df[!, column_name][i] == treatments[2]])

    # Assume binomial uncertainty for n_actions_t0 out of n_games * n_possible actions
    println(treatments[1], " ", n_actions_t0, " ", n_t0)
    println(treatments[2], " ", n_actions_t1, " ", n_t1)
    N0 = n_possible_actions * n_t0
    N1 = n_possible_actions * n_t1
    k0 = n_actions_t0
    k1 = n_actions_t1
    p0 = k0 / N0
    p1= k1 / N1
    err0 = sqrt(N0 * p0 * (1 - p0))
    err1 = sqrt(N1 * p1 * (1 - p1))

    k_tot = k0 + k1
    frac = (k0 - k1) / k_tot
    frac_err = sqrt( (2 * k1 * err0/ k_tot^2 )^2 + (2 * k0 * err1/ k_tot^2 )^2  )

    # Return for 95% confidence level
    return k0, err0 * 1.96, k1, err1 * 1.96, (frac + 1) * 0.5, frac_err * 0.5 * 1.96, n_t0, n_t1
end

using PlotlyJS

function create_uncertain_pie(df, column_name, treatments, move)

    k0, err0, k1, err1, y, y_err, nt0, nt1 = calc_treatment_counts(df, column_name, treatments; move=move)

    # Outer ring data (Lower estimate: y - y_err)
    trace_outer = pie(
        title = "Actions/Game",
        titlefont=attr(size=25),
        values = [y - y_err, 1 - y + y_err],
        hole = 0.6, 
        sort = false,
        direction = "clockwise",
        textinfo = "none",
        marker = attr(colors = ["rgba(31, 119, 180, 0.5)", "rgba(255, 127, 14, 0.5)"]),
        domain = attr(x = [0.0, 1.0], y = [0.0, 1.0]),
        text = false,
        name = "Outer Ring"
    )

    # Middle ring data (Point estimate: y)
    trace_mid = pie(
        labels = [treatments[1], treatments[2]],
        values = [y, 1 - y],
        hole = 0.6,
        sort = false,
        direction = "clockwise",
        textinfo = "label",
        textposition = "inside",
        textfont = attr(size = 25),
        marker = attr(colors = ["rgba(31, 119, 180, 1.0)", "rgba(255, 127, 14, 1.0)"]),
        domain = attr(x = [0.05, 0.95], y = [0.05, 0.95]),
        text = false,
        name = "Middle Ring"
    )

    # Inner ring data (Upper estimate: y + y_err)
    trace_inner = pie(
        values = [y + y_err, 1 - y - y_err],
        hole = 0.75,
        sort = false,
        direction = "clockwise",
        textinfo = "none",
        marker = attr(colors = ["rgba(31, 119, 180, 0.5)", "rgba(255, 127, 14, 0.5)"]),
        domain = attr(x = [0.2, 0.8], y = [0.2, 0.8]),
        name = "Inner Ring",
    )


    s0 = string(round(k0 / nt0, digits=2)) * " +- " * string(round(err0 / nt0, digits=2))
    s1 = string(round(k1 / nt1, digits=2)) * " +- " * string(round(err1 / nt1, digits=2))

    layout = Layout(
        showlegend = false,
        shapes = [
            attr(
                type = "line",
                x0 = 0.5,
                y0 = 0.0,
                x1 = 0.5,
                y1 = 0.3,
                line = attr(
                    color = "black",
                    width = 4,
                    dash = "dash"
                )
            )
        ],
        annotations = [
            attr(
                text = s1 * "            " * s0,  # Replace with the desired text
                x = 0.5,                  # Centered horizontally
                y = -0.05,                # Position below the plot; adjust as needed
                xref = "paper",
                yref = "paper",
                showarrow = false,
                font = attr(size = 25),   # Adjust font size as needed
                align = "center",
                valign = "top"            # Align text towards the top of the bounding box
            ),
            attr(
                text = treatments[2] * "               " * treatments[1],
                x = 0.5,
                y = 1.1,
                xref = "paper",
                yref = "paper",
                showarrow = false,
                font = attr(size = 25),
                align = "center",
                valign = "top"
            )
        ]
    )

    plt = PlotlyJS.plot([trace_outer, trace_inner, trace_mid], layout)
    # display(plt)

    return plt
end

p = create_uncertain_pie(df_real_aug24, ai_column_name, ai_accuracies, 1)
PlotlyJS.savefig(p, "number_actions_treatment_ai_acc.png", scale=3)
p = create_uncertain_pie(df_real_aug24, train_column_name, train_quality, 1)
PlotlyJS.savefig(p, "number_actions_treatment_ai_train.png", scale=3)
p = create_uncertain_pie(df_real_aug24, china_column_name, china_treatments, 2)
PlotlyJS.savefig(p, "number_actions_treatment_china_posture.png", scale=3)