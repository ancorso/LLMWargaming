
using Random
using Plots
using StatsBase
using CSV
using DataFrames
using Formatting
using Plots.PlotMeasures

include("../src/game.jl")
include("../src/config.jl")


function create_transition_matrices(df_0)
    # Focus on extreme actions
    options1 = move_1_2_options_desc() #[[1, 4, 2, 3]]
    short_options1 = move_1_2_options_shortdesc()#[[1, 4, 2, 3]]
    # Remove superscript-action "Military Aciton"
    options2 = move_2_2_options_desc()[[2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]] # [2:end] 

    res_move1 = df_0[!, options1]
    res_move2 = df_0[!, options2]

    n_cols1 = size(res_move1, 2)
    n_cols2 = size(res_move2, 2)
    p_trans = zeros(Float64, n_cols1, n_cols2)
    p_trans_noti = zeros(Float64, n_cols1, n_cols2)

    for (j, col_j) in enumerate(eachcol(res_move2))
        vec_j = Vector(col_j)
        for (i, col_i) in enumerate(eachcol(res_move1))
            vec_i = Vector(col_i)
            p_ij = mean(vec_i .* vec_j)
            p_trans[i, j] = p_ij

            p_ij_noti = mean( (.!vec_i) .* vec_j)
            p_trans_noti[i, j] = p_ij_noti
        end
    end

    for trans_mat in [p_trans, p_trans_noti]
        col_lab = [first(s, 15) for s in options2]
        row_lab = short_options1 #[first(s, 19) for s in short_options1]
        rounded_df = DataFrame(round.(trans_mat, digits=2), col_lab)
        rounded_df[!, :Row] = row_lab
        rounded_df = select(rounded_df, :Row, :) 
        display(rounded_df)

    end
end

println("HUMANS")
create_transition_matrices(df_real_feb24)

println("GPT3.5")
create_transition_matrices(df_gpt35_dialog3_fix)

println("GPT4.0")
create_transition_matrices(df_gpt4_dialog3_fix)


function create_transition_aggro(df_0, do_china_treat=0)

    # The validity of these splits is limited by the experiment design/availabel actions
    # While it provides some signal, I would view them with a bit of care
    move_1_aggro = [1, 0, 0, 1, 1, 1, 0]
    move_1_paci = [0, 1, 1, 0, 0, 0, 1]
    # move_2_aggro = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]
    # Remove superscript-action "Military Action"
    move_2_aggro = [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]

    move_2_viol = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    move_2_ambi = [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    move_2_nonv = [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1]

    # # Check for hard military in 1 and military in 2 + economic sanctions
    # move_1_aggro = [1, 0, 0, 1, 1, 0, 0]
    # move_1_paci = [0, 1, 1, 0, 0, 1, 1]
    # # move_2_aggro = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]
    # # Remove superscript-action "Military Action"
    # move_2_aggro = [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]


    # Focus on extreme actions
    options1 = move_1_2_options_desc() 
    short_options1 = move_1_2_options_shortdesc() 
    # Remove superscript-action "Military Action"
    options2 = move_2_2_options_desc()[2:end]
    # options2 = move_2_2_options_desc()

    df_list = [df_0]
    if do_china_treat != 0
        df_01 = df_0[df_0[!, china_column_name] .== china_treatments[1], :]
        df_02 = df_0[df_0[!, china_column_name] .== china_treatments[2], :]
        df_list = [df_01, df_02]
    end
    
    p1 = plot()
    categories = ["p (aggro_2 | aggro_1)", "p (aggro_2 | pacif_1)", "p (viol_2 | aggro_1)", "p (viol_2 | pacif_1)",
    "p (ambi_2 | aggro_1)", "p (ambi_2 | pacif_1)", "p (nonviol_2 | aggro_1)", "p (nonviol_2 | pacif_1)"]
    x_positions = 1:length(categories)

    for (df_i, df) in enumerate(df_list)

        # DIRECT, NO BOOTSTRAP
        res_move1 = df[!, options1]
        res_move2 = df[!, options2]
        p_aggro_aggro = []
        p_pacif_aggro = []

        p_aggro_viol = []
        p_pacif_viol = []
        p_aggro_ambi = []
        p_pacif_ambi = []
        p_aggro_nonv = []
        p_pacif_nonv = []
        for (k, row_k) in enumerate(eachrow(res_move2))
            # MULTIPLY WITH MASK HERE? Then check if sum is 1 or larger
            vec2_k = Vector(row_k)
            is_aggro2 = sum(vec2_k .* move_2_aggro) >= 1 ? 1 : 0
            is_viol2 = sum(vec2_k .* move_2_viol) >= 1 ? 1 : 0
            is_ambi2 = sum(vec2_k .* move_2_ambi) >= 1 ? 1 : 0
            is_nonv2 = sum(vec2_k .* move_2_nonv) >= 1 ? 1 : 0

            vec1_k = Vector(res_move1[k, :])
            is_aggro1 = sum(vec1_k .* move_1_aggro) >= 1 ? 1 : 0
            is_pacif1 = sum(vec1_k .* move_1_paci) >= 1 ? 1 : 0

            aggro_aggro = is_aggro2 * is_aggro1
            pacif_aggro = is_aggro2 * is_pacif1

            push!(p_aggro_aggro, aggro_aggro)
            push!(p_pacif_aggro, pacif_aggro)

            aggro_viol = is_viol2 * is_aggro1
            pacif_viol = is_viol2 * is_pacif1
            aggro_ambi = is_ambi2 * is_aggro1
            pacif_ambi = is_ambi2 * is_pacif1
            aggro_nonv = is_nonv2 * is_aggro1
            pacif_nonv = is_nonv2 * is_pacif1

            push!(p_aggro_viol, aggro_viol)
            push!(p_pacif_viol, pacif_viol)
            push!(p_aggro_ambi, aggro_ambi)
            push!(p_pacif_ambi, pacif_ambi)
            push!(p_aggro_nonv, aggro_nonv)
            push!(p_pacif_nonv, pacif_nonv)
        end
        n_dat = length(p_aggro_aggro)
        println(n_dat)
        p_aggro_aggro = mean(p_aggro_aggro)
        p_pacif_aggro = mean(p_pacif_aggro)

        p_aggro_viol = mean(p_aggro_viol)
        p_pacif_viol = mean(p_pacif_viol)
        p_aggro_ambi = mean(p_aggro_ambi)
        p_pacif_ambi = mean(p_pacif_ambi)
        p_aggro_nonv = mean(p_aggro_nonv)
        p_pacif_nonv = mean(p_pacif_nonv)

        # BOOTSTRAP
        n_b = 10000
        Random.seed!(SEED)

        boot_aggro_aggro = []
        boot_pacif_aggro = []

        boot_aggro_viol = []
        boot_pacif_viol = []
        boot_aggro_ambi = []
        boot_pacif_ambi = []
        boot_aggro_nonv = []
        boot_pacif_nonv = []
        for b in range(1, n_b)
            sampled_indices = StatsBase.sample(1:nrow(df), n_dat, replace=true)
            boot_0 = df[sampled_indices, :]

            res_move1 = boot_0[!, options1]
            res_move2 = boot_0[!, options2]
            temp_p_aggro_aggro = []
            temp_p_pacif_aggro = []
            temp_p_aggro_viol = []
            temp_p_pacif_viol = []
            temp_p_aggro_ambi = []
            temp_p_pacif_ambi = []
            temp_p_aggro_nonv = []
            temp_p_pacif_nonv = []
            for (k, row_k) in enumerate(eachrow(res_move2))
                vec2_k = Vector(row_k)
                is_aggro2 = sum(vec2_k .* move_2_aggro) >= 1 ? 1 : 0
                is_viol2 = sum(vec2_k .* move_2_viol) >= 1 ? 1 : 0
                is_ambi2 = sum(vec2_k .* move_2_ambi) >= 1 ? 1 : 0
                is_nonv2 = sum(vec2_k .* move_2_nonv) >= 1 ? 1 : 0

                vec1_k = Vector(res_move1[k, :])
                is_aggro1 = sum(vec1_k .* move_1_aggro) >= 1 ? 1 : 0
                is_pacif1 = sum(vec1_k .* move_1_paci) >= 1 ? 1 : 0

                aggro_aggro = is_aggro2 * is_aggro1
                pacif_aggro = is_aggro2 * is_pacif1

                push!(temp_p_aggro_aggro, aggro_aggro)
                push!(temp_p_pacif_aggro, pacif_aggro)

                aggro_viol = is_viol2 * is_aggro1
                pacif_viol = is_viol2 * is_pacif1
                aggro_ambi = is_ambi2 * is_aggro1
                pacif_ambi = is_ambi2 * is_pacif1
                aggro_nonv = is_nonv2 * is_aggro1
                pacif_nonv = is_nonv2 * is_pacif1

                push!(temp_p_aggro_viol, aggro_viol)
                push!(temp_p_pacif_viol, pacif_viol)
                push!(temp_p_aggro_ambi, aggro_ambi)
                push!(temp_p_pacif_ambi, pacif_ambi)
                push!(temp_p_aggro_nonv, aggro_nonv)
                push!(temp_p_pacif_nonv, pacif_nonv)
            end
            temp_p_aggro_aggro = mean(temp_p_aggro_aggro)
            temp_p_pacif_aggro = mean(temp_p_pacif_aggro)

            temp_p_aggro_viol = mean(temp_p_aggro_viol)
            temp_p_pacif_viol = mean(temp_p_pacif_viol)
            temp_p_aggro_ambi = mean(temp_p_aggro_ambi)
            temp_p_pacif_ambi = mean(temp_p_pacif_ambi)
            temp_p_aggro_nonv = mean(temp_p_aggro_nonv)
            temp_p_pacif_nonv = mean(temp_p_pacif_nonv)
        
            append!(boot_aggro_aggro, temp_p_aggro_aggro)
            append!(boot_pacif_aggro, temp_p_pacif_aggro)

            append!(boot_aggro_viol, temp_p_aggro_viol)
            append!(boot_pacif_viol, temp_p_pacif_viol)
            append!(boot_aggro_ambi, temp_p_aggro_ambi)
            append!(boot_pacif_ambi, temp_p_pacif_ambi)
            append!(boot_aggro_nonv, temp_p_aggro_nonv)
            append!(boot_pacif_nonv, temp_p_pacif_nonv)
        end

        lower = sort(boot_aggro_aggro)[round(Int, n_b * 0.025)]
        upper = sort(boot_aggro_aggro)[round(Int, n_b * 0.975)] 
        errors_aggro_aggro = (p_aggro_aggro - lower, upper - p_aggro_aggro)

        lower = sort(boot_pacif_aggro)[round(Int, n_b * 0.025)]
        upper = sort(boot_pacif_aggro)[round(Int, n_b * 0.975)] 
        errors_pacif_aggro = (p_pacif_aggro - lower, upper - p_pacif_aggro)

        println("p_aggro_aggro = $(p_aggro_aggro) +- $(errors_aggro_aggro) [$(p_aggro_aggro - errors_aggro_aggro[1]), $(p_aggro_aggro + errors_aggro_aggro[2])]")
        println("p_pacif_aggro = $(p_pacif_aggro) +- $(errors_pacif_aggro) [$(p_pacif_aggro - errors_pacif_aggro[1]), $(p_pacif_aggro + errors_pacif_aggro[2])]")

        lower = sort(boot_aggro_viol)[round(Int, n_b * 0.025)]
        upper = sort(boot_aggro_viol)[round(Int, n_b * 0.975)] 
        errors_aggro_viol = (p_aggro_viol - lower, upper - p_aggro_viol)

        lower = sort(boot_pacif_viol)[round(Int, n_b * 0.025)]
        upper = sort(boot_pacif_viol)[round(Int, n_b * 0.975)] 
        errors_pacif_viol = (p_pacif_viol - lower, upper - p_pacif_viol)

        lower = sort(boot_aggro_ambi)[round(Int, n_b * 0.025)]
        upper = sort(boot_aggro_ambi)[round(Int, n_b * 0.975)] 
        errors_aggro_ambi = (p_aggro_ambi - lower, upper - p_aggro_ambi)

        lower = sort(boot_pacif_ambi)[round(Int, n_b * 0.025)]
        upper = sort(boot_pacif_ambi)[round(Int, n_b * 0.975)] 
        errors_pacif_ambi = (p_pacif_ambi - lower, upper - p_pacif_ambi)

        lower = sort(boot_aggro_nonv)[round(Int, n_b * 0.025)]
        upper = sort(boot_aggro_nonv)[round(Int, n_b * 0.975)] 
        errors_aggro_nonv = (p_aggro_nonv - lower, upper - p_aggro_nonv)

        lower = sort(boot_pacif_nonv)[round(Int, n_b * 0.025)]
        upper = sort(boot_pacif_nonv)[round(Int, n_b * 0.975)] 
        errors_pacif_nonv = (p_pacif_nonv - lower, upper - p_pacif_nonv)

        println("p_aggro_viol = $(p_aggro_viol) +- $(errors_aggro_viol) [$(p_aggro_viol - errors_aggro_viol[1]), $(p_aggro_viol + errors_aggro_viol[2])]")
        println("p_pacif_viol = $(p_pacif_viol) +- $(errors_pacif_viol) [$(p_pacif_viol - errors_pacif_viol[1]), $(p_pacif_viol + errors_pacif_viol[2])]")
        println("p_aggro_ambi = $(p_aggro_ambi) +- $(errors_aggro_ambi) [$(p_aggro_ambi - errors_aggro_ambi[1]), $(p_aggro_ambi + errors_aggro_ambi[2])]")
        println("p_pacif_ambi = $(p_pacif_ambi) +- $(errors_pacif_ambi) [$(p_pacif_ambi - errors_pacif_ambi[1]), $(p_pacif_ambi + errors_pacif_ambi[2])]")
        println("p_aggro_nonv = $(p_aggro_nonv) +- $(errors_aggro_nonv) [$(p_aggro_nonv - errors_aggro_nonv[1]), $(p_aggro_nonv + errors_aggro_nonv[2])]")
        println("p_pacif_nonv = $(p_pacif_nonv) +- $(errors_pacif_nonv) [$(p_pacif_nonv - errors_pacif_nonv[1]), $(p_pacif_nonv + errors_pacif_nonv[2])]")

        label_str = "Cond. prop."
        if do_china_treat != 0
            label_str =  china_treatments[df_i]
        end
        scatter!(
            x_positions,
            [p_aggro_aggro, p_pacif_aggro, p_aggro_viol, p_pacif_viol, p_aggro_ambi, p_pacif_ambi, p_aggro_nonv, p_pacif_nonv],
            yerror=[errors_aggro_aggro, errors_pacif_aggro, errors_aggro_viol, errors_pacif_viol, errors_aggro_ambi, errors_pacif_ambi, errors_aggro_nonv, errors_pacif_nonv],
            marker=true,
            label=label_str,
            dpi=300,
            color=df_i,
            xrot=45,
            bottom_margin=15mm,
            left_margin=15mm,
            ylims=[0., 1.], 
            # ylabel="p_aggro_2",
            xticks=(x_positions, categories)
        )
        x_positions = x_positions .+ 0.1
    end

    # return p_aggro_aggro, errors_aggro_aggro, p_pacif_aggro, errors_pacif_aggro
    return p1
end

create_transition_aggro(df_real_aug24)
# savefig("cross_move_enhanced_cats.png")
p2 = create_transition_aggro(df_real_aug24, 1)
# savefig("cross_move_enhanced_cats_treats.png")



function run_crossmove_aggro()
    labs = ["Humans", "GPT3.5", "GPT4.0", "GPT4o", "Random"]
    # res_hum = create_transition_aggro(df_real_feb24)
    res_hum = create_transition_aggro(df_real_aug24)
    res_gpt35 = create_transition_aggro(df_gpt35_dialog3_fix)
    res_gpt4 = create_transition_aggro(df_gpt4_dialog3_fix)
    res_gpt4o = create_transition_aggro(df_gpt4o_dialog3_fix)


    random_df = DataFrame(rand(Bool, (500, size(df_gpt35_dialog3_fix)[2])) .* 1, names(df_gpt35_dialog3_fix))
    res_random = create_transition_aggro(random_df)

    scatter(
        labs,
        [res_hum[1], res_gpt35[1], res_gpt4[1],  res_gpt4o[1], res_random[1]],
        yerror=[res_hum[2], res_gpt35[2], res_gpt4[2], res_gpt4o[2], res_random[2]],
        marker=true,
        label="p ( aggro_2 | aggro_1)",
        dpi=300,
        color=1,
        ylabel="p_aggro_2",
    )
    scatter!(
        labs,
        [res_hum[3], res_gpt35[3], res_gpt4[3], res_gpt4o[3], res_random[3]],
        yerror=[res_hum[4], res_gpt35[4], res_gpt4[4], res_gpt4o[4], res_random[4]],
        marker=true,
        label="p ( aggro_2 | pacif_1)",
        dpi=300,
        color=0,
    )
end
# Needs change back to multi output instead of jsut p1 for later functions (bad practice, i know)
# run_crossmove_aggro()
# savefig("cross_move_aggro_probs.png")

# Aug24
# 48 (Human)
# p_aggro_aggro = 0.9375 +- (0.08333333333333337, 0.0625) [0.8541666666666666, 1.0]
# p_pacif_aggro = 0.6458333333333334 +- (0.14583333333333337, 0.125) [0.5, 0.7708333333333334]
# 80 (GPT3.5)
# p_aggro_aggro = 0.975 +- (0.03749999999999998, 0.025000000000000022) [0.9375, 1.0]
# p_pacif_aggro = 0.85 +- (0.08750000000000002, 0.07500000000000007) [0.7625, 0.925]
# 79 (GPT4)
# p_aggro_aggro = 0.9873417721518988 +- (0.025316455696202556, 0.012658227848101222) [0.9620253164556962, 1.0]
# p_pacif_aggro = 0.7341772151898734 +- (0.10126582278481011, 0.10126582278481011) [0.6329113924050633, 0.8354430379746836]
# 500 (Random)
# p_aggro_aggro = 0.932 +- (0.02400000000000002, 0.02199999999999991) [0.908, 0.954]
# p_pacif_aggro = 0.854 +- (0.03200000000000003, 0.030000000000000027) [0.822, 0.884]

# GPT4o
# 80
# p_aggro_aggro = 1.0 +- (0.0, 0.0) [1.0, 1.0]
# p_pacif_aggro = 0.8625 +- (0.07500000000000007, 0.07499999999999996) [0.7875, 0.9375]
















function create_transition_aggro_diff(df_0)

    move_1_aggro = [1, 0, 0, 1, 1, 1, 0]
    move_1_paci = [0, 1, 1, 0, 0, 0, 1]
    # move_2_aggro = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]
    # Remove superscript-action "Military Action"
    move_2_aggro = [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]


    # Focus on extreme actions
    options1 = move_1_2_options_desc() 
    short_options1 = move_1_2_options_shortdesc() 
    # Remove superscript-action "Military Action"
    options2 = move_2_2_options_desc()[2:end]
    
    # DIRECT, NO BOOTSTRAP
    res_move1 = df_0[!, options1]
    res_move2 = df_0[!, options2]
    p_aggro_diff = []
    for (k, row_k) in enumerate(eachrow(res_move2))
        # MULTIPLY WITH MASK HERE? Then check if sum is 1 or larger
        vec2_k = Vector(row_k)
        is_aggro2 = sum(vec2_k .* move_2_aggro) >= 1 ? 1 : 0

        vec1_k = Vector(res_move1[k, :])
        is_aggro1 = sum(vec1_k .* move_1_aggro) >= 1 ? 1 : 0
        is_pacif1 = sum(vec1_k .* move_1_paci) >= 1 ? 1 : 0

        aggro_aggro = is_aggro2 * is_aggro1
        pacif_aggro = is_aggro2 * is_pacif1

        push!(p_aggro_diff, aggro_aggro - pacif_aggro)
    end
    n_dat = length(p_aggro_diff)
    println(n_dat)
    p_aggro_diff = mean(p_aggro_diff)

    # BOOTSTRAP
    n_b = 10000
    Random.seed!(SEED)

    boot_aggro_diff = []
    for b in range(1, n_b)
        # boot_0 = StatsBase.sample(df_0, n_dat, replace=true)
        sampled_indices = StatsBase.sample(1:nrow(df_0), n_dat, replace=true)
        boot_0 = df_0[sampled_indices, :]

        res_move1 = boot_0[!, options1]
        res_move2 = boot_0[!, options2]
        temp_p_aggro_diff= []
        for (k, row_k) in enumerate(eachrow(res_move2))
            # MULTIPLY WITH MASK HERE? Then check if sum is 1 or larger
            vec2_k = Vector(row_k)
            is_aggro2 = sum(vec2_k .* move_2_aggro) >= 1 ? 1 : 0

            vec1_k = Vector(res_move1[k, :])
            is_aggro1 = sum(vec1_k .* move_1_aggro) >= 1 ? 1 : 0
            is_pacif1 = sum(vec1_k .* move_1_paci) >= 1 ? 1 : 0

            aggro_aggro = is_aggro2 * is_aggro1
            pacif_aggro = is_aggro2 * is_pacif1

            push!(temp_p_aggro_diff, aggro_aggro - pacif_aggro)
        end
        temp_p_aggro_diff = mean(temp_p_aggro_diff)
    
        append!(boot_aggro_diff, temp_p_aggro_diff)
    end

    lower = sort(boot_aggro_diff)[round(Int, n_b * 0.025)]
    upper = sort(boot_aggro_diff)[round(Int, n_b * 0.975)] 
    errors_aggro_diff = (p_aggro_diff - lower, upper - p_aggro_diff)
    println("p_aggro_aggro = $(p_aggro_diff) +- $(errors_aggro_diff)")

    return p_aggro_diff, errors_aggro_diff
end

function run_crossmove_aggro_diff()
    labs = ["Humans", "GPT3.5", "GPT4.0", "Random"]
    res_hum = create_transition_aggro_diff(df_real_aug24)
    res_gpt35 = create_transition_aggro_diff(df_gpt35_dialog3_fix)
    res_gpt4 = create_transition_aggro_diff(df_gpt4_dialog3_fix)

    random_df = DataFrame(rand(Bool, (500, size(df_gpt35_dialog3_fix)[2])) .* 1, names(df_gpt35_dialog3_fix))
    res_random = create_transition_aggro_diff(random_df)

    scatter(
        labs,
        [res_hum[1], res_gpt35[1], res_gpt4[1], res_random[1]],
        yerror=[res_hum[2], res_gpt35[2], res_gpt4[2], res_random[2]],
        marker=true,
        label="p ( aggro_2 | aggro_1) - p ( aggro_2 | pacif_1)",
        dpi=300,
        color=1,
        ylabel="p_aggro_diff",
    )

end
run_crossmove_aggro_diff()








function phi(x::Vector{<:Integer}, y::Vector{<:Integer})
    a = sum((x .== 1) .& (y .== 1))
    b = sum((x .== 1) .& (y .== 0))
    c = sum((x .== 0) .& (y .== 1))
    d = sum((x .== 0) .& (y .== 0))
    numerator = (a * d) - (b * c)
    denominator = sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return denominator != 0 ? numerator / denominator : 0.0
end

function create_correlation_matrix(df_0)
    # Focus on extreme actions
    options1 = move_1_2_options_desc()[[1, 2, 3, 4, 5, 6]] #[[1, 4, 2, 3]]
    short_options1 = move_1_2_options_shortdesc()[[1, 2, 3, 4, 5, 6]]#[[1, 4, 2, 3]]

    options2 = move_2_2_options_desc()[[2, 4, 5, 6, 8, 9, 10, 11, 13, 14]] #[2:end]
    res_move1 = df_0[!, options1]
    res_move2 = df_0[!, options2]
    n_cols1 = size(res_move1, 2)
    n_cols2 = size(res_move2, 2)
    # == ncol(df)?
    # Compute the Phi correlation matrix
    correlation_matrix1 = Matrix{Float64}(undef, n_cols1, n_cols1)
    correlation_matrix2 = Matrix{Float64}(undef, n_cols2, n_cols2)

    for (dat, mat) in [[res_move1, correlation_matrix1], [res_move2, correlation_matrix2]]
        for i in 1:ncol(dat)
            for j in i:ncol(dat)
                if i == j
                    mat[i, j] = 1.0
                else
                    # Compute Phi coefficient
                    x = dat[!, i]
                    y = dat[!, j]
                    mat[i, j] = phi(x, y)
                    mat[j, i] = mat[i, j]  # Symmetric
                end
            end
        end
    end

    correlation_df1 = DataFrame(correlation_matrix1, :auto)
    correlation_df2 = DataFrame(correlation_matrix2, :auto)

    rename!(correlation_df1, short_options1)
    rename!(correlation_df2, [first(s, 15) for s in options2])
    # display(correlation_df1)
    # display(correlation_df2)

    col_lab = short_options1
    row_lab = short_options1
    rounded_df = DataFrame(round.(correlation_df1, digits=2))
    rounded_df[!, :Row] = row_lab
    rounded_df = select(rounded_df, :Row, :) 
    display(rounded_df)

    col_lab = options2
    row_lab = [first(s, 15) for s in options2]
    rounded_df = DataFrame(round.(correlation_df2, digits=2))
    rounded_df[!, :Row] = row_lab
    rounded_df = select(rounded_df, :Row, :) 
    display(rounded_df)


    # return correlation_df
end

println("HUMANS")
create_correlation_matrix(df_real_feb24)







# using DataFrames, Statistics, HypothesisTests


# function create_correlation_matrix(df_0; alpha=0.05)
#     # Focus on extreme actions
#     options1 = move_1_2_options_desc() #[[1, 4, 2, 3]]
#     short_options1 = move_1_2_options_shortdesc()#[[1, 4, 2, 3]]

#     res_move1 = df_0[!, options1]

#     n_cols1 = size(res_move1, 2)
    
#     # Initialize correlation matrix and p-value matrix
#     correlation_matrix = Matrix{Float64}(undef, n_cols1, n_cols1)
#     p_value_matrix = Matrix{Float64}(undef, n_cols1, n_cols1)
    
#     # Data structure to store p-values for multiple testing correction
#     p_values_list = Float64[]
#     pairs_list = Tuple{String, String}[]

#     for i in 1:n_cols1
#         for j in i:n_cols1
#             if i == j
#                 correlation_matrix[i, j] = 1.0
#                 p_value_matrix[i, j] = 0.0  # p-value for self-correlation is 0
#             else
#                 # Extract binary vectors
#                 x = res_move1[!, i]
#                 y = res_move1[!, j]
                
#                 # Compute Phi coefficient
#                 phi_coeff = phi(x, y)
#                 correlation_matrix[i, j] = phi_coeff
#                 correlation_matrix[j, i] = phi_coeff  # Symmetric
                
#                 # Create contingency table
#                 a = sum((x .== 1) .& (y .== 1))
#                 b = sum((x .== 1) .& (y .== 0))
#                 c = sum((x .== 0) .& (y .== 1))
#                 d = sum((x .== 0) .& (y .== 0))
#                 contingency_table = [
#                     a  b
#                     c  d
#                 ]
                
#                 # Perform Chi-Square Test of Independence
#                 test = ChiSquareTest(contingency_table)
#                 p_val = pvalue(test)
#                 p_value_matrix[i, j] = p_val
#                 p_value_matrix[j, i] = p_val  # Symmetric
                
#                 # Store p-value and pair for multiple testing correction
#                 push!(p_values_list, p_val)
#                 push!(pairs_list, (short_options1[i], short_options1[j]))
#             end
#         end
#     end

#     # Convert to DataFrame for better readability
#     correlation_df = DataFrame(correlation_matrix, :auto)
#     rename!(correlation_df, short_options1)

#     # Multiple Testing Correction using Bonferroni
#     m = length(p_values_list)  # Number of tests
#     adjusted_p_values = [min(p * m, 1.0) for p in p_values_list]  # Bonferroni correction
    
#     # Alternatively, Benjamini-Hochberg Procedure for FDR control
#     function benjamini_hochberg(pvals, alpha=0.05)
#         sorted_indices = sortperm(pvals)
#         sorted_pvals = pvals[sorted_indices]
#         m = length(pvals)
#         thresholds = [ (i/m) * alpha for i in 1:m ]
#         significant = findall(i -> sorted_pvals[i] <= thresholds[i], 1:m)
#         if isempty(significant)
#             return Int[]
#         else
#             max_i = maximum(significant)
#             return sorted_indices[1:max_i]
#         end
#     end

#     significant_indices = benjamini_hochberg(p_values_list, alpha)
#     adjusted_p_values_fdr = similar(adjusted_p_values)
#     for i in 1:length(p_values_list)
#         adjusted_p_values_fdr[i] = p_values_list[i]  # Placeholder if needed
#     end

#     # Determine significant pairs based on Bonferroni correction
#     significant_pairs = pairs_list[findall(p -> p * m < alpha, p_values_list)]
    
#     # Alternatively, determine significant pairs based on Benjamini-Hochberg
#     significant_pairs_fdr = pairs_list[significant_indices]
    
#     # Display Correlation Matrix
#     for trans_mat in [correlation_matrix]
#         col_lab = short_options1
#         row_lab = short_options1 #[first(s, 19) for s in short_options1]
#         rounded_df = DataFrame(round.(trans_mat, digits=2), col_lab)
#         rounded_df[!, :Row] = row_lab
#         rounded_df = select(rounded_df, :Row, :) 
#         display(rounded_df)
#     end

#     # Display P-Value Matrix (Optional)
#     # You can similarly display the p-value matrix if desired

#     # Display Significant Pairs (Bonferroni)
#     println("\nSignificant Pairs after Bonferroni Correction (α = $alpha):")
#     for pair in significant_pairs
#         println(pair)
#     end

#     # Display Significant Pairs (Benjamini-Hochberg)
#     println("\nSignificant Pairs after Benjamini-Hochberg FDR Correction (α = $alpha):")
#     for pair in significant_pairs_fdr
#         println(pair)
#     end

#     # Optionally, return the correlation matrix and significant pairs
#     return (correlation_matrix=correlation_matrix,
#             p_value_matrix=p_value_matrix,
#             significant_pairs_bonferroni=significant_pairs,
#             significant_pairs_fdr=significant_pairs_fdr)
# end

# println("HUMANS")
# results = create_correlation_matrix(df_real_feb24)

# # Accessing the results
# correlation_matrix = results.correlation_matrix
# p_value_matrix = results.p_value_matrix
# significant_bonferroni = results.significant_pairs_bonferroni
# significant_fdr = results.significant_pairs_fdr