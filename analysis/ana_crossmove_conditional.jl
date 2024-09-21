
using Random
using Plots
using StatsBase
using CSV
using DataFrames
using Formatting

include("../src/game.jl")
include("../src/config.jl")


function create_transition_matrices(df_0)
    # Focus on extreme actions
    options1 = move_1_2_options_desc()[[1, 4, 2, 3]]
    short_options1 = move_1_2_options_shortdesc()[[1, 4, 2, 3]]
    # Remove superscript-action "Military Aciton"
    options2 = move_2_2_options_desc()[[2, 4, 5, 6, 9, 12, 13, 14]] # [2:end] 

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


function create_transition_aggro(df_0)

    move_1_aggro = [1, 0, 0, 1, 1, 1, 0]
    move_1_paci = [0, 1, 1, 0, 0, 0, 1]
    # move_2_aggro = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]
    # Remove superscript-action "Military Action"
    move_2_aggro = [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]

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
    
    # DIRECT, NO BOOTSTRAP
    res_move1 = df_0[!, options1]
    res_move2 = df_0[!, options2]
    p_aggro_aggro = []
    p_pacif_aggro = []
    for (k, row_k) in enumerate(eachrow(res_move2))
        # MULTIPLY WITH MASK HERE? Then check if sum is 1 or larger
        vec2_k = Vector(row_k)
        is_aggro2 = sum(vec2_k .* move_2_aggro) >= 1 ? 1 : 0

        vec1_k = Vector(res_move1[k, :])
        is_aggro1 = sum(vec1_k .* move_1_aggro) >= 1 ? 1 : 0
        is_pacif1 = sum(vec1_k .* move_1_paci) >= 1 ? 1 : 0

        aggro_aggro = is_aggro2 * is_aggro1
        pacif_aggro = is_aggro2 * is_pacif1

        push!(p_aggro_aggro, aggro_aggro)
        push!(p_pacif_aggro, pacif_aggro)
    end
    n_dat = length(p_aggro_aggro)
    println(n_dat)
    p_aggro_aggro = mean(p_aggro_aggro)
    p_pacif_aggro = mean(p_pacif_aggro)

    # BOOTSTRAP
    n_b = 10000
    Random.seed!(SEED)

    boot_aggro_aggro = []
    boot_pacif_aggro = []
    for b in range(1, n_b)
        sampled_indices = StatsBase.sample(1:nrow(df_0), n_dat, replace=true)
        boot_0 = df_0[sampled_indices, :]

        res_move1 = boot_0[!, options1]
        res_move2 = boot_0[!, options2]
        temp_p_aggro_aggro = []
        temp_p_pacif_aggro = []
        for (k, row_k) in enumerate(eachrow(res_move2))
            vec2_k = Vector(row_k)
            is_aggro2 = sum(vec2_k .* move_2_aggro) >= 1 ? 1 : 0

            vec1_k = Vector(res_move1[k, :])
            is_aggro1 = sum(vec1_k .* move_1_aggro) >= 1 ? 1 : 0
            is_pacif1 = sum(vec1_k .* move_1_paci) >= 1 ? 1 : 0

            aggro_aggro = is_aggro2 * is_aggro1
            pacif_aggro = is_aggro2 * is_pacif1

            push!(temp_p_aggro_aggro, aggro_aggro)
            push!(temp_p_pacif_aggro, pacif_aggro)
        end
        temp_p_aggro_aggro = mean(temp_p_aggro_aggro)
        temp_p_pacif_aggro = mean(temp_p_pacif_aggro)
    
        append!(boot_aggro_aggro, temp_p_aggro_aggro)
        append!(boot_pacif_aggro, temp_p_pacif_aggro)
    end

    lower = sort(boot_aggro_aggro)[round(Int, n_b * 0.025)]
    upper = sort(boot_aggro_aggro)[round(Int, n_b * 0.975)] 
    errors_aggro_aggro = (p_aggro_aggro - lower, upper - p_aggro_aggro)

    lower = sort(boot_pacif_aggro)[round(Int, n_b * 0.025)]
    upper = sort(boot_pacif_aggro)[round(Int, n_b * 0.975)] 
    errors_pacif_aggro = (p_pacif_aggro - lower, upper - p_pacif_aggro)

    println("p_aggro_aggro = $(p_aggro_aggro) +- $(errors_aggro_aggro) [$(p_aggro_aggro - errors_aggro_aggro[1]), $(p_aggro_aggro + errors_aggro_aggro[2])]")
    println("p_pacif_aggro = $(p_pacif_aggro) +- $(errors_pacif_aggro) [$(p_pacif_aggro - errors_pacif_aggro[1]), $(p_pacif_aggro + errors_pacif_aggro[2])]")

    return p_aggro_aggro, errors_aggro_aggro, p_pacif_aggro, errors_pacif_aggro
end

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
run_crossmove_aggro()

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


