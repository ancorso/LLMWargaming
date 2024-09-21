
using Random
using Plots
using CSV
using DataFrames
using Statistics
using TSne
using MultivariateStats
using LinearAlgebra

include("../src/game.jl")
include("../src/config.jl")

# Function to calculate the mean of the dataset
function calculate_mean(data)
    return mean(data, dims=1)
end

# Function to calculate the covariance matrix
function calculate_covariance(data, mean)
    n = size(data, 1)
    return (data .- mean)' * (data .- mean) / n
end

# Function to plot the 95% confidence ellipse
function plot_confidence_ellipse(mean, covariance, ax, col; label="95% Conf.", primary=true)
    eigenvalues, eigenvectors = eigen(covariance)
    angle = atan(eigenvectors[2,1], eigenvectors[1,1])
    chisquare_val = 2.4477  # 95% confidence interval
    theta = LinRange(0, 2 * π, 100)

    a = sqrt(eigenvalues[1]) * chisquare_val
    b = sqrt(eigenvalues[2]) * chisquare_val

    ellipse_x = mean[1] .+ a * cos.(theta) * cos(angle) - b * sin.(theta) * sin(angle)
    ellipse_y = mean[2] .+ a * cos.(theta) * sin(angle) + b * sin.(theta) * cos(angle)

    plot!(ax, ellipse_x, ellipse_y, label=label, seriestype=:shape, color=col, fill=(0.2, col), alpha=0.2, primary=primary)
end

function create_pca_plot(df_0, df_1, tit, plot_labs; move=1, method="LDA", add_noise=true, add_random=false)

    Random.seed!(SEED)

    plot_title = "Move $(move): " * tit
    if move == 1
        short_options = move_1_2_options_shortdesc()
        options = move_1_2_options_desc()
    elseif move == 2
        options = move_2_2_options_desc()
        short_options = options
    elseif move == 0
        options = [move_1_2_options_desc()..., move_2_2_options_desc()...]
        plot_title = "Both Moves: " * tit
    end

    # Convert DataFrame columns to arrays and then to a matrix
    res_0_mat = Matrix(df_0[:, options])
    res_1_mat = Matrix(df_1[:, options])

    rescale(A; dims=1) = (A .- Statistics.mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())

    if add_random
        append!(plot_labs, ["Random"])

        # Half of total data will be random
        # n_rand = size(res_0_mat)[1] + size(res_1_mat)[1]
        # third of total data will be random
        n_rand = round(Int, (size(res_0_mat)[1] + size(res_1_mat)[1]) * 0.5)


        res_rand = [rand(0:1, size(res_1_mat)[2]) for _ in 1:n_rand]
        res_rand = hcat(res_rand...)
        rand_labels = ones(n_rand) * -1

        res_tot = transpose(vcat(res_0_mat, res_1_mat, transpose(res_rand)))
        res_labels = [ones(size(res_0_mat)[1])..., 2 .+ zeros(size(res_1_mat)[1])..., rand_labels...]

    else 
        res_tot = transpose(vcat(res_0_mat, res_1_mat))
        res_labels = [ones(size(res_0_mat)[1])..., 2 .+ zeros(size(res_1_mat)[1])...]

    end

    res_tot = rescale(res_tot)


    if method == "LDA"
        # Do LDA
        lda = fit(MulticlassLDA, res_tot, res_labels; outdim=2)
        preds = predict(lda, res_tot)
    elseif method == "PCA"
        # Do PCA
        m_pca = fit(PCA, res_tot; maxoutdim=2)
        preds = predict(m_pca, res_tot)
    elseif method == "tSNE"
        # Do tSNE
        preds = transpose(rescale(tsne(transpose(res_tot), 2)))
    else
        @error "Not implemented method $(method)"
    end

    if add_noise
        preds = preds + 0.025 .* randn(size(preds))  
    end

    s = plot(
        # yticks=(1:length(opts), opts),
        # xrot=60,
        # bottom_margin=15mm,
        # left_margin=12mm,
        # xlabel="$(method) dim 1 [ ]",
        # ylabel="$(method) dim 2 [ ]",
        xlabel="Response Vector Projection Dim 1 [a.u.]",
        ylabel="Response Vector Projection Dim 2 [a.u.]",
        # xlims=[-0.82, 0.82],
        title=plot_title,
        legend=:topleft,
    )
    # vline!([0], linestyle=:dash, linecolor="#100B00", label=nothing)


    for (lab_i, lab) in enumerate(unique(res_labels))
        data = transpose(preds[:, res_labels .== lab])
        data_mean = calculate_mean(data)
        covariance = calculate_covariance(data, data_mean)

        if lab != -1.
            col = Int(lab)
            plot_confidence_ellipse(data_mean, covariance, s, col; primary=false)
            scatter!(
                preds[1, :][res_labels .== lab],
                preds[2, :][res_labels .== lab],
                # collect(1:length(options)),
                # xerror=errors,
                label=plot_labs[lab_i] * " Data (95% Conf.)",
                color=col,
                dpi=300,
            )
        else
            plot_confidence_ellipse(data_mean, covariance, s, 4; label="Random Data (95% Conf.)")
        end

    end

    return s
end

function create_pca_plot(df_0, df_1, df_2, df_3, tit, plot_labs; move=1, method="LDA", add_noise=true, add_random=false)

    Random.seed!(SEED)

    plot_title = "Move $(move): " * tit
    if move == 1
        short_options = move_1_2_options_shortdesc()
        options = move_1_2_options_desc()
    elseif move == 2
        options = move_2_2_options_desc()
        short_options = options
    elseif move == 0
        options = [move_1_2_options_desc()..., move_2_2_options_desc()...]
        # plot_title = "Both Moves: " * tit
        plot_title = tit
    end

    # Convert DataFrame columns to arrays and then to a matrix
    res_0_mat = Matrix(df_0[:, options])
    res_1_mat = Matrix(df_1[:, options])
    res_2_mat = Matrix(df_2[:, options])
    res_3_mat = Matrix(df_3[:, options])
    rescale(A; dims=1) = (A .- Statistics.mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())

    if add_random
        # insert!(plot_labs, 3, "Random")
        append!(plot_labs, ["Random"])

        # Quarter of total data will be random
        n_rand = round(Int, (size(res_0_mat)[1] + size(res_1_mat)[1] + size(res_2_mat)[1] + size(res_3_mat)[1]) * 0.25)

        res_rand = [rand(0:1, size(res_1_mat)[2]) for _ in 1:n_rand]
        res_rand = hcat(res_rand...)
        rand_labels = ones(n_rand) * -1

        res_tot = transpose(vcat(res_0_mat, res_1_mat, res_2_mat, res_3_mat, transpose(res_rand)))
        res_labels = [zeros(size(res_0_mat)[1])..., ones(size(res_1_mat)[1])..., 2 .+ zeros(size(res_2_mat)[1])..., 3 .+ zeros(size(res_3_mat)[1])..., rand_labels...]

    else 
        res_tot = transpose(vcat(res_0_mat, res_1_mat, res_2_mat, res_3_mat))
        res_labels = [zeros(size(res_0_mat)[1])..., ones(size(res_1_mat)[1])..., 2 .+ zeros(size(res_2_mat)[1])..., 3 .+ zeros(size(res_3_mat)[1])...]
    end

    res_tot = rescale(res_tot)

    if method == "LDA"
        # Do LDA
        lda = fit(MulticlassLDA, res_tot, res_labels; outdim=2)
        preds = predict(lda, res_tot)
    elseif method == "PCA"
        # Do PCA
        m_pca = fit(PCA, res_tot; maxoutdim=2)
        preds = predict(m_pca, res_tot)
    elseif method == "tSNE"
        # Do tSNE
        preds = transpose(rescale(tsne(transpose(res_tot), 2)))
    else
        @error "Not implemented method $(method)"
    end

    if add_noise
        preds = preds + 0.025 .* randn(size(preds))  
    end

    s = plot(
        xlabel="Response Vector Projection Dim 1 [a.u.]",
        ylabel="Response Vector Projection Dim 2 [a.u.]",
        # xlims=[-0.82, 0.82],
        title=plot_title,
        legend=:topleft,
    )

    for (lab_i, lab) in enumerate(unique(res_labels))
        data = transpose(preds[:, res_labels .== lab])
        data_mean = calculate_mean(data)
        covariance = calculate_covariance(data, data_mean)

        if lab != -1.
            plot_confidence_ellipse(data_mean, covariance, s, cols[lab]; primary=false)
            scatter!(
                preds[1, :][res_labels .== lab],
                preds[2, :][res_labels .== lab],
                # collect(1:length(options)),
                # xerror=errors,
                label=plot_labs[lab_i] * " Data (95% Conf.)",
                color=cols[lab],
                # markershapes=:xcross,
                # markersize=4,
                dpi=300,
                # alpha=0.7
            )
        else
            plot_confidence_ellipse(data_mean, covariance, s, cols[lab]; label="Random Data (95% Conf.)")
        end

    end

    return s
end

create_pca_plot(df_real_aug24, df_gpt35_dialog3_fix, df_gpt4_dialog3_fix, df_gpt4o_dialog3_fix, "", ["Human", "GPT-3.5", "GPT-4", "GPT-4o"]; move=0, add_random=true)
savefig("lda_bothmoves_all5_Aug24.png")
savefig("lda_bothmoves_all5_Aug24.pdf")

# create_pca_plot(df_gpt35_dialog3_fix, df_real_feb24, df_gpt4_dialog3_fix, "", ["GPT-3.5", "GPT-4", "Human"]; move=0, method=meth, add_random=true)
# # savefig(meth * "_bothmoves_gpt3_v_human_v_gpt4_v_random_Feb24_fixed.png")
# # savefig(meth * "_bothmoves_gpt3_v_human_v_gpt4_v_random_Feb24_fixed.pdf")
# create_pca_plot(df_gpt35_dialog3_fix, df_real_aug24, df_gpt4_dialog3_fix, "", ["GPT-3.5", "GPT-4", "Human"]; move=0, method=meth, add_random=true)
# # savefig(meth * "_bothmoves_gpt3_v_human_v_gpt4_v_random_Aug24_fixed.png")
# # savefig(meth * "_bothmoves_gpt3_v_human_v_gpt4_v_random_Aug24_fixed.pdf")
