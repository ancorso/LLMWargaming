
using Random
using Plots
using CSV
using DataFrames
using Statistics
using TSne
using MultivariateStats
using LinearAlgebra
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
function plot_confidence_ellipse(mean, covariance, ax, col; label="95% Conf.")
    eigenvalues, eigenvectors = eigen(covariance)
    angle = atan(eigenvectors[2,1], eigenvectors[1,1])
    chisquare_val = 2.4477  # 95% confidence interval
    theta = LinRange(0, 2 * π, 100)

    a = sqrt(eigenvalues[1]) * chisquare_val
    b = sqrt(eigenvalues[2]) * chisquare_val

    ellipse_x = mean[1] .+ a * cos.(theta) * cos(angle) - b * sin.(theta) * sin(angle)
    ellipse_y = mean[2] .+ a * cos.(theta) * sin(angle) + b * sin.(theta) * cos(angle)

    plot!(ax, ellipse_x, ellipse_y, label=label, seriestype=:shape, color=col, fill=(0.2, col), alpha=0.2)
end

function create_pca_plot(df_0, df_1, tit, plot_labs; move=1, method="LDA", add_noise=true, add_random=false)

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
        res_labels = [ones(size(res_0_mat)[1])..., zeros(size(res_1_mat)[1])..., rand_labels...]

    else 
        res_tot = transpose(vcat(res_0_mat, res_1_mat))
        res_labels = [ones(size(res_0_mat)[1])..., zeros(size(res_1_mat)[1])...]
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
        preds = preds + 0.02 .* randn(size(preds))  
    end

    s = plot(
        # yticks=(1:length(opts), opts),
        # xrot=60,
        # bottom_margin=15mm,
        # left_margin=12mm,
        xlabel="$(method) dim 1 [ ]",
        ylabel="$(method) dim 2 [ ]",
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
            plot_confidence_ellipse(data_mean, covariance, s, lab_i)
            scatter!(
                preds[1, :][res_labels .== lab],
                preds[2, :][res_labels .== lab],
                # collect(1:length(options)),
                # xerror=errors,
                label=plot_labs[lab_i] * " Data",
                color=lab_i,
                dpi=300,
            )
        else
            plot_confidence_ellipse(data_mean, covariance, s, lab_i; label="95% Conf. Random Data")
        end

    end

    return s
end

meth = "LDA"
create_pca_plot(df_dialogno, df_real, "no Dialog/Human", ["no Dialog", "Human"]; move=0, method=meth)
create_pca_plot(df_dialog3, df_real, "Dialog3/Human", ["Dialog3", "Human"]; move=0, method=meth)
# savefig(meth * "_bothmoves_dialog3_v_human.png")
create_pca_plot(df_dialog3, df_real, "Dialog3/Human", ["Dialog3", "Human"]; move=0, method=meth, add_random=true)
# savefig(meth * "_bothmoves_dialog3_v_human_v_random.png")
create_pca_plot(df_dialog6, df_real, "Dialog6/Human", ["Dialog6", "Human"]; move=0, method=meth)

create_pca_plot(df_dialogno, df_real, "no Dialog/Human", ["no Dialog", "Human"]; move=1, method=meth)
create_pca_plot(df_dialog3, df_real, "Dialog3/Human", ["Dialog3", "Human"]; move=1, method=meth)
# savefig(meth * "_move1_dialog3_v_human.png")
create_pca_plot(df_dialog3, df_real, "Dialog3/Human", ["Dialog3", "Human"]; move=1, method=meth, add_random=true)
# savefig(meth * "_move1_dialog3_v_human_v_random.png")
create_pca_plot(df_dialog6, df_real, "Dialog6/Human", ["Dialog6", "Human"]; move=1, method=meth)

create_pca_plot(df_dialogno, df_real, "no Dialog/Human", ["no Dialog", "Human"]; move=2, method=meth)
create_pca_plot(df_dialog3, df_real, "Dialog3/Human", ["Dialog3", "Human"]; move=2, method=meth)
# savefig(meth * "_move2_dialog3_v_human.png")
create_pca_plot(df_dialog3, df_real, "Dialog3/Human", ["Dialog3", "Human"]; move=2, method=meth, add_random=true)
# savefig(meth * "_move2_dialog3_v_human_v_random.png")
create_pca_plot(df_dialog6, df_real, "Dialog6/Human", ["Dialog6", "Human"]; move=2, method=meth)

