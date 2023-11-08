include("src/game.jl")
using DataFrames
using CSV

# Prepare the connection to the GPT model
secret_key = "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"
model = "gpt-3.5-turbo-16k"
chat_setup = ChatSetup(secret_key, model)

# Setup the results dataframe and game dir
wargame_dir = "wargame/"
res = results_df(USPRCCrisisSimulation)

# Generate Treatments
treatments = []
for AI_accuracy in ["70-85%", "95-99%"]
    for AI_system_training in [:basic, :significant]
        for china_status in [:revisionist, :status_quo]
            push!(treatments, USPRCCrisisSimulation(wargame_dir, AI_accuracy, AI_system_training, china_status))
        end
    end
end

# Generate one team (for now)
teams = [[Player() for i in 1:6] for i=1:10]

# # Print a sample:
# print_prompts(treatments[1], team)

# # Run a sample game
# result = run_game(treatments[1], team, chat_setup, verbose=true)

# Run the games
for (j,team) in enumerate(teams)
    println("Team: ", j)
    for (i, game) in enumerate(treatments)
        println("Running treatment: ", i)
        try 
            results = run_game(game, team, chat_setup)
            push!(res, results)
            CSV.write("results/v1_results.csv", res)
        catch e
            println(e)
            sleep(10)
        end
    end
end

# Write the results DataFrame to a CSV file
CSV.write("results/sample_results.csv", res)


using StatsPlots
using Plots.Measures
df = CSV.read("results/sample_results.csv", DataFrame)


function compare_treatments_move1(df, column_name, treatments)
    short_options = move_1_2_options_shortdesc()
    options = move_1_2_options_desc()

    group = repeat(treatments, inner=length(options))

    y = [sum(df[df[!, column_name] .== t, o]) for t in treatments for o in options]
    names = repeat(short_options, outer=2)
    groupedbar(names, y, group=group, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 1")
end

function compare_treatments_move2(df, column_name, treatments)
    options = move_2_2_options_desc()

    group = repeat(treatments, inner=length(options))

    y = [sum(df[df[!, column_name] .== t, o]) for t in treatments for o in options]
    names = repeat(options, outer=2)
    groupedbar(names, y, group=group, xrot=60, bottom_margin=15mm, ylabel="Counts", title="Move 2")
end


AI_column_name = "AI Accuracy"
AI_Accuracies = ["70-85%", "95-99%"]

china_column_name = "China Status"
china_treatments = ["revisionist", "status_quo"]

compare_treatments_move1(df, AI_column_name, AI_Accuracies)
compare_treatments_move2(df, china_column_name, china_treatments)