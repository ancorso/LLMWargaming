include("src/game.jl")
using DataFrames
using CSV
using Serialization

# Prepare the connection to the GPT model
secret_key = "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"
model = "gpt-3.5-turbo-16k"
chat_setup = ChatSetup(secret_key, model)

# Setup the results dataframe and game dir
wargame_dir = "wargame/"
res = results_df(USPRCCrisisSimulation)
boostrap_players = true

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
if boostrap_players
    loaded_player_data = deserialize("wargame/player_data.jls")
    teams = [[rand(loaded_player_data) for i in 1:6] for i=1:10]
else
    teams = [[Player() for i in 1:6] for i=1:10]
end

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
