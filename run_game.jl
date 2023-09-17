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
team = [Player() for i in 1:6]

# Run the games
for (i, game) in enumerate(treatments)
    println("Running treatment: ", i)
    results = run_game(game, team, chat_setup)
    push!(res, results)
end

# Write the results DataFrame to a CSV file
CSV.write("results/sample_results.csv", res)
