include("src/game.jl")
using DataFrames
using CSV
using Serialization


# Struct for simulation config parameters
@with_kw struct SimulationConfig
    model::String = "gpt-3.5-turbo-16k"
    secret_key::String = get(ENV, "OPENAI_API_KEY", "")
    wargame_dir::String = "wargame/"
    out_csv_file::String = "sample_results.csv"
    boostrap_players::Bool = true
    do_dialog::Bool = true # placeholder
    verbose::Bool = false
    save_results_to_csv::Bool = true
    run_test_game::Bool = false
    n_teams::Int = 10
    n_players::Int = 6
    n_dialog_steps::Int = 3 # placeholder
end


function run_simulation(config::SimulationConfig)

    # Prepare the connection to the GPT model
    if config.secret_key == ""
        @warn "OPENAI_API_KEY not set in ENV"
    end
    chat_setup = ChatSetup(config.secret_key, config.model)

    # Setup the results dataframe and game dir
    res = results_df(USPRCCrisisSimulation)

    # Generate Treatments
    treatments = []
    for AI_accuracy in ["70-85%", "95-99%"]
        for AI_system_training in [:basic, :significant]
            for china_status in [:revisionist, :status_quo]
                push!(treatments, USPRCCrisisSimulation(config.wargame_dir, AI_accuracy, AI_system_training, china_status))
            end
        end
    end

    # Generate teams
    if config.boostrap_players
        loaded_player_data = deserialize(config.wargame_dir * "player_data.jls")
        teams = [[rand(loaded_player_data) for i in 1:config.n_players] for i=1:config.n_teams]
    else
        teams = [[Player() for i in 1:config.n_players] for i=1:config.n_teams]
    end

    # Run a test game or all treatments * teams
    if config.run_test_game
        result = run_game(treatments[1], teams[1], chat_setup, verbose=config.verbose)
        push!(res, result)

        # Write the results DataFrame to a CSV file
        if config.save_results_to_csv
            CSV.write("results/" * config.out_csv_file, res)
        end
    else
        # Run the games
        for (j, team) in enumerate(teams)
            println("Team: ", j)
            for (i, game) in enumerate(treatments)
                println("Running treatment: ", i)
                try 
                    result = run_game(game, team, chat_setup)
                    push!(res, result)

                    # Write the results DataFrame to a CSV file
                    if config.save_results_to_csv
                        CSV.write("results/" * config.out_csv_file, res)
                    end
                catch e
                    println(e)
                    sleep(10)
                end
            end
        end
    end

    return res
end

# run_simulation(SimulationConfig())
run_simulation(SimulationConfig(run_test_game=true))
