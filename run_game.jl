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
    verbose::Bool = false
    save_results_to_csv::Bool = true
    run_test_game::Bool = false
    n_teams::Int = 10
    n_players::Int = 6
    n_dialog_steps::Int = 3
end


function results_df(cnf::SimulationConfig)
    df = DataFrame(
        "directory" => String[], 
        "AI Accuracy" => String[],
        "AI System Training" => String[],
        "China Status" => String[],
        "N Dialog Steps" => String[],
        "Player 1" => String[],
        "Player 2" => String[],
        "Player 3" => String[],
        "Player 4" => String[],
        "Player 5" => String[],
        "Player 6" => String[],
        ["Dialogue 1-$(i)" => String[] for i in 1:cnf.n_dialog_steps]...,
        "Move 1 Question 1" => String[],
        "Move 1 Question 2" => String[],
        [o => String[] for o in move_1_2_options_desc()]...,
        "Move 1 to Move 2 Transition Response" => String[],
        ["Dialogue 2-$(i)" => String[] for i in 1:cnf.n_dialog_steps]...,
        "Move 2 Question 1" => String[],
        "Move 2 Question 2" => String[],
        [o => String[] for o in move_2_2_options_desc()]...,
        "Move 2 Question 3" => String[]
    )
    return df
end


function run_simulation(config::SimulationConfig)

    # Prepare the connection to the GPT model
    if config.secret_key == ""
        @warn "OPENAI_API_KEY not set in ENV"
    end
    chat_setup = ChatSetup(config.secret_key, config.model)

    # Setup the results dataframe and game dir
    res = results_df(config)

    # Generate Treatments
    treatments = []
    for AI_accuracy in ["70-85%", "95-99%"]
        for AI_system_training in [:basic, :significant]
            for china_status in [:revisionist, :status_quo]
                push!(treatments, USPRCCrisisSimulation(config.wargame_dir, AI_accuracy, AI_system_training, china_status, config.n_dialog_steps))
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
run_simulation(SimulationConfig(run_test_game=true, verbose=true))
