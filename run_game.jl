include("src/game.jl")
include("src/utils.jl")
using DataFrames
using CSV
using Serialization
using ProgressBars

# TODO keyword passing between config structs
# TODO automated config attributes into results data frame for logging
# TODO add debug mode to return empty strings instead of use GPT
# TODO set and variate text generation length?
# TODO variate roleplying as chiefs (focus on human backgrounds)?
# TODO create no_dialog option (direct recommendation)
# TODO add safety check of answers (did not catch "b, c, e; probably should check that at least one answer is made)
# TODO create fixed test data set functions (work around ablation parameter changes!)

# Struct for simulation config parameters
@with_kw struct SimulationConfig
    model::String = "gpt-3.5-turbo-16k"
    secret_key::String = get(ENV, "OPENAI_API_KEY", "")
    wargame_dir::String = "wargame/"
    output_dir::String = "results/"
    out_csv_file::String = ""
    use_dummygpt::Bool = false
    no_dialog::Bool = false # placeholder
    no_chiefs::Bool = false # placeholder
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
    chat_setup = ChatSetup(config.secret_key, config.model, config.use_dummygpt)

    # Setup the results dataframe and game dir
    res = results_df(config)

    # Setup outpuf file names
    if config.out_csv_file == ""
        ending = create_file_ending(config.output_dir)
        data_filename = config.output_dir * "data" * ending * ".csv"
    else
        data_filename = config.output_dir * config.out_csv_file
    end 

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
            CSV.write(data_filename, res)
        end
    else
        # Run the games
        for (j, team) in ProgressBar(enumerate(teams))
            println("Team: ", j)
            for (i, game) in ProgressBar(enumerate(treatments))
                println("Running treatment: ", i)
                try 
                    result = run_game(game, team, chat_setup)
                    push!(res, result)

                    # Write the results DataFrame to a CSV file
                    if config.save_results_to_csv
                        CSV.write(data_filename, res)
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
run_simulation(SimulationConfig(run_test_game=true, verbose=true, use_dummygpt=true))
