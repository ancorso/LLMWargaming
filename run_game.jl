
include("src/game.jl")
include("src/simulation.jl")
include("src/utils.jl")
using DataFrames
using CSV
using Serialization
using ProgressBars


# TODO set and variate text generation length?
# TODO variate roleplying as chiefs (focus on human backgrounds)?
# TODO create no_dialog option (direct recommendation)
# TODO add safety check of answers (did not catch "b, c, e"; probably should check that at least one answer is made)

# Everything not set will result in usage of default values 
conf = init_sim_conf(
    # model="gpt-3.5-turbo-16k",
    # secret_key=get(ENV, "OPENAI_API_KEY", ""),
    # wargame_dir="wargame/",
    # output_dir="results/",
    # out_csv_file="",
    use_dummygpt=true,
    use_bench_players=true,
    no_dialog=false, # placeholder
    no_chiefs=false, # placeholder
    boostrap_players=true,
    verbose=true,
    # save_results_to_csv=true,
    run_test_game=true,
    n_teams=10,
    n_players=6,
    n_dialog_steps=2,
)

function run_simulation(config::SimulationConfig)

    # Prepare the connection to the GPT model
    # TODO? turn into assert and combine with && !config.use_dummygpt
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

    if conf.use_bench_players
        test_data = deserialize("wargame/test_data.jls")
        treatments = test_data[1]
        teams = test_data[2]
    else
        # Generate Treatments
        treatments = gen_all_treatments(config)

        # Generate teams
        teams = gen_teams(config)
    end

    # Run a test game or all treatments * teams
    if config.run_test_game
        result = run_game(config, treatments[1], teams[1], chat_setup)
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
                    result = run_game(config, game, team, chat_setup)
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
# run_simulation(SimulationConfig(run_test_game=true, verbose=true, use_dummygpt=true))
run_simulation(conf)
