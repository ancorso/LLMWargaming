
using Parameters


# Struct for simulation config parameters
@with_kw struct SimulationConfig
    model::String = "gpt-3.5-turbo-16k"
    secret_key::String = get(ENV, "OPENAI_API_KEY", "")
    wargame_dir::String = "wargame/"
    output_dir::String = "results/"
    out_csv_file::String = ""
    use_dummygpt::Bool = false
    use_bench_players::Bool = false
    no_dialog::Bool = false 
    no_chiefs::Bool = false 
    boostrap_players::Bool = true
    pacificsm::Bool = false
    sociopaths::Bool = false
    verbose::Bool = false
    save_results_to_csv::Bool = true
    run_test_game::Bool = false
    n_teams::Int = 10
    n_players::Int = 6
    n_dialog_steps::Int = 3
end

function init_sim_conf(; kwargs...)
    # Create a default instance
    default_conf = SimulationConfig()

    # Convert the default instance to a NamedTuple
    default_conf_nt = (; (name => getfield(default_conf, name) for name in fieldnames(SimulationConfig))...)

    # Merge the default values with the provided keyword arguments
    merged_kwargs = merge(default_conf_nt, kwargs)

    # Create a new SimulationConfig instance with the merged keyword arguments
    return SimulationConfig(merged_kwargs...)
end

# remove secret_key from logging and output data
function get_pars4store(fil_fields=[:secret_key])
    fields = fieldnames(SimulationConfig)
    store_fields = [f in fil_fields ? nothing : f for f in fields]
    filter!(x -> x !== nothing, store_fields)
    store_names = ["$f" for f in store_fields]

    @assert length(store_fields) == length(store_names) "Length mismatch for stored fields $(length(store_fields)), $(length(store_names))"

    return store_fields, store_names
end

function results_df(cnf::SimulationConfig)
    df = DataFrame(
        [o => String[] for o in get_pars4store()[2]]...,
        "AI Accuracy" => String[],
        "AI System Training" => String[],
        "China Status" => String[],
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
