include("src/game.jl")

game = USPRCCrisisSimulation()
team = [Player() for i in 1:6]

# Flow of the game
println(game_setup_prompt(game, team))

println(move_1_1_prompt(game))

println(move_1_2_prompt(game))

# TODO This prompt isn't working very well, it starts to just answer the question
println(global_response(game))

println(move_2_1_prompt(game))

println(move_2_2_prompt(game))

println(move_2_3_prompt(game))
