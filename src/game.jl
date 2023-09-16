using Parameters
include("utils.jl")
include("players.jl")

@with_kw struct USPRCCrisisSimulation
    dir = "wargame"
    AI_accuracy_range = "95-99%" # "70-85%" or 
    AI_system_training = :basic # :basic or :significant
    china_status = :revisionist # :revisionist or :status_quo
end

function AI_accuracy_prompt(game::USPRCCrisisSimulation)
    s = readfile(game.dir, "AI_accuracy.txt")
    s = replace(s, "AI_ACCURACY_RANGE" => game.AI_accuracy_range)
    if game.AI_system_training == :basic
        s = s * readfile(game.dir, "system_training_basic.txt")
    elseif game.AI_system_training == :significant
        s = s * readfile(game.dir, "system_training_significant.txt")
    else
        error("Invalid AI system training option")
    end
    return s
end

function game_setup_prompt(game::USPRCCrisisSimulation, team::Vector{Player})
    s = readfile(game.dir, "context.txt") * "\n\n"
    s = s * team_description(team) * "\n\n"
    s = s * readfile(game.dir, "scenario.txt") * "\n\n"
    s = s * readfile(game.dir, "incident.txt") * "\n\n"
    s = s * readfile(game.dir, "roles.txt") * "\n\n"
    s = s * readfile(game.dir, "available_forces.txt") * "\n\n"
    s = s * AI_accuracy_prompt(game) * "\n\n"
    s = s * readfile(game.dir, "move1_option_summary.txt") * "\n\n"
    return s
end

function pose_question(question)
    s = "Now answer the following question from the perspective of the team. Only respond to the question do not simulate any more dialogue.\n\n"
    return s * question * "\n\n"
end

function move_1_1_prompt(game::USPRCCrisisSimulation)
    return pose_question(readfile(game.dir, "move1-1.txt"))
end

function move_1_2_prompt(game::USPRCCrisisSimulation)
    return pose_question(readfile(game.dir, "move1-2.txt"))
end

function global_response(game::USPRCCrisisSimulation)
    s = readfile(game.dir, "global_response_move2.txt") * "\n\n"
    if game.china_status == :revisionist
        s = s * readfile(game.dir, "revisionist_china.txt") * "\n\n"
    elseif game.china_status == :status_quo
        s = s * readfile(game.dir, "status_quo_china.txt") * "\n\n"
    else
        error("Invalid China status option")
    end
    s = s * readfile(game.dir, "move2_option_summary.txt") * "\n\n"
    return s
end

function move_2_1_prompt(game::USPRCCrisisSimulation)
    return pose_question(readfile(game.dir, "move2-1.txt"))
end

function move_2_2_prompt(game::USPRCCrisisSimulation)
    return pose_question(readfile(game.dir, "move2-2.txt"))
end

function move_2_3_prompt(game::USPRCCrisisSimulation)
    return pose_question(readfile(game.dir, "move2-3.txt"))
end

