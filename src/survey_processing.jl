
include("utils.jl") 
include("players.jl")

using CSV
using DataFrames
using Serialization


# Import player survey data
survey = CSV.read("wargame/Survey_October24.csv", DataFrame) 
c_names = names(survey)

# Helper function
function check_attribute(input, options, label="att")
    attribute = ""
    for att in options
        # This links the description in players.jl and survey answers for age
        if label == "age"
            att = att * " years old"
        end
        if input == att
            attribute = att
        end
    end
    if attribute == ""
        @warn "Unknown $(label) for row $row: $(input)"
    end

    return attribute
end

function create_player_data()

    # List of player data to fill
    player_data = []
    # Iterate over each player's information
    for row in range(3, size(survey)[1])

        # Safety check for "Finished" category
        if survey[row, "Finished"] != "TRUE"
            println("Unfinished $(row): ", survey[row, "Finished"])
            continue
        end

        # Do player and group id [2, 3] ?

        # Get age
        age_range = check_attribute(survey[row, "Q2"], age_range_options(), "age")

        # Get gender
        gender = check_attribute(survey[row, "Q4"], gender_options(), "gender")

        # Get experience
        experience = check_attribute(survey[row, "Q24"], experience_options(), "XP")

        # Get AI familiarity
        AI_familiarity = check_attribute(survey[row, "Q15"], AI_familiarity_options(), "aifam")

        # Get China familiarity
        China_military_familiarity = check_attribute(survey[row, "Q16"], China_military_familiarity_options(), "chinafam")

        # Get US military familiarity
        US_military_familiarity = check_attribute(survey[row, "Q17"], US_military_familiarity_options(), "usfam")

        # Get professional background (one-hot encoding)
        given_back = split(survey[row, "Q23"], ",")
        affil = 0
        if "Government" in given_back
            government_affiliation = true
            affil += 1
        else
            government_affiliation = false
        end
        if "Academic" in given_back
            academic_affiliation = true
            affil += 1
        else
            academic_affiliation = false
        end
        if "Military" in given_back
            military_affiliation = true
            affil += 1
        else
            military_affiliation = false
        end
        if "Private Industry" in given_back
            private_industry_affiliation = true
            affil += 1
        else
            private_industry_affiliation = false
        end
        if "Non-Governmental Organization" in given_back
            non_governmental_organization_affiliation = true
            affil += 1
        else
            non_governmental_organization_affiliation = false
        end
        other_affiliation = ""
        if "Other" in given_back && affil <= 0
            other_affiliation = "Other"
        elseif affil <= 0
            println("UNKNOWN BACK $(row): ", survey[row, "Q23"])
            continue
        end

        new_player = Player(
            age_range=age_range,
            gender=gender,
            government_affiliation=government_affiliation,
            academic_affiliation=academic_affiliation,
            military_affiliation=military_affiliation,
            private_industry_affiliation=private_industry_affiliation,
            non_governmental_organization_affiliation=non_governmental_organization_affiliation,
            other_affiliation=other_affiliation,
            experience=experience,
            AI_familiarity=AI_familiarity,
            China_military_familiarity=China_military_familiarity,
            US_military_familiarity=US_military_familiarity
        )
        # println(new_player)
        append!(player_data, [new_player])

    end

    return player_data
end

player_data = create_player_data()

# serialize("wargame/player_data.jls", player_data)
# loaded_player_data = deserialize("wargame/player_data.jls")
# rand(loaded_player_data)
