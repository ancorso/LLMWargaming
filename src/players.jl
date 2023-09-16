using Parameters
using Distributions

function age_range_options()
    return ["under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
end

function gender_options()
    return ["Male", "Female", "Non-binary / third gender", "Prefer not to say"]
end

function experience_options()
    return ["Student", "Entry level professional (up to 5 years experince)", "Mid-level professional (5-15 years experience)", "Senior professional (15+ years experience)"]
end

function AI_familiarity_options()
    return ["No familiarity", "Routine understanding of concepts behind AI", "AI policy expert", "AI technical expert", "AI policy and technical expert"]
end

function China_military_familiarity_options()
    return ["No familiarity", "Routine understanding", "Policy expert", "Technical expert", "Policy and technical expert"]
end

function US_military_familiarity_options()
    return ["No familiarity", "Routine understanding", "Policy expert", "Technical expert", "Policy and technical expert"]
end

@with_kw struct Player
    description = ""
    age_range = rand(age_range_options())
    gender = rand(gender_options())
    country_of_citizenship = "United States"
    government_affiliation = rand(Bernoulli(0.2))
    academic_affiliation = rand(Bernoulli(0.2))
    military_affiliation = rand(Bernoulli(0.2))
    private_industry_affiliation = rand(Bernoulli(0.2))
    non_governmental_organization_affiliation = rand(Bernoulli(0.2))
    other_affiliation = !any([government_affiliation, academic_affiliation, military_affiliation, private_industry_affiliation, non_governmental_organization_affiliation]) ? "Other" : ""
    experience = rand(experience_options())
    AI_familiarity = rand(AI_familiarity_options())
    China_military_familiarity = rand(China_military_familiarity_options())
    US_military_familiarity = rand(US_military_familiarity_options())
end

function player_description(p::Player)
    s = ""
    if p.description != ""
        s = s * "Short Description: " * p.description * "\n"
    end
    s = s * "Age Range: " * p.age_range * "\n"
    s = s * "Gender: " * p.gender * "\n"
    s = s * "Country of Citizenship: " * p.country_of_citizenship * "\n"
    if p.government_affiliation
        s = s * "Government Affiliation: Yes\n"
    end
    if p.academic_affiliation
        s = s * "Academic Affiliation: Yes\n"
    end
    if p.military_affiliation
        s = s * "Military Affiliation: Yes\n"
    end
    if p.private_industry_affiliation
        s = s * "Private Industry Affiliation: Yes\n"
    end
    if p.non_governmental_organization_affiliation
        s = s * "Non-Governmental Organization Affiliation: Yes\n"
    end
    if p.other_affiliation != ""
        s = s * "Other Affiliation: " * p.other_affiliation * "\n"
    end
    s = s * "Professional Experience: " * p.experience * "\n"
    s = s * "AI Familiarity: " * p.AI_familiarity * "\n"
    s = s * "China Military Familiarity: " * p.China_military_familiarity * "\n"
    s = s * "US Military Familiarity: " * p.US_military_familiarity * "\n"
    return s
end

function team_description(team::Vector{Player})
    s = "The team consists of the $(length(team)) players. Each player answered an online questionaire with the following information:\n\n"
    for (i,p) in enumerate(team)
        s = s * "Player $i: \n"
        s = s * player_description(p) * "\n"
    end
    return s
end