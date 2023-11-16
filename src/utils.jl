using OpenAI
using Dates

struct ChatSetup
    secret_key::String
    model::String
end

function chat!(setup, chat_hist, prompt)
    push!(chat_hist, Dict("role" => "user", "content"=> prompt))
    r = create_chat(
        setup.secret_key, 
        setup.model,
        chat_hist
      )
    message = r.response[:choices][1][:message]
    push!(chat_hist, Dict("role" => message[:role], "content"=> message[:content]))
end

function readfile(dir, filename)
    s = ""
    open(joinpath(dir, filename)) do f
        s = read(f, String)
    end
    return s
end

function create_file_ending(dir)
    tod = string(today())
    file_ind = length(filter(e->occursin(tod, e), readdir(dir)))

    return tod * "_" * string(file_ind)
end

