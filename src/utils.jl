# using OpenAI
# secret_key = "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"
# model = "gpt-3.5-turbo-16k"
# # rate_limit = Chat	200 RPD	3 RPM	40,000 TPM

# prompt =  "Say \"this is a twest\""

# r = create_chat(
#     secret_key, 
#     model,
#     [Dict("role" => "user", "content"=> prompt)]
#   )
# println(r.response[:choices][begin][:message][:content])

function readfile(dir, filename)
    s = ""
    open(joinpath(dir, filename)) do f
        s = read(f, String)
    end
    return s
end

