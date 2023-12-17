
include("../src/utils.jl") 
include("../src/players.jl")
using CSV
using DataFrames
using Serialization

# Import player survey data
data = CSV.read("results/sensitivity_studies/data_dialog3.csv", DataFrame) 
c_names = names(data)
n_dialog = data[1, c_names[15]]

# find all columns with Dialog
# oppose, against, however, disapprove, reject, refuse, dispute, counter, contradict
words = ["No,", "disagree", "admit", "Admit", "wrong", "incorrect", "oppose", "disapprove", "reject", "refuse", "dispute", "contradict"]
for row in range(1, size(data)[1])
    if data[row, c_names[31]] == false
        continue
    end
    buffer = "\n\t--$(row)--\n"
    for i in range(1, 3)
        buffer *= "\n\t++$(i)++\n"
        buffer *= data[row, c_names[24 + i]]
    end
    flag = false
    for word in words
        if occursin(word, buffer)
            println(word)
            flag = true
        end
    end
    if flag
        println(buffer)
        println("\t---\n\n")
    end
end


