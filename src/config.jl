
# Fix parameters globally
SEED = 42

# Treatment variables
ai_column_name = "AI Accuracy"
ai_accuracies = ["70-85%", "95-99%"]
train_column_name = "AI System Training"
train_quality = ["basic", "significant"]
china_column_name = "China Status"
china_treatments = ["revisionist", "status_quo"]

# Fixed colors for plotting
# Random, Human, GPT3.5, GPT4, GPT4o
# cols = Dict(-1 => "#35393C", 0 => "#FF9505", 1 => "#69995D", 2 => "#EC4E20", 3 => "#016FB9")
cols = Dict(-1 => "#35393C", 0 => "#E69F00", 1 => "#69995D", 2 => "#CC79A7", 3 => "#0072B2")
# cols = {
#     "EI": "#0072B2",
#     "PG": "#E69F00",
#     "PPONorm (Markovian Training)": "#CC79A7", # "#D55E00",
# }


# Run imports for fixed names
# Human data (in chronological order)
df_real = CSV.read("data/ganz_data_full.csv", DataFrame)
df_real_feb24 = CSV.read("data/ganz_data_full_updateFeb24.csv", DataFrame)
df_real_aug24 = CSV.read("data/ganz_data_full_updateAug24.csv", DataFrame)

# LLM Data
df_gpt4_dialogno_fix = CSV.read("results/post_fix/data_gpt4_dialog0.csv", DataFrame)
df_gpt4_dialog1_fix = CSV.read("results/post_fix/data_gpt4_dialog1.csv", DataFrame)
df_gpt4_dialog2_fix = CSV.read("results/post_fix/data_gpt4_dialog2.csv", DataFrame)
df_gpt4_dialog3_fix = CSV.read("results/post_fix/data_gpt4_dialog3.csv", DataFrame)
df_gpt4_dialog4_fix = CSV.read("results/post_fix/data_gpt4_dialog4.csv", DataFrame)
df_gpt4_dialog5_fix = CSV.read("results/post_fix/data_gpt4_dialog5.csv", DataFrame)
df_gpt4_dialog6_fix = CSV.read("results/post_fix/data_gpt4_dialog6.csv", DataFrame)

df_gpt35_dialogno_fix = CSV.read("results/post_fix/data_gpt35_dialog0.csv", DataFrame)
df_gpt35_dialog1_fix = CSV.read("results/post_fix/data_gpt35_dialog1.csv", DataFrame)
df_gpt35_dialog2_fix = CSV.read("results/post_fix/data_gpt35_dialog2.csv", DataFrame)
df_gpt35_dialog3_fix = CSV.read("results/post_fix/data_gpt35_dialog3.csv", DataFrame)
df_gpt35_dialog4_fix = CSV.read("results/post_fix/data_gpt35_dialog4.csv", DataFrame)
df_gpt35_dialog5_fix = CSV.read("results/post_fix/data_gpt35_dialog5.csv", DataFrame)
df_gpt35_dialog6_fix = CSV.read("results/post_fix/data_gpt35_dialog6.csv", DataFrame)

df_gpt4o_dialogno_fix = CSV.read("results/gpt4o/data_gpt4o_dialog0.csv", DataFrame)
df_gpt4o_dialog1_fix = CSV.read("results/gpt4o/data_gpt4o_dialog1.csv", DataFrame)
df_gpt4o_dialog2_fix = CSV.read("results/gpt4o/data_gpt4o_dialog2.csv", DataFrame)
df_gpt4o_dialog3_fix = CSV.read("results/gpt4o/data_gpt4o_dialog3.csv", DataFrame)
df_gpt4o_dialog4_fix = CSV.read("results/gpt4o/data_gpt4o_dialog4.csv", DataFrame)
df_gpt4o_dialog5_fix = CSV.read("results/gpt4o/data_gpt4o_dialog5.csv", DataFrame)
df_gpt4o_dialog6_fix = CSV.read("results/gpt4o/data_gpt4o_dialog6.csv", DataFrame)
df_gpt4o_dialog3_noinstr = CSV.read("results/gpt4o/data_gpt4o_dialog3_noinstruct.csv", DataFrame)

# LLM Data Without Presidential Instructions (Old Naming Convention)


# Sensitivity Studies
df_nochief = CSV.read("results/sensitivity_studies/data_nochiefs.csv", DataFrame)
df_playeruniform = CSV.read("results/sensitivity_studies/data_playeruniform.csv", DataFrame)
df_pacifism = CSV.read("results/sensitivity_studies_new/data_pacifism.csv", DataFrame)
df_sociopath = CSV.read("results/sensitivity_studies_new/data_sociopath.csv", DataFrame)
df_more_disagreement = CSV.read("results/sensitivity_studies_new/data_moredisagree.csv", DataFrame)
df_gpt4_pacifism = CSV.read("results/gpt4_turbo/data_pacifism.csv", DataFrame)
df_gpt4_sociopath = CSV.read("results/gpt4_turbo/data_sociopaths.csv", DataFrame)