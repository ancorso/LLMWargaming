
# WIP, not validated
# Requires hbm_sample_data.pkl (gitignored, ~400 MB).
# Regenerate with: python analysis/ana_demographic_impact.py  (from repo root)
import pickle
import ana_demographic_impact
import numpy as np


def main():
    # Import cached data
    with open('hbm_sample_data.pkl', 'rb') as outp:
        imported_data = pickle.load(outp)

    n_trends_found = 0

    for cat, test_label in [
        ("Gender", "Gender"),
        ("DeathPenaltySupport", "DeathPenaltySupport0"),
        ("DeathPenaltySupport", "DeathPenaltySupport1"),
        ("Background", "Background0"),
        ("Background", "Background1"),
        ("Background", "Background2"),
        ("Age", "Age0"),
        ("Age", "Age1"),
        ("Age", "Age2"),
        ("Professionality", "Professionality"),
        ("AIExp", "AIExp"),
        ("ChinaMilExp", "ChinaMilExp0"),
        ("ChinaMilExp", "ChinaMilExp1"),
        ("ChinaMilExp", "ChinaMilExp2"),
        ("USMilExp", "USMilExp0"),
        ("USMilExp", "USMilExp1"),
        ("USMilExp", "USMilExp2"),
    ]:
        n_trends_found_local = 0
        print(f"\n------- {test_label} ------- \n")

        for move in [1, 2]:
            print(f"--- MOVE {move} --- \n")
            if move == 1:
                action_list = ana_demographic_impact.move_1_2_options_desc()
            else:
                action_list = ana_demographic_impact.move_2_2_options_desc()
            action_inds = list(range(len(action_list)))
            for action_ind in action_inds:
                action_lab = action_list[action_ind]

                # Unpack improted data
                hbm_samples, res = imported_data[cat][move][action_ind]
                X_data, Y_data = ana_demographic_impact.HBM_data_helper(res)
                X_mean = X_data.mean(axis=0)
                p_base = Y_data.mean()
                cat_vals = list(res.keys())

                # print(cat, cat_vals, X_mean)

                X_eval = {"base": X_mean}
                if test_label == "Gender":
                    more_label = f"more {cat_vals[0]}, less {cat_vals[1]}"
                    delta = 0.20
                    cache = X_mean.copy()
                    cache[0] += delta
                    cache[1] += -delta
                    X_eval["more"] = cache

                    cache = X_mean.copy()
                    cache[0] += -delta
                    cache[1] += delta
                    X_eval["less"] = cache
                elif test_label == "DeathPenaltySupport0":
                    more_label = f"more DeathPen {cat_vals[1]} + {cat_vals[3]}, less DeathPen {cat_vals[0]} + {cat_vals[2]}"
                    delta = 0.08
                    cache = X_mean.copy()
                    cache[1] += delta
                    cache[3] += delta
                    cache[0] += -delta
                    cache[2] += -delta
                    X_eval["more"] = cache

                    cache = X_mean.copy()
                    cache[0] += delta
                    cache[2] += delta
                    cache[1] += -delta
                    cache[3] += -delta
                    X_eval["less"] = cache
                elif test_label == "DeathPenaltySupport1":
                    more_label = f"more DeathPen {cat_vals[2]} + {cat_vals[3]}, less DeathPen {cat_vals[0]} + {cat_vals[1]}"
                    delta = 0.08
                    cache = X_mean.copy()
                    cache[2] += delta
                    cache[3] += delta
                    cache[0] += -delta
                    cache[1] += -delta
                    X_eval["more"] = cache

                    cache = X_mean.copy()
                    cache[0] += delta
                    cache[1] += delta
                    cache[2] += -delta
                    cache[3] += -delta
                    X_eval["less"] = cache
                elif test_label == "Background0":
                    more_label = f"more {cat_vals[2]}, less {cat_vals[0]}"
                    delta = 0.20
                    cache = X_mean.copy()
                    cache[2] += delta
                    cache[0] += -delta
                    X_eval["more"] = cache

                    cache = X_mean.copy()
                    cache[2] += -delta
                    cache[0] += delta
                    X_eval["less"] = cache
                elif test_label == "Background1":
                    more_label = f"more {cat_vals[2]}, less {cat_vals[1]}"
                    delta = 0.20
                    cache = X_mean.copy()
                    cache[2] += delta
                    cache[1] += -delta
                    X_eval["more"] = cache

                    cache = X_mean.copy()
                    cache[2] += -delta
                    cache[1] += delta
                    X_eval["less"] = cache
                elif test_label == "Background2":
                    more_label = f"more {cat_vals[0]}, less {cat_vals[1]}"
                    delta = 0.20
                    cache = X_mean.copy()
                    cache[0] += delta
                    cache[1] += -delta
                    X_eval["more"] = cache

                    cache = X_mean.copy()
                    cache[0] += -delta
                    cache[1] += delta
                    X_eval["less"] = cache
                elif test_label == "Age0":
                    more_label = f"more {cat_vals[0]}, less {cat_vals[1]}"
                    delta = 0.15
                    cache = X_mean.copy()
                    cache[0] += delta
                    cache[1] += -delta
                    X_eval["more"] = cache

                    cache = X_mean.copy()
                    cache[0] += -delta
                    cache[1] += delta
                    X_eval["less"] = cache
                elif test_label == "Age1":
                    more_label = f"more {cat_vals[0]}, less {cat_vals[3]}"
                    delta = 0.15
                    cache = X_mean.copy()
                    cache[0] += delta
                    cache[3] += -delta
                    X_eval["more"] = cache

                    cache = X_mean.copy()
                    cache[0] += -delta
                    cache[3] += delta
                    X_eval["less"] = cache
                elif test_label == "Age2":
                    more_label = f"more {cat_vals[1]}, less {cat_vals[3]}"
                    delta = 0.15
                    cache = X_mean.copy()
                    cache[1] += delta
                    cache[3] += -delta
                    X_eval["more"] = cache

                    cache = X_mean.copy()
                    cache[1] += -delta
                    cache[3] += delta
                    X_eval["less"] = cache
                elif test_label == "Professionality":
                    more_label = f"more {cat_vals[0]}, less {cat_vals[1]}"
                    delta = 0.2
                    cache = X_mean.copy()
                    cache[0] += delta
                    cache[1] += -delta
                    X_eval["more"] = cache

                    cache = X_mean.copy()
                    cache[0] += -delta
                    cache[1] += delta
                    X_eval["less"] = cache
                elif test_label == "AIExp":
                    more_label = f"more {cat_vals[0]}, less {cat_vals[1]}"
                    delta = 0.1
                    cache = X_mean.copy()
                    cache[0] += delta
                    cache[1] += -delta
                    X_eval["more"] = cache

                    cache = X_mean.copy()
                    cache[0] += -delta
                    cache[1] += delta
                    X_eval["less"] = cache
                elif test_label == "ChinaMilExp0":
                    more_label = f"more {cat_vals[2]}, less {cat_vals[1]}"
                    delta = 0.08
                    cache = X_mean.copy()
                    cache[2] += delta
                    cache[1] += -delta
                    X_eval["more"] = cache

                    cache = X_mean.copy()
                    cache[2] += -delta
                    cache[1] += delta
                    X_eval["less"] = cache
                elif test_label == "ChinaMilExp1":
                    more_label = f"more {cat_vals[2]}, less {cat_vals[3]}"
                    delta = 0.08
                    cache = X_mean.copy()
                    cache[2] += delta
                    cache[3] += -delta
                    X_eval["more"] = cache

                    cache = X_mean.copy()
                    cache[2] += -delta
                    cache[3] += delta
                    X_eval["less"] = cache
                elif test_label == "ChinaMilExp2":
                    more_label = f"more {cat_vals[2]}, less {cat_vals[1]} + {cat_vals[3]}"
                    delta = 0.08
                    cache = X_mean.copy()
                    cache[2] += delta
                    cache[1] += -delta
                    cache[3] += -delta
                    X_eval["more"] = cache

                    cache = X_mean.copy()
                    cache[2] += -delta
                    cache[1] += delta
                    cache[3] += delta
                    X_eval["less"] = cache     
                elif test_label == "USMilExp0":
                    more_label = f"more {cat_vals[0]}, less {cat_vals[2]}"
                    delta = 0.1
                    cache = X_mean.copy()
                    cache[0] += delta
                    cache[2] += -delta
                    X_eval["more"] = cache

                    cache = X_mean.copy()
                    cache[0] += -delta
                    cache[2] += delta
                    X_eval["less"] = cache
                elif test_label == "USMilExp1":
                    more_label = f"more {cat_vals[0]}, less {cat_vals[4]}"
                    delta = 0.1
                    cache = X_mean.copy()
                    cache[0] += delta
                    cache[4] += -delta
                    X_eval["more"] = cache

                    cache = X_mean.copy()
                    cache[0] += -delta
                    cache[4] += delta
                    X_eval["less"] = cache
                elif test_label == "USMilExp2":
                    more_label = f"more {cat_vals[0]}, less {cat_vals[2]} + {cat_vals[4]}"
                    delta = 0.1
                    cache = X_mean.copy()
                    cache[0] += delta
                    cache[2] += -delta
                    cache[4] += -delta
                    X_eval["more"] = cache

                    cache = X_mean.copy()
                    cache[0] += -delta
                    cache[2] += delta
                    cache[4] += delta
                    X_eval["less"] = cache             
                else:
                    NotImplementedError(f"Invalid category: {cat}")

                eval_res = {
                    key: ana_demographic_impact.evaluate_new_points(X_eval[key], hbm_samples)[0] for key in X_eval
                }

                downward = eval_res["more"] < eval_res["base"] and eval_res["less"] >= eval_res["base"]
                upward = eval_res["less"] < eval_res["base"] and eval_res["more"] >= eval_res["base"]
                if upward or downward:
                    residuals = ana_demographic_impact.calculate_residuals(hbm_samples, X_data, Y_data)
                    mae = np.abs(residuals).mean()
                    threshhold = np.sqrt(2 * (mae **2 )) + 0.01
                    trend_diff = eval_res["more"] - eval_res["less"]
                    if np.abs(trend_diff) > threshhold:
                        n_trends_found += 1
                        n_trends_found_local += 1
                        print(action_lab)
                        print(
                            "Trend! " 
                            + (f"Increased likelihood when [{more_label}]" if upward else f"Decreased likelihood when [{more_label}]")
                            + f"\tDelta = {trend_diff:0.3f}, MAE/Thresh: {mae:0.3f}/{threshhold:0.3f}"
                        )
                        print(f"Baseline mean probability of action being picked p ={p_base:0.3f}")
                        # print(f"Baseline mean fraction across {cat_vals}", X_mean)
                        # print("Tested fractions: ", X_eval)
                        # print("Results: ", eval_res)
                        print()
        # print(n_trends_found_local)
    print("\n#Trends found in total: ", n_trends_found)
            

if __name__ == "__main__":
    main()

