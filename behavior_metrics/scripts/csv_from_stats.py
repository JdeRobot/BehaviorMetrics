import os.path
import pandas as pd
import yaml
import json
from copy import copy
import numpy as np

if __name__ == "__main__":
    experiment_main_file = "/home/frivas/devel/mio/github/BehaviorMetrics/behavior_metrics/configs/default-multiple4.yml"

    with open(experiment_main_file, 'r') as f:
        doc = yaml.load(f)

    stats = json.load(open("/home/frivas/devel/mio/github/BehaviorMetrics/behavior_metrics/bag_analysis/stats.json"))

    n_models = len(doc['Behaviors']['Robot']['Parameters']['Model'])

    all_data_all_list_of_dict = []
    mean_data_list_of_dict = []

    n_repetitions = doc["Behaviors"]["Experiment"]["Repetitions"]

    for circuit in stats:
        for idx, model in enumerate(doc['Behaviors']['Robot']['Parameters']['Model']):

            current_stat = {"circuit": circuit,
                            "v_classes": "N/A",
                            "w_classes": "N/A"}
            current_stat["model"] = model
            if isinstance(model, list):
                model_info = yaml.load(open(os.path.join(model[0], "config.yaml")))
                model_info_v = yaml.load(open(os.path.join(model[1], "config.yaml")))
                model_name = list(model_info.keys())[0]
                model_name_v = list(model_info_v.keys())[0]

                if model_info[model_name]["mode"] == "classification":
                    current_stat["w_classes"] = len(model_info[model_name]["classification_data"]["w"])
                    current_stat["v_classes"] = len(model_info_v[model_name_v]["classification_data"]["v"])
            else:
                model_info = yaml.load(open(os.path.join(model, "config.yaml")))
                model_name = list(model_info.keys())[0]
                if model_info[model_name]["mode"] == "classification":
                    current_stat["w_classes"] = len(model_info[model_name]["classification_data"]["w"])
                    current_stat["v_classes"] = len(model_info[model_name]["classification_data"]["v"])

            current_stat["mode"] = model_info[model_name]["mode"]
            current_stat["backend"] = model_info[model_name]["backend"]
            current_stat["fc_head"] = model_info[model_name]["fc_head"]
            current_stat["pretrained"] = model_info[model_name]["pretrained"]
            current_stat["weight_loss"] = model_info[model_name]["weight_loss"]
            current_stat["base_size"] = model_info[model_name]["base_size"]
            current_stat["batch_size"] = model_info[model_name]["batch_size"]
            current_stat["apply_vertical_flip"] = model_info[model_name].get("apply_vertical_flip", True)
            current_stat["non_common_samples_mult_factor"] = model_info[model_name].get("non_common_samples_mult_factor", 0)
            current_stat["loss_reduction"] = model_info[model_name].get("loss_reduction", "mean")


            acc = {"v": 0, "w": 0}

            paths_to_evaluate = []
            if isinstance(model, list):
                paths_to_evaluate += model
            else:
                paths_to_evaluate.append(model)

            for path_to_evaluate in paths_to_evaluate:
                current_stats_file = os.path.join(path_to_evaluate, "stats.json")
                current_stats_data = json.load(open(current_stats_file))

                for controller in ["w", "v"]:
                    if controller in current_stats_data:
                        if model_info[model_name]["mode"] == "classification":
                            acc[controller] = current_stats_data[controller]["acc1"]
                        else:
                            acc[controller] = current_stats_data[controller]["rmse"]

            current_stat["acc_v"] = acc["v"]
            current_stat["acc_w"] = acc["w"]
            current_stat["acc"] = (acc["v"] + acc["w"]) /2

            current_experiment_runs = []
            for idx2, i in enumerate(range(idx*n_repetitions, idx*n_repetitions + n_repetitions)):
                current_stat["run"] = idx2
                for stat in stats[circuit]:
                    if stat == "time_meta":
                        continue
                        for time_stat in stats[circuit][stat][i]:
                            current_stat[time_stat] = stats[circuit][stat][i][time_stat]
                    else:
                        current_stat[stat] = stats[circuit][stat][i]
                current_experiment_runs.append(copy(current_stat))

            all_data_all_list_of_dict += current_experiment_runs
            #computing mean value for each experiment
            for current_key in current_experiment_runs[0]:
                if isinstance(current_experiment_runs[0][current_key], str) or isinstance(current_experiment_runs[0][current_key], int):
                    pass

                else:
                    current_stat[current_key] = np.mean([x[current_key] for x in current_experiment_runs if x["lap_seconds"] > 0])

            current_stat["Success ratio"] = (len([ x for x in current_experiment_runs if x["lap_seconds"] > 0]) / len(current_experiment_runs)) * 100

            mean_data_list_of_dict += [current_stat]

            if current_stat["Success ratio"] != 100:
                print("hola")

            print("hola")


        # break


    df = pd.DataFrame(all_data_all_list_of_dict)
    df_mean = pd.DataFrame(mean_data_list_of_dict)


    df.to_csv("all_stats.csv")
    df_mean.to_csv("mean_stats.csv")

    print(model)

