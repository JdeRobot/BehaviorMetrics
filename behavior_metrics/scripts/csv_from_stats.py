import os.path
import pandas as pd
import yaml
import json
from copy import copy

if __name__ == "__main__":
    experiment_main_file = "/home/frivas/devel/mio/github/BehaviorMetrics/behavior_metrics/configs/default-multiple2.yml"

    with open(experiment_main_file, 'r') as f:
        doc = yaml.load(f)

    stats = json.load(open("/home/frivas/devel/mio/github/BehaviorMetrics/behavior_metrics/bag_analysis/stats.json"))

    n_models = len(doc['Behaviors']['Robot']['Parameters']['Model'])

    all_data_all_list_of_dict = []

    for circuit in stats:
        current_stat = {"circuit": circuit,
                        "v_classes": "N/A",
                         "w_classes": "N/A"}
        for idx, model in enumerate(doc['Behaviors']['Robot']['Parameters']['Model']):
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

            for idx2, i in enumerate(range(idx*2, idx*2 + 2)):
                current_stat["run"] = idx2
                for stat in stats[circuit]:
                    if stat == "time_meta":
                        for time_stat in stats[circuit][stat][i]:
                            current_stat[time_stat] = stats[circuit][stat][i][time_stat]
                    else:
                        current_stat[stat] = stats[circuit][stat][i]
                all_data_all_list_of_dict.append(copy(current_stat))

        # break


    df = pd.DataFrame(all_data_all_list_of_dict)

    df.to_csv("all_stats.csv")
    print(model)

