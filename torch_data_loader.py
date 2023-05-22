import torch


torch.manual_seed(42)


def torch_data_loader(data, wells, labels, features):
    wells.sort()

    loader_dict, loader_labels, feature_dict, search_idx, tensor_exists, append_last = [{}, [], {}, 0, False, True]
    loader_dict.update(dict.fromkeys(features, []))
    feature_dict.update(dict.fromkeys(features, torch.tensor([])))
    for well_idx, _ in enumerate(data["WellId"], 0):
        if data["WellId"][well_idx] == wells[search_idx]:
            if tensor_exists:
                for feature in features:
                    feature_dict[feature] = torch.cat((feature_dict[feature],
                                                       torch.tensor([data[feature][well_idx]])))
            else:
                for feature in features:
                    feature_dict[feature] = torch.tensor([data[feature][well_idx]])
                tensor_exists = True
        elif tensor_exists:
            for feature in features:
                loader_dict[feature] = loader_dict[feature] + [feature_dict[feature]]
            loader_labels.append(labels[wells[search_idx]])
            search_idx += 1

            tensor_exists = False

        if search_idx == len(wells):
            append_last = False
            break

    if append_last:
        for feature in features:
            loader_dict[feature] = loader_dict[feature] + [feature_dict[feature]]
        loader_labels.append(labels[wells[search_idx]])

    return loader_dict, loader_labels
