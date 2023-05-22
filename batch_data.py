import torch


torch.manual_seed(42)


def batch_data(features, data, labels, batch_size):

    batch_dict, batch_list_dict = [{} for _ in range(2)]
    labels_batch, labels_batch_list, data_batch_cat, data_batch_cat_list = [[] for _ in range(4)]

    batch_dict.update(dict.fromkeys(features, torch.tensor([])))
    batch_list_dict.update(dict.fromkeys(features, []))

    feature_batch = [batch_dict[feature] for feature in features]
    feature_batch_list = [batch_list_dict[feature] for feature in features]

    batch_exists, batch_counter = [False, 0]
    for idx, _ in enumerate(data[features[0]], 0):
        if batch_exists and (batch_counter % batch_size) == 0:
            for list_idx in range(len(features)):
                feature_batch_list[list_idx] = feature_batch_list[list_idx] + [feature_batch[list_idx]]

            labels_batch_list.append(labels_batch)

            for feature_idx in range(len(features)):
                feature_batch[feature_idx] = data[features[feature_idx]][idx]

            data_batch_cat_list.append(data_batch_cat)

            data_batch_cat = torch.tensor([])
            data_batch = torch.tensor([batch_counter % batch_size])
            data_batch = data_batch.repeat(data[features[0]][idx].shape[0])
            data_batch_cat = torch.cat((data_batch_cat, data_batch))

            labels_batch = torch.tensor([labels[idx]])
            batch_counter += 1

        elif batch_exists:
            for feature_idx in range(len(features)):
                feature_batch[feature_idx] = torch.cat((feature_batch[feature_idx],
                                                        data[features[feature_idx]][idx]))

            data_batch = torch.tensor([batch_counter % batch_size])
            data_batch = data_batch.repeat(data[features[0]][idx].shape[0])
            data_batch_cat = torch.cat((data_batch_cat, data_batch))

            labels_batch = torch.cat((labels_batch, torch.tensor([labels[idx]])))
            batch_counter += 1
        else:
            for feature_idx in range(len(features)):
                feature_batch[feature_idx] = data[features[feature_idx]][idx]

            data_batch_cat = torch.tensor([])
            data_batch = torch.tensor([batch_counter % batch_size])
            data_batch = data_batch.repeat(data[features[0]][idx].shape[0])
            data_batch_cat = torch.cat((data_batch_cat, data_batch))

            labels_batch = torch.tensor([labels[idx]])
            batch_exists = True
            batch_counter += 1

    if batch_exists:
        for feature_idx in range(len(features)):
            feature_batch_list[feature_idx] = feature_batch_list[feature_idx] + [feature_batch[feature_idx]]

        data_batch_cat_list.append(data_batch_cat)
        labels_batch_list.append(labels_batch)

    return feature_batch_list, labels_batch_list, data_batch_cat_list
