def well_gen(letter_idx, num_idx):
    well_id = []
    for letter in letter_idx:
        for number in num_idx:
            well_str = letter + f"{number}"
            if len(well_str) == 2:
                well_str = well_str[0] + "0" + well_str[1]
            well_id.append(" " + well_str)

    return well_id


def normalise_features(data, features):

    new_data = {}
    for feature in features:

        feature_list = []
        max_element = max(data[feature])
        min_element = min([item for item in data[feature]])

        for data_idx, data_item in enumerate(data[feature], 0):
            data_item -= min_element
            data_item /= (max_element - min_element)

            feature_list.append(data_item)

        new_data[feature] = feature_list

    return data


def pre_process_data(data, features, compounds):

    data = normalise_features(data=data, features=features)
    letters = ["C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]

    # Generate well labels & insert into a dictionary
    well_2331 = well_gen(letter_idx=letters, num_idx=[4, 5, 6])
    well_8752 = well_gen(letter_idx=letters, num_idx=[7, 8, 9])
    well_4951 = well_gen(letter_idx=letters, num_idx=[10, 11, 12])
    well_1529 = well_gen(letter_idx=letters, num_idx=[13, 14, 15])
    well_1854 = well_gen(letter_idx=letters, num_idx=[16, 17, 18])
    well_4184 = well_gen(letter_idx=letters, num_idx=[19, 20, 21])
    well_om = well_gen(letter_idx=letters, num_idx=[2, 23])
    well_om_dmso = well_gen(letter_idx=letters, num_idx=[3, 22])

    starting_concentration = 1
    concentration_dict = {}
    for letter_idx, letter in enumerate(letters, 0):
        well_concentration = well_gen(letter_idx=[letter],
                                      num_idx=range(2, 24))
        concentration = starting_concentration / (3 ** letter_idx)
        concentration_dict.update(dict.fromkeys(well_concentration, concentration))

    labels = []
    for compound in compounds:
        labels.append(eval(compound))
    well_id = [i for i in range(len(labels))]

    well_list = []
    for compound_labels in labels:
        for label in compound_labels:
            well_list.append(label)

    label_dict = {}
    for idx, label in enumerate(labels, 0):
        label_dict.update(dict.fromkeys(label, well_id[idx]))

    return data, well_list, concentration_dict, label_dict
