from sklearn.model_selection import train_test_split


def cross_validation(cv_folds, well_list, label_dict, concentration_dict):

    cv_wells = []
    for i in range(cv_folds - 1):
        label_list = [label_dict[well] for well in well_list]
        concentration_list = [concentration_dict[well] for well in well_list]
        for well_idx, well in enumerate(well_list, 0):
            if concentration_dict[well] > 1 / (3 ** 4):
                concentration_list[well_idx] = 0
            elif concentration_dict[well] > 1 / (3 ** 8):
                concentration_list[well_idx] = 10
            else:
                concentration_list[well_idx] = 20

        stratify_list = []
        for idx, _ in enumerate(well_list, 0):
            stratify_list.append(concentration_list[idx] + label_list[idx])

        well_list, test_wells = train_test_split(well_list, test_size=1 / (cv_folds - i), stratify=stratify_list,
                                                 random_state=42)

        cv_wells.append(test_wells)

    cv_wells.append(well_list)

    return cv_wells
