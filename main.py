from pre_process_data import pre_process_data
from torch_data_loader import torch_data_loader
from np_data_loader import np_data_loader
from batch_data import batch_data
from cross_validation import cross_validation

import torch
from torch_cluster import knn_graph, radius_graph
from torch.nn import Sequential, Linear, ReLU, Dropout, Module, Softmax
from torch_geometric.nn import MessagePassing, global_max_pool, GATConv
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import os
import time
import random
import numpy as np
import pandas as pd
from scipy import stats

start_time = time.time()

cv_folds = 5
model_used = 'gnn'
graph_building_method = 'ball_query'
graph_building_param = 25
batch_size = 8
dropout_rate = 0
stopping_criteria = 1e-6
learning_rate = 1e-4
h_features = ['ObjectAreaCh1', 'ObjectShapeP2ACh1', 'ObjectShapeLWRCh1', 'ObjectTotalIntenCh1', 'ObjectAvgIntenCh1',
              'ObjectVarIntenCh1', 'SpotCountCh2', 'SpotTotalAreaCh2', 'SpotAvgAreaCh2', 'SpotTotalIntenCh2',
              'SpotAvgIntenCh2', 'TotalIntenCh2', 'AvgIntenCh2']
pos_features = ['X', 'Y']
eval_data = False

train_prop = 0.8    # proportion of data used to train the model
test_prop = 1 - train_prop  # proportion of data used to test the model

task = 'train'  # 'train' or 'validate'


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels + len(pos_features), out_channels),
                              ReLU(),
                              Linear(out_channels, out_channels))

    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_j, pos_i, pos_j):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]
        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_j, input], dim=-1)

        return self.mlp(input)  # Apply our final MLP.


class PointNet(Module):
    def __init__(self, selected_features, compounds):
        super().__init__()

        self.conv1 = PointNetLayer(len(selected_features), 32)
        self.conv2 = PointNetLayer(32, 32)
        self.att_conv = GATConv(32, 32, concat=False, heads=1, dropout=0.6)
        self.classifier = Linear(32, len(compounds))

        self.dropout = Dropout(dropout_rate)
        self.softmax = Softmax(dim=1)

    def forward(self, h, pos, batch):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        edge_index = None
        if graph_building_method == 'knn':
            edge_index = knn_graph(pos, k=graph_building_param, batch=batch, loop=True)
        elif graph_building_method == 'ball_query':
            edge_index = radius_graph(pos, r=graph_building_param, batch=batch, loop=True, max_num_neighbors=8192)

        # 3. Start bipartite message passing.
        h = self.conv1(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)

        if model_used == 'gat':
            h = h.relu()
            h = self.att_conv(h, edge_index=edge_index)

        h = self.dropout(h)
        h = h.relu()

        # 4. Global Pooling.
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

        # 5. Classifier.
        return self.classifier(h)


def evaluate_data():
    # Read csv data
    data = pd.read_csv(f'/content/drive/MyDrive/Cell.csv')

    data, well_list, concentration_dict, label_dict = pre_process_data(data=data, features=h_features + pos_features,
                                                                       compounds=['well_om', 'well_om_dmso'])
    print(f'Data pre-processed in {round(time.time() - start_time, 4)} seconds.')

    om_wells = well_list[:int(len(well_list) / 2)]
    om_dmso_wells = well_list[int(len(well_list) / 2):]

    om_data, _ = torch_data_loader(data=data, wells=om_wells, labels=label_dict, features=['SpotCountCh2'])
    om_dmso_data, _ = torch_data_loader(data=data, wells=om_dmso_wells, labels=label_dict, features=['SpotCountCh2'])
    om_data, om_dmso_data = np_data_loader(train_data=om_data, test_data=om_dmso_data, h_features=['SpotCountCh2'])

    p_value = stats.ttest_rel(om_data, om_dmso_data)[1][0]

    print(f"The result of the paired t-test between the OM and OM+DMSO controls provide a p-value of"
          f" p={p_value}.")
    if p_value > 0.05:
        print("Conclusion: insufficient evidence for difference in mean SpotCountCh2")
    else:
        print("Conclusion: sufficient evidence for difference in mean SpotCountCh2")
    exit()


def concat_features(h_feature_batch_loader, pos_feature_batch_loader, idx):
    h_features_cat, pos_features_cat = [None for _ in range(2)]
    concat_h = True
    for feature_batch_loader in (h_feature_batch_loader, pos_feature_batch_loader):
        features_cat = torch.tensor([])
        features_cat_exists = False
        for batch_loader in feature_batch_loader:
            batch_loader[idx] = torch.reshape(batch_loader[idx], (batch_loader[idx].shape[0], 1))
            if features_cat_exists:
                features_cat = torch.cat((features_cat, batch_loader[idx]), dim=1)
            else:
                features_cat = batch_loader[idx]
                features_cat_exists = True
        if concat_h:
            h_features_cat = features_cat
            concat_h = False
        else:
            pos_features_cat = features_cat

    return h_features_cat, pos_features_cat


def train(model, device, criterion, optimizer, features, data, labels):
    feature_batch_loader, labels_batch_loader, data_batch_cat_loader = \
        batch_data(features=features, data=data, labels=labels, batch_size=batch_size)

    h_feature_batch_loader = feature_batch_loader[:len(features) - len(pos_features)]
    pos_feature_batch_loader = feature_batch_loader[len(features) - len(pos_features):]

    model.train()
    total_loss = 0
    for idx, _ in enumerate(data_batch_cat_loader, 0):
        h_features_cat, pos_features_cat = concat_features(h_feature_batch_loader=h_feature_batch_loader,
                                                           pos_feature_batch_loader=pos_feature_batch_loader, idx=idx)
        optimizer.zero_grad()  # Clear gradients.

        logits = model(h=h_features_cat.float().to(device), pos=pos_features_cat.float().to(device),
                       batch=data_batch_cat_loader[idx].type(torch.LongTensor).to(device))  # Forward pass.
        loss = criterion(logits, labels_batch_loader[idx].type(torch.LongTensor).to(device))  # Loss computation.
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.

        total_loss += loss.item() * 10

    return total_loss / len(features[0])


def test(model, device, features, data, labels):
    feature_batch_loader, labels_batch_loader, data_batch_cat_loader = \
        batch_data(features=features, data=data, labels=labels, batch_size=batch_size)

    h_feature_batch_loader = feature_batch_loader[:len(features) - len(pos_features)]
    pos_feature_batch_loader = feature_batch_loader[len(features) - len(pos_features):]

    model.eval()

    label_pred, label_true, logits_list = [[] for _ in range(3)]
    for idx, _ in enumerate(data_batch_cat_loader, 0):
        h_features_cat, pos_features_cat = concat_features(h_feature_batch_loader=h_feature_batch_loader,
                                                           pos_feature_batch_loader=pos_feature_batch_loader, idx=idx)

        logits = model(h=h_features_cat.float().to(device), pos=pos_features_cat.float().to(device),
                       batch=data_batch_cat_loader[idx].type(torch.LongTensor).to(device))  # Forward pass.

        label_pred += logits.argmax(dim=-1).tolist()
        label_true += labels_batch_loader[idx].type(torch.LongTensor).tolist()

        logits_list += Softmax(dim=1)(logits).tolist()

    total_correct = 0
    for idx, label in enumerate(label_pred, 0):
        if label == label_true[idx]:
            total_correct += 1

    roc_auc = roc_auc_score(np.array(label_true), np.array(logits_list)[:, 1])

    return total_correct / len(label_pred), roc_auc


def run_class(compounds):

    print(f'Task: Classify between {compounds}')

    train_data_list, test_data_list, train_labels_list, test_labels_list, selected_features_list,\
        benchmark_accuracy_list, test_accuracy_list, roc_auc_list = [[] for _ in range(8)]

    data = pd.read_csv(f'/content/drive/MyDrive/Cell.csv')
    data, well_list, concentration_dict, label_dict = pre_process_data(data=data, features=h_features + pos_features,
                                                                       compounds=compounds)
    print(f'Data pre-processed in {round(time.time() - start_time, 4)} seconds.')

    cv_wells = cross_validation(cv_folds=round(1 / test_prop), well_list=well_list, label_dict=label_dict,
                                concentration_dict=concentration_dict)

    data_wells = []
    for wells in cv_wells[:-1]:
        data_wells += wells
    val_wells = cv_wells[-1]

    cv_wells = cross_validation(cv_folds=cv_folds, well_list=data_wells, label_dict=label_dict,
                                concentration_dict=concentration_dict)

    print(f'CV completed in {round(time.time() - start_time, 4)} seconds.')

    for i in range(cv_folds):
        train_idx = [idx for idx in range(cv_folds)][:int(cv_folds * train_prop)]
        test_idx = [idx for idx in range(cv_folds)][int(cv_folds * train_prop):]

        train_idx = [(idx + i) % cv_folds for idx in train_idx]
        test_idx = [(idx + i) % cv_folds for idx in test_idx]

        train_wells, test_wells = [[] for _ in range(2)]

        train_wells_list = [cv_wells[idx] for idx in train_idx]
        for wells in train_wells_list:
            train_wells += wells

        test_wells_list = [cv_wells[idx] for idx in test_idx]
        for wells in test_wells_list:
            test_wells += wells

        if task == 'validate':
            train_wells = data_wells
            test_wells = val_wells

        random.shuffle(train_wells)
        random.shuffle(test_wells)

        train_data, train_labels = torch_data_loader(data=data, wells=train_wells, labels=label_dict,
                                                     features=h_features + pos_features)
        test_data, test_labels = torch_data_loader(data=data, wells=test_wells, labels=label_dict,
                                                   features=h_features + pos_features)

        train_1, train_1_labels = torch_data_loader(data=data, wells=train_wells[:round(len(train_wells) * 0.6)],
                                                    labels=label_dict, features=h_features + pos_features)
        train_2, train_2_labels = torch_data_loader(data=data, wells=train_wells[round(len(train_wells) * 0.6):],
                                                    labels=label_dict, features=h_features + pos_features)

        train_1, train_2 = np_data_loader(train_data=train_1, test_data=train_2, h_features=h_features)

        forest = RandomForestClassifier(random_state=42)
        forest.fit(train_1, train_1_labels)

        result = permutation_importance(forest, train_2, train_2_labels, n_repeats=10, random_state=42,
                                        n_jobs=-1)
        forest_importance = pd.Series(result.importances_mean, index=h_features)

        feature_importance = [(forest_importance[feature], feature) for feature in h_features]
        feature_importance.sort()

        cutoff_idx = 0
        for importance, feature in feature_importance:
            if importance > 0:
                break
            else:
                cutoff_idx += 1

        if cutoff_idx != len(feature_importance):
            selected_features = [feature_importance[idx][1] for idx in range(cutoff_idx, len(feature_importance))]
        else:
            selected_features = [feature_importance[idx][1] for idx in range(len(feature_importance) - 3,
                                                                             len(feature_importance))]

        train_data_list.append(train_data)
        test_data_list.append(test_data)
        train_labels_list.append(train_labels)
        test_labels_list.append(test_labels)
        selected_features_list.append(selected_features)

        if task == 'validate':
            break

    feature_count_dict = {}
    for features in selected_features_list:
        for feature in features:
            if feature not in feature_count_dict:
                feature_count_dict[feature] = 1
            else:
                feature_count_dict[feature] += 1

    selected_features, backup_features = [[] for _ in range(2)]
    max_value = 0
    for key, value in feature_count_dict.items():
        if value > cv_folds / 2:
            selected_features.append(key)
        elif task == 'validate' and value > 0:
            selected_features.append(key)

        if value > max_value:
            max_value = value
            backup_features = [key]
        elif value == max_value:
            backup_features.append(key)

    if not selected_features:
        selected_features = backup_features

    training_dict, training_dict_empty = [{}, True]
    for i in range(cv_folds):
        train_data_np, test_data_np = np_data_loader(train_data=train_data_list[i], test_data=test_data_list[i],
                                                     h_features=selected_features)
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(train_data_np, train_labels_list[i])
        np_pred = clf.predict(test_data_np)

        total_correct = 0
        for pred_idx, _ in enumerate(np_pred, 0):
            if np_pred[pred_idx] == test_labels_list[i][pred_idx]:
                total_correct += 1

        benchmark_accuracy = total_correct / len(test_labels_list[i])
        benchmark_accuracy_list.append(benchmark_accuracy)

        if task == 'train':
            print(f'Fold: {i + 1}, Benchmark Accuracy: {round(benchmark_accuracy, 2)},'
                  f' Time: {round(time.time() - start_time, 4)} seconds.')
        elif task == 'validate':
            print(f'Benchmark Accuracy: {round(benchmark_accuracy, 2)},'
                  f' Time: {round(time.time() - start_time, 4)} seconds.')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PointNet(selected_features=selected_features, compounds=compounds).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss().to(device)  # Define loss criterion.

        test_acc, roc_auc, converged, prev_loss, epoch = [None, None, False, [], 0]
        while not converged and epoch < 10000:
            loss = train(model, device, criterion, optimizer, selected_features + pos_features,
                         train_data_list[i], train_labels_list[i])
            test_acc, roc_auc = test(model, device, selected_features + pos_features, test_data_list[i],
                                     test_labels_list[i])

            if task == 'train':
                print(f'Fold {i + 1}, Epoch: {epoch + 1:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f},'
                      f' ROC AUC Score: {roc_auc:.4f}, Time:'f' {time.time() - start_time:.2f} seconds')
            elif task == 'validate':
                print(f'Epoch: {epoch + 1:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f},'
                      f' ROC AUC Score: {roc_auc:.4f}, Time:'f' {time.time() - start_time:.2f} seconds')

            epoch += 1
            prev_loss.append(loss)
            if len(prev_loss) >= 5:
                std = np.std(prev_loss[-5:])
                if std < stopping_criteria:
                    converged = True

            if training_dict_empty:
                training_dict = {'fold': [i + 1], 'epoch': [epoch], 'test_acc': [test_acc], 'roc_auc': [roc_auc],
                                 'time': [time.time() - start_time]}
                training_dict_empty = False
            else:
                training_dict['fold'] = training_dict['fold'] + [i + 1]
                training_dict['epoch'] = training_dict['epoch'] + [epoch]
                training_dict['test_acc'] = training_dict['test_acc'] + [test_acc]
                training_dict['roc_auc'] = training_dict['roc_auc'] + [roc_auc]
                training_dict['time'] = training_dict['time'] + [time.time() - start_time]

        print(f'Final metrics after {epoch:02d} epochs: Test Accuracy: {test_acc:.4f}, ROC AUC Score: {roc_auc:.4f},'
              f' Time:'f' {time.time() - start_time:.2f} seconds')

        test_accuracy_list.append(test_acc)
        roc_auc_list.append(roc_auc)

        if task == 'validate':
            break

    print(f'Model used: {model_used}')
    print(f'Graph-building features: {pos_features}')
    print(f'Selected features: {selected_features}')

    if task == 'train':
        print(f'Accuracy metric: {cv_folds}-fold CV')
    elif task == 'validate':
        print(f'Accuracy metric: Validation Accuracy')

    print(f'Batch size: {batch_size}')
    print(f'Dropout rate: {dropout_rate}')
    print(f'Stopping criteria: {stopping_criteria}')
    print(f'Learning rate: {learning_rate}')
    print(f'Graph-building method: {graph_building_method} with parameter={graph_building_param}')
    print(f'Benchmark test accuracy after {cv_folds}-fold CV:'
          f' {sum(benchmark_accuracy_list) / len(benchmark_accuracy_list):.4f}')
    print(f'Final test accuracy after {cv_folds}-fold CV: {sum(test_accuracy_list) / len(test_accuracy_list):.4f}')
    print(f'Final ROC AUC score after {cv_folds}-fold CV: {sum(roc_auc_list) / len(roc_auc_list):.4f}')
    print(f'Runtime: {time.time() - start_time:.2f} seconds')
    print("")

    model_description = {'model_used': model_used, 'pos': [pos_features], 'selected_features': [selected_features],
                         'cv_folds': cv_folds, 'batch_size': batch_size, 'dropout_rate': dropout_rate,
                         'stopping_criteria': stopping_criteria, 'learning_rate': learning_rate,
                         'graph_building_method': graph_building_method, 'graph_building_param': graph_building_param,
                         'benchmark_test_acc': sum(benchmark_accuracy_list) / len(benchmark_accuracy_list),
                         'test_acc': sum(test_accuracy_list) / len(test_accuracy_list),
                         'roc_auc_score': sum(roc_auc_list) / len(roc_auc_list), 'time': time.time() - start_time}

    if not os.path.exists('output/compounds_out'):
        os.makedirs('output/compounds_out')

    pd.DataFrame(model_description).to_csv(f'output/compounds_out/model_out_{compounds}.csv', index=False)
    pd.DataFrame(training_dict).to_csv(f'output/compounds_out/training_out_{compounds}.csv', index=False)

    return sum(benchmark_accuracy_list) / len(benchmark_accuracy_list),\
        sum(test_accuracy_list) / len(test_accuracy_list), sum(roc_auc_list) / len(roc_auc_list)


if eval_data:
    evaluate_data()

print(f'Task: {task}')

compounds_list = ['well_2331', 'well_8752', 'well_4951', 'well_1529', 'well_1854', 'well_4184', 'well_om',
                  'well_om_dmso']

final_benchmark_accuracy_list, final_test_accuracy_list, final_roc_auc_list = [[] for _ in range(3)]
for compound in compounds_list[:6]:
    final_benchmark_accuracy, final_test_accuracy, final_roc_auc_score = run_class(compounds=[compound, 'well_om'])
    final_benchmark_accuracy_list.append(final_benchmark_accuracy)
    final_test_accuracy_list.append(final_test_accuracy)
    final_roc_auc_list.append(final_roc_auc_score)

print(f'Model used: {model_used}')
print(f'Accuracy metric: {cv_folds}-fold CV')
print(f'Batch size: {batch_size}')
print(f'Dropout rate: {dropout_rate}')
print(f'Stopping criteria: {stopping_criteria}')
print(f'Learning rate: {learning_rate}')
print(f'Graph-building method: {graph_building_method} with parameter={graph_building_param}')
print(f'Benchmark test accuracy after {cv_folds}-fold CV:'
      f' {sum(final_benchmark_accuracy_list) / len(final_benchmark_accuracy_list):.4f}')
print(f'Final test accuracy after {cv_folds}-fold CV:'
      f' {sum(final_test_accuracy_list) / len(final_test_accuracy_list):.4f}')
print(f'Final ROC AUC score after {cv_folds}-fold CV: {sum(final_roc_auc_list) / len(final_roc_auc_list):.4f}')
print(f'Runtime: {time.time() - start_time:.2f} seconds')

final_model_description = {'model_used': model_used, 'pos_features': [pos_features], 'cv_folds': cv_folds,
                           'batch_size': batch_size, 'dropout_rate': dropout_rate,
                           'stopping_criteria': stopping_criteria, 'learning_rate': learning_rate,
                           'graph_building_method': graph_building_method, 'graph_building_param': graph_building_param,
                           'benchmark_test_acc':
                               sum(final_benchmark_accuracy_list) / len(final_benchmark_accuracy_list),
                           'test_acc': sum(final_test_accuracy_list) / len(final_test_accuracy_list),
                           'roc_auc_score': sum(final_roc_auc_list) / len(final_roc_auc_list),
                           'time': time.time() - start_time}

if not os.path.exists('output/ovr_out'):
    os.makedirs('output/ovr_out')

pd.DataFrame(final_model_description).to_csv(f'output/ovr_out/final_model_out.csv', index=False)
