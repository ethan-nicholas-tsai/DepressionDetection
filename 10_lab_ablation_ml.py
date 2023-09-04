def calc_metrics_binary(model, X_test, y_test):
    from sklearn.metrics import (
        classification_report,
        accuracy_score,
        f1_score,
        roc_auc_score,
        recall_score,
        precision_score,
    )

    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=["Normal", "Depressed"], digits=4
    )
    return report


from datetime import datetime


def train_TimeSeriesForest(X_train, y_train, X_test, y_test):
    start_time = datetime.now()
    # base
    print("***TimeSeriesForestClassifier***")
    from sklearn.pipeline import Pipeline
    from sktime.classification.interval_based import TimeSeriesForestClassifier
    from sktime.classification.compose import ColumnEnsembleClassifier
    from sktime.transformations.panel.compose import ColumnConcatenator

    steps = [
        ("concatenate", ColumnConcatenator()),
        ("classify", TimeSeriesForestClassifier(n_estimators=100)),
    ]
    clf = Pipeline(steps)
    clf.fit(X_train, y_train)

    report = calc_metrics_binary(clf, X_test, y_test)

    print(report)

    print(str(datetime.now() - start_time))

    return clf


def train_ROCKETClassifier(X_train, y_train, X_test, y_test):
    start_time = datetime.now()
    # 2020
    print("***ROCKETClassifier***")
    from sktime.classification.kernel_based import ROCKETClassifier

    clf = ROCKETClassifier(num_kernels=500)
    clf.fit(X_train, y_train)

    report = calc_metrics_binary(clf, X_test, y_test)

    print(report)

    print(str(datetime.now() - start_time))

    return clf


def train_Signature(X_train, y_train, X_test, y_test):
    start_time = datetime.now()

    # 2020
    print("***SignatureClassifier***")
    from sktime.classification.feature_based import SignatureClassifier

    clf = SignatureClassifier()
    clf.fit(X_train, y_train)

    report = calc_metrics_binary(clf, X_test, y_test)

    print(report)

    print(str(datetime.now() - start_time))

    return clf


def train_Arsenal(X_train, y_train, X_test, y_test):
    start_time = datetime.now()
    # uni
    print("***Arsenal***")
    from sktime.classification.kernel_based import Arsenal

    clf = Arsenal(num_kernels=200, n_estimators=5)
    clf.fit(X_train, y_train)

    report = calc_metrics_binary(clf, X_test, y_test)

    print(report)
    print(str(datetime.now() - start_time))
    start_time = datetime.now()

    return clf


def train_TSFresh(X_train, y_train, X_test, y_test):
    start_time = datetime.now()
    # 2018
    print("***TSFreshClassifier***")
    from sktime.classification.feature_based import TSFreshClassifier

    clf = TSFreshClassifier()
    clf.fit(X_train, y_train)

    report = calc_metrics_binary(clf, X_test, y_test)

    print(report)

    print(str(datetime.now() - start_time))

    return clf


def train_HIVECOTEV2(X_train, y_train, X_test, y_test):
    start_time = datetime.now()
    # No module named sktime.classificastion.shapelet_based.mrseql.mrseql
    print("***HIVECOTEV2***")
    from sktime.classification.hybrid import HIVECOTEV2
    from sktime.contrib.vector_classifiers._rotation_forest import RotationForest

    clf = HIVECOTEV2(
        stc_params={
            "estimator": RotationForest(n_estimators=3),
            "n_shapelet_samples": 500,
            "max_shapelets": 20,
            "batch_size": 100,
        },
        drcif_params={"n_estimators": 10},
        arsenal_params={"num_kernels": 100, "n_estimators": 5},
        tde_params={
            "n_parameter_samples": 25,
            "max_ensemble_size": 5,
            "randomly_selected_params": 10,
        },
    )
    clf.fit(X_train, y_train)

    report = calc_metrics_binary(clf, X_test, y_test)

    print(report)
    print(str(datetime.now() - start_time))
    start_time = datetime.now()

    return clf


def train_ShapeletTransform(X_train, y_train, X_test, y_test):
    start_time = datetime.now()
    # shapelet
    print("***ShapeletTransformClassifier***")
    from sktime.classification.shapelet_based import ShapeletTransformClassifier
    from sktime.contrib.vector_classifiers._rotation_forest import RotationForest

    clf = ShapeletTransformClassifier(
        estimator=RotationForest(n_estimators=3),
        n_shapelet_samples=500,
        max_shapelets=20,
        batch_size=100,
    )
    clf.fit(X_train, y_train)

    report = calc_metrics_binary(clf, X_test, y_test)

    print(report)
    print(str(datetime.now() - start_time))
    start_time = datetime.now()

    return clf


model_list = ["bst", "rocket", "gs", "hc2", "tsf"]

if __name__ == "__main__":
    import os
    import numpy as np
    from sktime.utils.data_io import load_from_tsfile_to_dataframe
    from utils.symptom import symptoms_dsm_5 as symptoms

    # 情绪
    feat_emo = [
        "depressive_mood",
        "retardation_or_agitation",
        "panic_and_anxious",
    ]  # sad, agi, pan
    # 认知
    feat_cog = [
        "interest_pleasure_loss",
        "self_blame",
        "suicidal_ideation",
        "concentration_problem",
    ]  # int, sel(low-esteem), sui, con
    # 躯体
    feat_bod = [
        "appetite_weight_problem",
        "insomnia_or_hypersomnia",
        "energy_loss",
        "sympathetic_arousal",
    ]  # app, ins, ene, sym
    # 行为？

    feat_emo_dim = []
    feat_cog_dim = []
    feat_bod_dim = []

    for i, k in enumerate(symptoms):
        print(i, k)
        feat_id = "dim_{}".format(i)
        if k in feat_emo:
            feat_emo_dim.append(feat_id)
        elif k in feat_cog:
            feat_cog_dim.append(feat_id)
        elif k in feat_bod:
            feat_bod_dim.append(feat_id)

    print(feat_emo_dim, feat_cog_dim, feat_bod_dim)

    feat_group = [feat_emo_dim, feat_cog_dim, feat_bod_dim]


    flag = 1
    for r in range(10):

        # load data
        data_dir = "dataset/swdd-7k_ts_origin_500_0"

        X_train_all, y_train = load_from_tsfile_to_dataframe(
            os.path.join(data_dir, "train.ts")
        )
        X_test_all, y_test = load_from_tsfile_to_dataframe(
            os.path.join(data_dir, "test.ts")
        )

        print(X_train_all.shape, y_train.shape, X_test_all.shape, y_test.shape)
        print(np.unique(y_train))

        for i in range(len(feat_group)):
            if flag and i < 1:
                continue

            feat_group_train = []
            for j in range(len(feat_group)):
                if i != j:
                    feat_group_train += feat_group[j]
            print(feat_group_train)

            save_dir = "results/swdd-7k_model_500_0_remove_feat_group_{}".format(i)

            print(data_dir, save_dir)

            # ablation
            X_train = X_train_all[feat_group_train]
            X_test = X_test_all[feat_group_train]

            for cls in model_list:
                if flag and cls != "hc2":
                    continue
                if flag:
                    flag = 0
                print(cls)
                if cls == "bst":
                    try:
                        clf = train_ShapeletTransform(
                            X_train, y_train, X_test, y_test
                        )  # need at least one array to concatenate
                    except Exception as e:
                        print(e)
                elif cls == "rocket":
                    clf = train_ROCKETClassifier(X_train, y_train, X_test, y_test)
                elif cls == "gs":
                    # try:
                    #     clf = train_TSFresh(
                    #         X_train, y_train, X_test, y_test
                    #     )
                    # except Exception as e:
                    #     print(e)
                    try:
                        # clf = train_Signature(X_train, y_train, X_test, y_test)
                        clf = train_Arsenal(X_train, y_train, X_test, y_test)
                    except Exception as e:
                        print(e)
                elif cls == "hc2":
                    try:
                        clf = train_HIVECOTEV2(
                            X_train, y_train, X_test, y_test
                        )  # not work
                    except Exception as e:
                        print(e)
                elif cls == "tsf":
                    clf = train_TimeSeriesForest(X_train, y_train, X_test, y_test)

                # # analyze model
                report = calc_metrics_binary(clf, X_test, y_test)

                res_save_path = os.path.join(save_dir, cls + ".txt")
                with open(res_save_path, "a+") as f:
                    f.write(report)
                    f.write("\n" + "*" * 15 + "\n")

