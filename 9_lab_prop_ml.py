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

    flag = 1
    for r in range(9):
        for i in range(10, 100, 10):
            # TODO:
            if flag and i < 10:
                continue

            data_dir = "dataset/swdd-4k_{}_ts_origin_500_0".format(i)
            save_dir = "results/swdd-4k_{}_model_500_0".format(i)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            print(data_dir, save_dir)

            # load data
            X_train, y_train = load_from_tsfile_to_dataframe(
                os.path.join(data_dir, "train.ts")
            )
            X_test, y_test = load_from_tsfile_to_dataframe(
                os.path.join(data_dir, "test.ts")
            )

            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            print(np.unique(y_train))

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
