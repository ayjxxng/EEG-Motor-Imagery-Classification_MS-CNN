import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from keras.utils import to_categorical


def run_classification(train_data, DE_input, NPS_input, train_labels,
                       val_data, val_DE, val_NPS, val_labels):
    kf = KFold(n_splits=10, shuffle=True) #splits=10
    classification_acc = pd.DataFrame()
    batch_size = 16 #16
    epochs = 500 #500

    train_labels = np.array(train_labels).flatten()
    train_labels -= 1
    train_labels = to_categorical(train_labels, num_classes=2)

    val_labels = np.array(val_labels).flatten()
    val_labels -= 1
    val_labels = to_categorical(val_labels, num_classes=2)

    # kflod
    for fold, (train_index, test_index) in enumerate(kf.split(train_data)):
        print(f"Fold: {fold + 1}")

        # load data and labels
        x_train, x_test = train_data[train_index], train_data[test_index]
        x_train_DE, x_test_DE = DE_input[train_index], DE_input[test_index]
        x_train_NPS, x_test_NPS = NPS_input[train_index], NPS_input[test_index]
        y_train, y_test = train_labels[train_index], train_labels[test_index]


        # CNN train
        print('Training CNN ------------')
        model_CNN = build_model_CNN()
        hist_CNN = model_CNN.fit([x_train], y_train, validation_data=(x_test, y_test),
                              batch_size=batch_size, epochs=epochs, verbose=1)
        print(f"CNN_Fold{fold + 1}_loss: {hist_CNN.history['val_loss'][-1]:.4f}")
        print(f"CNN_Fold{fold + 1}_accuracy: {hist_CNN.history['val_accuracy'][-1]:.4f}")


        # model_0 train
        print('Training model_0 ------------')
        model_0 = build_model_0()
        hist_0 = model_0.fit([x_train], y_train, validation_data=(x_test, y_test),
                              batch_size=batch_size, epochs=epochs, verbose=1)
        print(f"model_0_Fold{fold + 1}_loss: {hist_0.history['val_loss'][-1]:.4f}")
        print(f"model_0_Fold{fold + 1}_accuracy: {hist_0.history['val_accuracy'][-1]:.4f}")


        # model_1 train
        print('Training model_1 ------------')
        model_1 = build_model_1()
        hist_1 = model_1.fit([x_train, x_train_DE], y_train, validation_data=([x_test, x_test_DE], y_test),
                              batch_size=batch_size, epochs=epochs, verbose=1)
        print(f"model_1_Fold{fold + 1}_loss: {hist_1.history['val_loss'][-1]:.4f}")
        print(f"model_1_Fold{fold + 1}_accuracy: {hist_1.history['val_accuracy'][-1]:.4f}")


        # model_2 train
        print('Training model_2  ------------')
        model_2 = build_model_2()
        hist_2 = model_2.fit([x_train, x_train_NPS], y_train, validation_data=([x_test, x_test_NPS], y_test),
                             batch_size=batch_size, epochs=epochs, verbose=1)
        print(f"model_2_Fold{fold + 1}_loss: {hist_2.history['val_loss'][-1]:.4f}")
        print(f"model_2_Fold{fold + 1}_accuracy: {hist_2.history['val_accuracy'][-1]:.4f}")


        # model_3 train
        print('Training model_3 ------------')
        model_3 = build_model_3()
        hist_3 = model_3.fit([x_train, x_train_DE, x_train_NPS], y_train, validation_data=([x_test, x_test_DE, x_test_NPS], y_test),
                             batch_size=batch_size, epochs=epochs, verbose=1)
        print(f"model_3_Fold{fold + 1}_loss: {hist_3.history['val_loss'][-1]:.4f}")
        print(f"model_3_Fold{fold + 1}_accuracy: {hist_3.history['val_accuracy'][-1]:.4f}")



    result_CNN, roc_CNN = validate_model_CNN(model_CNN, val_data, val_labels)
    print(f"CNN Validation Loss: {result_CNN[0]:.4f}, CNN Validation Accuracy: {result_CNN[1]:.4f}, CNN Validation F1: {result_CNN[2]:.4f}")

    result_0, roc_0 = validate_model_0(model_0, val_data, val_labels)
    print(f"Model_0 Validation Loss: {result_0[0]:.4f}, Model_0 Validation Accuracy: {result_0[1]:.4f}, Model_0 Validation F1: {result_0[2]:.4f}")

    result_1, roc_1 = validate_model_1(model_1, val_data, val_DE, val_labels)
    print(f"Model_1 Validation Loss: {result_1[0]:.4f}, Model_1 Validation Accuracy: {result_1[1]:.4f}, Model_1 Validation F1: {result_1[2]:.4f}")

    result_2, roc_2 = validate_model_2(model_2, val_data, val_NPS, val_labels)
    print(f"Model_2 Validation Loss: {result_2[0]:.4f}, Model_2 Validation Accuracy: {result_2[1]:.4f}, Model_2 Validation F1: {result_2[2]:.4f}")

    result_3, roc_3 = validate_model_3(model_3, val_data, val_DE, val_NPS, val_labels)
    print(f"Model_3 Validation Loss: {result_3[0]:.4f}, Model_3 Validation Accuracy: {result_3[1]:.4f}, Model_3 Validation F1: {result_3[2]:.4f}")

    return model_CNN, model_0, model_1, model_2, model_3, result_CNN, result_0, result_1, result_2, result_3, hist_CNN, hist_0, hist_1, hist_2, hist_3, roc_CNN, roc_0, roc_1, roc_2, roc_3