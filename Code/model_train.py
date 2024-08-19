import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from getData import *
from DProcess import convertRawToXY
from capsule_keras import *
from assessment import *

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from keras.layers import merge
import keras.layers.convolutional as conv
from tensorflow.keras import backend as K, regularizers
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from  scipy import interpolate
from numpy import interp
from tensorflow.keras.callbacks import ModelCheckpoint


seq_len = 69
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.70)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# Load specific-specious data
oneofkey_pos, oneofkey_neg = get_data_oneofk_stru(r'../data/structure_data/Toxoplasma gondii.txt', winnum=seq_len // 2, label=True)
# Shulffle
oneofkey_pos, oneofkey_neg = shuffle_(oneofkey_pos, oneofkey_neg)
# Pos concat neg
data_oneofk = pd.concat([oneofkey_pos, oneofkey_neg]) 
data_oneofk = shuffle_PosNeg(data_oneofk)
data_oneofk = pd.DataFrame(data_oneofk)
data_seq = data_oneofk.iloc[:, :70]
data_struc = pd.concat([data_oneofk.iloc[:, 0], data_oneofk.iloc[:, 70:139]], axis=1)
x_seq, y_seq = convertRawToXY(np.array(data_seq), codingMode=0)
x_stru, y_stru = convertRawToXY(np.array(data_struc), codingMode=6)
x_seq = x_seq.reshape(x_seq.shape[0], x_seq.shape[2], x_seq.shape[3])
x_stru = x_stru.reshape(x_stru.shape[0], x_stru.shape[2], x_stru.shape[3])


X_seqence_train, X_seqence_test, Y_seqence_train, Y_seqence_test = train_test_split(x_seq, y_seq, test_size=0.1, random_state=1)
X_structure_train, X_structure_test, Y_structure_train, Y_structure_test = train_test_split(x_stru, y_stru, test_size=0.1, random_state=1)


checkpoint_filepath = 'Toxo.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

def expand_dim_backend(x):
    x1 = K.reshape(x, (-1, 1, 256))
    return x1


def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
            1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)


def mul_model():
    optimizer = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)
    input_seq = Input(shape=(seq_len, 21))  # sequence 
    input_stru = Input(shape=(seq_len, 4))  # structure 

    # seq feature extraction
    x_seq = conv.Conv1D(filters=256, kernel_size=7,
                        activation='relu',
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(0),
                        )(input_seq)
    x_seq = conv.Conv1D(filters=256, kernel_size=11,
                        activation='relu',
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(0),
                        )(x_seq)
    x_seq = Dropout(0.3)(x_seq)
    squeeze_seq = GlobalAveragePooling1D()(x_seq)
    squeeze_seq = Lambda(expand_dim_backend)(squeeze_seq)
    # squeeze = Lambda(expand_dim_backend)(o_x)
    excitation_seq = Conv1D(filters=256, kernel_size=1, strides=1, padding='valid', activation='relu')(squeeze_seq)
    excitation_seq = Conv1D(filters=256, kernel_size=1, strides=1, padding='valid', activation='sigmoid')(excitation_seq)
    x_seq_md = Lambda(multiply)([x_seq, excitation_seq])
    print(x_seq_md.shape)
    seq = Model(inputs=input_seq, outputs=x_seq_md)

    # stru feature extraction
    x_stru = conv.Conv1D(filters=256, kernel_size=7,
                        activation='relu',
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(0),
                        )(input_stru)
    x_stru = conv.Conv1D(filters=256, kernel_size=11,
                        activation='relu',
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(0),
                        )(x_stru)
    x_stru = Dropout(0.3)(x_stru)
    squeeze_stru = GlobalAveragePooling1D()(x_stru)
    squeeze_stru = Lambda(expand_dim_backend)(squeeze_stru)

    excitation_stru = Conv1D(filters=256, kernel_size=1, strides=1, padding='valid', activation='relu')(squeeze_stru)
    excitation_stru = Conv1D(filters=256, kernel_size=1, strides=1, padding='valid', activation='sigmoid')(
        excitation_stru)
    x_stru_md = Lambda(multiply)([x_stru, excitation_stru])
    print(x_stru_md.shape)
    stru = Model(inputs=input_stru, outputs=x_stru_md)

    x_seq_stru_md = concatenate([seq.output, stru.output], axis=2)  
    print(x_seq_stru_md.shape)

    out = Capsule(1, 10, 3, True)(x_seq_stru_md)
    output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(1,))
    out = output(out)

    cnn_caps = Model(inputs=[seq.input, stru.input], outputs=out)
    cnn_caps.compile(loss=margin_loss, optimizer=optimizer, metrics=["accuracy"])
  
    return cnn_caps

avg_acc = 0
avg_sensitivity = 0
avg_specificity = 0
avg_mcc = 0
avg_f1 = 0
n_split = 10  # k-fold
time = 0
tprs = []
aucs = []
fprs = []
mean_fpr = np.linspace(0, 1, 100) 
true_label = list(Y_seqence_test)

for train_index, val_index in KFold(n_splits=n_split).split(X_seqence_train):
    X_seq_train, X_seq_val = X_seqence_train[train_index], X_seqence_train[val_index]
    Y_seq_train, Y_seq_val = Y_seqence_train[train_index], Y_seqence_train[val_index]

    X_stru_train, X_stru_val = X_structure_train[train_index], X_structure_train[val_index]
    Y_stru_train, Y_stru_val = Y_structure_train[train_index], Y_structure_train[val_index]

    models = mul_model()
    history = models.fit([X_seq_train, X_stru_train], Y_seq_train, batch_size=40, epochs=45,
                         validation_data=([X_seq_val, X_stru_val], Y_seq_val),
                         verbose=1,callbacks=[model_checkpoint_callback])
    pred_proba = models.predict([X_seqence_test, X_structure_test], batch_size=40)
    pred_class = []
    for i in pred_proba:
        if i >= 0.5:
            i = 1
            pred_class.append(i)
        else:
            i = 0
            pred_class.append(i)
    
    acc, sensitivity, specificity, mcc, f1 = calculate_performance(len(X_seqence_test), true_label, pred_class)
    
    # for avg calculate
    avg_acc += acc
    avg_sensitivity += sensitivity
    avg_specificity += specificity
    avg_mcc += mcc
    avg_f1 += f1
    
    # AUC 
    fpr, tpr, thresholds = roc_curve(true_label, pred_proba)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    print('AUC:', roc_auc)
    with open("structure_result/Toxoplasma-0715.csv", "a+") as f:
        f.write('length:' + str(seq_len) + ' acc:' + str(acc) + '  sen:' + str(sensitivity) + '  spe:' + str(
            specificity) + '  mcc:' + str(mcc) + '  f1:' + str(f1) + '  AUC:' + str(roc_auc) + '\n')
