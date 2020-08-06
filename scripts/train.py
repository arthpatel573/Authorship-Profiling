import numpy as np 
import pandas as pd
import datetime

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline 

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Bidirectional
from tensorflow.keras.layers import Dense, Activation, Reshape, Input, Dropout
from tensorflow.keras.layers import Embedding, BatchNormalization

EPOCHS = 20
WORD_INDEX = None

def prepare_data():
    '''
    Prepare dataset with train test split
    for training Deep Network classifier
    '''
    global WORD_INDEX

    train_df = pd.read_csv('../training_data_dl.csv')
    train_df['gender'] = train_df['gender'].apply(lambda x: 1 if x=='male' else 0)

    Xml_train, Xml_test, y_train, y_test = train_test_split(xml_df['id'].values, xml_df['gender'].values,
                                                            random_state=123,
                                                            shuffle=True, 
                                                            test_size=0.2,
                                                            stratify=xml_df['gender'].values)

    train_df.dropna(subset=['document'], inplace=True)

    # get the maximum size of document to get an estimate of padding at a later stage
    train_df['document'].apply(lambda x:len(str(x).split())).max()


    xtrain = train_df.loc[train_df['id'].isin(Xml_train),'document'].values
    ytrain = train_df.loc[train_df['id'].isin(Xml_train),'gender'].values

    xvalid = train_df.loc[~train_df['id'].isin(Xml_train),'document'].values
    yvalid = train_df.loc[~train_df['id'].isin(Xml_train),'gender'].values

    # using keras tokenizer here
    token = text.Tokenizer(num_words=None)

    token.fit_on_texts(list(xtrain) + list(xvalid))
    xtrain_seq = token.texts_to_sequences(xtrain)
    xvalid_seq = token.texts_to_sequences(xvalid)

    #zero pad the sequences
    xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
    xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

    WORD_INDEX = token.word_index

    return xtrain_pad, ytrain, xvalid_pad, yvalid 


class Classifier(Model):
    def __init__(self, vocab_size, embedding_dim, input_length, rate=0.2):
        '''
        Classifier that predict the gender based on the text
        information from tweets

        Args:
            vocab_size: size of vocabulary
                dtype: int
            embedding_dim: dimension for embeddings
                dtype: int
            input_length: length of input
                dtype: int
            rate: dilation rate for layers
                default: 0.2
                dtype: float32  
        '''
        super(Classifier, self).__init__()

        # Custom Embedding layer
        self.embedding = Embedding(
            vocab_size, embedding_dim, input_length = input_length
        )

        # Bidirectional LSTM layers with different size
        self.bidirectional_128 = Bidirectional(
            LSTM(128, dropout=rate, return_sequences=True)
        )
        self.bidirectional_64 = Bidirectional(
            LSTM(64, dropout=rate, return_sequences=True)
        )

        # Dense layers 
        self.fc_32 = Dense(32, activation = 'relu') 
        self.fc_16 = Dense(16, activation = 'relu')
        self.fc_1 = Dense(1, activation = 'sigmoid')
        
        # dropout layers
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(0.4)
    
    def call(self, x):
        # embedding input layer
        x = self.embedding(x)

        # bidirectional lstm layers
        x = self.bidirectional_128(x)
        x = self.bidirectional_64(x)

        # fully connected dense layers
        x = self.fc_32(x)
        x = self.dropout1(x)
        x = self.fc_16(x)
        x = self.dropout2(x)
        x = self.fc_1(x)

        return x

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=2000):
        '''
        Custom scheduler with 2000 warmup steps decreasing
        learning rate at rate of 1.5
        '''
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        

if __name__ = '__main__':
    # prepare dataset
    xtrain_pad, ytrain, xvalid_pad, yvalid = prepare_data()

    # instantiate gender classifier
    gender_Classifier = Classifier(
        vocab_size = len(WORD_INDEX) + 1, embedding_dim = 300, input_length = 45
    )

    learning_rate = CustomSchedule(d_model)

    # optimizer with learning rate scheduler
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.85, beta_2=0.97, 
                                        epsilon=1e-7)

    # loss object function
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    gender_Classifier.compile(loss=loss_object, optimizer=optimizer, metrics=['accuracy'])

    gender_Classifier.fit(x_train, y_train, epochs=1)

    # train the model
    gender_Classifier.fit(xtrain_pad, ytrain, epochs=EPOCHS, batch_size=128, 
                    validation_data=(xvalid_pad, yvalid),
                    verbose=1)
