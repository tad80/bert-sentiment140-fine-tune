import nltk
from nltk import word_tokenize
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds
from tqdm.auto import tqdm
from transformers import BertTokenizerFast
from transformers import TFBertForSequenceClassification, TFTrainer, TFTrainingArguments
from transformers import logging as hf_logging
from sklearn.metrics import classification_report
from official.nlp import optimization

hf_logging.set_verbosity_info()
nltk.download('punkt')

PREPROCESSOR_NAME = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
NUM_WARMUP_STEPS = 500
LEARNING_RATE = 1e-5
EPOCHS = 2
MODEL = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2'
BATCH_SIZE = 64
OUTPUT_DIR='gs://sdm-research-tweets/bert-output/huggingface_sentiment140_v1/'

# Modified From: https://github.com/ForeverZyh/certified_lstms/blob/81ca8c66c6e0d1a15f8abcd513e9370bd95dfb8b/src/text_classification.py
def prepare_ds(ds):
    text_list = []
    label_list = []
    num_pos = 0
    num_neg = 0
    num_words = 0
    for features in tqdm(tfds.as_numpy(ds), total=len(ds)):
        sentence, label = features["text"], 1 if features["polarity"] == 4 else 0
        tokens = word_tokenize(sentence.decode('UTF-8').lower())
        text_list.append(' '.join(tokens))
        label_list.append(label)
        num_pos += label == 1
        num_neg += label == 0
        num_words += len(tokens)

    avg_words = num_words / len(text_list)
    print('Read %d examples (+%d, -%d), average length %d words' % (
        len(text_list), num_pos, num_neg, avg_words))
    return tf.data.Dataset.from_tensor_slices((text_list, label_list))

# Load from Tensorflow Datasets
train, valid = tfds.load(
    name="sentiment140",
    with_info=False,
    split=['train', 'test']
)

print()
print('Building Training Data')
train_dataset = prepare_ds(train).shuffle(1000).batch(BATCH_SIZE)
print()
print('Building Validation Data')
valid_dataset = prepare_ds(valid).batch(BATCH_SIZE)
print()


model_output_dir='gs://sdm-research-tweets/saved_models/huggingface_sentiment140_tf_v1/'

def build_classifier_model(train_dataset):
    text_input = tf.keras.layers.Input(shape=[], dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(PREPROCESSOR_NAME, name='preprocessing', )
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(MODEL, trainable=True, name='BERT_encoder', )
    outputs = encoder(encoder_inputs)

    net = outputs['pooled_output']
    net = tf.keras.layers.Dense(
                int(256),
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.002),
                activation="relu",
                name="pre_classifier"
            )(net)

    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(EPOCHS, activation="sigmoid", use_bias=True, name='classifier')(net)
    model = tf.keras.Model(text_input, net, name='sentiment_classification')

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
    num_train_steps = steps_per_epoch * EPOCHS
    num_warmup_steps = NUM_WARMUP_STEPS or int(0.1*num_train_steps)

    optimizer = optimization.create_optimizer(init_lr=LEARNING_RATE,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    model.compile(optimizer=optimizer,
                            loss=loss,
                            metrics=['accuracy'])

    model.summary()
    return model



model = build_classifier_model(train_dataset)

callbacks = [ModelCheckpoint(filepath=OUTPUT_DIR, 
                             verbose=1,
                             save_freq='epoch',
                             monitor='val_accuracy',
                             save_best_only=True, 
                             mode='max', 
                             save_weights_only=True),
             EarlyStopping(patience=3, monitor='val_loss', mode='min')
            ]


# Training model...
history = model.fit(train_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks, validation_data=valid_dataset)

model = build_classifier_model(train_dataset)
model.load_weights(OUTPUT_DIR)

tf.saved_model.save(
    model,
    model_output_dir,
    signatures=None
)

