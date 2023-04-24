import tensorflow as tf
from keras import layers
import keras_nlp
import keras
from preprocess import train_ds, val_ds


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    #Restrict Tensorflow to only allocate 7gb of memory on the first GPU
   try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7172)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
   except RuntimeError as e:
       #virtual devices must be set before GPUs have been initialized
        print(e)



VOCAB_SIZE = 30000
MAX_SEQ = 512
EMB_DIM = 3072
INTER_DIM = 512
NUM_HEAD = 12
NUM_LAYERS = 12


inputs = train_ds

input = layers.Input(shape=(None,))

encoding = keras_nlp.layers.TokenAndPositionEmbedding(vocabulary_size=VOCAB_SIZE, sequence_length=MAX_SEQ, embedding_dim=EMB_DIM, mask_zero=True)

x = encoding(input)

for x in range(NUM_LAYERS):
    decoder = keras_nlp.layers.TransformerDecoder(intermediate_dim=INTER_DIM, num_heads=NUM_HEAD, dropout=0.1) 
    x = decoder(x)


outputs = layers.Dense(VOCAB_SIZE)(x)
model = keras.Model(input=inputs, outputs=outputs)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)
model.compile(optimizer="adam", loss=loss, metrics=[metric])

model.fit(train_ds, validation_data=val_ds, verbose=2, epochs=1)