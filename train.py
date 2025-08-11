import argparse, pathlib, os
import tensorflow as tf

from tensorflow.keras import layers, models

parser = argparse.ArgumentParser()
args = parser.parse_args()

IMG_SIZE = (160, 160)

BATCH = 128
ALPHA = 1

OUTDIR = pathlib.Path.cwd() / "model"
OUTDIR.mkdir(parents=True, exist_ok=True)



CLASS_ORDER = ['A', 'B', 'C', 'D', "E", 'F', 'G', "H", "I", 'J', "K", "L", 'M', 'N', "Nothing", "O", "P", "Q", "R", "S", "Space", "T", "U", "V", 'W', 'X', 'Y', "Z"]

train_dir = 'C:\\Users\\snagunur\\Desktop\\ASL_Classification\\data\\Train\\'

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, 
    labels='inferred',
    label_mode='int',
    class_names=CLASS_ORDER,
    validation_split=0.2,
    subset='training',
    seed=1337,
    image_size=(160,160), 
    batch_size=128, 
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, 
    labels='inferred',
    label_mode='int',
    class_names=CLASS_ORDER,
    validation_split=0.2,
    subset='validation',
    seed=1337,
    image_size=(160,160), 
    batch_size=128, 
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(128 *4).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
num_classes = len(CLASS_ORDER)

# Model

inputs = layers.Input(shape=(160, 160, 3), name="image")
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.05)(x)
x = layers.RandomZoom(0.10)(x)
x = layers.RandomContrast(0.10)(x)
x = layers.Rescaling(1.0/127.5, offset=-1)(x)

base = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="C:\\Users\\snagunur\\Desktop\\ASL_Classification\\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5",
    alpha=1
)
base.trainable = False

x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
model = models.Model(inputs, outputs)
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(OUTDIR / "best_stage1.keras", save_best_only=True, monitor='val_accuracy'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2)
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0005),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

print("fitting")
model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks, verbose=2)

print("done fitting")

# Fine-tune

base.trainable = True
cut = int(len(base.layers)* (1- 20/100.0))
for layer in base.layers[:cut]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0005),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

callbacks_ft = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(OUTDIR / "best_stage2.keras", save_best_only=True, monitor='val_accuracy'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2)
]

print("we fitting again")
model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks_ft, verbose=2)

saved_model_dir = OUTDIR / "saved_model"

model.save(OUTDIR / "model.keras")
print("saving")

converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
tflite_float = converter.convert()
(OUTDIR/ "model_float.tflite").write_bytes(tflite_float)
print("Saved Float Tflite")

def rep_dataset(ds, limit=300):
    i = 0
    for img, _ in ds.unbatch():
        yield [tf.cast(tf.expland_dims(img,0), tf.float32)]
        i+=1
        if i >= limit:
            break

converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

converter.optimizations = tf.lite.Optimize.DEFAULT
converter.representative_dataset = lambda: rep_dataset(train_ds, limit=300)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_int8 = converter.convert()

(OUTDIR / "model_int8.tflite").write_bytes(tflite_int8)

with open(OUTDIR / "labels.txt", "w") as f:
    for c in CLASS_ORDER:
        f.write(c+"\n")
print("labels saved")