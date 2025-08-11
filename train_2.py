from tensorflow.keras import layers, models
import tensorflow as tf
import pathlib

OUTDIR = pathlib.Path.cwd() / "model"
saved_model_dir = OUTDIR / "saved_model"

CLASS_ORDER = ['A', 'B', 'C', 'D', "E", 'F', 'G', "H", "I", 'J', "K", "L", 'M', 'N', "Nothing", "O", "P", "Q", "R", "S", "Space", "T", "U", "V", 'W', 'X', 'Y', "Z"]
num_classes = len(CLASS_ORDER)

inputs = layers.Input(shape=(160, 160, 3), name="image")
x = layers.Rescaling(1.0/127.5, offset=-1)(inputs)

base = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights=None,  # We'll load weights manually
    alpha=1
)
base.trainable = True  # Fine-tuned already

x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
model = models.Model(inputs, outputs)

# Load weights from your trained model
model.load_weights(OUTDIR / "best_stage2.keras")

# Export to SavedModel format
model.export(saved_model_dir)
print("Model exported to SavedModel format.")

converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
tflite_float = converter.convert()
(OUTDIR / "model_float.tflite").write_bytes(tflite_float)
print("Saved Float TFLite model.")


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

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(512).prefetch(AUTOTUNE)


def rep_dataset(ds, limit=300):
    i = 0
    for img, _ in ds.unbatch():
        yield [tf.cast(tf.expand_dims(img, 0), tf.float32)]
        i += 1
        if i >= limit:
            break

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = lambda: rep_dataset(train_ds, limit=300)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_int8 = converter.convert()
(OUTDIR / "model_int8.tflite").write_bytes(tflite_int8)
print("Saved INT8 TFLite model.")
