
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op

from sklearn.utils.class_weight import compute_class_weight

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, glob, pathlib, random

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

i16min, i16max = -2**15, 2**15 - 1
fsamp = 16000
wave_length_samps = 16000

window_size_ms = 30
window_step_ms = 20
num_filters = 40

EPOCHS = 40
batch_size = 32

commands = ['dog', 'basketball']
label_list = ['_silence', '_unknown'] + commands
label_to_id = {l:i for i,l in enumerate(label_list)}


#data_dir = pathlib.Path(os.getcwd()) / 'data' / 'mini_speech_commands'
data_dir = pathlib.Path(os.getcwd()) / 'data' 


standard_files = []
for d in glob.glob(os.path.join(str(data_dir), '*')):
    if not os.path.isdir(d): 
        continue
    if '_background_noise_' in d:
        continue

    files = glob.glob(os.path.join(d, '*.wav'))
    random.shuffle(files)
    standard_files.extend(files[:120])
max_unknown = 800  

unknown_files = []
command_files = []

for f in standard_files:
    label = os.path.basename(os.path.dirname(f)).lower()
    if label in commands:
        command_files.append(f)
    else:
        unknown_files.append(f)

random.shuffle(unknown_files)
unknown_files = unknown_files[:max_unknown]

standard_files = command_files + unknown_files

print(f"After undersampling: {len(command_files)} command, {len(unknown_files)} unknown")

basketball_dir = os.path.join(os.getcwd(), 'samples_audio_basketball')
basketball_files = glob.glob(os.path.join(basketball_dir, '*.wav'))

random.shuffle(standard_files)
random.shuffle(basketball_files)

def split(files):
    n = len(files)
    return files[:int(0.8*n)], files[int(0.8*n):int(0.9*n)], files[int(0.9*n):]

std_train, std_val, std_test = split(standard_files)
bsk_train, bsk_val, bsk_test = split(basketball_files)

train_files = std_train + bsk_train
val_files   = std_val + bsk_val
test_files  = std_test + bsk_test

def get_label(file_path):
    path = tf.strings.lower(file_path)

    is_bsk = tf.strings.regex_full_match(path, ".*basketball.*")
    folder = tf.strings.split(file_path, os.path.sep)[-2]
    is_cmd = tf.reduce_any(tf.equal(folder, commands))

    return tf.case([
        (is_bsk, lambda: tf.constant('basketball')),
        (is_cmd, lambda: folder),
    ], default=lambda: tf.constant('_unknown'))

def add_noise(waveform, noise_factor=0.0001):
    noise = tf.random.normal(shape=tf.shape(waveform), mean=0.0, stddev=1.0)
    return waveform + noise_factor * noise

def random_gain(waveform, min_gain=0.1, max_gain=0.2):
    gain = tf.random.uniform([], min_gain, max_gain)
    return waveform * gain

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def augment_waveform(waveform, label):
    waveform = tf.cast(waveform, tf.float32)

    waveform = add_noise(waveform)
    waveform = random_gain(waveform)

    return waveform, label

def get_waveform_and_label(file_path):
    return decode_audio(tf.io.read_file(file_path)), get_label(file_path)

def get_spectrogram(waveform):
    waveform = waveform[:wave_length_samps]
    padding = tf.zeros([wave_length_samps] - tf.shape(waveform), dtype=tf.int16)

    waveform = tf.cast(0.5 * waveform * (i16max - i16min), tf.int16)
    waveform = tf.concat([waveform, padding], 0)

    return frontend_op.audio_microfrontend(
        waveform,
        sample_rate=fsamp,
        num_channels=num_filters,
        window_size=window_size_ms,
        window_step=window_step_ms
    )

def create_silence(n=200):
    waves = np.random.normal(0, 0.02, (n, wave_length_samps)).astype(np.float32)
    labels = ['_silence'] * n
    return tf.data.Dataset.from_tensor_slices((waves, labels))

def wavds2specds(ds):
    specs, labels = [], []

    for wav, label in ds:
        spec = get_spectrogram(wav)
        spec = tf.expand_dims(tf.expand_dims(spec, 0), -1)

        specs.append(spec)

        label_str = label.numpy().decode()
        labels.append(label_to_id[label_str])

    return tf.data.Dataset.from_tensor_slices(
        (np.vstack(specs), np.array(labels))
    )

def preprocess(files):
    ds = tf.data.Dataset.from_tensor_slices(files)

    ds = ds.map(get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(augment_waveform, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.concatenate(create_silence(150))

    return wavds2specds(ds)

train_ds = preprocess(train_files).shuffle(2000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds   = preprocess(val_files).batch(batch_size)
test_ds  = preprocess(test_files).batch(batch_size)

for spec, _ in train_ds.take(1):
    input_shape = spec.shape[1:]

print("Input shape:", input_shape)

model = models.Sequential([
    layers.Input(shape=input_shape),

    layers.Conv2D(16, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((1,2)),
    layers.BatchNormalization(),

    layers.SeparableConv2D(32, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.SeparableConv2D(32, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),

    layers.Dropout(0.2),
    layers.GlobalMaxPooling2D(),

    layers.Dense(len(label_list))
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)


model.summary()
y_train = []
for _, label in train_ds.unbatch():
    y_train.append(label.numpy())

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))
class_weights = {k: min(v, 5) for k, v in class_weights.items()}

print("Class weights:", class_weights)

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=class_weights)

model.save("kws_model.h5")

def collect_preds(ds):
    y_true, y_pred = [], []

    for audio, label in ds.unbatch():
        pred = model.predict(np.expand_dims(audio.numpy(), 0), verbose=0)
        y_pred.append(np.argmax(pred))
        y_true.append(label.numpy())

    return np.array(y_true), np.array(y_pred)


def compute_cm(y_true, y_pred):
    return tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=len(label_list)
    ).numpy()


def print_metrics(cm, name):
    print(f"\n{name} Metrics")
    print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'Support':<8}")
    print("-" * 55)

    for i, label in enumerate(label_list):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"{label:<12} | {precision:<10.2f} | {recall:<10.2f} | {support:<8}")


def evaluate(ds, name):
    y_true, y_pred = collect_preds(ds)
    cm = compute_cm(y_true, y_pred)

    acc = np.mean(y_true == y_pred)

    print(f"\n{name} Accuracy: {acc:.4f}")
    print_metrics(cm, name)

    return cm, acc


cm_train, train_acc = evaluate(train_ds, "Training")
cm_val, val_acc     = evaluate(val_ds, "Validation")
cm_test, test_acc   = evaluate(test_ds, "Test")


def plot_cm(cm, title, filename):
    plt.figure(figsize=(7,6))
    sns.heatmap(
        cm,
        xticklabels=label_list,
        yticklabels=label_list,
        annot=True,
        fmt='g',
        cmap='Blues'
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


plot_cm(cm_train, "Training Confusion Matrix", "train_cm.png")
plot_cm(cm_val, "Validation Confusion Matrix", "val_cm.png")
plot_cm(cm_test, "Test Confusion Matrix", "test_cm.png")


print("\nFINAL SUMMARY")
print(f"Training Accuracy:   {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy:       {test_acc:.4f}")

def print_fpr_frr(cm, name):
    print(f"\n{name} FPR / FRR")

    for i, label in enumerate(label_list):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - (tp + fp + fn)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0

        print(f"{label:<12} | FPR: {fpr:.3f} | FRR: {frr:.3f}")

print_fpr_frr(cm_train, "Training")

print_fpr_frr(cm_val, "Validation")

print_fpr_frr(cm_test, "Test")


plt.figure()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("accuracy_plot.png")
plt.show()