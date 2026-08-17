import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

N_SAMPLES = 2000
EPSILONS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
PGD_STEPS = 10

X_test = np.load("data/X_test.npy").astype("float32")[:N_SAMPLES]
y_test = np.load("data/y_test.npy")[:N_SAMPLES]
target_names = joblib.load("data/label_encoder.pkl").classes_.tolist()
n_classes = len(target_names)
model = load_model("models/Model.keras")
loss_fn = SparseCategoricalCrossentropy()


def fgsm_attack(x, y, epsilon):
    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = loss_fn(y, model(x, training=False))
    grad = tape.gradient(loss, x)
    x_adv = x + epsilon * tf.sign(grad)
    return tf.clip_by_value(x_adv, 0.0, 1.0)


def pgd_attack(x, y, epsilon, steps=PGD_STEPS):
    x_orig = tf.convert_to_tensor(x)
    alpha = epsilon / steps * 2.5
    x_adv = tf.identity(x_orig)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            loss = loss_fn(y, model(x_adv, training=False))
        grad = tape.gradient(loss, x_adv)
        x_adv = x_adv + alpha * tf.sign(grad)
        x_adv = tf.clip_by_value(x_adv, x_orig - epsilon, x_orig + epsilon)
        x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
    return x_adv


def accuracy_from_preds(preds, y_true):
    return float((preds.argmax(axis=1) == y_true).mean())


def per_class_accuracy(preds, y_true, n_classes):
    correct = preds.argmax(axis=1) == y_true
    return np.array([
        correct[y_true == c].mean() if (y_true == c).any() else np.nan
        for c in range(n_classes)
    ])


fgsm_acc, pgd_acc = [], []
fgsm_per_class = np.zeros((len(EPSILONS), n_classes))
pgd_per_class = np.zeros((len(EPSILONS), n_classes))

for idx, epsilon in enumerate(EPSILONS):
    if epsilon == 0.0:
        fgsm_preds = pgd_preds = model.predict(X_test, verbose=0)
    else:
        fgsm_preds = model.predict(fgsm_attack(X_test, y_test, epsilon).numpy(), verbose=0)
        pgd_preds = model.predict(pgd_attack(X_test, y_test, epsilon).numpy(), verbose=0)

    fgsm_acc.append(accuracy_from_preds(fgsm_preds, y_test))
    pgd_acc.append(accuracy_from_preds(pgd_preds, y_test))
    fgsm_per_class[idx] = per_class_accuracy(fgsm_preds, y_test, n_classes)
    pgd_per_class[idx] = per_class_accuracy(pgd_preds, y_test, n_classes)
    print(f"epsilon={epsilon}: FGSM acc={fgsm_acc[-1]:.3f}  PGD acc={pgd_acc[-1]:.3f}")

plt.figure(figsize=(6, 5))
plt.plot(EPSILONS, fgsm_acc, 'o-', label='FGSM')
plt.plot(EPSILONS, pgd_acc, 'o-', label=f'PGD ({PGD_STEPS} steps)')
plt.xlabel('epsilon (L-inf perturbation)')
plt.ylabel('accuracy')
plt.title('Adversarial robustness')
plt.legend()
plt.tight_layout()
plt.savefig('Results/adversarial_robustness.png')
print("Saved adversarial robustness plot to Results/adversarial_robustness.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, per_class, title in [
    (axes[0], fgsm_per_class, 'FGSM'),
    (axes[1], pgd_per_class, f'PGD ({PGD_STEPS} steps)'),
]:
    for c, name in enumerate(target_names):
        ax.plot(EPSILONS, per_class[:, c], 'o-', label=name)
    ax.set_xlabel('epsilon (L-inf perturbation)')
    ax.set_title(title)
axes[0].set_ylabel('per-class accuracy')
axes[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.savefig('Results/adversarial_per_class.png')
print("Saved per-class adversarial accuracy plot to Results/adversarial_per_class.png")
