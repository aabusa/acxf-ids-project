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


def accuracy(x):
    preds = model.predict(x, verbose=0)
    return float((preds.argmax(axis=1) == y_test).mean())


fgsm_acc, pgd_acc = [], []
for epsilon in EPSILONS:
    if epsilon == 0.0:
        fgsm_acc.append(accuracy(X_test))
        pgd_acc.append(fgsm_acc[-1])
        continue
    fgsm_acc.append(accuracy(fgsm_attack(X_test, y_test, epsilon).numpy()))
    pgd_acc.append(accuracy(pgd_attack(X_test, y_test, epsilon).numpy()))
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
