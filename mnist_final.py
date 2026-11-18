"""MNIST assignment: original MLP + one CNN + visualizations + my handwriting.

Run: python mnist_final_simple.py
The converted handwritten_mnist.npz can be next to this script or in a data folder.
"""

from pathlib import Path
import csv
import webbrowser
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')  # Save figures even when there is no display window.
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / 'mnist_results'
RESULT_DIR.mkdir(exist_ok=True)
EPOCHS = 10

# ============================================================
# 1. Original MNIST data preparation (ex2_mnist.py)
# ============================================================
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print('Train data X:', train_dataset.data.shape)
print('Train data Y:', train_dataset.targets.shape)
print(train_dataset.data[0].shape)
print(train_dataset.targets[0].shape)

X_train = train_dataset.data.float() / 255.0
X_test = test_dataset.data.float() / 255.0
y_train = F.one_hot(train_dataset.targets, num_classes=10).float()
y_test = F.one_hot(test_dataset.targets, num_classes=10).float()

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)

# ============================================================
# 2. Original MLP (architecture and training method unchanged)
# ============================================================
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 512),
    nn.Tanh(),
    nn.Linear(512, 10),
    nn.Softmax(dim=1)
)

mlp_train_loss, mlp_test_loss, mlp_accuracy = [], [], []
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(EPOCHS):
    total_loss = 0.0

    for images, labels in train_loader:
        outputs = model(images)
        loss = -torch.sum(
            labels * torch.log(outputs.clamp_min(1e-8)), dim=1
        ).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = -torch.sum(
                labels * torch.log(outputs.clamp_min(1e-8)), dim=1
            ).mean()
            val_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            ground_truth = torch.argmax(labels, dim=1)
            correct += (predicted == ground_truth).sum().item()
            total += labels.size(0)

    val_loss /= len(test_loader)
    val_accuracy = 100 * correct / total
    mlp_train_loss.append(train_loss)  # Added only to draw graphs.
    mlp_test_loss.append(val_loss)
    mlp_accuracy.append(val_accuracy)
    print(f'MLP Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | '
          f'Test Loss: {val_loss:.4f} | Test Accuracy: {val_accuracy:.2f}%')

# Save the original MLP after the final epoch.
torch.save(model.state_dict(), RESULT_DIR / 'mlp.pth')

# ============================================================
# 3. My CNN (same nn.Sequential / training style as the MLP)
# ============================================================
model_student = nn.Sequential(
    nn.Flatten(),
    nn.Unflatten(1, (1, 28, 28)),
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)
)

cnn_train_loss, cnn_test_loss, cnn_accuracy = [], [], []
optimizer = torch.optim.Adam(model_student.parameters(), lr=0.001)
for epoch in range(EPOCHS):
    total_loss = 0.0
    model_student.train()

    for images, labels in train_loader:
        outputs = model_student(images)
        loss = -torch.sum(
            labels * torch.log(outputs.clamp_min(1e-8)), dim=1
        ).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    model_student.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model_student(images)
            loss = -torch.sum(
                labels * torch.log(outputs.clamp_min(1e-8)), dim=1
            ).mean()
            val_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            ground_truth = torch.argmax(labels, dim=1)
            correct += (predicted == ground_truth).sum().item()
            total += labels.size(0)

    val_loss /= len(test_loader)
    val_accuracy = 100 * correct / total
    cnn_train_loss.append(train_loss)
    cnn_test_loss.append(val_loss)
    cnn_accuracy.append(val_accuracy)
    print(f'CNN Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | '
          f'Test Loss: {val_loss:.4f} | Test Accuracy: {val_accuracy:.2f}%')

torch.save(model_student.state_dict(), RESULT_DIR / 'cnn.pth')

# Record final results for the PPT report.
with open(RESULT_DIR / 'comparison.csv', 'w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(['Model', 'Test accuracy (%)', 'Test loss', 'Parameters'])
    writer.writerow(['MLP', round(mlp_accuracy[-1], 2), round(mlp_test_loss[-1], 4),
                     sum(p.numel() for p in model.parameters())])
    writer.writerow(['CNN', round(cnn_accuracy[-1], 2), round(cnn_test_loss[-1], 4),
                     sum(p.numel() for p in model_student.parameters())])

# ============================================================
# 4. Clear visualization: overall accuracy and loss
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
axes[0].bar(['MLP', 'CNN'], [mlp_accuracy[-1], cnn_accuracy[-1]],
            color=['#5274cf', '#eb844e'])
axes[0].set_ylim(0, 105)
axes[0].set_title('Final test accuracy (10,000 MNIST digits)')
axes[0].set_ylabel('Accuracy (%)')
for i, acc in enumerate([mlp_accuracy[-1], cnn_accuracy[-1]]):
    axes[0].text(i, acc + 1, f'{acc:.2f}%', ha='center', weight='bold')

axes[1].plot(range(1, EPOCHS + 1), mlp_accuracy, marker='o', label='MLP')
axes[1].plot(range(1, EPOCHS + 1), cnn_accuracy, marker='o', label='CNN')
axes[1].set_title('Test accuracy by epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()

axes[2].plot(range(1, EPOCHS + 1), mlp_test_loss, marker='o', label='MLP')
axes[2].plot(range(1, EPOCHS + 1), cnn_test_loss, marker='o', label='CNN')
axes[2].set_title('Test loss by epoch')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Cross-entropy')
axes[2].legend()
for ax in axes:
    ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(RESULT_DIR / '01_model_comparison.png', dpi=160)
plt.close(fig)

# ============================================================
# 5. Compare predictions for the same MNIST test images
# ============================================================
model.eval()
model_student.eval()
test_images, labels = next(iter(test_loader))
with torch.no_grad():
    mlp_outputs = model(test_images)
    cnn_outputs = model_student(test_images)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    actual = int(labels[i].argmax())
    mlp_digit = int(mlp_outputs[i].argmax())
    cnn_digit = int(cnn_outputs[i].argmax())
    ax.imshow(test_images[i], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Actual: {actual}\nMLP: {mlp_digit} | CNN: {cnn_digit}',
                 color='green' if mlp_digit == cnn_digit == actual else 'darkred', fontsize=11)
    ax.axis('off')
fig.suptitle('Same MNIST images | MLP vs CNN', weight='bold')
fig.tight_layout()
fig.savefig(RESULT_DIR / '02_mnist_predictions.png', dpi=160)
plt.close(fig)

# ============================================================
# 6. Optional: classify my converted handwritten digits (0-9)
# ============================================================
handwriting_paths = [
    BASE_DIR / 'handwritten_mnist.npz',
    BASE_DIR / 'data' / 'handwritten_mnist.npz',
    BASE_DIR.parent / 'data' / 'handwritten_mnist.npz',
    BASE_DIR.parent.parent / 'data' / 'handwritten_mnist.npz',
    BASE_DIR / 'my_handwritten_mnist_0_to_9' / 'handwritten_mnist.npz',
]
handwriting_path = next((p for p in handwriting_paths if p.exists()), None)
handwriting_result = None

if handwriting_path is not None:
    with np.load(handwriting_path, allow_pickle=False) as data:
        my_images = data['images'].astype(np.float32)
        my_labels = data['labels'].astype(np.int64)
    if my_images.ndim != 3 or my_images.shape[1:] != (28, 28) or my_labels.shape != (len(my_images),):
        raise ValueError('Handwritten data must have images [N,28,28] and labels [N].')
    if my_images.max() > 1:
        my_images /= 255.0
    my_tensor = torch.tensor(my_images, dtype=torch.float32)

    with torch.no_grad():
        my_mlp = model(my_tensor).numpy()
        my_cnn = model_student(my_tensor).numpy()

    mlp_guess = my_mlp.argmax(axis=1)
    cnn_guess = my_cnn.argmax(axis=1)
    mlp_correct = int((mlp_guess == my_labels).sum())
    cnn_correct = int((cnn_guess == my_labels).sum())
    handwriting_result = (mlp_correct, cnn_correct, len(my_labels))

    fig, axes = plt.subplots(2, 5, figsize=(14, 6.5))
    for i, ax in enumerate(axes.flat):
        if i >= len(my_images):
            ax.axis('off')
            continue
        actual = int(my_labels[i])
        mp, cp = int(mlp_guess[i]), int(cnn_guess[i])
        mc, cc = 100 * my_mlp[i, mp], 100 * my_cnn[i, cp]
        ax.imshow(my_images[i], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'My digit: {actual}', weight='bold')
        ax.text(0.5, -0.06, f'MLP: {mp} ({mc:.1f}%)', transform=ax.transAxes,
                ha='center', color='green' if mp == actual else 'red', fontsize=10)
        ax.text(0.5, -0.16, f'CNN: {cp} ({cc:.1f}%)', transform=ax.transAxes,
                ha='center', color='green' if cp == actual else 'red', fontsize=10)
        ax.axis('off')
    fig.suptitle('My handwritten digits | MLP vs CNN', fontsize=17, weight='bold')
    fig.subplots_adjust(top=.88, bottom=.10, hspace=.8, wspace=.3)
    fig.savefig(RESULT_DIR / '03_my_handwriting.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    with open(RESULT_DIR / 'my_handwriting_predictions.csv', 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['Actual', 'MLP predicted', 'MLP confidence (%)', 'CNN predicted', 'CNN confidence (%)'])
        for i in range(len(my_images)):
            writer.writerow([int(my_labels[i]), int(mlp_guess[i]), round(float(100 * my_mlp[i].max()), 2),
                             int(cnn_guess[i]), round(float(100 * my_cnn[i].max()), 2)])
    print(f'My handwriting: MLP {mlp_correct}/{len(my_labels)} | CNN {cnn_correct}/{len(my_labels)}')
else:
    print('Optional: handwritten_mnist.npz not found. Skipping personal handwriting.')

# ============================================================
# 7. Simple HTML report (no confusion matrix)
# ============================================================
score_line = ''
extra_section = ''
if handwriting_result is not None:
    m, c, n = handwriting_result
    score_line = f'<p><b>My handwriting:</b> MLP {m}/{n} | CNN {c}/{n}</p>'
    extra_section = '<h2>My handwritten digits (optional)</h2><img src="03_my_handwriting.png">'

html = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>MNIST MLP vs CNN</title><style>
body{{font-family:Arial,'Malgun Gothic',sans-serif;max-width:1000px;margin:35px auto;padding:0 20px;color:#202a35}}
img{{max-width:100%;height:auto}}h1{{margin-bottom:8px}}section{{padding:15px 0;border-bottom:1px solid #ddd}}
</style></head><body>
<h1>MNIST | MLP vs CNN</h1>
<p><b>Official MNIST test:</b> MLP {mlp_accuracy[-1]:.2f}% | CNN {cnn_accuracy[-1]:.2f}% |
Difference {cnn_accuracy[-1]-mlp_accuracy[-1]:+.2f} percentage points</p>
{score_line}
<section><h2>Accuracy and loss</h2><img src="01_model_comparison.png"></section>
<section><h2>MNIST prediction examples</h2><img src="02_mnist_predictions.png"></section>
<section>{extra_section}</section>
<p>Same training settings: 60,000 train images, 10,000 test images, 10 epochs, batch 64, Adam, learning rate 0.001.</p>
</body></html>'''
report = RESULT_DIR / 'report.html'
report.write_text(html, encoding='utf-8')
print(f'Results saved in: {RESULT_DIR}')
webbrowser.open(report.resolve().as_uri())
