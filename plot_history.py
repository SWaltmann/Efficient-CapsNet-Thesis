import json
import matplotlib.pyplot as plt

with open("training_history.json", "r") as f:
    loaded_history = json.load(f)

with open("test_history.json", "r") as f:
    test_history = json.load(f)


loss = loaded_history['loss']
acc = loaded_history['categorical_accuracy']

test_acc = test_history['accuracy']
print(f"Min training loss: {min(loss):.4f}")
print(f"Max accuracy: {max(acc):.4f}")
print(f"Max test accuracy: {max(test_acc):.4f}")



plt.plot(test_acc, label='Test Accuracy')
plt.plot(loaded_history['categorical_accuracy'], label='Training Accuracy')
plt.plot(loaded_history['loss'], label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()