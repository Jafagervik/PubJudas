from tinygrad import Device

print(Device.DEFAULT)

from tinygrad import Tensor, nn, TinyJit

class MNIST:
    def __init__(self):
        self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.l3 = nn.Linear(1600, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x).relu().max_pool2d((2,2))
        x = self.l2(x).relu().max_pool2d((2,2))
        return self.l3(x.flatten(1).dropout(0.5))

from tinygrad.nn.datasets import mnist

X_train, Y_train, X_test, Y_test = mnist()
#print(X_train.shape)

model = MNIST()

#acc= (model(X_test).argmax(axis=1) == Y_test).mean()

optim = nn.optim.Adam(nn.state.get_parameters(model))

batch_size = 128

@TinyJit
def jit_step():
    Tensor.training = True
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    X, Y = X_train[samples], Y_train[samples]
    optim.zero_grad()

    loss = model(X).sparse_categorical_crossentropy(Y).backward()
    optim.step()
    print(loss.item())
    return loss

losses = [0.0] * 70
accs = [0.0] * 70
epochs = range(1, 7000 + 1, 100)

i = 0
for step in range(7000):
    loss = jit_step()
    if step % 100 == 0:
        Tensor.training = False
        acc= (model(X_test).argmax(axis=1) == Y_test).mean().item()
        losses[i] = loss.item()
        accs[i] = acc
        print(f"step {step:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")
        i += 1


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, losses, 'r', label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# Plotting the accuracies
plt.subplot(1, 2, 2)
plt.plot(epochs, accs, 'b', label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()