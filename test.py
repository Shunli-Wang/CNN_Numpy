"#codeing=utf-8"
import numpy as np
import matplotlib.pyplot as plt
from Convolution2 import Convolution, Pooling, FCLayer, CEloss, ReLU
from data_load import get_dataset, dataset


def test_forward(model, **kwargs):
    test_dataset = enumerate(dataset(kwargs['x'], kwargs['y'], kwargs['bs']))

    acc = 0
    for i, [x, y] in test_dataset:
        conv1_a = model[0].forward(x)
        pool1_a = model[1].forward(conv1_a)
        relu1_a = model[2].forward(pool1_a)
        conv2_a = model[3].forward(relu1_a)
        pool2_a = model[4].forward(conv2_a)
        relu2_a = model[5].forward(pool2_a)
        net_output = relu2_a.reshape((kwargs['bs'], -1))
        fc_out = model[6].forward(net_output)
        q = model[7].forward(fc_out)
        acc_batch = np.mean(np.argmax(q, axis=1) == np.argmax(y.T, axis=0))
        acc += acc_batch

    return acc / i


if __name__ == "__main__":

    # x:(?,784)  y:(?,10)
    [train_x, test_x, train_y, test_y] = get_dataset()

    # Hyper
    batch_size = 64
    lr_list = [2e-2, 1e-2, 1e-3]
    step = [0, 3000, 3500]
    epoch = 5
    cost, train_acc, test_acc = [], [], []

    k1, k2 = 5, 5

    # Create network
    conv1 = Convolution(1, 6, kernel_size=(k1, k1), stride=1, pad=0)
    pool1 = Pooling(2, 2, 2)
    relu1 = ReLU()
    conv2 = Convolution(6, 16, kernel_size=(k2, k2), stride=1, pad=0)
    pool2 = Pooling(2, 2, 2)
    relu2 = ReLU()
    fc = FCLayer(256, 10)
    loss = CEloss()

    for epoch_cnt in range(epoch):

        # train:
        train_dataset = enumerate(dataset(train_x, train_y, batch_size))
        for i, [x, y] in train_dataset:

            # Forward
            conv1_a = conv1.forward(x)
            pool1_a = pool1.forward(conv1_a)
            relu1_a = relu1.forward(pool1_a)
            conv2_a = conv2.forward(relu1_a)
            pool2_a = pool2.forward(conv2_a)
            relu2_a = relu2.forward(pool2_a)
            net_output = relu2_a.reshape((batch_size, -1))
            fc_out = fc.forward(net_output)
            q, loss_ = loss.forward(fc_out, target=y)

            print('\t Loss: ', loss_)

            # Backword
            dx = loss.backword()
            dfc = fc.backword(dx)
            dfc = dfc.reshape(pool2_a.shape)
            drelu2_a = relu2.backword(dfc)
            dpool2_a = pool2.backward(drelu2_a)
            dconv2a = conv2.backward(dpool2_a)
            drelu1_a = relu1.backword(dconv2a)
            dpool1_a = pool1.backward(drelu1_a)
            dconv1a = conv1.backward(dpool1_a)

            # Update LR
            if i in range(step[0], step[1]):
                lr = lr_list[0]
            elif i in range(step[1], step[2]):
                lr = lr_list[1]
            else:
                lr = lr_list[2]

            # Update all parameters
            fc.weight -= lr * fc.wgrad
            fc.bias -= lr * fc.bgrad
            conv2.W -= lr * conv2.dW
            conv2.b -= lr * conv2.db
            conv1.W -= lr * conv1.dW
            conv1.b -= lr * conv1.db

            # test
            if (i + 1) % 100 == 0:
                model = [conv1, pool1, relu1, conv2, pool2, relu2, fc, loss]

                test_acc_ = test_forward(model, x=test_x, y=test_y, bs=batch_size)
                print('TEST:', test_acc_)
                test_acc.append(test_acc_)

            # Record
            train_acc_ = np.mean(np.argmax(q, axis=1) == np.argmax(y.T, axis=0))
            print('Batch:', i, ' LR:', lr, ' Precison: ', train_acc_)
            cost.append(loss_)
            train_acc.append(train_acc_)

    plt.figure()
    plt.plot(cost)
    plt.title("Training loss")
    plt.xlabel("Batch number")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig('./result/' + '_' + str(k1) + str(k2) + '/' + 'loss.jpg')

    plt.figure()
    plt.plot(train_acc)
    plt.title("Training accuracy of every batch")
    plt.xlabel("Batch number")
    plt.ylabel("Training accuracy")
    plt.grid()
    plt.savefig('./result/' + '_' + str(k1) + str(k2) + '/' + 'train_acc.jpg')

    plt.figure()
    plt.plot(test_acc, '-*')
    plt.title("Test accuracy")
    plt.xlabel("Batch number / 100")
    plt.ylabel("Test accuracy")
    plt.grid()
    plt.savefig('./result/' + '_' + str(k1) + str(k2) + '/' + 'test_acc.jpg')

    plt.show()
