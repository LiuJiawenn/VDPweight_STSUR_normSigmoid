import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from model.stsur import VDPWeightSTSUR
import time
import traceback
from file_operation.data_generator import data_generator


def training(model, patch_rate, epoch_nb=30):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.000001)
    criterion = nn.MSELoss()

    minibatch_mae = []
    minibatch_mse = []
    batch_mae_history = []
    batch_mse_history = []
    truth_sur = []
    predicted_sur = []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1)
    line1, = ax1.plot(batch_mae_history, label='train_MAE')
    line2, = ax1.plot(batch_mse_history, label='train_MSE')
    line3, = ax2.plot(truth_sur, label='truth_sur')
    line4, = ax2.plot(predicted_sur, label='predicted_sur')
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.set_title("Training Label")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("truth/predicted SUR")
    ax2.legend()

    # tag = str(int((model.patch_per_frame + patch_rate) * 10))
    count = 0

    try:
        for epoch in range(epoch_nb):
            generator = data_generator(patch_per_frame=model.patch_per_frame, patch_rate=patch_rate,
                                       key_frame_nb=model.key_frame_nb)
            # data(s,c,o,v)
            for X_train, y_train in generator:
                count += 1
                # 将数据移动到正确的设备（例如 GPU）
                X_train_gpu, y_train_gpu = tuple(x.to(device) for x in X_train), torch.tensor(y_train).float().to(device)
                # 执行训练步骤
                optimizer.zero_grad()
                outputs = model(X_train_gpu)
                loss = criterion(outputs, y_train_gpu)
                loss.backward()
                optimizer.step()

                print("******************** 当前压缩视频预测标签： ", outputs.item(), "真值： ", y_train_gpu.item())
                truth_sur.append(y_train)
                predicted_sur.append(outputs.item())

                mae = abs(outputs.item() - y_train)
                mse = loss.item()

                minibatch_mae.append(mae)
                minibatch_mse.append(mse)

                if len(minibatch_mse) == 11:
                    m_mse = np.mean(minibatch_mse)
                    m_mae = np.mean(minibatch_mae)
                    print("*****************************  MAE: ", m_mae, "   MSE: ", m_mse,"   当前视频： ", count)
                    batch_mae_history.append(m_mae)
                    batch_mse_history.append(m_mse)
                    minibatch_mse = []
                    minibatch_mae = []

                # 实时绘图
                line3.set_ydata(truth_sur[-30:])
                line3.set_xdata(range(len(truth_sur[-30:])))
                line4.set_ydata(predicted_sur[-30:])
                line4.set_xdata(range(len(truth_sur[-30:])))
                ax2.relim()
                ax2.autoscale_view()
                line1.set_ydata(batch_mae_history)
                line1.set_xdata(range(len(batch_mae_history)))
                line2.set_ydata(batch_mse_history)
                line2.set_xdata(range(len(batch_mse_history)))
                ax1.relim()
                ax1.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

            np.save('loss_history/0203batch_loss_epoch'+str(epoch+104)+'.npy', batch_mse_history)
            torch.save(model.state_dict(), 'weights/0203model_weights_epoch'+str(epoch+104)+'.pth')

    except Exception as err:
        print(f"训练过程中发生了错误：{err}", "     当前已训练视频：", count)
        traceback.print_exc()
    finally:
        np.save('loss_history/0203batch_loss_final.npy', batch_mse_history)
        torch.save(model.state_dict(), 'weights/0203model_weights_final.pth')
        # return model.state_dict(), batch_mse_history, predicted_sur, truth_sur

def truncated_normal_(tensor, mean=0.0, std=1.0, trunc_std=2):
    """
    Fill the input Tensor with values drawn from a truncated normal distribution.
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def initialize_weights(model):
    """
    Initialize the weights of the model with the truncated normal distribution.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            truncated_normal_(m.weight,mean=0, std=0.0001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.__version__)

    r = 0.4
    patch_per_frame = 64
    key_frame_nb = 12

    model = VDPWeightSTSUR(patch_per_frame=patch_per_frame, key_frame_nb=key_frame_nb)
    # initialize_weights(model)
    model.load_state_dict(torch.load('weights/0203model_weights_epoch103.pth'))
    model.to(device)
    # state, mse, predicted, truth = training(model, patch_rate=r, epoch_nb=200)
    # torch.save(state, 'weight/model_weight_' + str(int((r + patch_per_frame) * 10)) + '.pth')
    # np.save('lossHistory/mse_' + str(int((r + patch_per_frame) * 10)) + '.npy', mse)
    # np.save('lossHistory/predict_' + str(int((r + patch_per_frame) * 10)) + '.npy', predicted)
    # np.save('lossHistory/truth_' + str(int((r + patch_per_frame) * 10)) + '.npy', truth)

    training(model, patch_rate=r, epoch_nb=200)