import paddle
import paddle.nn as nn

class MSELoss(nn.Layer):
    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, label):

        # print('pred = ',pred)
        # print('label = ',label)

        # 确保输入是Paddle张量
        if not isinstance(pred, paddle.Tensor):
            pred = paddle.to_tensor(pred)
        if not isinstance(label, paddle.Tensor):
            label = paddle.to_tensor(label)
        loss = self.mse_loss(pred, label)
        return {"MSELoss": loss}