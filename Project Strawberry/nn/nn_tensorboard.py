from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("../logs_")
for i in range(100):
    # writer.add_image()
    writer.add_scalar(tag="y = x", scalar_value=i, global_step=i)

writer.close()

# 神中神
"""tensorboard --logdir=../logs --port=8006"""
"""tensorboard --logdir=../logs_ --port=8006"""
"""用上一级目录，../logs"""
