import torch


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()


        self.conv1 = torch.nn.Linear(args.num_features, args.output)


    def forward(self, F1):

        z = self.conv1(F1)


        return z
