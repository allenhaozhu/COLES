# NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
import argparse
from time import perf_counter
import numpy as np
import torch

from classification import classify
from ssgc import Net
from utils import load_adj_neg, load_dataset_adj_lap

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='pubmed',
                    help='dataset')
parser.add_argument('--seed', type=int, default=123,
                    help='seed')
parser.add_argument('--nhid', type=int, default=256,
                    help='hidden size')
parser.add_argument('--output', type=int, default=256,
                    help='output size')
parser.add_argument('--lr', type=float, default=0.02,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=40,
                    help='maximum number of epochs')
parser.add_argument('--sample', type=int, default=3,
                    help='    ')
parser.add_argument('--num_nodes', type=int, default=19717,
                    help='    ')
parser.add_argument('--num_features', type=int, default=500,
                    help='    ')

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feature, adj_normalized, lap_normalized= load_dataset_adj_lap(args.dataset)
feature = feature.to(device)
adj_normalized = adj_normalized.to(device)
lap_normalized = lap_normalized.to(device)
t = perf_counter()
K = 8
emb = feature
for i in range(K):
    feature = torch.mm(adj_normalized, feature)
    emb = emb + feature
emb/=K
neg_sample = torch.from_numpy(load_adj_neg(args.num_nodes, args.sample)).float().to(device)

model = Net(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
model.train()
Lambda = 1
for epoch in range(args.epochs):

    optimizer.zero_grad()
    out = model(emb)
    loss = ((Lambda*torch.trace(torch.mm(torch.mm(torch.transpose(out, 0, 1), neg_sample), out)) - torch.trace(
        torch.mm(torch.mm(torch.transpose(out, 0, 1), lap_normalized), out)))/out.shape[0])
    print(loss)
    loss.backward()
    optimizer.step()
print(perf_counter()-t)
model.eval()
emb = model(emb).cpu().detach().numpy()
np.save('embedding.npy', emb)
classify(emb, args.dataset, per_class='20')
classify(emb, args.dataset, per_class='5')

#K=8
# 77.33673521457368
# 1.8841861205625545
# 65.94036673947275
# 5.2056865384995055
