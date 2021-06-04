# NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
import torch
from utils import load_adj_neg, load_dataset_adj_lap
from ssgc import Net
import argparse
import numpy as np
from classification import classify

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='cora_full',
                    help='dataset')
parser.add_argument('--seed', type=int, default=123,
                    help='seed')
parser.add_argument('--nhid', type=int, default=512,
                    help='hidden size')
parser.add_argument('--output', type=int, default=512,
                    help='output size')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=30,
                    help='maximum number of epochs')
parser.add_argument('--sample', type=int, default=3,
                    help='    ')
parser.add_argument('--num_nodes', type=int, default=19793,
                    help='    ')
parser.add_argument('--num_features', type=int, default=8710,
                    help='    ')

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feature, adj_normalized, lap_normalized= load_dataset_adj_lap(args.dataset)
feature = feature.to(device)
adj_normalized = adj_normalized.to(device)
lap_normalized = lap_normalized.to(device)
K = 2
emb = feature
for i in range(K):
    feature = torch.mm(adj_normalized, feature)
    emb += feature
emb/=K
neg_sample = torch.from_numpy(load_adj_neg(args.num_nodes, args.sample)).float().to(device)

model = Net(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
model.train()

for epoch in range(args.epochs):

    optimizer.zero_grad()
    out = model(emb)
    loss = torch.nn.functional.softplus((torch.trace(torch.mm(torch.mm(torch.transpose(out, 0, 1), neg_sample), out)) - torch.trace(
        torch.mm(torch.mm(torch.transpose(out, 0, 1), lap_normalized), out)))/out.shape[0])
    print(loss)
    loss.backward()
    optimizer.step()

emb = model(emb).cpu().detach().numpy()
np.save('embedding.npy', emb)
classify(emb, args.dataset, per_class='20')
classify(emb, args.dataset, per_class='5')
#K = 5
# 50.02639296187683
# 1.3780714721662846
# 60.387520525451556
# 0.500275536439848
#K = 4
# 50.54733975701717
# 1.3802282020626642
# 61.017945109078106
# 0.4755433475305339
# K=3
#50.8843736908253
# 1.363444353917577
# 61.4808820079756
# 0.4674891907768461
# K = 2
# 50.75324675324675
# 1.432164399477703
# 61.819258737977954
# 0.47992284895776816
