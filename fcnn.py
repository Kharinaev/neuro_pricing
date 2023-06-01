import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm


class ANN(nn.Module):
    def __init__(self, struct, bn=False, device=None):
        super().__init__()
        self.struct = struct
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        layers = []
        for i,l in enumerate(struct[:-1]):
            if bn:
                layers.append(nn.BatchNorm1d(struct[i]))
            
            layers += [
                nn.Linear(struct[i], struct[i+1]),
                nn.ReLU()
            ]

        self.net = nn.Sequential(*layers[:-1])

    def forward(self, x):
        return self.net(x)
    
    def train_network(self, optimizer, criterion, dataloader_train, dataloader_test, num_epochs, noise=None, writer=None):
        logs = {
            'train_loss' : [],
            'test_loss' : []
        }

        self.train()
        for epoch in tqdm(range(num_epochs), desc='Epoch'):
            train_running_loss, grad_norm = 0.0, 0.0
            for iter_num, (X,y) in enumerate(dataloader_train):
                X = X.to(self.device)
                y = y.to(self.device).unsqueeze(1)
                if noise is not None:
                    y = y + torch.rand_like(y) * noise
                optimizer.zero_grad()
                out = self(X)
                loss = criterion(out, y)
                train_running_loss += loss.item()
                loss.backward()
                
                optimizer.step()

                if writer is not None:
                    writer.add_scalars("loss_iter", {'train' : loss}, iter_num + epoch * len(dataloader_train))

            logs['train_loss'].append(train_running_loss/len(dataloader_train))

            running_loss = 0.0
            for iter_num, (X,y) in enumerate(dataloader_test):
                with torch.no_grad():
                    X = X.to(self.device)
                    y = y.to(self.device).unsqueeze(1)
                    out = self(X)
                    loss = criterion(out, y)
                    running_loss += loss.item()
                    if writer is not None:
                        writer.add_scalars('loss_iter', {'test' : loss}, iter_num + epoch * len(dataloader_test))

            logs['test_loss'].append(running_loss/len(dataloader_test))
            
            if writer is not None:
                writer.add_scalars('loss_epoch', {
                    'train': train_running_loss/len(dataloader_train),
                    'test': running_loss/len(dataloader_test),
                }, epoch)

        if writer is not None:
            writer.close()
            
        return logs
    
    def iv_surface(self, strikes, maturities, stock_price, opt_prices=None):
        if self.struct[0] == 4 and opt_prices is None:
            raise ValueError('Neural Network type A should get option prices')
        
        self.eval()
        surface = np.zeros((len(maturities), len(strikes)), dtype=float)

        for i, t in enumerate(zip(maturities)):
            if self.struct[0] == 4:
                cur_batch = torch.tensor(np.vstack([
                    np.log(stock_price / strikes),
                    np.repeat(t, len(strikes)),
                    np.repeat(0.05, len(strikes)),
                    np.log(opt_prices[i] / strikes),
                ]), dtype=torch.float32).T.to(self.device)

            elif self.struct[0] == 3:
                cur_batch = torch.tensor(np.vstack([
                    np.log(stock_price / strikes),
                    np.repeat(t, len(strikes)),
                    np.repeat(0.05, len(strikes)),
                ]), dtype=torch.float32).T.to(self.device)
    
            with torch.no_grad():
                surface[i,:] = self(cur_batch).cpu().detach().numpy().squeeze(1)

        return surface


    def MSE_all(self, df_list, maturities, stock_price, r):
        self.eval()
        sse, cnt = 0, 0
        for i, (t, df) in enumerate(zip(maturities, df_list)):
            cnt += len(df)

            if self.struct[0] == 4:
                cur_batch = torch.tensor(np.vstack([
                    np.log(stock_price / df.strike),
                    np.repeat(t, len(df.strike)),
                    np.repeat(r, len(df.strike)),
                    np.log((df.bid + df.ask) / 2 / df.strike),
                ]), dtype=torch.float32).T.to(self.device)

            elif self.struct[0] == 3:
                cur_batch = torch.tensor(np.vstack([
                    np.log(stock_price / df.strike),
                    np.repeat(t, len(df.strike)),
                    np.repeat(r, len(df.strike)),
                ]), dtype=torch.float32).T.to(self.device)
            
            with torch.no_grad():
                nn_vols = self(cur_batch).cpu().detach().numpy().squeeze(1)
            
            sse += np.sum(np.square(nn_vols - df.impliedVolatility))

        return sse / cnt
    