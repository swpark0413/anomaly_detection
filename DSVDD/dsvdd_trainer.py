from sklearn.metrics import roc_auc_score
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class AETrainer(nn.Module):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super(AETrainer, self).__init__()

        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        

    def train(self, train_loader, ae_net):
        # logger = logging.getLogger()

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        # logger.info('Starting pretraining...')
        print('Starting pretraining...')
        start_time = time.time()
        loss_plot = []
        ae_net.train()
        for epoch in range(self.n_epochs):

            if epoch in self.lr_milestones:
                # logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                loss = torch.mean(rec_loss)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            # logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
            #            f'| Train Loss: {epoch_loss / n_batches:.6f} |')
            # print(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
            #             f'| Train Loss: {epoch_loss / n_batches:.6f} |')
            
            scheduler.step()
            loss_plot.append(epoch_loss / n_batches)

        pretrain_time = time.time() - start_time
        # logger.info('Pretraining Time: {:.3f}s'.format(self.train_time))
        print('Pretraining Time: {:.3f}s'.format(pretrain_time))
        # logger.info('Finished pretraining.')
        print('Finished pretraining.')
        return ae_net, loss_plot

    def test(self, test_loader, ae_net):
        # logger = logging.getLogger()

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device for network
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Testing
        # logger.info('Testing autoencoder...')
        print('Testing autoencoder...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        label_score = []
        
        ae_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels  = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                scores = torch.mean(rec_loss, dim=tuple(range(1, rec.dim())))

                # Save (label, score) in a list
                label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss = torch.mean(rec_loss)
                epoch_loss += loss.item()
                n_batches += 1

        test_time = time.time() - start_time

        # Compute AUC
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        # Log results
        # logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        # logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        # logger.info('Test Time: {:.3f}s'.format(self.test_time))
        # logger.info('Finished testing autoencoder.')
        print('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        print('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        print('Test Time: {:.3f}s'.format(test_time))
        print('Finished testing autoencoder.')
        
        
class DeepSVDDTrainer(nn.Module):
    
    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super(DeepSVDDTrainer, self).__init__()
        
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        
        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        
        
        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu


        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, train_loader, net):
        # logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            # logger.info('Initializing center c...')
            print('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            # logger.info('Center c initialized.')
            print('Center c initialized.')

        # Training
        # logger.info('Starting training...')
        print('Starting training...')
        start_time = time.time()
        loss_plot = []
        net.train()
        for epoch in range(self.n_epochs):

            if epoch in self.lr_milestones:
                # logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                    
                # losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                #  loss = torch.mean(losses)
                loss.backward()
                optimizer.step()
                
                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)


                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            loss_plot.append(epoch_loss)
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            # logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
            #            f'| Train Loss: {epoch_loss / n_batches:.6f} |')
            print(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time
        # logger.info('Training Time: {:.3f}s'.format(self.train_time))
        # logger.info('Finished training.')
        print('Training Time: {:.3f}s'.format(self.train_time))
        print('Finished training.')

        return net, loss_plot

    def test(self, test_loader, net):
        # logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Testing
        # logger.info('Starting testing...')
        print('Starting testing...')
        n_batches = 0
        start_time = time.time()
        label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist
                    
                # losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                # loss = torch.mean(losses)
                #  scores = dist

                # Save (label, score) in a list
                label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = label_score

        # Compute AUC
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        # Log results
        # logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        # logger.info('Test Time: {:.3f}s'.format(self.test_time))
        # logger.info('Finished testing.')
        print('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        print('Test Time: {:.3f}s'.format(self.test_time))
        print('Finished testing.')

    def init_center_c(self, train_loader, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
    
    
def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)