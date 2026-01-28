import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim = 512, hidden_dim = 512, output_dim = 4):
        super(MultiLayerPerceptron, self).__init__()

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU()
        )

        self.classifier = torch.nn.Linear(512,output_dim)


    def forward(self, x, feats_only = False):

        if x.size(0) == 0:
            return (torch.empty(0, self.classifier.out_features, device = x.device), 
                    torch.empty(0, self.feature_extractor[-2].out_features, device=x.device))
        elif x.dim() > 2:
            x = x.view(x.size(0), -1)
        feats = self.feature_extractor(x)
        logits = self.classifier(feats)
        if feats_only:
            return feats
        else:
            return logits, feats
    

    def _train_step(self,
                    args,
                    source_data,
                    source_labels):
        
        self.train().to(args.device)
        
        tot_loss, tot_acc = 0.0, 0.0
        dataset_size = source_data.size(0)
        num_batches = (dataset_size + args.batch_size - 1) // args.batch_size
        pbar = tqdm(range(num_batches), desc='Training', leave=False)
        for i in pbar:
            start = i * args.batch_size
            end = min(start + args.batch_size, dataset_size)
            source_data_batch = source_data[start:end].to(args.device)
            source_labels_batch = source_labels[start:end].to(args.device)
            args.optimizer.zero_grad()
            logits, _ = self(source_data_batch, feats_only=False)
            #task_loss = _weighted_loss(logits, source_labels_batch)
            task_loss = args.criterion(logits, source_labels_batch)
            accuracy = (logits.argmax(dim=1) == source_labels_batch).float().mean().item()
            loss = task_loss
            loss.backward()
            args.optimizer.step()
            tot_loss += task_loss.item()
            tot_acc += accuracy
            _tot_loss = tot_loss / (i + 1)
            _tot_acc = tot_acc / (i + 1)
            pbar.set_description(f"Train Acc:{_tot_acc * 100: 4f}, Task Loss:{_tot_loss: 4f}")

        return {'train_loss': _tot_loss,
                'train_acc': _tot_acc}
    
    def _test_step( self, 
                    args, 
                    source_data,
                    source_labels,
                    desc = '[S]'):
        
        self.eval().to(args.device)
        tot_loss, tot_acc = 0.0, 0.0
        dataset_size = source_data.size(0)
        num_batches = (dataset_size + args.batch_size - 1) // args.batch_size
        pbar = tqdm(range(num_batches), desc = 'Eval', leave= False)
        with torch.no_grad():
            for i in pbar:
                start = i * args.batch_size
                end = min(start + args.batch_size, dataset_size)
                source_data_batch = source_data[start:end].to(args.device)
                source_labels_batch = source_labels[start:end].to(args.device)
                logits, _ = self(source_data_batch, feats_only = False)
                #task_loss = _weighted_loss(logits, source_labels_batch)
                task_loss = args.criterion(logits, source_labels_batch)
                accuracy = (logits.argmax(dim = 1) == source_labels_batch).float().mean().item()
                tot_loss += task_loss.item()
                tot_acc += accuracy
                _tot_loss = tot_loss / (i+1)
                _tot_acc = tot_acc / (i+1)
                pbar.set_description(f" Test Acc {desc}:{_tot_acc * 100: 4f}, Test Loss {desc}:{_tot_loss: 4f}")
        
        return{'test_loss {desc}': _tot_loss,
                'test_acc {desc}': _tot_acc}
    
    def _trainer_fit(self,
                     args,
                     source_data_train,
                     #source_data_test,
                     source_labels_train,
                     #source_labels_test, 
                     target_data, 
                     target_labels, test = True):
        
        history = {'train_loss':[],
                    'train_acc':[],
                    's_test_loss':[],
                    's_test_acc': [],
                    't_test_loss':[],
                    't_test_acc':[] }
        
        for epoch in range(args.epochs):
            #print(f"Epoch {epoch + 1}/{args.epochs}")
            train_metrics = self._train_step(args, source_data_train, source_labels_train)
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_acc'].append(train_metrics['train_acc'])
            if test:
                #s_test_metrics = self._test_step(args, source_data_test, source_labels_test, desc = '[S]')
                test_metrics = self._test_step(args, target_data, target_labels, desc = ' [T]')
                #history['s_test_loss'].append(s_test_metrics['test_loss {desc}'])
                #history['s_test_acc'].append(s_test_metrics['test_acc {desc}'])
                history['t_test_loss'].append(test_metrics['test_loss {desc}'])
                history['t_test_acc'].append(test_metrics['test_acc {desc}'])
                if hasattr(args, "scheduler") and hasattr(args.scheduler, "step"):
                    args.scheduler.step()
                current_lr = args.optimizer.param_groups[0]['lr']
                #print(f"    Learning Rate: {current_lr:.6f}")

        return history
    
def _weighted_loss(outputs, targets):
    N, C = outputs.size()

    preds = torch.argmax(outputs, dim=1)
    weights = torch.where(preds != targets, torch.tensor(2.0, device=outputs.device), torch.tensor(1.0, device=outputs.device))
    log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=C).float()
    ce_per_sample = -(targets_one_hot * log_probs).sum(dim=1)
    loss = (weights * ce_per_sample).mean()
    return loss
