import os
import yaml

class Config():
    def __init__(self, path):
        with open(path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            
    def __getattr__(self, name):
        return self.config[name]
    
    def __str__(self):
        s = ''

        s += "-------Config Settings-------\n"
        for k, v in self.config.items():
            s += str(k) + ": " +  str(v) + "\n"
        s += "-----------------------------\n"
        
        
        return s
    
    
def eval_accuracy(model, dataloader, device):
    with torch.no_grad():
        correct = 0.
        total = 0.
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted==y).sum().cpu().numpy()
            total += len(y)
        return correct / total