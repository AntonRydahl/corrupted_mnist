from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self,n_in=784,n_out=10):
        super().__init__()
        # Defining the whole thing in one go with sequential
        self.n_in = n_in
        self.n_out = n_out
        self.n_hid_1 = 512
        self.n_hid_2 = 256
        self.n_hid_3 = 128
        self.n_hid_4 = 64
        self.output = nn.Sequential(
            nn.Linear(self.n_in,self.n_hid_1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.n_hid_1,self.n_hid_2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.n_hid_2,self.n_hid_3),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.n_hid_3,self.n_hid_4),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.n_hid_4,self.n_out),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)       
        return self.output(x)