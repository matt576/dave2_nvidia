import torch
from torch import nn
class Policy(nn.Module):
    def __init__(self, a_dim):
        super(Policy, self).__init__()
        """
        This function initializes the different layers of the architecture.

        input:
            a_dim (type: int): dimension of the output
        """
        # Convolutional layers of the DAVE-2 architecture:

        self.cnn1 = nn.Conv2d(3, 24, 5, stride=2)
        self.relu = nn.ELU(self.cnn1),
        self.cnn2 = nn.Conv2d(24, 36, 5, stride=2)
        self.relu = nn.ELU(self.cnn2)
        self.cnn3 = nn.Conv2d(36, 48, 5, stride=2)
        self.relu = nn.ELU(self.cnn3)
        self.cnn4 = nn.Conv2d(48, 64, 3)
        self.relu = nn.ELU(self.cnn4)
        self.cnn5 = nn.Conv2d(64, 64, 3)
        self.relu = nn.ELU(self.cnn5)

       # Fully connected layers of the DAVE-2 architecture:

        self.fcn1 = nn.Linear(in_features=1152, out_features=100)
        self.relu = nn.ELU(self.fcn1)
        # nn.Dropout(0.2)
        self.fcn2 = nn.Linear(in_features=100, out_features=50)
        self.relu = nn.ELU(self.fcn2)
        # nn.Dropout(0.2)
        self.fcn3 = nn.Linear(in_features=50, out_features=10)
        self.relu = nn.ELU(self.fcn3)
        self.last_layer = nn.Linear(in_features=10, out_features=a_dim)

        # Last activation function:
        # Vehicle actions are defined between [0, 1]
        
        self.last_activation = nn.Sigmoid()

        self.relu = nn.ReLU(inplace=True)
        self.loss = nn.MSELoss(reduction='sum')
        print(self)

    def forward(self, x):
        x_cnn1 = self.relu(self.cnn1(x))
        x_cnn2 = self.relu(self.cnn2(x_cnn1))
        x_cnn3 = self.relu(self.cnn3(x_cnn2))
        x_cnn4 = self.relu(self.cnn4(x_cnn3))
        x_cnn5 = self.relu(self.cnn5(x_cnn4))
        x_flatten = x_cnn5.view(x_cnn5.shape[0], -1)
        x_fcn1 = self.relu(self.fcn1(x_flatten))
        x_fcn2 = self.relu(self.fcn2(x_fcn1))
        x_fcn3 = self.relu(self.fcn3(x_fcn2))
        vehicle_control = self.last_activation(self.last_layer(x_fcn3))

        return vehicle_control

    def criterion(self, a_imitator_, a_expert_):
        loss = self.loss(a_imitator_, a_expert_)
        return loss


if __name__ == '__main__':
    # Hyperparameters
    BATCH_SIZE = 1
    A_DIM = 2

    # Initialize imitator policy
    imitator_policy = Policy(a_dim=A_DIM) 

    # Generate random test tensors for action and state
    a_expert = torch.rand(size=(BATCH_SIZE, 2))
    state_testing = torch.rand(size=(BATCH_SIZE, 3, 66, 200))

    a_imitator = imitator_policy(state_testing) # Calculate predicted action

    print('Predicted action: ', a_imitator)

    loss = imitator_policy.criterion(a_imitator, a_expert) # Calculate MSE Loss
    print('Calculated loss: ', loss)

