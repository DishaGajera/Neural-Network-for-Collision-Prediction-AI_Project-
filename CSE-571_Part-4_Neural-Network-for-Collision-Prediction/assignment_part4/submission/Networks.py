import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self, input_size = 6, hidden_size = 1, output_size = 1):
        # STUDENTS: __init__() must initiatize nn.Module and define your network's
        # custom architecture

        super(Action_Conditioned_FF, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # STUDENTS: forward() must complete a single forward pass through your network
        # and return the output which should be a tensor
        x = self.layer1(input)
        x = self.relu(x)
        x = self.layer2(x)
        return x


    def evaluate(self, model, test_loader, loss_function):
        # STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
        # mind that we do not need to keep track of any gradients while evaluating the
        # model. loss_function will be a PyTorch loss function which takes as argument the model's
        # output and the desired output.

        model.eval()
        totalLoss = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch['input'], batch['label']
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                totalLoss += loss.item()
        averageLoss = totalLoss / len(test_loader)
        return averageLoss

def main():
    model = Action_Conditioned_FF()

    torch.save(model.state_dict(), 'saved_model.pkl')

if __name__ == '__main__':
    main()
