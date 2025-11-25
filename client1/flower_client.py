from collections import OrderedDict
import flwr as fl
import torch

from train import (
    load_model,
    load_data,
    validate,
    train,
    num_classes,
    path_dataset_train,
    path_dataset_val,
    batch_size,
    num_workers,
    resize_shape,
    model_path
)


model_name = 'faster_rcnn_v2'                                              
model_path  = "/models/model_flower_1_model_faster_rcnn_v2_SGD_1400x1050_00001.pt"
num_classes = 12                                                             
resize_shape = (1400, 1050)                                                     
pretrained = True

# Train parameters
num_epochs = 1             # Number of epochs
batch_size = 2             # Batch size
lr = 0.0001                # Learning rate
use_gpu = True       

# Other parameters (do not change)
num_workers = 0
lr_momentum = 0.9
lr_decay = 0.005
lr_factor = 0.1
lr_patience = 10
lr_threshold = 1e-4
lr_min = 1e-10
max_grad_norm = 1.0 if model_name == 'ssd_v1' else 0

print("[FLOWER] Loading model...")

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

model, optimizer, initial_epoch, device = load_model(model_name, model_path, num_classes, lr, lr_momentum, lr_decay, pretrained, use_gpu)
train_loader, val_loader = load_data(path_dataset_train, path_dataset_val, resize_shape, model, batch_size, num_workers)

class FlowerClient(fl.client.NumPyClient):
    round_num = 0  # Initialize round counter

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def fit(self, parameters, config):
        self.round_num += 1  # Increment round counter
        set_parameters(model, parameters)
        train(model,optimizer, device, model_path,lr_factor,lr_patience,lr_threshold,lr_min, num_epochs,train_loader,val_loader,max_grad_norm,initial_epoch,self.round_num)

        return self.get_parameters({}), len(train_loader), {}

    def evaluate(self, parameters, config):
        set_parameters(model, parameters)
        loss, accuracy = validate(model, val_loader, device, num_classes)  # Make sure num_classes is provided
        return float(loss), len(val_loader), {"accuracy": accuracy}


import grpc
import flwr as fl

fl.client.start_client(
    server_address='flower_server:8080',
    client=FlowerClient().to_client(),  # Use client_fn to initialize the client
    grpc_max_message_length=512 * 1024 * 1024,  # Ensure the message size is adequate
    insecure=True  # Set to False if a secure connection is needed
)
