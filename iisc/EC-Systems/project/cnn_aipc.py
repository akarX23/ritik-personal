import openvino as ov
import torch
import math
from torchvision.models import alexnet, AlexNet_Weights
from torchvision import transforms, datasets
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset

from lenet import DenseNet

def extract_model_analytics(model: ov.Model, core: ov.Core, bytes_per_element=4):
    graph_nodes = model.get_ordered_ops()
    
    layer_metrics = []
    e_compute = 1e-9  # Energy per MAC operation in Joules (1 nJ)
    e_memory = 1e-12  # Energy per memory access in Joules (1 pJ)

    npu_support = core.query_model(model, "NPU")
    gpu_support = core.query_model(model, "GPU")
    cpu_support = core.query_model(model, "CPU")

    for node in graph_nodes:
        
        curr_mac = 0
        memory_traffic = 0
        node_type = node.get_type_name()
        
        can_run_npu = npu_support.get(node.get_friendly_name(), False)
        can_run_gpu = gpu_support.get(node.get_friendly_name(), False)
        can_run_cpu = cpu_support.get(node.get_friendly_name(), False)
        
        if node_type == "Convolution":
            input_shape = node.input(0).get_shape()
            kernel_shape = node.input(1).get_shape()
            output_shape = node.output(0).get_shape()
            
            # Batch * (In_Channels * K_H * K_W) * (Out_H * Out_W * Out_Channels)
            curr_mac = input_shape[0] * (kernel_shape[1] * kernel_shape[2] * kernel_shape[3]) * (output_shape[2] * output_shape[3] * output_shape[1])
            memory_traffic = (math.prod(input_shape) + math.prod(kernel_shape) + math.prod(output_shape)) * bytes_per_element
        
        elif node_type == "MatMul":
            input_shape = node.input(0).get_shape()
            weights_shape = node.input(1).get_shape()
            output_shape = node.output(0).get_shape()
            
            # Batch * In_Features * Out_Features
            curr_mac = input_shape[0] * input_shape[1] * output_shape[1]
            memory_traffic = (math.prod(input_shape) + math.prod(weights_shape) + math.prod(output_shape)) * bytes_per_element
        
        arith_intensity = curr_mac / memory_traffic if memory_traffic != 0 else 0
        energy_consumption = curr_mac * e_compute + memory_traffic * e_memory
        
        layer_metrics.append({
            "name": node.get_friendly_name(),
            "type": node_type,
            "macs": curr_mac,
            "memory_traffic": memory_traffic,
            "arith_intensity": arith_intensity,
            "energy_consumption": energy_consumption,
            "supported_on": ("NPU " if can_run_npu else "") + ("GPU " if can_run_gpu else "") + ("CPU" if can_run_cpu else "")
        })
    
    return layer_metrics

class LeNet_AIPC:
    def __init__(self, batch_size=1):
        self.model = DenseNet()
        self.ov_model = ov.convert_model(self.model, example_input=torch.randn(1, 1, 32, 32))
        self.core = ov.Core()
        self.ov_model.reshape([batch_size, 1, 32, 32])
        self.analytics = extract_model_analytics(self.ov_model, self.core)
        
    def preprocess_raw_images(self, raw_images):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        preprocessed_images = [transform(img) for img in raw_images]
        return torch.stack(preprocessed_images).numpy()
    
    def init_model_infer_object(self, device="CPU"):
        self.compiled_model = self.core.compile_model(self.ov_model, device)
        self.infer_request = self.compiled_model.create_infer_request()
    
    def predict_batch(self, preprocessed_images):
        self.infer_request.infer({0: preprocessed_images})
        output = self.infer_request.get_output_tensor().data
        return output
    
    def setup_training(self, learning_rate=0.001, momentum=0.9, device="CPU"):
        self.train_device = torch.device(device.lower())
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.model.to(self.train_device)
        self.criterion = nn.CrossEntropyLoss().to(self.train_device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def load_train_val_datasets(self, batch_size=64):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    def finetune_epoch(self):
        total_loss = 0
        self.model.train()
        for images, labels in self.train_dataloader:
            images, labels = images.to(self.train_device), labels.to(self.train_device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss
    
    def evaluate_accuracy(self):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.test_dataloader:
                images, labels = images.to(self.train_device), labels.to(self.train_device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total
    
# Create the same class above for another CNN
class AlexNet_AIPC:
    def __init__(self, batch_size=1):
        self.model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.ov_model = ov.convert_model(self.model, example_input=torch.randn(1, 3, 224, 224))
        self.core = ov.Core()
        self.ov_model.reshape([batch_size, 3, 224, 224])
        self.analytics = extract_model_analytics(self.ov_model, self.core)

    def preprocess_raw_images(self, raw_images):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        preprocessed_images = [transform(img) for img in raw_images]
        return torch.stack(preprocessed_images).numpy()

    def init_model_infer_object(self, device="CPU"):
        self.compiled_model = self.core.compile_model(self.ov_model, device)
        self.infer_request = self.compiled_model.create_infer_request()

    def predict_batch(self, preprocessed_images):
        self.infer_request.infer({0: preprocessed_images})
        output = self.infer_request.get_output_tensor().data
        return output
    
    def setup_training(self, learning_rate=0.001, momentum=0.9, device="CPU"):
        self.train_device = torch.device(device.lower())
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.model.to(self.train_device)
        self.criterion = nn.CrossEntropyLoss().to(self.train_device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def load_train_val_datasets(self, batch_size=64, max_samples=25000):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        def make_torch_dataset(hf_split):
            class HFImageDataset(torch.utils.data.Dataset):
                def __init__(self, hf_data, transform):
                    self.hf_data = hf_data
                    self.transform = transform

                def __len__(self):
                    return len(self.hf_data)

                def __getitem__(self, idx):
                    item = self.hf_data[idx]
                    image = item["image"].convert("RGB")
                    label = item["label"]
                    image = self.transform(image)
                    return image, label

            return HFImageDataset(hf_split, transform)

        hf_dataset = load_dataset("timm/mini-imagenet", cache_dir="./data", token="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")
        train_dataset = make_torch_dataset(hf_dataset["train"].take(max_samples))
        test_dataset = make_torch_dataset(hf_dataset["validation"])
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    def finetune_epoch(self):
        total_loss = 0
        self.model.train()
        for images, labels in self.train_dataloader:
            images, labels = images.to(self.train_device), labels.to(self.train_device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss
    
    def evaluate_accuracy(self):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.test_dataloader:
                images, labels = images.to(self.train_device), labels.to(self.train_device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total
