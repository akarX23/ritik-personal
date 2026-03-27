from lenet import DenseNet
import openvino as ov
import torch
import math
import numpy as np
import time
import openvino.properties as props
import openvino.properties.device as device

model = DenseNet()
ov_model = ov.convert_model(model, example_input=torch.randn(1, 1, 32, 32))
ov_model.reshape([1, 1, 32, 32])
core = ov.Core()
core.set_property({props.cache_dir: "./cache_dir"})
caching_supported = 'EXPORT_IMPORT' in core.get_property("NPU", device.capabilities)
print("Caching supported on NPU: ", caching_supported)

graph_nodes = ov_model.get_ordered_ops()
# for node in graph_nodes:
#     print("Type: ", node.get_type_name(), " Friendly Name: ", node.get_friendly_name())
#     print("Attributes: ", node.get_attributes())
    
#     for i, input_port in enumerate(node.inputs()):
#         shape = input_port.get_partial_shape()
#         print(f"Input {i} Shape: {shape}")

#     for i, output_port in enumerate(node.outputs()):
#         shape = output_port.get_partial_shape()
#         print(f"Output {i} Shape: {shape}")
#     print("==================================")

# Calculate MACs for each Convolution/MatMul layer
layer_metrics = []
bytes_per_element = 4  # Assuming 32-bit floats
e_compute = 1e-9  # Energy per MAC operation in Joules (1 nJ)
e_memory = 1e-12  # Energy per memory access in Joules (1 pJ)

npu_support = core.query_model(ov_model, "NPU")
gpu_support = core.query_model(ov_model, "GPU")
cpu_support = core.query_model(ov_model, "CPU")

for node in graph_nodes:
    
    curr_mac = 0
    memory_traffic = 0
    node_type = node.get_type_name()
    
    can_run_gpu = gpu_support.get(node.get_friendly_name(), False)
    can_run_cpu = cpu_support.get(node.get_friendly_name(), False)
    can_run_npu = npu_support.get(node.get_friendly_name(), False)
    
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
    
# Save the metrics to a CSV
# import pandas as pd
# df = pd.DataFrame(layer_metrics)
# df.to_csv("layer_metrics.csv", index=False)

ov_model.reshape([500, 1, 32, 32])

compile_start_time = time.perf_counter()
compiled_model = core.compile_model(ov_model, "NPU")
compile_end_time = time.perf_counter()
print("Model compilation time: ", compile_end_time - compile_start_time, "seconds")

infer_request = compiled_model.create_infer_request()

start_time = time.perf_counter()
infer_request.infer({"input": np.random.rand(500, 1, 32, 32).astype('float32')})
end_time = time.perf_counter()

output = infer_request.get_output_tensor().data
print("Inference output shape: ", output.shape)
print("Inference time: ", end_time - start_time, "seconds")
