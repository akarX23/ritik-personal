import openvino as ov
import openvino.properties as props
import time
import numpy as np

from lenet import DenseNet

# model = DenseNet()
# ov_model = ov.convert_model(model, example_input=np.random.rand(1, 1, 32, 32).astype('float32'))
# ov_model.reshape([2000, 1, 32, 32])
# ov.save_model(ov_model, "lenet.xml")

core = ov.Core()
core.set_property({props.cache_dir: "./cache_dir"})

compile_start_time = time.perf_counter()
compiled_model = core.compile_model("./lenet.xml", "NPU")
compile_end_time = time.perf_counter()
print("Model compilation time: ", compile_end_time - compile_start_time, "seconds")

infer_request = compiled_model.create_infer_request()

start_time = time.perf_counter()
infer_request.infer({"input": np.random.rand(2000, 1, 32, 32).astype('int8')})
end_time = time.perf_counter()

output = infer_request.get_output_tensor().data
print("Inference output shape: ", output.shape)
print("Inference time: ", end_time - start_time, "seconds")