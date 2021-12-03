import torch
import coremltools as ct



# Load scripted model
# model = torch.jit.load('melgan-neurips_cpu.pt')
model = torch.jit.load('steps_30000_cpu.pt')

# Create the input image type
input_image_src = ct.ImageType(name="my_input_src", shape=(1, 80, 128), scale=1/255)
input_image_tgt = ct.ImageType(name="my_input_tgt", shape=(1, 80, 128), scale=1/255)

# Convert the model
# coreml_model = ct.convert(model, inputs=[input_image_src, input_image_tgt])
coreml_model = ct.convert(model, inputs=[ct.TensorType(shape=(1, 80, 128)), ct.TensorType(shape=(1, 80, 128))])

# NOTE: melgan cannot be easily converted:
# RuntimeError: PyTorch convert function for op '_weight_norm' not implemented.
# DONE: removed weight norm, melgan converted.
# coreml_model = ct.convert(model, inputs=[ct.TensorType(shape=(1, 80, 128))])

# Modify the output's name to "my_output" in the spec
spec = coreml_model.get_spec()
print(spec)
# ct.utils.rename_feature(spec, "81", "my_output")

# Re-create the model from the updated spec
coreml_model_updated = ct.models.MLModel(spec)

# Save the CoreML model
# coreml_model_updated.save('steps_300000.mlmodel')