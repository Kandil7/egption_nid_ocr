"""Inspect ONNX model structure and metadata."""
import onnx
from onnx import numpy_helper
import numpy as np

# Load the model
model = onnx.load('weights/field_detector.onnx')

# Print model info
print('=== MODEL INFO ===')
print(f'Input count: {len(model.graph.input)}')
print(f'Output count: {len(model.graph.output)}')
print(f'Node count: {len(model.graph.node)}')

# Check inputs
print('\n=== INPUTS ===')
for inp in model.graph.input:
    print(f'Name: {inp.name}')
    dims = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
    print(f'Shape: {dims}')

# Check outputs
print('\n=== OUTPUTS ===')
for out in model.graph.output:
    print(f'Name: {out.name}')
    dims = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
    print(f'Shape: {dims}')

# Check metadata
print('\n=== METADATA ===')
for meta in model.metadata_props:
    print(f'{meta.key}: {meta.value}')

# Check initializers (weights)
print('\n=== INITIALIZERS (first 10) ===')
for init in model.graph.initializer[:10]:
    print(f'{init.name}: {init.dims}')
