# Example Scripts for Edge Nodes

This directory contains example Python scripts that can be loaded and executed dynamically on edge nodes.

## How to Use

1. Place your script in `scripts/{node_id}/your_script.py`
2. Load the script on the node:
   ```bash
   ros2 topic pub /{node_id}/script/load std_msgs/String "data: 'your_script'"
   ```
3. The script functions will be available for execution

## Available Context

Scripts have access to:
- `torch` - PyTorch
- `np` - NumPy
- `rclpy` - ROS2 Python

## Example Functions

### process_sensor_data(sensor_data_list)
Process sensor data before compression.

### adaptive_compression_level(quality_score, bandwidth_estimate)
Calculate optimal compression level.

### motor_command_filter(motor_commands, previous_commands)
Filter motor commands to reduce jitter.

## Script Structure

```python
def process_sensor_data(sensor_data_list):
    # Your processing code
    return processed_data

def your_custom_function(*args, **kwargs):
    # Your custom function
    return result
```

## Best Practices

1. Keep functions fast (< 10ms)
2. Use PyTorch tensors for efficiency
3. Handle errors gracefully
4. Document your functions

