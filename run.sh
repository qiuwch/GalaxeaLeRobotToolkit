#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$SCRIPT_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate humble

# Set up ROS2 workspace environment
# Manually set library paths and Python paths for hdas_msg
INSTALL_DIR="$WORKSPACE_DIR/install"
if [ -d "$INSTALL_DIR/hdas_msg" ]; then
    # Add workspace install directory to AMENT_PREFIX_PATH (ROS2 package discovery)
    export AMENT_PREFIX_PATH="$INSTALL_DIR:${AMENT_PREFIX_PATH:-}"
    
    # Add hdas_msg library directory to DYLD_LIBRARY_PATH (macOS)
    export DYLD_LIBRARY_PATH="$INSTALL_DIR/hdas_msg/lib:${DYLD_LIBRARY_PATH:-}"
    
    # Add hdas_msg Python package to PYTHONPATH
    HDAS_MSG_PYTHON_DIR="$INSTALL_DIR/hdas_msg/lib/python3.11/site-packages"
    if [ -d "$HDAS_MSG_PYTHON_DIR" ]; then
        export PYTHONPATH="$HDAS_MSG_PYTHON_DIR:${PYTHONPATH:-}"
    fi
    
    # Try to source the workspace setup if available (may have issues, but try anyway)
    if [ -f "$INSTALL_DIR/local_setup.bash" ]; then
        # Suppress errors from missing files in setup script
        source "$INSTALL_DIR/local_setup.bash" 2>/dev/null || true
    fi
else
    echo "Warning: Could not find install/hdas_msg. Make sure the workspace is built."
fi

dataset_name=debug
# input_dir=/Users/qiuwch/code/gym_control/asset/mcap_test
input_dir=/Users/qiuwch/code/gym_control/asset/mcap_test/hw1217/recording_20251216_095533_@_
output_dir=/Users/qiuwch/code/gym_control/asset/mcap_test
robot_type=R1Pro # options: R1Pro, R1Lite

export SAVE_VIDEO=1 
export USE_H264=0
export USE_COMPRESSION=0
export IS_COMPUTE_EPISODE_STATS_IMAGE=1
export MAX_PROCESSES=1
export USE_ROS1=0
export USE_TRANSLATION=0

# Change to workspace directory to run the module
cd "$WORKSPACE_DIR"

python -m dataset_converter \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --robot_type $robot_type \
    --dataset_name $dataset_name