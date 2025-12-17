"""
Topic Mapping Configuration Module

This module defines the TopicMappingConfig class and related functions for managing
mappings between ROS2 topics and lerobot dataset keys.
"""
import os
import yaml
from loguru import logger
from channels_definition import *

# Import ROS2 modules for topic resolution
try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
except ImportError:
    SequentialReader = None
    StorageOptions = None
    ConverterOptions = None


class TopicMappingConfig:
    """
    Configuration class that defines the mapping between ROS2 topics and lerobot dataset keys.
    This separates the configuration from the conversion logic.
    """
    
    def __init__(self, robot_type, rgb_wrist_left_topic, rgb_wrist_right_topic, arm_dof, save_video=False):
        self.robot_type = robot_type
        self.arm_dof = arm_dof
        self.save_video = save_video
        
        # Store original topic names for serialization
        self.rgb_wrist_left_topic = rgb_wrist_left_topic
        self.rgb_wrist_right_topic = rgb_wrist_right_topic
        
        # Topic lists by processing type
        self.rgb_topics = [
            rgb_wrist_left_topic,
            rgb_wrist_right_topic,
            RGB_HEAD_RIGHT_TOPIC
        ]
        self.depth_topics = [
            DEPTH_HEAD_TOPIC, 
            DEPTH_LEFT_TOPIC, 
            DEPTH_RIGHT_TOPIC
        ]
        self.joint_topics = [
            JOINT_OBS_LEFT_TOPIC, 
            JOINT_OBS_RIGHT_TOPIC, 
            GRIPPER_OBS_LEFT_TOPIC, 
            GRIPPER_OBS_RIGHT_TOPIC, 
            CHASSIS_OBS_TOPIC, 
            TORSO_OBS_TOPIC, 
            JOINT_ACTION_LEFT_TOPIC, 
            JOINT_ACTION_RIGHT_TOPIC, 
            TORSO_ACTION_TOPIC
        ]
        self.pose_topics = [
            EE_POSE_OBS_LEFT_TOPIC, 
            EE_POSE_OBS_RIGHT_TOPIC, 
        ]
        if robot_type == "r1pro":
            self.pose_topics.extend([
                EE_POSE_ACTION_LEFT_TOPIC,
                EE_POSE_ACTION_RIGHT_TOPIC
            ])
        
        self.gripper_topics = [
            GRIPPER_ACTION_LEFT_TOPIC, 
            GRIPPER_ACTION_RIGHT_TOPIC
        ]
        self.twist_topics = [CHASSIS_ACTION_TOPIC]
        if robot_type == "r1lite":
            self.twist_topics.append(TORSO_ACTION_SPEED_TOPIC)
        
        self.control_topics = [
            JOINT_CONTROL_ACTION_LEFT_TOPIC, 
            JOINT_CONTROL_ACTION_RIGHT_TOPIC, 
            GRIPPER_CONTROL_ACTION_LEFT_TOPIC, 
            GRIPPER_CONTROL_ACTION_RIGHT_TOPIC, 
            CHASSIS_CONTROL_ACTION_TOPIC, 
            TORSO_CONTROL_ACTION_TOPIC
        ]
        self.imu_topics = [CHASSIS_IMU_TOPIC]
        
        # All target topics (topics to extract)
        self.target_topics = {
            GRIPPER_ACTION_LEFT_TOPIC: [],
            GRIPPER_ACTION_RIGHT_TOPIC: [],
            EE_POSE_OBS_LEFT_TOPIC: [],
            EE_POSE_OBS_RIGHT_TOPIC: [],
            JOINT_OBS_LEFT_TOPIC: [],
            JOINT_OBS_RIGHT_TOPIC: [],
            JOINT_ACTION_LEFT_TOPIC: [],
            JOINT_ACTION_RIGHT_TOPIC: [],
            GRIPPER_OBS_LEFT_TOPIC: [],
            GRIPPER_OBS_RIGHT_TOPIC: [],
            RGB_HEAD_LEFT_TOPIC: [],
            RGB_HEAD_RIGHT_TOPIC: [],
            rgb_wrist_left_topic: [],
            rgb_wrist_right_topic: [],
            DEPTH_HEAD_TOPIC: [],
            DEPTH_LEFT_TOPIC: [],
            DEPTH_RIGHT_TOPIC: [],
            CHASSIS_ACTION_TOPIC: [],
            TORSO_ACTION_TOPIC: [],
            CHASSIS_OBS_TOPIC: [],
            CHASSIS_IMU_TOPIC: [],
            TORSO_OBS_TOPIC: [],
            JOINT_CONTROL_ACTION_LEFT_TOPIC: [],
            JOINT_CONTROL_ACTION_RIGHT_TOPIC: [],
            GRIPPER_CONTROL_ACTION_LEFT_TOPIC: [],
            GRIPPER_CONTROL_ACTION_RIGHT_TOPIC: [],
            CHASSIS_CONTROL_ACTION_TOPIC: [],
            TORSO_CONTROL_ACTION_TOPIC: []
        }
        if robot_type == "r1pro":
            self.target_topics[EE_POSE_ACTION_LEFT_TOPIC] = []
            self.target_topics[EE_POSE_ACTION_RIGHT_TOPIC] = []
        if robot_type == "r1lite":
            self.target_topics[TORSO_ACTION_SPEED_TOPIC] = []
        
        # Topic to lerobot key mappings
        self.topic_to_key = {
            RGB_HEAD_LEFT_TOPIC: "observation.images.head_rgb",
            RGB_HEAD_RIGHT_TOPIC: "observation.images.head_right_rgb",
            rgb_wrist_left_topic: "observation.images.left_wrist_rgb",
            rgb_wrist_right_topic: "observation.images.right_wrist_rgb",
            
            JOINT_OBS_LEFT_TOPIC: {
                "position": "observation.state.left_arm",
                "velocity": "observation.state.left_arm.velocities"
            },
            JOINT_OBS_RIGHT_TOPIC: {
                "position": "observation.state.right_arm",
                "velocity": "observation.state.right_arm.velocities"
            },
            GRIPPER_OBS_LEFT_TOPIC: "observation.state.left_gripper",
            GRIPPER_OBS_RIGHT_TOPIC: "observation.state.right_gripper",
            
            CHASSIS_IMU_TOPIC: "observation.state.chassis.imu",
            CHASSIS_OBS_TOPIC: {
                "position": "observation.state.chassis",
                "velocity": "observation.state.chassis.velocities"
            },
            TORSO_OBS_TOPIC: {
                "position": "observation.state.torso",
                "velocity": "observation.state.torso.velocities"
            },
            
            EE_POSE_OBS_LEFT_TOPIC: "observation.state.left_ee_pose",
            EE_POSE_OBS_RIGHT_TOPIC: "observation.state.right_ee_pose",
            
            # Actions
            GRIPPER_ACTION_LEFT_TOPIC: "action.left_gripper",
            GRIPPER_ACTION_RIGHT_TOPIC: "action.right_gripper",
            JOINT_ACTION_LEFT_TOPIC: "action.left_arm",
            JOINT_ACTION_RIGHT_TOPIC: "action.right_arm",
            CHASSIS_ACTION_TOPIC: "action.chassis.velocities",
            TORSO_ACTION_TOPIC: "action.torso",
            TORSO_ACTION_SPEED_TOPIC: "action.torso.velocities",
        }
        
        if robot_type == "r1pro":
            self.topic_to_key[EE_POSE_ACTION_LEFT_TOPIC] = "action.left_ee_pose"
            self.topic_to_key[EE_POSE_ACTION_RIGHT_TOPIC] = "action.right_ee_pose"
        
        # Processing type for each topic
        self.topic_processing_type = {}
        for topic in self.rgb_topics + [RGB_HEAD_LEFT_TOPIC]:
            self.topic_processing_type[topic] = "image"
        for topic in self.joint_topics:
            self.topic_processing_type[topic] = "joint"
        for topic in self.pose_topics:
            self.topic_processing_type[topic] = "pose"
        for topic in self.gripper_topics:
            self.topic_processing_type[topic] = "gripper"
        for topic in self.twist_topics:
            self.topic_processing_type[topic] = "twist"
        for topic in self.imu_topics:
            self.topic_processing_type[topic] = "imu"
        for topic in self.control_topics:
            self.topic_processing_type[topic] = "control"
    
    def get_lerobot_key(self, topic, field=None):
        """Get the lerobot dataset key for a given topic and optional field."""
        mapping = self.topic_to_key.get(topic)
        if isinstance(mapping, dict):
            return mapping.get(field) if field else mapping
        return mapping
    
    def get_processing_type(self, topic):
        """Get the processing type for a topic."""
        return self.topic_processing_type.get(topic, "unknown")
    
    def to_dict(self):
        """Convert config to dictionary for YAML serialization."""
        return {
            "robot_type": self.robot_type,
            "arm_dof": self.arm_dof,
            "save_video": self.save_video,
            "rgb_wrist_left_topic": self.rgb_wrist_left_topic,
            "rgb_wrist_right_topic": self.rgb_wrist_right_topic,
            "rgb_topics": self.rgb_topics,
            "depth_topics": self.depth_topics,
            "joint_topics": self.joint_topics,
            "pose_topics": self.pose_topics,
            "gripper_topics": self.gripper_topics,
            "twist_topics": self.twist_topics,
            "control_topics": self.control_topics,
            "imu_topics": self.imu_topics,
            "target_topics": list(self.target_topics.keys()),
            "topic_to_key": self.topic_to_key,
            "topic_processing_type": self.topic_processing_type,
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create TopicMappingConfig from dictionary."""
        # Extract basic parameters
        robot_type = config_dict["robot_type"]
        arm_dof = config_dict["arm_dof"]
        save_video = config_dict.get("save_video", False)
        rgb_wrist_left_topic = config_dict["rgb_wrist_left_topic"]
        rgb_wrist_right_topic = config_dict["rgb_wrist_right_topic"]
        
        # Create config instance
        config = cls(
            robot_type=robot_type,
            rgb_wrist_left_topic=rgb_wrist_left_topic,
            rgb_wrist_right_topic=rgb_wrist_right_topic,
            arm_dof=arm_dof,
            save_video=save_video
        )
        
        # Override with loaded values if they differ (for custom configs)
        if "rgb_topics" in config_dict:
            config.rgb_topics = config_dict["rgb_topics"]
        if "depth_topics" in config_dict:
            config.depth_topics = config_dict["depth_topics"]
        if "joint_topics" in config_dict:
            config.joint_topics = config_dict["joint_topics"]
        if "pose_topics" in config_dict:
            config.pose_topics = config_dict["pose_topics"]
        if "gripper_topics" in config_dict:
            config.gripper_topics = config_dict["gripper_topics"]
        if "twist_topics" in config_dict:
            config.twist_topics = config_dict["twist_topics"]
        if "control_topics" in config_dict:
            config.control_topics = config_dict["control_topics"]
        if "imu_topics" in config_dict:
            config.imu_topics = config_dict["imu_topics"]
        if "topic_to_key" in config_dict:
            config.topic_to_key = config_dict["topic_to_key"]
        if "topic_processing_type" in config_dict:
            config.topic_processing_type = config_dict["topic_processing_type"]
        if "target_topics" in config_dict:
            # Reconstruct target_topics dict from list of keys
            config.target_topics = {topic: [] for topic in config_dict["target_topics"]}
        
        return config


def save_topic_mapping_config(config: TopicMappingConfig, filepath: str):
    """
    Save TopicMappingConfig to a YAML file.
    
    Args:
        config: TopicMappingConfig instance to save
        filepath: Path to the YAML file to save to
        
    Raises:
        IOError: If file cannot be written
        ValueError: If config is invalid
    """
    if not isinstance(config, TopicMappingConfig):
        raise ValueError(f"Expected TopicMappingConfig instance, got {type(config)}")
    
    config_dict = config.to_dict()
    
    # Create directory if it doesn't exist
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    logger.info(f"TopicMappingConfig saved to {filepath}")


def load_topic_mapping_config(filepath: str) -> TopicMappingConfig:
    """
    Load TopicMappingConfig from a YAML file.
    
    Args:
        filepath: Path to the YAML file to load from
        
    Returns:
        TopicMappingConfig instance
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
        KeyError: If required fields are missing
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        raise ValueError(f"Empty or invalid YAML file: {filepath}")
    
    # Validate required fields
    required_fields = ["robot_type", "arm_dof", "rgb_wrist_left_topic", "rgb_wrist_right_topic"]
    missing_fields = [field for field in required_fields if field not in config_dict]
    if missing_fields:
        raise ValueError(f"Missing required fields in config: {missing_fields}")
    
    config = TopicMappingConfig.from_dict(config_dict)
    logger.info(f"TopicMappingConfig loaded from {filepath}")
    
    return config


def create_topic_mapping_config(
    robot_type: str,
    sample_mcap_path: str,
    use_ros1: bool = False,
    save_video: bool = False
) -> TopicMappingConfig:
    """
    Create TopicMappingConfig by resolving topics from sample bag/mcap file.
    
    Args:
        robot_type: Robot type ('r1pro', 'r1', 'r1lite')
        sample_mcap_path: Path to sample mcap/bag file to resolve topics
        use_ros1: Whether using ROS1 (False for ROS2)
        save_video: Whether to save video format
        
    Returns:
        TopicMappingConfig instance
    """
    # Determine arm DOF
    if robot_type == "r1pro":
        arm_dof = 7
    else:
        arm_dof = 6

    # Resolve wrist camera topics (check for "_rect" variant)
    rgb_wrist_left_topic = RGB_WRIST_LEFT_TOPIC
    rgb_wrist_right_topic = RGB_WRIST_RIGHT_TOPIC
    
    if not use_ros1:
        # check if wrist camera topic has "_rect", can be cleaned by ATC 2.1.5
        if SequentialReader is None or StorageOptions is None or ConverterOptions is None:
            logger.warning("ROS2 modules not available, cannot check wrist camera topics.")
        else:
            reader = SequentialReader()
            storage_options = StorageOptions(uri=sample_mcap_path, storage_id="mcap")
            converter_options = ConverterOptions()
            reader.open(storage_options, converter_options)
            all_topics = [topic.name for topic in reader.get_all_topics_and_types()]
            if rgb_wrist_left_topic not in all_topics or rgb_wrist_right_topic not in all_topics:
                rgb_wrist_left_topic = rgb_wrist_left_topic.replace("image_raw", "image_rect_raw")
                rgb_wrist_right_topic = rgb_wrist_right_topic.replace("image_raw", "image_rect_raw")
            assert rgb_wrist_left_topic in all_topics and rgb_wrist_right_topic in all_topics
    
    # Create configuration mapping
    config = TopicMappingConfig(
        robot_type=robot_type,
        rgb_wrist_left_topic=rgb_wrist_left_topic,
        rgb_wrist_right_topic=rgb_wrist_right_topic,
        arm_dof=arm_dof,
        save_video=save_video
    )
    
    return config

