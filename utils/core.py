import torch
import json
import textwrap
from pathlib import Path
import inspect
import pandas as pd


def find_project_root() -> Path:
    current_path = Path(__file__).resolve()
    while current_path != current_path.root:
        if (current_path / 'utils').exists():  # Check if 'utils' directory exists
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError("Project root not found")


class QuizManager:
    """Utility class for managing interactive quizzes"""

    def __init__(self, session: str):
        self.session = session
        project_root = find_project_root()
        self.quizzes_path = project_root / 'utils/quizzes.json'

        # Load quizzes if the file exists, otherwise create a default empty structure
        if self.quizzes_path.exists():
            with open(self.quizzes_path, 'r') as f:
                self.quizzes = json.load(f)
        else:
            self.quizzes = {}
            print("📝 Note: quizzes.json not found. Using default empty quizzes.")

    def run_quiz(self, quiz_number: int):
        """Run a quiz by its number"""
        try:
            quiz_key = f"quiz_{quiz_number}"
            if self.session not in self.quizzes or quiz_key not in self.quizzes[self.session]:
                print(
                    f"❌ Quiz {quiz_number} not found for session {self.session}")
                return

            quiz = self.quizzes[self.session][quiz_key]

            print("-" * 80)
            print(f"📋 {quiz['title']}")
            print("-" * 80)

            # Display the question and options
            print(quiz["question"])
            print()

            for i, option in enumerate(quiz["options"]):
                # Split options by newlines to make them more readable
                option_lines = option.split('\n')
                first_line = option_lines[0]
                print(f"{chr(65 + i)}. {first_line}")

                # Print remaining lines with proper indentation
                for line in option_lines[1:]:
                    print(f"   {line}")
                print()  # Extra spacing between options

            # Get the answer from the user (wrapped in try-catch for interrupt handling)
            while True:
                try:
                    answer = input(
                        "Enter your answer (A, B, C, etc.): ").strip().upper()
                    if answer and answer in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(quiz["options"])]:
                        break
                    else:
                        print(
                            f"Please enter a valid option (A-{chr(64 + len(quiz['options']))})")
                except KeyboardInterrupt:
                    print("\nQuiz aborted.")
                    return

            # Check the answer
            correct_index = quiz["correct"]
            correct_letter = chr(65 + correct_index)

            if answer == correct_letter:
                print("\n✅ Correct!")
            else:
                print(
                    f"\n❌ Incorrect. The correct answer is {correct_letter}.")

            # Display the explanation with better formatting
            print("\n📚 Explanation:")

            # Split explanation by newlines and print each line
            wrapped_lines = textwrap.wrap(quiz["explanation"], width=80)
            for line in wrapped_lines:
                print(line)

            print("-" * 80)

        except Exception as e:
            print(f"❌ Error running quiz: {str(e)}")


class ExerciseChecker:
    """Utility class for checking workshop exercises"""

    def __init__(self, session: str):
        self.session = session
        project_root = find_project_root()
        self.solutions_path = project_root / 'utils/solutions.json'
        with open(self.solutions_path, 'r') as f:
            self.answers = json.load(f)
        self.hints_shown = set()

    @staticmethod
    def _print_success(msg: str) -> None:
        print(f"✅ {msg}")

    @staticmethod
    def _print_error(msg: str) -> None:
        print(f"❌ {msg}")

    @staticmethod
    def _print_hint(msg: str) -> None:
        print(f"💡 Hint: {msg}")

    def check_exercise(self, exercise: int, student_answer: dict) -> None:
        """
        Check student answer against solution.
        Handles various types including tensors, models and metrics.
        """
        print('-' * 80)
        exercise_key = f"exercise_{exercise}"
        exercise_data = self.answers[self.session][exercise_key]
        all_correct = True

        for key, expected in exercise_data["answers"].items():
            if key not in student_answer:
                self._print_error(f"Missing {key}")
                self._show_relevant_hint(exercise_data["hints"], key)
                all_correct = False
                continue

            try:
                value_correct = self._check_value(
                    student_value=student_answer[key],
                    expected=expected,
                    key=key,
                    exercise_data=exercise_data
                )

                if value_correct:
                    self._print_success(f"{key} is correct!")
                else:
                    all_correct = False

            except Exception as e:
                self._print_error(f"Error checking {key}: {str(e)}")
                self._show_relevant_hint(exercise_data["hints"], key)
                all_correct = False

        if all_correct:
            print("\n🎉 Excellent! All parts are correct!")

    def _check_value(self, student_value, expected, key: str, exercise_data: dict) -> bool:
        """Check a single value against expected solution"""

        # Handle functions (like sigmoid)
        if "test_cases" in expected:
            return self._check_function(student_value, expected, key, exercise_data)

        # Handle regular tensors
        elif expected.get("dtype", "").startswith("torch."):
            return self._check_tensor(student_value, expected, key, exercise_data)

        # Handle PyTorch models
        elif isinstance(student_value, torch.nn.Module):
            return self._check_model(student_value, expected, key, exercise_data)

        # Handle tensor shapes (special case for torch.Size objects)
        elif isinstance(student_value, torch.Size):
            if "shape" in expected:
                expected_shape = tuple(expected["shape"])
                if student_value != expected_shape:
                    self._print_error(
                        f"{key} has wrong shape. Expected {expected_shape}, got {student_value}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
            return True

        # Handle weight initialization checks specific to SE03 exercise 2
        elif key.endswith("_weight_init") or key.endswith("_bias_init"):
            return self._check_weight_initialization(student_value, expected, key, exercise_data)

        # Handle CNN-specific checks (added for SE04)
        elif key.endswith("_transform") and "transform_type" in expected:
            return self._check_transforms(student_value, expected, key, exercise_data)

        elif "model_type" in expected and key.endswith("_architecture"):
            return self._check_model_architecture(student_value, expected, key, exercise_data)

        elif key == "conv_layer" or key == "conv_layers":
            return self._check_conv_layers(student_value, expected, key, exercise_data)

        elif key == "flatten_operation":
            return self._check_flatten_operation(student_value, expected, key, exercise_data)

        elif key == "pooling_layers":
            return self._check_pooling_layers(student_value, expected, key, exercise_data)

        elif key == "dropout_layers":
            return self._check_dropout_layers(student_value, expected, key, exercise_data)

        elif key == "batch_norm_layers":
            return self._check_batch_norm_layers(student_value, expected, key, exercise_data)

        elif key == "linear_layers":
            return self._check_linear_layers(student_value, expected, key, exercise_data)

        elif key == "activation":
            return self._check_activation_layers(student_value, expected, key, exercise_data)

        elif key == "depthwise_blocks":
            return self._check_depthwise_blocks(student_value, expected, key, exercise_data)

        elif key == "global_avg_pool":
            return self._check_adaptive_pooling(student_value, expected, key, exercise_data)

        elif key.endswith("dataloader") and "loader_type" in expected:
            return self._check_dataloader(student_value, expected, key, exercise_data)

        elif key == "classification_report" and "contains_metrics" in expected:
            return self._check_classification_report(student_value, expected, key, exercise_data)

        # Handle metrics with min/max range or expected value with tolerance
        elif "expected" in expected or "min_val" in expected or "max_val" in expected:
            # Handle exact value with tolerance
            if "expected" in expected:
                tolerance = expected.get("tolerance", 1e-4)
                if abs(student_value - expected["expected"]) > tolerance:
                    self._print_error(
                        f"{key} has incorrect value. "
                        f"Expected {expected['expected']:.4f}, got {student_value:.4f}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

            # Handle min/max range checks
            if "min_val" in expected and student_value < expected["min_val"]:
                self._print_error(
                    f"{key} is too small: {student_value:.4f}. Should be at least {expected['min_val']:.4f}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

            if "max_val" in expected and student_value > expected["max_val"]:
                self._print_error(
                    f"{key} is too large: {student_value:.4f}. Should be at most {expected['max_val']:.4f}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

            return True

        # Handle metrics specifically (can contain dictionaries or ranges)
        elif "metrics" in expected:
            return self._check_metrics(student_value, expected, key, exercise_data)

        return True

    def _check_transforms(self, student_transform, expected, key: str, exercise_data: dict) -> bool:
        """Validate image transformation configurations"""
        import torchvision.transforms as T

        # Check transform type
        if not isinstance(student_transform, T.Compose):
            self._print_error(f"{key} should be a transforms.Compose object")
            self._show_relevant_hint(exercise_data["hints"], key)
            return False

        # Check for required transform components
        transform_list = student_transform.transforms
        transform_names = [t.__class__.__name__ for t in transform_list]

        for required_transform in expected.get("contains", []):
            if required_transform not in transform_names:
                self._print_error(
                    f"{key} missing {required_transform} transform")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        # Check specific transform parameters
        if "resize_size" in expected:
            resize_transforms = [
                t for t in transform_list if isinstance(t, T.Resize)]
            if not resize_transforms:
                self._print_error(f"{key} should include a Resize transform")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
            resize_size = resize_transforms[0].size
            if resize_size != tuple(expected["resize_size"]) and resize_size != expected["resize_size"][0]:
                self._print_error(
                    f"{key} has incorrect resize size. Expected {expected['resize_size']}")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        if "crop_size" in expected:
            crop_transforms = [t for t in transform_list
                               if isinstance(t, T.CenterCrop) or isinstance(t, T.RandomCrop)]
            if not crop_transforms:
                self._print_error(f"{key} should include a crop transform")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
            crop_size = crop_transforms[0].size[0]
            if crop_size != expected["crop_size"]:
                self._print_error(f"{key} has incorrect crop size")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        if "flip_probability" in expected:
            flip_transforms = [t for t in transform_list if isinstance(
                t, T.RandomHorizontalFlip)]
            if not flip_transforms:
                self._print_error(
                    f"{key} should include a RandomHorizontalFlip transform")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
            flip_prob = flip_transforms[0].p
            if abs(flip_prob - expected["flip_probability"]) > 1e-5:
                self._print_error(f"{key} has incorrect flip probability")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        if "rotation_degrees" in expected:
            rotate_transforms = [
                t for t in transform_list if isinstance(t, T.RandomRotation)]
            if not rotate_transforms:
                self._print_error(
                    f"{key} should include a RandomRotation transform")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
            degrees = rotate_transforms[0].degrees
            if isinstance(degrees, (tuple, list)):
                degrees = max(degrees)
            if degrees != expected["rotation_degrees"]:
                self._print_error(f"{key} has incorrect rotation degrees")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        if "norm_mean" in expected or "norm_std" in expected:
            norm_transforms = [
                t for t in transform_list if isinstance(t, T.Normalize)]
            if not norm_transforms:
                self._print_error(
                    f"{key} should include a Normalize transform")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

            if "norm_mean" in expected:
                mean = norm_transforms[0].mean
                if not all(abs(m1 - m2) < 1e-5 for m1, m2 in zip(mean, expected["norm_mean"])):
                    self._print_error(
                        f"{key} has incorrect normalization mean")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

            if "norm_std" in expected:
                std = norm_transforms[0].std
                if not all(abs(s1 - s2) < 1e-5 for s1, s2 in zip(std, expected["norm_std"])):
                    self._print_error(f"{key} has incorrect normalization std")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

        return True

    def _check_model_architecture(self, student_model, expected, key: str, exercise_data: dict) -> bool:
        """Validate CNN model architecture"""
        if not isinstance(student_model, torch.nn.Module):
            self._print_error(f"{key} should be a PyTorch model (nn.Module)")
            self._show_relevant_hint(exercise_data["hints"], key)
            return False

        if expected.get("has_init", False) and not hasattr(student_model, "__init__"):
            self._print_error(f"{key} should have an __init__ method")
            self._show_relevant_hint(exercise_data["hints"], key)
            return False

        if expected.get("has_forward", False) and not hasattr(student_model, "forward"):
            self._print_error(f"{key} should have a forward method")
            self._show_relevant_hint(exercise_data["hints"], key)
            return False

        if "layers_count" in expected:
            # Count layers by inspecting module children
            layers_count = len(list(student_model.modules()))
            min_count = expected["layers_count"]["min"]
            max_count = expected["layers_count"]["max"]

            if layers_count < min_count or layers_count > max_count:
                self._print_error(
                    f"{key} should have between {min_count} and {max_count} layers, found {layers_count}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        return True

    def _check_conv_layers(self, student_model, expected, key: str, exercise_data: dict) -> bool:
        """Validate convolutional layers in a CNN model"""
        # Find all Conv2d layers in the model
        conv_layers = []
        for name, module in student_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append(module)

        # Check if we have expected number of convolutional layers
        if "count" in expected:
            min_count = expected["count"]["min"]
            max_count = expected["count"]["max"]

            if len(conv_layers) < min_count or len(conv_layers) > max_count:
                self._print_error(
                    f"{key} should have between {min_count} and {max_count} Conv2d layers, found {len(conv_layers)}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        # Check first conv layer parameters
        if conv_layers and "channels" in expected and "first_layer" in expected["channels"]:
            first_layer = conv_layers[0]

            if "in" in expected["channels"]["first_layer"] and first_layer.in_channels != expected["channels"]["first_layer"]["in"]:
                self._print_error(
                    f"First conv layer should have {expected['channels']['first_layer']['in']} input channels, found {first_layer.in_channels}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

            if "out" in expected["channels"]["first_layer"]:
                out_spec = expected["channels"]["first_layer"]["out"]
                if isinstance(out_spec, dict):
                    min_out = out_spec["min"]
                    max_out = out_spec["max"]
                    if first_layer.out_channels < min_out or first_layer.out_channels > max_out:
                        self._print_error(
                            f"First conv layer should have between {min_out} and {max_out} output channels, found {first_layer.out_channels}"
                        )
                        self._show_relevant_hint(exercise_data["hints"], key)
                        return False
                else:
                    if first_layer.out_channels != out_spec:
                        self._print_error(
                            f"First conv layer should have {out_spec} output channels, found {first_layer.out_channels}"
                        )
                        self._show_relevant_hint(exercise_data["hints"], key)
                        return False

        # Check for increasing channel pattern
        if len(conv_layers) > 1 and "channels" in expected and expected["channels"].get("increasing_pattern", False):
            for i in range(1, len(conv_layers)):
                if conv_layers[i].out_channels <= conv_layers[i-1].out_channels:
                    self._print_error(
                        f"Conv layers should have increasing number of channels, found {conv_layers[i-1].out_channels} → {conv_layers[i].out_channels}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

        # Check kernel sizes
        if "kernel_sizes" in expected:
            min_size = expected["kernel_sizes"]["min"]
            max_size = expected["kernel_sizes"]["max"]

            for i, layer in enumerate(conv_layers):
                if isinstance(layer.kernel_size, int):
                    k_size = layer.kernel_size
                else:
                    k_size = layer.kernel_size[0]  # Assume square kernels

                if k_size < min_size or k_size > max_size:
                    self._print_error(
                        f"Conv layer {i} should have kernel size between {min_size} and {max_size}, found {k_size}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

        return True

    def _check_flatten_operation(self, student_model, expected, key: str, exercise_data: dict) -> bool:
        """Validate flattening operation in CNN model"""
        # If we're checking a boolean flag for flatten operation
        if isinstance(student_model, bool):
            # If student_model is True, it means flattening is implemented
            return student_model

        # Original code for checking model objects
        has_flatten = False
        for name, module in student_model.named_modules():
            if isinstance(module, torch.nn.Flatten):
                has_flatten = True
                break

        # Check for view or reshape operations in forward method if no Flatten layer
        if not has_flatten:
            forward_code = inspect.getsource(student_model.forward)
            if "view" not in forward_code and "reshape" not in forward_code and "flatten" not in forward_code.lower():
                self._print_error(
                    f"{key} missing flatten/reshape operation before linear layers")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        return True

    def _check_pooling_layers(self, student_model, expected, key: str, exercise_data: dict) -> bool:
        """Validate pooling layers in CNN model"""
        pooling_layers = []
        for name, module in student_model.named_modules():
            if isinstance(module, torch.nn.MaxPool2d) or isinstance(module, torch.nn.AvgPool2d):
                pooling_layers.append(module)

        # Check number of pooling layers
        if "count" in expected:
            min_count = expected["count"]["min"]
            max_count = expected["count"]["max"]

            if len(pooling_layers) < min_count or len(pooling_layers) > max_count:
                self._print_error(
                    f"{key} should have between {min_count} and {max_count} pooling layers, found {len(pooling_layers)}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        # Check pooling type
        if pooling_layers and "type" in expected:
            for i, layer in enumerate(pooling_layers):
                if layer.__class__.__name__ != expected["type"]:
                    self._print_error(
                        f"Pooling layer {i} should be {expected['type']}, found {layer.__class__.__name__}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

        # Check kernel size
        if pooling_layers and "kernel_size" in expected:
            min_size = expected["kernel_size"]["min"]
            max_size = expected["kernel_size"]["max"]

            for i, layer in enumerate(pooling_layers):
                if isinstance(layer.kernel_size, int):
                    k_size = layer.kernel_size
                else:
                    k_size = layer.kernel_size[0]  # Assume square kernels

                if k_size < min_size or k_size > max_size:
                    self._print_error(
                        f"Pooling layer {i} should have kernel size between {min_size} and {max_size}, found {k_size}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

        return True

    def _check_batch_norm_layers(self, student_model, expected, key: str, exercise_data: dict) -> bool:
        """Validate batch normalization layers in CNN model"""
        # Handle case where we're given a list of batch norm layers
        if isinstance(student_model, list):
            # Check if all elements are batch norm layers
            if not all(isinstance(layer, torch.nn.BatchNorm2d) for layer in student_model):
                self._print_error(f"{key} should contain only BatchNorm2d layers")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
                
            # Check count of batch norm layers
            if "count" in expected:
                if len(student_model) < expected["count"]["min"] or len(student_model) > expected["count"]["max"]:
                    self._print_error(
                        f"{key} should have between {expected['count']['min']} and {expected['count']['max']} batch norm layers, found {len(student_model)}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
                                
            # Check momentum if specified
            if "momentum" in expected:
                for i, layer in enumerate(student_model):
                    min_momentum = expected["momentum"]["min"] if isinstance(expected["momentum"], dict) else expected["momentum"]
                    max_momentum = expected["momentum"]["max"] if isinstance(expected["momentum"], dict) else expected["momentum"]
                    
                    if layer.momentum < min_momentum or layer.momentum > max_momentum:
                        self._print_error(f"Batch norm layer {i} should have momentum between {min_momentum} and {max_momentum}")
                        self._show_relevant_hint(exercise_data["hints"], key)
                        return False
            
            return True
    
        bn_layers = []
        for name, module in student_model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                bn_layers.append(module)

        # Check if batch normalization is present
        if expected.get("present", False) and not bn_layers:
            self._print_error(
                f"{key} should include batch normalization layers")
            self._show_relevant_hint(exercise_data["hints"], key)
            return False

        # Check number of batch norm layers
        if "count" in expected:
            min_count = expected["count"]["min"]
            max_count = expected["count"]["max"]

            if len(bn_layers) < min_count or len(bn_layers) > max_count:
                self._print_error(
                    f"{key} should have between {min_count} and {max_count} batch normalization layers, found {len(bn_layers)}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        # Check momentum
        if bn_layers and "momentum" in expected:
            min_momentum = expected["momentum"]["min"]
            max_momentum = expected["momentum"]["max"]

            for i, layer in enumerate(bn_layers):
                if layer.momentum < min_momentum or layer.momentum > max_momentum:
                    self._print_error(
                        f"Batch norm layer {i} should have momentum between {min_momentum} and {max_momentum}, found {layer.momentum}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

        return True

    def _check_dropout_layers(self, student_model, expected, key: str, exercise_data: dict) -> bool:
        """Validate dropout layers in CNN model"""
        # Handle case where we're given a list of dropout layers
        if isinstance(student_model, list):
            # Check if all elements are dropout layers
            if not all(isinstance(layer, torch.nn.Dropout) for layer in student_model):
                self._print_error(f"{key} should contain only Dropout layers")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
                
            # Check count of dropout layers
            if "count" in expected:
                if len(student_model) < expected["count"]["min"] or len(student_model) > expected["count"]["max"]:
                    self._print_error(f"{key} should have {expected['count']} dropout layers, found {len(student_model)}")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
                
            # Check dropout rate if specified
            if "rate" in expected:
                for i, layer in enumerate(student_model):
                    min_rate = expected["rate"]["min"] if isinstance(expected["rate"], dict) else expected["rate"]
                    max_rate = expected["rate"]["max"] if isinstance(expected["rate"], dict) else expected["rate"]
                    
                    if layer.p < min_rate or layer.p > max_rate:
                        self._print_error(f"Dropout layer {i} should have rate between {min_rate} and {max_rate}")
                        self._show_relevant_hint(exercise_data["hints"], key)
                        return False
            
            return True
        
        dropout_layers = []
        for name, module in student_model.named_modules():
            if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
                dropout_layers.append(module)

        # Check if dropout is present
        if expected.get("present", False) and not dropout_layers:
            self._print_error(
                f"{key} should include dropout layers for regularization")
            self._show_relevant_hint(exercise_data["hints"], key)
            return False

        # Check number of dropout layers
        if "count" in expected:
            min_count = expected["count"]["min"]
            max_count = expected["count"]["max"]

            if len(dropout_layers) < min_count or len(dropout_layers) > max_count:
                self._print_error(
                    f"{key} should have between {min_count} and {max_count} dropout layers, found {len(dropout_layers)}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        # Check dropout rate
        if dropout_layers and "rate" in expected:
            min_rate = expected["rate"]["min"]
            max_rate = expected["rate"]["max"]

            for i, layer in enumerate(dropout_layers):
                if layer.p < min_rate or layer.p > max_rate:
                    self._print_error(
                        f"Dropout layer {i} should have rate between {min_rate} and {max_rate}, found {layer.p}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

        return True

    def _check_linear_layers(self, student_model, expected, key: str, exercise_data: dict) -> bool:
        """Validate linear layers in CNN model"""
        # If we're checking a dictionary containing linear layer information
        if isinstance(student_model, dict):
            # Check count of linear layers
            if "count" in expected and "count" in student_model:
                min_count = expected["count"]["min"] if isinstance(expected["count"], dict) else expected["count"]
                max_count = expected["count"]["max"] if isinstance(expected["count"], dict) else expected["count"]
                
                if student_model["count"] < min_count or student_model["count"] > max_count:
                    self._print_error(
                        f"{key} should have between {min_count} and {max_count} linear layers, found {student_model['count']}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
                    
            # Check output features
            if "output_features" in expected and "output_features" in student_model:
                if student_model["output_features"] != expected["output_features"]:
                    self._print_error(
                        f"Output layer should have {expected['output_features']} output features, found {student_model['output_features']}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
            
            return True
            
        # Original code for checking model objects
        linear_layers = []
        for name, module in student_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_layers.append(module)
                
        # Check number of linear layers
        if "count" in expected:
            min_count = expected["count"]["min"] if isinstance(expected["count"], dict) else expected["count"]
            max_count = expected["count"]["max"] if isinstance(expected["count"], dict) else expected["count"]
            
            if len(linear_layers) < min_count or len(linear_layers) > max_count:
                self._print_error(
                    f"{key} should have between {min_count} and {max_count} linear layers, found {len(linear_layers)}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
                
        # Check output dimensions
        if linear_layers and "output_features" in expected:
            last_layer = linear_layers[-1]
            if last_layer.out_features != expected["output_features"]:
                self._print_error(
                    f"Output layer should have {expected['output_features']} output features, found {last_layer.out_features}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
                
        return True

    def _check_activation_layers(self, student_model, expected, key: str, exercise_data: dict) -> bool:
        """Validate activation functions in CNN model"""
        # If we're checking a dictionary containing activation information
        if isinstance(student_model, dict):
            # Check activation function type
            if "function" in expected and "function" in student_model:
                if student_model["function"] != expected["function"]:
                    self._print_error(
                        f"{key} should use {expected['function']} activation function, found {student_model['function']}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
                    
            # Check count of activation functions
            if "count" in expected and "count" in student_model:
                min_count = expected["count"]["min"] if isinstance(expected["count"], dict) else expected["count"]
                max_count = expected["count"]["max"] if isinstance(expected["count"], dict) else expected["count"]
                
                if student_model["count"] < min_count or student_model["count"] > max_count:
                    self._print_error(
                        f"{key} should have between {min_count} and {max_count} activation functions, found {student_model['count']}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
                    
            return True
        
        # Original code for checking model objects
        activation_layers = []
        for name, module in student_model.named_modules():
            if isinstance(module, torch.nn.ReLU) or isinstance(module, torch.nn.LeakyReLU) or \
            isinstance(module, torch.nn.Sigmoid) or isinstance(module, torch.nn.Tanh):
                activation_layers.append(module)
                
        # Check if using expected activation function
        if "function" in expected and activation_layers:
            correct_function = False
            for layer in activation_layers:
                if layer.__class__.__name__ == expected["function"]:
                    correct_function = True
                    break
                    
            if not correct_function:
                self._print_error(f"{key} should use {expected['function']} activation function")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
                
        # Check number of activation layers
        if "count" in expected:
            min_count = expected["count"]["min"] if isinstance(expected["count"], dict) else expected["count"]
            max_count = expected["count"]["max"] if isinstance(expected["count"], dict) else expected["count"]
            
            if len(activation_layers) < min_count or len(activation_layers) > max_count:
                self._print_error(
                    f"{key} should have between {min_count} and {max_count} activation layers, found {len(activation_layers)}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
                
        # If no explicit activation layers, check for functional calls in forward method
        if not activation_layers:
            forward_code = inspect.getsource(student_model.forward)
            if "relu" not in forward_code.lower() and "sigmoid" not in forward_code.lower() and \
            "tanh" not in forward_code.lower() and "leaky_relu" not in forward_code.lower():
                self._print_error(f"{key} should include activation functions")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
                
        return True


    def _check_depthwise_blocks(self, student_model, expected, key: str, exercise_data: dict) -> bool:
        """Validate depthwise separable convolution pattern in a MobileNet-style model."""
        if not isinstance(student_model, torch.nn.Module):
            self._print_error(f"{key} should be a PyTorch model (nn.Module)")
            self._show_relevant_hint(exercise_data["hints"], key)
            return False

        # Depthwise: Conv2d where groups == in_channels and groups > 1
        depthwise_convs = [
            m for m in student_model.modules()
            if isinstance(m, torch.nn.Conv2d) and m.groups == m.in_channels and m.groups > 1
        ]
        # Pointwise: 1×1 Conv2d
        pointwise_convs = [
            m for m in student_model.modules()
            if isinstance(m, torch.nn.Conv2d)
            and (m.kernel_size == (1, 1) or m.kernel_size == 1)
        ]

        min_blocks = expected.get("min_blocks", 1)
        if len(depthwise_convs) < min_blocks:
            self._print_error(
                f"{key} should have at least {min_blocks} depthwise Conv2d layer(s) "
                f"(groups=in_channels), found {len(depthwise_convs)}. "
                f"Set groups=in_channels to create a depthwise convolution."
            )
            self._show_relevant_hint(exercise_data["hints"], key)
            return False

        if expected.get("has_pointwise", False) and not pointwise_convs:
            self._print_error(
                f"{key} should include at least one pointwise (1×1) Conv2d layer"
            )
            self._show_relevant_hint(exercise_data["hints"], key)
            return False

        return True

    def _check_adaptive_pooling(self, student_model, expected, key: str, exercise_data: dict) -> bool:
        """Validate that the model contains AdaptiveAvgPool2d (global average pooling)."""
        if not isinstance(student_model, torch.nn.Module):
            self._print_error(f"{key} should be a PyTorch model (nn.Module)")
            self._show_relevant_hint(exercise_data["hints"], key)
            return False

        has_adaptive = any(
            isinstance(m, torch.nn.AdaptiveAvgPool2d)
            for m in student_model.modules()
        )
        if not has_adaptive:
            self._print_error(
                f"{key} should include torch.nn.AdaptiveAvgPool2d for global average pooling. "
                f"Use AdaptiveAvgPool2d((1, 1)) to reduce each feature map to a single value."
            )
            self._show_relevant_hint(exercise_data["hints"], key)
            return False

        return True

    def _check_dataloader(self, student_dataloader, expected, key: str, exercise_data: dict) -> bool:
        """Validate PyTorch DataLoader configuration"""
        from torch.utils.data import DataLoader

        if not isinstance(student_dataloader, DataLoader):
            self._print_error(f"{key} should be a torch.utils.data.DataLoader")
            self._show_relevant_hint(exercise_data["hints"], key)
            return False

        # Check batch size
        if "batch_size" in expected:
            min_batch = expected["batch_size"]["min"]
            max_batch = expected["batch_size"]["max"]

            if student_dataloader.batch_size < min_batch or student_dataloader.batch_size > max_batch:
                self._print_error(
                    f"{key} should have batch size between {min_batch} and {max_batch}, found {student_dataloader.batch_size}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        # Check shuffle parameter
        # if "shuffle" in expected and student_dataloader.shuffle != expected["shuffle"]:
        #     expected_state = "enabled" if expected["shuffle"] else "disabled"
        #     actual_state = "enabled" if student_dataloader.shuffle else "disabled"

        #     self._print_error(f"{key} should have shuffle {expected_state}, found {actual_state}")
        #     self._show_relevant_hint(exercise_data["hints"], key)
        #     return False

        return True

    def _check_classification_report(self, student_report, expected, key: str, exercise_data: dict) -> bool:
        """Validate classification report content and metrics"""
        # Check if it's a string report from sklearn
        if isinstance(student_report, str):
            # Check for required metrics in the report string
            for metric in expected.get("contains_metrics", []):
                if metric not in student_report:
                    self._print_error(
                        f"{key} should contain {metric} in the report")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

            # Check for required classes in the report string
            for class_name in expected.get("classes", []):
                if class_name not in student_report:
                    self._print_error(
                        f"{key} should contain {class_name} in the report")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

            # Check if precision, recall and f1-score are reasonable
            # This is a basic check since parsing values from a string report is complex
            if "f1_score" in expected and "0.0" in student_report:
                min_f1 = expected["f1_score"]["min"]
                if "0.0" in student_report and min_f1 > 0:
                    self._print_error(
                        f"{key} has some very low F1 scores, check model performance")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

        # Handle case where student used a dictionary or DataFrame
        elif isinstance(student_report, (dict, pd.DataFrame)):
            # Implementation depends on actual format used
            pass

        return True

    def _check_weight_initialization(self, student_tensor, expected, key: str, exercise_data: dict) -> bool:
        """Validate weight or bias initialization"""
        if not isinstance(student_tensor, torch.Tensor):
            self._print_error(f"{key} should be a tensor")
            return False

        # Check mean close to zero for weights
        if key.endswith("_weight_init"):
            mean = student_tensor.mean().item()
            std = student_tensor.std().item()

            # Check if mean is close to zero (characteristic of proper initialization)
            if abs(mean) > 0.05:  # Allow small deviation from zero
                self._print_error(
                    f"{key} mean should be close to zero, got {mean:.4f}")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

            # Check if standard deviation is in reasonable range
            if "fc1" in key and (std < 0.1 or std > 0.5):
                self._print_error(
                    f"{key} standard deviation should be between 0.1 and 0.5 for proper He initialization, got {std:.4f}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
            elif "fc2" in key and (std < 0.1 or std > 0.5):
                self._print_error(
                    f"{key} standard deviation should be between 0.1 and 0.5 for proper Xavier initialization, got {std:.4f}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        # Check if biases are initialized to zero
        elif key.endswith("_bias_init"):
            if not torch.allclose(student_tensor, torch.zeros_like(student_tensor), atol=1e-5):
                self._print_error(f"{key} should be initialized to zeros")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        return True

    def _check_function(self, student_func, expected, key: str, exercise_data: dict) -> bool:
        """Validate function implementation using test cases"""
        if not callable(student_func):
            self._print_error(f"{key} should be a function")
            return False

        test_cases = expected.get("test_cases", [])
        tolerance = expected.get("tolerance", 1e-4)

        for case in test_cases:
            input_data = case["input"]
            expected_val = case["expected"]

            try:
                # Handle MSE loss function test case (special format for SE02.exercise_3)
                if isinstance(input_data[0], dict) and "predictions" in input_data[0]:
                    predictions = torch.tensor(
                        input_data[0]["predictions"], dtype=torch.float32)
                    targets = torch.tensor(
                        input_data[0]["targets"], dtype=torch.float32)
                    result = student_func(predictions, targets).item()
                else:
                    # Regular function test case
                    input_val = torch.tensor(input_data, dtype=torch.float32)
                    result = student_func(input_val).item()

                if abs(result - expected_val) > tolerance:
                    self._print_error(
                        f"{key} failed test case: expected {expected_val:.4f}, got {result:.4f}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
            except Exception as e:
                self._print_error(f"Error testing {key}: {str(e)}")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        return True

    def _check_tensor(self, student_tensor, expected, key: str, exercise_data: dict) -> bool:
        """Validate tensor properties and values"""
        if not isinstance(student_tensor, torch.Tensor):
            self._print_error(f"{key} should be a tensor")
            return False

        # Check shape if specified
        if "shape" in expected:
            if student_tensor.shape != tuple(expected["shape"]):
                self._print_error(
                    f"{key} has wrong shape. Expected {tuple(expected['shape'])}, got {student_tensor.shape}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        # Check value range if specified
        if "min_val" in expected and "max_val" in expected:
            if not (expected["min_val"] <= student_tensor.min().item() <= expected["max_val"] and
                    expected["min_val"] <= student_tensor.max().item() <= expected["max_val"]):
                self._print_error(
                    f"{key} values should be between {expected['min_val']} and {expected['max_val']}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        # Check exact values if specified
        if "value" in expected:
            expected_tensor = torch.tensor(
                expected["value"],
                dtype=getattr(torch, expected["dtype"].split('.')[1])
            )
            if not torch.allclose(student_tensor, expected_tensor, rtol=1e-3):
                self._print_error(f"{key} has incorrect values")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        return True

    def _check_model(self, student_model, expected, key: str, exercise_data: dict) -> bool:
        """Validate neural network model properties"""
        # Check architecture
        if "architecture" in expected:
            if not isinstance(student_model, torch.nn.Module):
                self._print_error(f"{key} has incorrect architecture")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        # Generic check for activation_type and optimizer type - compare class types generically
        if (key == "activation_type") or (key == "optimizer_type"):
            # For boolean expected value, we simply check if the type exists
            if expected.get("expected") is True:
                return True
            # Otherwise compare directly with expected type if available
            if "expected_type" in expected:
                expected_type = getattr(torch.nn, expected["expected_type"]) if key == "activation_type" else getattr(
                    torch.optim, expected["expected_type"])

                if not (student_model == expected_type or issubclass(student_model, expected_type)):
                    self._print_error(
                        f"{key} has incorrect type. Expected {expected['expected_type']}")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
            return True

        # Generic check for weight initialization parameters
        if key in ["weight_init_kaiming", "weight_init_xavier", "bias_init_zeros"]:
            if not student_model:
                self._print_error(f"{key} initialization is missing")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
            return True

        # Generic check for weight statistics
        if key.endswith("_weight_stats"):
            if not isinstance(student_model, dict):
                self._print_error(
                    f"{key} should be a dictionary with mean and std")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

            # Check if mean is close to zero
            if "mean" not in student_model:
                self._print_error(f"{key} missing 'mean' field")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

            if "mean_near_zero" in expected and expected["mean_near_zero"]:
                if abs(student_model["mean"]) > 0.1:
                    self._print_error(
                        f"{key} mean should be close to zero, got {student_model['mean']:.4f}")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

            # Check if std is in reasonable range
            if "std" not in student_model:
                self._print_error(f"{key} missing 'std' field")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

            if "std_range" in expected and len(expected["std_range"]) == 2:
                min_std, max_std = expected["std_range"]
                if student_model["std"] < min_std or student_model["std"] > max_std:
                    self._print_error(
                        f"{key} std should be between {min_std} and {max_std}, got {student_model['std']:.4f}")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

            return True

        # Generic model validation for any session/exercise
        if key == "model":
            # Check if model has expected input/output dimensions
            if hasattr(expected, "input_size") and hasattr(student_model, "fc1") and hasattr(student_model.fc1, "in_features"):
                if student_model.fc1.in_features != expected["input_size"]:
                    self._print_error(
                        f"Model input size should be {expected['input_size']}, got {student_model.fc1.in_features}")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

            # Check for appropriate hidden layer size to avoid overfitting
            if "max_hidden_size" in expected and hasattr(student_model, "fc1") and hasattr(student_model.fc1, "out_features"):
                if student_model.fc1.out_features > expected["max_hidden_size"]:
                    self._print_error(
                        f"Hidden layer size {student_model.fc1.out_features} is too large (max: {expected['max_hidden_size']}) and may overfit")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

        # Check layer properties
        if "layers" in expected:
            for layer_name, layer_props in expected["layers"].items():
                if not hasattr(student_model, layer_name):
                    self._print_error(f"{key} missing layer {layer_name}")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

                layer = getattr(student_model, layer_name)
                for prop_name, prop_value in layer_props.items():
                    if not hasattr(layer, prop_name) or getattr(layer, prop_name) != prop_value:
                        self._print_error(
                            f"{key} layer {layer_name} has incorrect {prop_name}")
                        self._show_relevant_hint(exercise_data["hints"], key)
                        return False

        # --- Dispatch to specialised CNN/MobileNet checkers -----------------
        if key.endswith("_architecture") and "model_type" in expected:
            return self._check_model_architecture(student_model, expected, key, exercise_data)

        if key in ("conv_layer", "conv_layers"):
            return self._check_conv_layers(student_model, expected, key, exercise_data)

        if key == "flatten_operation":
            return self._check_flatten_operation(student_model, expected, key, exercise_data)

        if key == "pooling_layers":
            return self._check_pooling_layers(student_model, expected, key, exercise_data)

        if key == "batch_norm_layers":
            return self._check_batch_norm_layers(student_model, expected, key, exercise_data)

        if key == "dropout_layers":
            return self._check_dropout_layers(student_model, expected, key, exercise_data)

        if key == "linear_layers":
            return self._check_linear_layers(student_model, expected, key, exercise_data)

        if key == "activation":
            return self._check_activation_layers(student_model, expected, key, exercise_data)

        if key == "depthwise_blocks":
            return self._check_depthwise_blocks(student_model, expected, key, exercise_data)

        if key == "global_avg_pool":
            return self._check_adaptive_pooling(student_model, expected, key, exercise_data)

        return True

    def _check_metrics(self, student_value, expected, key: str, exercise_data: dict) -> bool:
        """Validate training metrics like loss and accuracy"""
        metrics = expected["metrics"]

        # Handle dictionary of metrics
        if isinstance(student_value, dict):
            for metric_name, expected_range in metrics.items():
                if metric_name not in student_value:
                    self._print_error(f"{key} missing metric {metric_name}")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

                # Handle range as (min, max) tuple
                if isinstance(expected_range, (list, tuple)) and len(expected_range) == 2:
                    min_val, max_val = expected_range
                    if not (min_val <= student_value[metric_name] <= max_val):
                        self._print_error(
                            f"{key} {metric_name} should be between {min_val} and {max_val}, got {student_value[metric_name]:.4f}"
                        )
                        self._show_relevant_hint(exercise_data["hints"], key)
                        return False
                # Handle exact value with tolerance
                elif isinstance(expected_range, dict) and "expected" in expected_range:
                    expected_val = expected_range["expected"]
                    tolerance = expected_range.get("tolerance", 1e-4)
                    if abs(student_value[metric_name] - expected_val) > tolerance:
                        self._print_error(
                            f"{key} {metric_name} should be close to {expected_val}, got {student_value[metric_name]:.4f}"
                        )
                        self._show_relevant_hint(exercise_data["hints"], key)
                        return False

        # Handle single metric value with range
        else:
            # Check if metrics contains a "value" entry with (min, max) format
            if "value" in metrics and isinstance(metrics["value"], (list, tuple)) and len(metrics["value"]) == 2:
                min_val, max_val = metrics["value"]
                if not (min_val <= student_value <= max_val):
                    self._print_error(
                        f"{key} should be between {min_val} and {max_val}, got {student_value:.4f}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
            # Handle exact value with tolerance
            elif "expected" in metrics:
                expected_val = metrics["expected"]
                tolerance = metrics.get("tolerance", 1e-4)
                if abs(student_value - expected_val) > tolerance:
                    self._print_error(
                        f"{key} should be close to {expected_val}, got {student_value:.4f}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False

        return True

    def _show_relevant_hint(self, hints: list, key: str) -> None:
        """Show context-aware hints for the current error"""
        # Try to find a specific hint for this key
        specific_hint = next(
            (hint for hint in hints if key.lower() in hint.lower()),
            None
        )

        if specific_hint and specific_hint not in self.hints_shown:
            self._print_hint(specific_hint)
            self.hints_shown.add(specific_hint)
        elif hints and hints[0] not in self.hints_shown:
            self._print_hint(hints[0])
            self.hints_shown.add(hints[0])

    def display_hints(self, exercise: str) -> None:
        """Display hints for the exercise"""
        exercise_data = self.answers[self.session][f"exercise_{exercise}"]
        print("\n💡 Hints:")
        for hint in exercise_data["hints"]:
            print(f"- {hint}")
