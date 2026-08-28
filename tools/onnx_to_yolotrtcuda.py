#!/usr/bin/env python3
"""Build a baked FP16 YoloTrtCuda engine from a raw YOLO ONNX model.

The generated file is a YoloTrtCuda container, not a plain TensorRT engine.
It deliberately keeps YOLO confidence filtering and NMS outside the TensorRT
graph so the thresholds remain runtime-configurable.

The script automatically changes the graph input to UINT8 NHWC, inserts the
GPU-side layout/cast/normalization nodes, and converts the network to FP16
before TensorRT builds the engine. This is required by TensorRT 11, whose
builder is always strongly typed and no longer exposes the legacy
BuilderFlag.FP16 switch.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, NoReturn


MAGIC = b"YoloTrtCudaV1\0\0\0"
VERSION = 1
HEADER = struct.Struct("<16sIIQ")
HEARTBEAT_INTERVAL_SECONDS = 5.0
START_TIME = time.perf_counter()


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def progress(percent: int, message: str) -> None:
    elapsed = time.perf_counter() - START_TIME
    print(f"[{percent:3d}%][{elapsed:7.1f}s] {message}", flush=True)


def status(message: str) -> None:
    elapsed = time.perf_counter() - START_TIME
    print(f"[{elapsed:7.1f}s] {message}", flush=True)


def run_with_heartbeat(operation: Callable[[], Any], message: str) -> Any:
    stop_event = threading.Event()
    started_at = time.perf_counter()

    def heartbeat() -> None:
        while not stop_event.wait(HEARTBEAT_INTERVAL_SECONDS):
            elapsed = time.perf_counter() - started_at
            status(f"{message} is still running ({elapsed:.1f}s elapsed)")

    heartbeat_thread = threading.Thread(target=heartbeat, name="progress-heartbeat", daemon=True)
    heartbeat_thread.start()
    try:
        return operation()
    finally:
        stop_event.set()
        heartbeat_thread.join()


def import_tensorrt() -> Any:
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError(
            "TensorRT Python bindings are required. Install the TensorRT wheel "
            "from your TensorRT installation before running this script."
        ) from error
    return trt


def pick_unused_name(graph: Any, candidates: list[str]) -> str:
    used: set[str] = set()
    for node in graph.node:
        used.update(node.input)
        used.update(node.output)
    for value in (*graph.input, *graph.output, *graph.initializer):
        used.add(value.name)
    for name in candidates:
        if name not in used:
            return name
    fail("Unable to allocate names for the baked preprocessing nodes")


def topologically_sort_graph(graph: Any) -> None:
    available = {value.name for value in graph.input}
    available.update(value.name for value in graph.initializer)
    pending = list(graph.node)
    sorted_nodes = []

    while pending:
        remaining = []
        progress = False
        for node in pending:
            if all(not name or name in available for name in node.input):
                sorted_nodes.append(node)
                available.update(name for name in node.output if name)
                progress = True
            else:
                remaining.append(node)

        if not progress:
            unresolved = sorted({
                name
                for node in remaining
                for name in node.input
                if name and name not in available
            })
            fail(
                "Unable to topologically sort the ONNX graph; unresolved inputs: "
                + ", ".join(unresolved[:8])
            )
        pending = remaining

    del graph.node[:]
    graph.node.extend(sorted_nodes)


def onnx_value_shape(value: Any) -> list[int]:
    shape = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value") and dimension.dim_value > 0:
            shape.append(int(dimension.dim_value))
        else:
            shape.append(-1)
    return shape


def detection_feature_count(shape: list[int]) -> int | None:
    if len(shape) != 3:
        return None
    candidates = [dimension for dimension in shape[1:] if 5 <= dimension <= 512]
    return min(candidates) if candidates else None


def raw_output_layout(shape: list[int]) -> tuple[int, int] | None:
    """Return (concat axis, feature count) for a rank-3 raw YOLO output."""
    feature_count = detection_feature_count(shape)
    if feature_count is None:
        return None
    if shape[1] == feature_count and shape[2] != feature_count:
        return 2, feature_count
    if shape[2] == feature_count and shape[1] != feature_count:
        return 1, feature_count
    return None


def has_embedded_nms(model: Any) -> bool:
    nms_output_hints = (
        "num_dets",
        "num_detections",
        "detection_boxes",
        "detection_scores",
        "detection_classes",
    )
    has_num_dets = False
    has_detection_output = False
    for node in model.graph.node:
        operation = node.op_type.lower()
        if operation == "nonmaxsuppression" or "nms" in operation:
            return True
    for output in model.graph.output:
        name = output.name.lower()
        if "num_dets" in name or "num_detections" in name:
            has_num_dets = True
        if any(hint in name for hint in nms_output_hints[2:]):
            has_detection_output = True
    return has_num_dets and has_detection_output


def is_end_to_end_output(model: Any, output_name: str, shape: list[int]) -> bool:
    """Detect the NMS-free [x1, y1, x2, y2, score, class_id] YOLO output."""
    if len(shape) != 3 or shape[2] != 6:
        return False

    producers = {
        tensor_name: node
        for node in model.graph.node
        for tensor_name in node.output
        if tensor_name
    }
    output_producer = producers.get(output_name)
    if output_producer is None or output_producer.op_type != "Concat":
        return False

    operations = set()
    frontier = [output_name]
    visited = set()
    for _ in range(8):
        next_frontier = []
        for tensor_name in frontier:
            producer = producers.get(tensor_name)
            if producer is None or id(producer) in visited:
                continue
            visited.add(id(producer))
            operations.add(producer.op_type)
            next_frontier.extend(name for name in producer.input if name)
        frontier = next_frontier

    return "TopK" in operations and "GatherElements" in operations


def is_final_output_name(name: str) -> bool:
    lowered = name.lower().replace("/", ".")
    return lowered in {
        "output",
        "pred",
        "preds",
        "prediction",
        "predictions",
    }


def normalize_detection_outputs(model: Any) -> None:
    """Keep raw rank-3 detection outputs and remove unusable graph outputs.

    Some exporters expose the final concatenated prediction together with the
    three intermediate feature maps.  Those feature maps are not needed by
    YoloTrtCuda and must not remain graph outputs.  If an exporter exposes
    several already-flattened raw heads instead, keep all of them: the runtime
    binds and merges every detection output before the single NMS pass.
    """
    outputs = list(model.graph.output)
    if not outputs:
        fail("The ONNX graph has no output")
    if len(outputs) == 1:
        return

    producers = {
        output_name: node
        for node in model.graph.node
        for output_name in node.output
        if output_name
    }
    candidates = []
    for output in outputs:
        shape = onnx_value_shape(output)
        layout = raw_output_layout(shape)
        if layout is None:
            continue
        producer = producers.get(output.name)
        if producer is not None and "nms" in producer.op_type.lower():
            continue
        candidates.append((output, shape, layout, producer))

    if not candidates:
        if has_embedded_nms(model):
            fail(
                "The ONNX graph exposes EfficientNMS/NMS outputs but no raw YOLO "
                "output. Export with NMS and end-to-end postprocessing disabled."
            )
        fail(
            "The ONNX graph has multiple outputs but no rank-3 raw YOLO output "
            "that can be selected or concatenated automatically."
        )

    named_candidates = [candidate for candidate in candidates if is_final_output_name(candidate[0].name)]
    concat_candidates = [
        candidate
        for candidate in candidates
        if candidate[3] is not None and candidate[3].op_type == "Concat"
    ]

    selected = None
    action = "selected"
    if len(named_candidates) == 1:
        selected = named_candidates[0][0]
    elif len(concat_candidates) == 1:
        selected = concat_candidates[0][0]
    elif len(candidates) == 1:
        selected = candidates[0][0]

    if selected is None:
        feature_counts = {candidate[2][1] for candidate in candidates}
        if len(feature_counts) != 1:
            fail(
                "The ONNX graph has multiple raw detection outputs with "
                "different feature counts; all heads must use the same class layout."
            )
        del model.graph.output[:]
        model.graph.output.extend(candidate[0] for candidate in candidates)
        names = ", ".join(candidate[0].name for candidate in candidates)
        status(
            f"Detected {len(outputs)} ONNX outputs; keeping "
            f"{len(candidates)} raw detection outputs: {names}."
        )
        return

    del model.graph.output[:]
    model.graph.output.append(selected)
    status(
        f"Detected {len(outputs)} ONNX outputs; {action} raw detection output "
        f"'{selected.name}'."
    )


def bake_preprocess_model(model: Any, height: int, width: int, numpy: Any) -> Any:
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    if len(graph.input) != 1:
        fail("Baking requires an ONNX model with exactly one graph input")

    original = graph.input[0]
    tensor_type = original.type.tensor_type
    element_type = tensor_type.elem_type
    if element_type == TensorProto.UINT8:
        # Keep compatibility with models baked by an older standalone script.
        return model
    if element_type not in (TensorProto.FLOAT, TensorProto.FLOAT16):
        fail("Baking requires a float32/float16 NCHW input")

    dimensions = [dimension.dim_value for dimension in tensor_type.shape.dim]
    if len(dimensions) != 4 or dimensions[1] not in (1, 3):
        fail(f"Baking requires a rank-4 NCHW input with 1 or 3 channels: {dimensions}")
    if dimensions[0] not in (0, 1):
        fail("YoloTrtCuda engines use batch 1; export the ONNX model with batch=1")
    channels = dimensions[1]
    model_height = dimensions[2] if dimensions[2] > 0 else height
    model_width = dimensions[3] if dimensions[3] > 0 else width
    if model_height <= 0 or model_width <= 0:
        fail("Baking requires positive input height and width")

    dtype = numpy.float16 if element_type == TensorProto.FLOAT16 else numpy.float32
    scale_value = 1.0 if has_yolox_raw_head(model) else 1.0 / 255.0
    scale_name = pick_unused_name(graph, ["bake_scale"])
    graph.initializer.append(
        numpy_helper.from_array(numpy.array(scale_value, dtype=dtype), name=scale_name)
    )

    input_name = pick_unused_name(graph, ["image", "image_u8", "bake_image"])
    nchw_output = pick_unused_name(graph, ["bake_nchw"])
    cast_output = pick_unused_name(graph, ["bake_cast"])
    nodes = [
        # TensorRT does not accept UINT8 as the input type of Transpose, so
        # cast before changing NHWC to NCHW.
        helper.make_node(
            "Cast",
            [input_name],
            [cast_output],
            name="bake_Cast",
            to=element_type,
        ),
        helper.make_node(
            "Transpose",
            [cast_output],
            [nchw_output],
            name="bake_Transpose",
            perm=[0, 3, 1, 2],
        ),
        helper.make_node(
            "Mul",
            [nchw_output, scale_name],
            [original.name],
            name="bake_Mul",
        ),
    ]
    for node in reversed(nodes):
        graph.node.insert(0, node)

    del graph.input[:]
    graph.input.append(
        helper.make_tensor_value_info(
            input_name,
            TensorProto.UINT8,
            [1, model_height, model_width, channels],
        )
    )
    topologically_sort_graph(graph)
    return model


def convert_onnx_to_fp16(path: Path, height: int, width: int) -> tuple[bytes, Any]:
    try:
        import onnx
        import numpy
        from onnxruntime.transformers import float16
    except ImportError as error:
        raise RuntimeError(
            "FP16 conversion requires the Python packages 'onnx' and "
            "'onnxruntime' and 'numpy'. Install them with: "
            "py -3 -m pip install onnx onnxruntime numpy"
        ) from error

    try:
        model = run_with_heartbeat(
            lambda: onnx.load(str(path)),
            f"Loading ONNX model: {path}",
        )
        status(
            f"Loaded ONNX graph: {len(model.graph.node)} nodes, "
            f"{len(model.graph.input)} input(s), {len(model.graph.output)} output(s)."
        )
        status("Normalizing ONNX detection outputs")
        normalize_detection_outputs(model)
        status("Baking UINT8 NHWC preprocessing into the ONNX graph")
        baked = bake_preprocess_model(model, height, width, numpy)
        status("Checking the baked ONNX graph")
        run_with_heartbeat(
            lambda: onnx.checker.check_model(baked),
            "Checking the baked ONNX graph",
        )
        status("Converting the ONNX graph to FP16")
        converted = run_with_heartbeat(
            lambda: float16.convert_float_to_float16(baked, keep_io_types=False),
            "Converting the ONNX graph to FP16",
        )
        status("Topologically sorting the converted graph")
        topologically_sort_graph(converted.graph)
        status("Checking the converted FP16 ONNX graph")
        run_with_heartbeat(
            lambda: onnx.checker.check_model(converted),
            "Checking the converted FP16 ONNX graph",
        )
        status("Serializing the converted FP16 ONNX graph")
        converted_bytes = run_with_heartbeat(
            converted.SerializeToString,
            "Serializing the converted FP16 ONNX graph",
        )
        status(f"FP16 ONNX graph ready: {len(converted_bytes) / (1024**2):.2f} MiB")
        return converted_bytes, baked
    except Exception as error:
        raise RuntimeError(f"Unable to bake and convert the ONNX model to FP16: {error}") from error


def has_objectness_branch(model: Any) -> bool:
    for node in model.graph.node:
        names = (node.name, *node.input, *node.output)
        if any("objectness" in name.lower() or "obj_" in name.lower() for name in names):
            return True
    return False


def has_yolox_raw_head(model: Any) -> bool:
    names = []
    for node in model.graph.node:
        names.extend((node.name, *node.input, *node.output))
    lowered = [name.lower() for name in names]
    return (
        any("reg_preds" in name for name in lowered)
        and any("obj_preds" in name for name in lowered)
        and any("cls_preds" in name for name in lowered)
    )


def has_yolov5_style_head(model: Any) -> bool:
    """Detect the legacy YOLOv5 head when metadata does not expose objectness."""
    if has_yolox_raw_head(model):
        return False

    initializer_shapes = {
        initializer.name: tuple(int(dimension) for dimension in initializer.dims)
        for initializer in model.graph.initializer
    }
    first_conv_shape = None
    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        first_conv_shape = initializer_shapes.get(node.input[1])
        if first_conv_shape is not None:
            break
    if first_conv_shape is None or len(first_conv_shape) != 4:
        return False

    names = []
    for node in model.graph.node:
        names.extend((node.name, *node.input, *node.output))
    has_model_namespace = any("model." in name.lower() for name in names)
    has_decode_power = any(node.op_type == "Pow" for node in model.graph.node)
    return first_conv_shape[2:] == (6, 6) and has_model_namespace and has_decode_power


def shape_values(shape: Any) -> list[int]:
    return [int(value) for value in shape]


def build_profile_shape(
    shape: list[int],
    batch: tuple[int, int, int],
    height: int,
    width: int,
    layout: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], bool]:
    minimum = list(shape)
    optimum = list(shape)
    maximum = list(shape)
    dynamic = False
    for axis, value in enumerate(shape):
        if value > 0:
            continue
        dynamic = True
        if axis == 0:
            minimum[axis], optimum[axis], maximum[axis] = batch
        elif axis == (1 if layout == "NHWC" else 2):
            minimum[axis] = optimum[axis] = maximum[axis] = height
        elif axis == (2 if layout == "NHWC" else 3):
            minimum[axis] = optimum[axis] = maximum[axis] = width
        else:
            fail(f"Unsupported dynamic input dimension at axis {axis}: {shape}")
    return tuple(minimum), tuple(optimum), tuple(maximum), dynamic


def tensor_dtype_name(trt: Any, dtype: Any) -> str:
    if dtype == trt.DataType.FLOAT:
        return "float32"
    if dtype == trt.DataType.HALF:
        return "float16"
    if hasattr(trt.DataType, "UINT8") and dtype == trt.DataType.UINT8:
        return "uint8"
    return str(dtype)


def create_network(builder: Any, trt: Any) -> Any:
    try:
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        return builder.create_network(flags)
    except (AttributeError, TypeError):
        return builder.create_network()


def configure_workspace(config: Any, trt: Any, workspace_gb: float) -> None:
    workspace_bytes = int(workspace_gb * (1024**3))
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes


def build_engine(args: argparse.Namespace) -> None:
    progress(0, "Initializing TensorRT")
    trt = import_tensorrt()
    status(f"TensorRT Python bindings: {trt.__version__}")
    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = create_network(builder, trt)
    parser = trt.OnnxParser(network, logger)
    status("TensorRT builder, network, and ONNX parser initialized")

    progress(15, f"Baking preprocessing and converting ONNX to FP16: {args.input}")
    onnx_bytes, onnx_model = convert_onnx_to_fp16(args.input, args.height, args.width)
    status(f"Prepared ONNX input for TensorRT: {len(onnx_bytes) / (1024**2):.2f} MiB")
    progress(40, "Parsing the baked FP16 ONNX graph")
    parsed = run_with_heartbeat(
        lambda: parser.parse(onnx_bytes),
        "Parsing the baked FP16 ONNX graph",
    )
    if not parsed:
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        fail("TensorRT could not parse the ONNX model:\n" + "\n".join(errors))
    status(f"TensorRT parsed the graph: {network.num_inputs} input(s), {network.num_outputs} output(s)")

    progress(55, "Validating the single-input, multi-output detection graph")
    if network.num_inputs != 1:
        fail("YoloTrtCuda requires an ONNX model with exactly one input")
    if network.num_outputs <= 0:
        fail("YoloTrtCuda requires at least one raw YOLO detection output")

    input_tensor = network.get_input(0)
    input_shape = shape_values(input_tensor.shape)
    uint8_dtype = getattr(trt.DataType, "UINT8", None)
    baked_preprocess = uint8_dtype is not None and input_tensor.dtype == uint8_dtype
    if len(input_shape) != 4:
        fail(f"YoloTrtCuda requires a rank-4 input: {input_shape}")
    if input_shape[0] > 1:
        fail("YoloTrtCuda engines use batch 1; export the ONNX model with batch=1")
    if baked_preprocess:
        if input_shape[3] not in (1, 3, -1):
            fail(f"Baked YoloTrtCuda input must be NHWC with 1 or 3 channels: {input_shape}")
        input_layout = "NHWC"
    else:
        if input_shape[1] not in (1, 3, -1):
            fail(f"YoloTrtCuda requires an NCHW input with 1 or 3 channels: {input_shape}")
        if input_tensor.dtype not in (trt.DataType.FLOAT, trt.DataType.HALF):
            fail("YoloTrtCuda expects a float32/float16 NCHW input or a baked UINT8 NHWC input")
        input_layout = "NCHW"
    status(
        f"Validated input: name={input_tensor.name}, shape={input_shape}, "
        f"dtype={tensor_dtype_name(trt, input_tensor.dtype)}, layout={input_layout}"
    )

    output_tensors = [network.get_output(index) for index in range(network.num_outputs)]
    output_specs = []
    feature_counts = set()
    for output_tensor in output_tensors:
        output_shape = shape_values(output_tensor.shape)
        if len(output_shape) != 3:
            fail(
                "YoloTrtCuda requires every detection output to be rank 3; "
                f"got {output_tensor.name}: {output_shape}"
            )
        output_features = detection_feature_count(output_shape)
        if output_features is None:
            fail(
                "YoloTrtCuda could not infer the feature dimension of raw YOLO "
                f"output {output_tensor.name}: {output_shape}"
            )
        if output_features == 6 and has_embedded_nms(onnx_model):
            fail(
                "The ONNX graph exposes an end-to-end/NMS output with 6 values "
                "per box. Use the raw detection export so confidence and NMS "
                "stay outside the engine."
            )
        if output_tensor.dtype not in (trt.DataType.FLOAT, trt.DataType.HALF):
            fail(
                "YoloTrtCuda detection outputs must be float32 or float16: "
                f"{output_tensor.name}"
            )
        feature_counts.add(output_features)
        output_specs.append(
            {
                "name": output_tensor.name,
                "shape": output_shape,
                "dtype": tensor_dtype_name(trt, output_tensor.dtype),
            }
        )

    if len(feature_counts) != 1:
        fail(
            "YoloTrtCuda detection heads must use the same feature count; "
            f"got {sorted(feature_counts)}"
        )

    status(
        f"Validated detection outputs: {len(output_tensors)} output(s), "
        f"feature count={next(iter(feature_counts))}"
    )

    output_tensor = output_tensors[0]
    output_shape = output_specs[0]["shape"]
    output_features = next(iter(feature_counts))
    end_to_end = (
        len(output_tensors) == 1
        and is_end_to_end_output(onnx_model, output_tensor.name, output_shape)
    )

    if end_to_end:
        objectness = False
    elif args.objectness == "on":
        objectness = True
    elif args.objectness == "off":
        objectness = False
    else:
        # YOLOv5-style COCO heads are [x, y, w, h, objectness, 80 classes].
        # YOLOv8/YOLO11-style COCO heads omit objectness and have 84 values.
        objectness = (
            output_features == 85
            or has_objectness_branch(onnx_model)
            or has_yolov5_style_head(onnx_model)
        )
    yolox_decode = not end_to_end and has_yolox_raw_head(onnx_model)

    if end_to_end:
        progress(60, "Detected NMS-free end-to-end YOLO output")

    config = builder.create_builder_config()
    configure_workspace(config, trt, args.workspace_gb)
    status(f"TensorRT workspace limit: {args.workspace_gb:.2f} GiB")
    fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
    if fp16_flag is not None:
        config.set_flag(fp16_flag)
    elif input_tensor.dtype not in (trt.DataType.HALF, uint8_dtype):
        fail(
            "TensorRT does not expose BuilderFlag.FP16 and the converted ONNX "
            "input is neither float16 nor a baked UINT8 input"
        )

    progress(70, "Building the FP16 TensorRT engine (this may take a while)")

    minimum, optimum, maximum, dynamic = build_profile_shape(
        input_shape,
        (args.min_batch, args.opt_batch, args.max_batch),
        args.height,
        args.width,
        input_layout,
    )
    if dynamic:
        profile = builder.create_optimization_profile()
        if not profile.set_shape(input_tensor.name, minimum, optimum, maximum):
            fail("TensorRT rejected the input optimization profile")
        config.add_optimization_profile(profile)
        status(f"Optimization profile: min={minimum}, opt={optimum}, max={maximum}")
    else:
        status(f"Static input shape: {tuple(input_shape)}")

    serialized = run_with_heartbeat(
        lambda: builder.build_serialized_network(network, config),
        "Building the FP16 TensorRT engine",
    )
    if serialized is None:
        fail("TensorRT failed to build the FP16 engine")
    engine_bytes = bytes(serialized)
    status(f"TensorRT engine built: {len(engine_bytes) / (1024**2):.2f} MiB")

    metadata = {
        "format": "YoloTrtCuda",
        "version": VERSION,
        "precision": "fp16",
        "confidence_embedded": False,
        "nms_embedded": False,
        "objectness": objectness,
        "yolox_decode": yolox_decode,
        "raw_input": yolox_decode,
        "preprocess_baked": baked_preprocess,
        "end_to_end": end_to_end,
        "input_layout": input_layout,
        "input": {
            "name": input_tensor.name,
            "shape": input_shape,
            "dtype": tensor_dtype_name(trt, input_tensor.dtype),
            "profile": {
                "min": minimum,
                "opt": optimum,
                "max": maximum,
            },
        },
        "output": {
            "name": output_tensor.name,
            "shape": output_shape,
            "dtype": tensor_dtype_name(trt, output_tensor.dtype),
        },
        "outputs": output_specs,
    }
    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    header = HEADER.pack(MAGIC, VERSION, len(metadata_bytes), len(engine_bytes))
    progress(95, f"Writing YoloTrtCuda engine: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_with_heartbeat(
        lambda: args.output.write_bytes(header + metadata_bytes + engine_bytes),
        "Writing the YoloTrtCuda engine",
    )
    progress(100, f"Completed: {args.output}")
    status(f"Wrote {args.output}")
    status(f"  input : {input_tensor.name} {input_shape} {tensor_dtype_name(trt, input_tensor.dtype)}")
    for index, output_spec in enumerate(output_specs):
        status(
            f"  output[{index}]: {output_spec['name']} "
            f"{output_spec['shape']} {output_spec['dtype']}"
        )
    preprocess_description = "baked UINT8 NHWC" if baked_preprocess else "runtime float NCHW"
    status(f"  precision: FP16; input: {preprocess_description}; confidence/NMS remain runtime-configurable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, metavar="INPUT", help="Raw YOLO ONNX model")
    parser.add_argument("--height", type=int, default=640, help="Profile height for dynamic inputs")
    parser.add_argument("--width", type=int, default=640, help="Profile width for dynamic inputs")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=1)
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument(
        "--objectness",
        choices=("auto", "on", "off"),
        default="auto",
        help="Interpret the raw head as objectness + classes; auto detects common YOLOv5 heads",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    input_stem = args.input.stem
    if input_stem.endswith("_u8"):
        input_stem = input_stem[:-3]
    args.output = args.input.with_name(f"{input_stem}_fp16_u8.engine")

    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")
    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive")
    if not (args.min_batch == args.opt_batch == args.max_batch == 1):
        parser.error("YoloTrtCuda currently supports batch size 1 only")
    if args.workspace_gb <= 0:
        parser.error("--workspace-gb must be positive")
    return args


def main() -> int:
    try:
        build_engine(parse_args())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
