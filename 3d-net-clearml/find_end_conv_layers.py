"""
Auto-detect the correct Hailo parser end nodes from an ONNX model.

Walks the graph backwards from the per-scale detection head Concat nodes
to find the raw cv2 (box distribution) and cv3 (class) Conv node names,
which Hailo's nms_postprocess expects as model end points.

Prints a space-separated list of node names suitable for use with
`hailo parser onnx --end-node-names`.
"""

import sys
import onnx

model_path = sys.argv[1] if len(sys.argv) > 1 else "model.onnx"
m = onnx.load(model_path)

output_to_node = {out: n for n in m.graph.node for out in n.output}


def first_conv_upstream(name, depth=15):
    """Walk backwards from `name`, return the first Conv node name on each path."""
    found = []

    def walk(n, d):
        if d == 0:
            return
        node = output_to_node.get(n)
        if node is None:
            return
        if node.op_type == "Conv":
            found.append(node.name)
            return
        for inp in node.input:
            if inp in output_to_node:
                walk(inp, d - 1)

    walk(name, depth)
    return found


per_scale_concats = [
    n for n in m.graph.node
    if n.op_type == "Concat"
    and any("/model.23/Concat" in o for o in n.output)
    and not any(o == "output0" for o in n.output)
]

if not per_scale_concats:
    print("ERROR: could not find per-scale Concat nodes in /model.23", file=sys.stderr)
    sys.exit(1)

seen = set()
end_nodes = []
for pc in per_scale_concats:
    for inp in pc.input:
        for conv_out in first_conv_upstream(inp):
            if conv_out not in seen:
                end_nodes.append(conv_out)
                seen.add(conv_out)

if not end_nodes:
    print("ERROR: could not find upstream Conv nodes", file=sys.stderr)
    sys.exit(1)

print(" ".join(end_nodes))
