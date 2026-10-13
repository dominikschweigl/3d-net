

## Connection Setup

connect raspberry pi via ethernet directly to to laptop with ethernet to usb-c adapter.
In the ubuntu desktop pc go to network settings and configure for the ethernet connection
to the raspberrypi in the IPv4 section select the IPv4 Method "Shared to other computers".
This way the desktop acts as a DHCP server for the raspberry and assigns it an IP address.

Afterwards run the command

```bash
ip address
```

and find the link connected to the same mac address in the form of `enx<mac-address>`
as seen in the network settings before. In the following example the IP subnet
`10.42.0.1/24` is used for this ethernet link. 

```bash
129: enxd43a65185976: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UP group default qlen 1000
    link/ether d4:3a:65:18:59:76 brd ff:ff:ff:ff:ff:ff
    inet 10.42.0.1/24 brd 10.42.0.255 scope global noprefixroute enxd43a65185976
       valid_lft forever preferred_lft forever
    inet6 fe80::f51:8648:3f81:cce7/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever
```

To find out the IPv4 address assigned to the raspberry Pi from our desktop 
DHCP server, running the command `nmap -sn 10.42.0.0/24`, which searches all
available device IP's in a subnet and receive following output.

```bash
dominik-schweigl@dominik-schweigl-ThinkPad-T14s-Gen-5:~/code/3d-net-clearml$ nmap -sn 10.42.0.0/24
    Starting Nmap 7.95 ( https://nmap.org ) at 2026-03-25 08:08 CET
    Nmap scan report for dominik-schweigl-ThinkPad-T14s-Gen-5 (10.42.0.1)
    Host is up (0.00019s latency).
    Nmap scan report for 10.42.0.78
    Host is up (0.00038s latency).
    Nmap done: 256 IP addresses (2 hosts up) scanned in 2.68 seconds
```

The first entry is the host desktop and the second one is the raspberry Pi.
This we can confirm by running `ping 10.42.0.78` and then connecting to it
via ssh. The ssh credentials can be found in the company internal 1Password vault.

```bash
ssh <user>@10.42.0.78
```

## Camera Network Configuration

The WIRIS camera is configured with a static IP of `10.174.42.250`. Since the Raspberry Pi has a single Ethernet port used for both the laptop connection and the camera connection, all three devices must be on the same subnet.

Configure the laptop's Pi-facing Ethernet connection to use the `10.174.42.x` range and set it to option
"Shared to other computers" if the raspberry should receive internet from your:

```bash
nmcli connection modify "Wired connection 2" ipv4.method shared ipv4.addresses "10.174.42.1/24"
nmcli connection up "Wired connection 2"
```

Then configure the Pi's `eth0` to a static IP in the same subnet. This is important since the Pi cannot
reach the camera live streams if this is not properly configured in the same subnet.
Check which connection manager is active:

```bash
nmcli connection show
```

If NetworkManager is used:

```bash
sudo nmcli connection modify "Wired connection 1" ipv4.method manual ipv4.addresses "10.174.42.89/24" ipv4.gateway "10.174.42.1" ipv4.dns "8.8.8.8"
sudo nmcli connection up "Wired connection 1"
```

If dhcpcd is used, add to `/etc/dhcpcd.conf`:

```
interface eth0
static ip_address=10.174.42.100/24
```

The resulting network layout is:

| Device | Address |
|--------|---------|
| Laptop (eth) | `10.174.42.1` |
| Raspberry Pi | `10.174.42.89` |
| WIRIS Camera | `10.174.42.250` |

The camera's WIRIS control port for telnet is the default `2240`.

In o

## Hailo AI Kit

### Dataflow Compiler

The [Dataflow Compiler](https://hailo.ai/products/hailo-software/hailo-ai-software-suite/#sw-dc) is 
used to build optimized binary model files `.hef` for the hailo accelerator from standard model 
weight formats like `.pt` from pytorch or `.onnx`.

#### Installation

The Dataflow Compiler installation wheel can be downloaded on the [Software Downloads Page](https://hailo.ai/developer-zone/software-downloads/?product=ai_accelerators&device=hailo_8_8l) from Hailo. For our purposes the Hailo 8/8L version needs to be selected.
To download the Dataflow Compiler installation wheel a Hailo-account is needed.

Afterwards the compiler can be installed following the [Dataflow Compiler Installation Guide](https://hailo.ai/developer-zone/documentation/dataflow-compiler-v3-33-1/?sp_referrer=install/install.html) in order to compile hailo models.
It should be noted that the compiler only supports x86 system architecture and Linux OS, which means the compilation of
the models cannot be done on the raspberry Pi 5 itself. Also, only Python 3.10 is supported by the Dataflow compiler despite
the Installation guide mentioning support for 3.11 and 3.12.

#### Compilation

The Hailo Dataflow Compiler uses a **multi-stage pipeline** to convert your model into a deployable binary for Hailo hardware.
This section assumes the Dataflow Compiler was successfully installed and is available in the current environment.

##### Pipeline Flow

```
.onnx → [parser] → .har → [optimize] → .har → [compile] → .hef
```

##### Steps

| Step | Command | Output | Description |
|------|---------|--------|-------------|
| 1. Parse | `hailo parser onnx model.onnx` | `model.har` | Converts ONNX graph to Hailo internal format |
| 2. Optimize | `hailo optimize model.har` | `model.har` | Quantization and calibration (requires calibration data) |
| 3. Compile | `hailo compiler model.har` | `model.hef` | Compiles model for Hailo hardware |

---

##### Example Workflow

1. Parse (Already Completed)

```bash
hailo parser onnx yolo11n.onnx --hw-arch hailo8
```

Output:

```
yolo11n.har
```

---

2. Optimize

You must provide a calibration dataset:

```bash
hailo optimize yolo11n.har \
  --calib-set-path /path/to/calib/images
```

For quick testing (random calibration):

```bash
hailo optimize yolo11n.har \
  --use-random-calib-set
```

---

3. Compile

Compile for your specific Hailo chip:
  
```bash
hailo compiler yolo11n.har --hw-arch hailo8
```

or

```bash
hailo compiler yolo11n.har --hw-arch hailo8l
```

---

The `--hw-arch` flag **must match your hardware**:

- `hailo8` → Hailo-8  
- `hailo8l` → Hailo-8L  

Using the wrong architecture will result in an incompatible `.hef` file.

##### Docker Setup

Alternatively, one can run the Dataflow compiler by downloading the docker
image for the Hailo AI software suite from [Software Downloads Page](https://hailo.ai/developer-zone/software-downloads/?product=ai_accelerators&device=hailo_8_8l). This includes the hailo model zoo,
hailort runtime examples and a `hailomz` command line interface for the aggregation
of the compilation pipeline seen earlier. In addition, `hailomz` integrates with the standard
models from the modelzoo and simplifies compilation of standard models.

The download is a `.zip` file that needs to be extracted and the docker container is
created by executing the `hailo_ai_sw_suite_docker_run.sh` shell script. 
The `.onnx` model file and quantization calibration images must be shared to the docker
container by the automatically created `shared_with_docker` directory next to the
setup shell script. After this standard model compilation can be run for example for
yolov11 model with `hailomz compile yolov11n --calib-path /local/shared_with_docker/doc/Vis/`,
where available models can be found in the `hailo_model_zoo` directory inside the hailo container. 

However, there is little customization possible with the `hailomz` cli tool, especially
the compilation of non-standard model architectures is not possible. For this reason,
we have to use the lower-level `hailo` compile pipeline.

##### Notes

For custom models or non-standard models provided by the Hailo model zoo
mapping failures can occur during compilation.

```bash
[error] Mapping Failed (allocation time: 7s)
No successful assignments: concat18 errors:
        Agent infeasible


[error] Failed to produce compiled graph
[error] BackendAllocatorException: Compilation failed: No successful assignments: concat18 errors:
        Agent infeasible
```


### Raspberry Configuration

The Hailo system dependencies on the were configured according to the software prerequisites
section of the following Raspberry Pi AI software setup guide.

https://www.raspberrypi.com/documentation/computers/ai.html#getting-started

This setup includes

- The Hailo kernel device driver and firmware.
- Hailo RT middleware software.
- Hailo Tappas core post-processing libraries.

#### Inference 

After successful compilation you should have a `<model-name>.hef` file, which needs
to copied to the raspberry Pi, for example with `scp`.

Using the inference scripts, you should see the input and output format of the `.hef`
model. 

```bash
<user>@<host>:~/object_detection $ python inference_images.py 

--- Input VStreams ---
yolov11n/input_layer1               shape=(640, 640, 3) type=FormatType.UINT8

--- Output VStreams ---
yolov11n/yolov8_nms_postprocess     shape=(80, 5, 100) type=FormatType.FLOAT32
```

Make sure that this information displayed in the beginning of running the script
matches the virtual stream params provided in the inference script. Otherwise this
will lead to a failure.

```python
# --- CREATE VSTREAM PARAMS ---
input_vstreams_params = InputVStreamParams.make_from_network_group(
    network_group, quantized=False, format_type=FormatType.INT8
)
output_vstreams_params = OutputVStreamParams.make_from_network_group(
    network_group, quantized=False, format_type=FormatType.INT8
)
```



Thesis Motivation inspiration:
https://hailo.ai/blog/powerful-edge-ai-to-boost-intelligent-traffic-systems-its/