# Docker

First build the docker service using

```bash
docker compose build
```

To run the service with restart do 

```bash
docker compose up
```

to run the service once without restart (i.e. for evaluation runs)

```bash
docker compose run inference-service
```

## Performance monitor configuration of docker

environment configuration scheduler_monNITOR=1 enables monitoring, set to scheduler_monNITOR=0 to disable.
Monitoring has very little performance impact as specified by Hailo Forum.

To monitor running application use the command `hailortcli monitor`, which reads
current performance metrics from `/tmp/hmon_files`. This directory is mounted as
a volume in `docker-compose.yml` to allow monitoring of the docker inference service.



# Protobuf Setup & Usage

This project uses **Protocol Buffers (protobuf)** to decode Hailo monitor files using the schema defined in `scheduler_mon.proto`.

## Hailo Utilization monitoring

From file /etc/default/hailort_service:

#This file contains HailoRT's configurable environment variables for HailoRT Linux Service.
#The environment variables are set to their default values.
#To change an environment variable's value, follow the steps:
#1. Change the value of the selected environemt variable in this file
#2. Reload systemd unit files by running: `sudo systemctl daemon-reload`
#3. Copy this file to /etc/default/hailort_service
#4. Enable and start service by running: `sudo systemctl enable --now hailort.service`

[Service]
HAILORT_LOGGER_PATH="/var/log/hailo"
scheduler_monNITOR=1
HAILO_TRACE=0
HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=0
HAILO_TRACE_SIZE_IN_KB_BOUNDED_DUMP=0
HAILO_TRACE_PATH=""


---

## What is Protobuf?

Protocol Buffers is a serialization format that allows structured data to be encoded efficiently.  
The workflow consists of:

1. Define schema in `.proto` file  
2. Compile `.proto` → generated Python classes  
3. Import generated classes in Python code  
4. Decode binary data using generated classes  

The proto file for the hailo monitor utilization `scheduler_mon.proto` file was found in the open-source hailort github
repository at https://github.com/hailo-ai/hailort/blob/master/hailort/libhailort/scheduler_mon.proto.

---

## Installation

Install system protobuf:

```bash
sudo apt install protobuf-compiler
```

From the `hailo_monitor` directory containing `scheduler_mon.proto`, run to generate the python class:

```bash
protoc --python_out=. scheduler_mon.proto
```

This generates:

```
scheduler_mon_pb2.py
```

This file contains the generated Python classes used in `hailo_monitor/hailo_monitor_parser.py`.

---

## When to Regenerate

You only need to regenerate if:

- `scheduler_mon.proto` changes
- upgrading protobuf version (optional)

Otherwise, the generated file should be committed to git.

---




