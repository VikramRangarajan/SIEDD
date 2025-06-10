#!/bin/bash

# Get the startup script content and escape it properly for JSON
STARTUP_SCRIPT=$(cat {{ RUNPOD_STARTUP_SH }} | jq -Rs .)

# Your SSH public key for accessing the pod
PUBLIC_KEY=$(cat ~/.ssh/id_ed25519.pub)

# Read deploy key (base64 encoded)
DEPLOY_KEY=$(cat ~/.ssh/id_ed25519 | base64 -w 0)

# Grab wandb api key from ~/.netrc
WANDB_API_KEY=$(awk '/api.wandb.ai/{getline; getline; print $2}' ~/.netrc)

# Upload configs to b2 bucket, which can then be download by the pods
b2 sync --exclude-regex '.*\.out|.*\.sh' --include-regex '.*\.yaml' ./ ${B2_EXP_SCRATCH_PATH}

# Create a temporary JSON file with the payload
cat << EOF > payload.json
{
  "allowedCudaVersions": ["12.7"],
  "cloudType": "SECURE",
  "computeType": "GPU",
  "containerDiskInGb": 50,
  "containerRegistryAuthId": "",
  "countryCodes": [],
  "cpuFlavorIds": ["cpu3c"],
  "cpuFlavorPriority": "availability",
  "dataCenterPriority": "availability",
  "dockerEntrypoint": [],
  "dockerStartCmd": [
    "/bin/bash",
    "-c",
    ${STARTUP_SCRIPT}
  ],
  "env": {
    "PUBLIC_KEY": "${PUBLIC_KEY}",
    "DEPLOY_KEY": "${DEPLOY_KEY}",
    "WANDB_API_KEY": "${WANDB_API_KEY}",
    "B2_APPLICATION_KEY": "${B2_APPLICATION_KEY}",
    "B2_APPLICATION_KEY_ID": "${B2_APPLICATION_KEY_ID}",
    "B2_EXP_SCRATCH_PATH": "${B2_EXP_SCRATCH_PATH}",
  },
  "gpuCount": 1,
  "gpuTypeIds": ["NVIDIA RTX A6000"],
  "gpuTypePriority": "availability",
  "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
  "interruptible": false,
  "locked": false,
  "minDiskBandwidthMBps": 0,
  "minDownloadMbps": 0,
  "minRAMPerGPU": 8,
  "minUploadMbps": 0,
  "minVCPUPerGPU": 2,
  "name": "{{ POD_NAME }}",
  "networkVolumeId": "${RUNPOD_VOLUME_ID}",
  "ports": ["8888/http", "22/tcp"],
  "supportPublicIp": true,
  "templateId": "",
  "vcpuCount": 1,
  "volumeInGb": 100,
  "volumeMountPath": "/workspace"
}
EOF

# Send the request using the temporary JSON file
curl https://rest.runpod.io/v1/pods \
  --request POST \
  --header 'Content-Type: application/json' \
  --header "Authorization: Bearer {{ RUNPOD_API_KEY }}" \
  --data @payload.json | jq '.'

# Clean up the temporary file
rm payload.json
