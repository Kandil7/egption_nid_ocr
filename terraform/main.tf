# Egyptian ID OCR - Oracle Cloud Infrastructure Terraform Configuration
# Deploys a VM instance with all necessary setup for the OCR API
#
# Prerequisites:
# 1. Oracle Cloud Free Tier account
# 2. OCI CLI configured (oci setup config)
# 3. Terraform installed
#
# Usage:
#   terraform init
#   terraform plan
#   terraform apply

terraform {
  required_version = ">= 1.0.0"
  
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = "~> 5.0"
    }
  }
}

# Configure the Oracle Cloud Provider
provider "oci" {
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}

# Variables
variable "tenancy_ocid" {
  description = "OCID of your tenancy"
  type        = string
}

variable "user_ocid" {
  description = "OCID of the user"
  type        = string
}

variable "fingerprint" {
  description = "Fingerprint of the API key"
  type        = string
}

variable "private_key_path" {
  description = "Path to the private key"
  type        = string
  default     = "~/.oci/oci_api_key.pem"
}

variable "region" {
  description = "OCI region"
  type        = string
  default     = "eu-frankfurt-1"  # Free tier available regions: eu-frankfurt-1, ap-hyderabad-1, ap-chuncheon-1
}

variable "compartment_ocid" {
  description = "OCID of the compartment"
  type        = string
}

variable "ssh_public_key" {
  description = "SSH public key for VM access"
  type        = string
}

variable "vm_shape" {
  description = "VM shape (ARM Ampere A1 for free tier)"
  type        = string
  default     = "VM.Standard.A1.Flex"
}

variable "vm_cpus" {
  description = "Number of OCPUs (max 4 for free tier)"
  type        = number
  default     = 4
}

variable "vm_memory" {
  description = "Memory in GB (max 24 for free tier)"
  type        = number
  default     = 24
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "egyptian-ocr"
}

# Data sources
data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

# VCN and Networking
resource "oci_core_vcn" "ocr_vcn" {
  compartment_id = var.compartment_ocid
  cidr_blocks    = ["10.0.0.0/16"]
  display_name   = "${var.project_name}-vcn"
  dns_label      = "ocrvcn"
}

resource "oci_core_internet_gateway" "igw" {
  compartment_id = var.compartment_ocid
  display_name   = "${var.project_name}-igw"
  vcn_id         = oci_core_vcn.ocr_vcn.id
  enabled        = "true"
}

resource "oci_core_route_table" "rt" {
  compartment_id = var.compartment_ocid
  display_name   = "${var.project_name}-rt"
  vcn_id         = oci_core_vcn.ocr_vcn.id

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.igw.id
  }
}

resource "oci_core_security_list" "sl" {
  compartment_id = var.compartment_ocid
  display_name   = "${var.project_name}-sl"
  vcn_id         = oci_core_vcn.ocr_vcn.id

  # Allow SSH
  ingress_security_rules {
    protocol    = "6"  # TCP
    source      = "0.0.0.0/0"
    source_type = "CIDR_BLOCK"
    tcp_options {
      destination_port_range {
        max = 22
        min = 22
      }
    }
  }

  # Allow HTTP
  ingress_security_rules {
    protocol    = "6"
    source      = "0.0.0.0/0"
    source_type = "CIDR_BLOCK"
    tcp_options {
      destination_port_range {
        max = 80
        min = 80
      }
    }
  }

  # Allow HTTPS
  ingress_security_rules {
    protocol    = "6"
    source      = "0.0.0.0/0"
    source_type = "CIDR_BLOCK"
    tcp_options {
      destination_port_range {
        max = 443
        min = 443
      }
    }
  }

  # Allow API port (8000)
  ingress_security_rules {
    protocol    = "6"
    source      = "0.0.0.0/0"
    source_type = "CIDR_BLOCK"
    tcp_options {
      destination_port_range {
        max = 8000
        min = 8000
      }
    }
  }

  # Allow all outbound
  egress_security_rules {
    protocol    = "all"
    destination = "0.0.0.0/0"
    destination_type = "CIDR_BLOCK"
  }
}

resource "oci_core_subnet" "subnet" {
  compartment_id      = var.compartment_ocid
  display_name        = "${var.project_name}-subnet"
  vcn_id              = oci_core_vcn.ocr_vcn.id
  cidr_block          = "10.0.0.0/24"
  dns_label           = "ocrsubnet"
  security_list_ids   = [oci_core_security_list.sl.id]
  route_table_id      = oci_core_route_table.rt.id
  subnet_access       = "PUBLIC"
}

# Boot volume for persistent storage
resource "oci_core_boot_volume" "boot_volume" {
  compartment_id = var.compartment_ocid
  display_name   = "${var.project_name}-boot"
  size_in_gbs    = 200  # 200GB free tier storage
}

# VM Instance
resource "oci_core_instance" "ocr_vm" {
  compartment_id      = var.compartment_ocid
  availability_domain = data "oci_identity_availability_domains" "ads" "availability_domains" [0] "name"
  display_name        = "${var.project_name}-vm"
  shape               = var.vm_shape

  shape_config {
    ocpus         = var.vm_cpus
    memory_in_gbs = var.vm_memory
  }

  source_details {
    source_type               = "image"
    source_id                 = "ocid1.image.oc1..aaaaaaaasemea5qep6zjxqz3qz3qz3qz3qz3qz3qz3qz3qz3qz3qz3qz3"  # Ubuntu 22.04 ARM - update with actual OCID
    boot_volume_size_in_gbs   = "200"
    boot_volume_vpus_per_gb   = "10"
  }

  metadata = {
    ssh_authorized_keys = var.ssh_public_key
    user_data = base64encode(<<-EOF
    #!/bin/bash
    # Cloud-init script for initial setup
    apt-get update
    apt-get install -y docker.io docker-compose git curl wget
    usermod -aG docker ubuntu
    systemctl enable docker
    systemctl start docker
    EOF
    )
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.subnet.id
    display_name     = "${var.project_name}-vnic"
    assign_public_ip = true
  }

  agent_config {
    are_all_plugins_disabled = false
    is_management_disabled   = false
    is_monitoring_disabled   = false
  }
}

# Outputs
output "vm_public_ip" {
  description = "Public IP of the VM"
  value       = oci_core_instance.ocr_vm.public_ip
}

output "vm_private_ip" {
  description = "Private IP of the VM"
  value       = oci_core_instance.ocr_vm.private_ip
}

output "ssh_command" {
  description = "SSH command to connect to the VM"
  value       = "ssh -i ~/.ssh/id_rsa ubuntu@${oci_core_instance.ocr_vm.public_ip}"
}

output "api_url" {
  description = "API endpoint URL"
  value       = "http://${oci_core_instance.ocr_vm.public_ip}:8000"
}
