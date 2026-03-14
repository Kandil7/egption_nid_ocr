# Egyptian ID OCR - Terraform Variables
# Copy this file to terraform.tfvars and fill in your values

# Oracle Cloud Authentication
# Get these from OCI Console -> Profile -> API Keys
tenancy_ocid     = "ocid1.tenancy.oc1..aaaaaaaa..."
user_ocid        = "ocid1.user.oc1..aaaaaaaa..."
fingerprint      = "xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx"
private_key_path = "~/.oci/oci_api_key.pem"

# Region (choose one with free tier availability)
# Available: eu-frankfurt-1, ap-hyderabad-1, ap-chuncheon-1
region = "eu-frankfurt-1"

# Compartment OCID (usually same as tenancy for root compartment)
compartment_ocid = "ocid1.tenancy.oc1..aaaaaaaa..."

# SSH Public Key (generate with: ssh-keygen -t ed25519)
ssh_public_key = "ssh-ed25519 AAAA..."

# VM Configuration (Free Tier limits)
vm_shape  = "VM.Standard.A1.Flex"  # ARM Ampere A1
vm_cpus   = 4                       # Max 4 OCPUs free tier
vm_memory = 24                      # Max 24GB RAM free tier

# Project naming
project_name = "egyptian-ocr"
