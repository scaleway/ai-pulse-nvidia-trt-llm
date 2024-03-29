
##### INSTANCE ######
variable "gpu_zone" {
  type = string
}
variable "users_ips_lists" {
  type = list(string)
}
locals {
  gpu_image = "ubuntu_jammy_gpu_os_12"
  gpu_type  = "H100-2-80G"
}

resource "scaleway_instance_ip" "gpu_instance_ip" {
  zone = var.gpu_zone
}

resource "scaleway_instance_volume" "scratch_block_volume" {
  zone = var.gpu_zone
  type       = "scratch"
  size_in_gb = 3000
}
resource "scaleway_instance_server" "gpu_instance" {
  zone = var.gpu_zone
  name  = "trt-llm-instance"
  type  = local.gpu_type
  image = local.gpu_image
  tags  = local.resources_tags
  root_volume {
    volume_type = "b_ssd"
    size_in_gb  = 300
  }
  ip_id                 = scaleway_instance_ip.gpu_instance_ip.id
  security_group_id     = scaleway_instance_security_group.gpu_access_sg.id
  additional_volume_ids = [scaleway_instance_volume.scratch_block_volume.id]
 user_data = {
    cloud-init = <<-EOT
    #cloud-config
    bootcmd:
      - apt-get update
    runcmd:
      - curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
      - sudo apt-get install git-lfs
    EOT
  }
}


resource "scaleway_instance_security_group" "gpu_access_sg" {
  name = "gpu_access"
  zone = var.gpu_zone
  inbound_default_policy  = "drop"
  outbound_default_policy = "accept"

  dynamic "inbound_rule" {
    for_each = var.users_ips_lists
    content {
      action   = "accept"
      ip_range = "${inbound_rule.value}"
    }
  }
}
