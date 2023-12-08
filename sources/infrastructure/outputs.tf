output "instance_public_ip" {
  value = scaleway_instance_ip.gpu_instance_ip.address
}