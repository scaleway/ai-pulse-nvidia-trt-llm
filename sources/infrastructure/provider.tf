terraform {
  required_providers {
    scaleway = {
      source  = "scaleway/scaleway"
      version = "~> 2.33.0"
    }
  }
  required_version = ">= 1.3.6"
}
provider "scaleway" {
  zone = var.gpu_zone
}