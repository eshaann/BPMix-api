provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "BPMix"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}