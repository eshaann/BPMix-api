variable "aws_region" {
  type        = string
  description = "AWS Region for deployment"
  default     = "us-east-2"
}

variable "app_name" {
  type        = string
  description = "Application name prefix"
  default     = "bpmix"
}

variable "environment" {
  type        = string
  description = "Deployment environment"
  default     = "production"
}