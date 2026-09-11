# Query the existing Default VPC
data "aws_vpc" "default" {
  default = true
}

# Query all public subnets belonging to the Default VPC
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}