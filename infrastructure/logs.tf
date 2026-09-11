resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.app_name}-api"
  retention_in_days = 14
}