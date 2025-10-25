# Variables for fraud detection infrastructure

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.27"
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r5.large"
  
  validation {
    condition = can(regex("^db\\.(t3|r5|r6g)\\.(micro|small|large|xlarge|2xlarge|4xlarge|8xlarge)$", var.rds_instance_class))
    error_message = "RDS instance class must be a valid AWS RDS instance type."
  }
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.r6g.large"
  
  validation {
    condition = can(regex("^cache\\.(t3|r6g|r5)\\.(micro|small|medium|large|xlarge|2xlarge|4xlarge)$", var.redis_node_type))
    error_message = "Redis node type must be a valid AWS ElastiCache node type."
  }
}

variable "enable_monitoring" {
  description = "Enable enhanced monitoring and logging"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
  
  validation {
    condition     = var.backup_retention_days >= 1 && var.backup_retention_days <= 365
    error_message = "Backup retention days must be between 1 and 365."
  }
}

variable "allowed_cidr_blocks" {
  description = "List of CIDR blocks allowed to access the infrastructure"
  type        = list(string)
  default     = ["10.0.0.0/8"]
}

variable "ssl_certificate_arn" {
  description = "SSL certificate ARN for load balancer"
  type        = string
  default     = ""
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "frauddetection.company.com"
}

variable "enable_deletion_protection" {
  description = "Enable deletion protection for critical resources"
  type        = bool
  default     = true
}

variable "cost_center" {
  description = "Cost center for resource tagging"
  type        = string
  default     = "fraud-prevention"
}

variable "contact_email" {
  description = "Contact email for resource tagging"
  type        = string
  default     = "fraud-team@company.com"
}

# Environment-specific configurations
variable "env_configs" {
  description = "Environment-specific configurations"
  type = map(object({
    min_capacity = number
    max_capacity = number
    instance_types = list(string)
    storage_size = number
  }))
  
  default = {
    dev = {
      min_capacity   = 1
      max_capacity   = 5
      instance_types = ["t3.medium", "t3.large"]
      storage_size   = 20
    }
    staging = {
      min_capacity   = 2
      max_capacity   = 10
      instance_types = ["m5.large", "m5.xlarge"]
      storage_size   = 50
    }
    prod = {
      min_capacity   = 3
      max_capacity   = 50
      instance_types = ["m5.xlarge", "m5.2xlarge", "c5n.xlarge"]
      storage_size   = 100
    }
  }
}
