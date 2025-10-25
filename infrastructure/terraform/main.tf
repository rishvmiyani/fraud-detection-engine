# Main Terraform configuration for fraud detection infrastructure
terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }

  backend "s3" {
    bucket         = "fraud-detection-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "fraud-detection-terraform-locks"
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "fraud-detection"
      Environment = var.environment
      ManagedBy   = "terraform"
      Team        = "fraud-detection-team"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values
locals {
  name_prefix = "fraud-detection-"
  
  common_tags = {
    Project     = "fraud-detection"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
  
  # Network configuration
  vpc_cidr = "10.0.0.0/16"
  azs      = slice(data.aws_availability_zones.available.names, 0, 3)
  
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  # Database configuration
  database_subnets = ["10.0.201.0/24", "10.0.202.0/24", "10.0.203.0/24"]
}

# VPC Module
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "-vpc"
  cidr = local.vpc_cidr

  azs              = local.azs
  private_subnets  = local.private_subnets
  public_subnets   = local.public_subnets
  database_subnets = local.database_subnets

  # NAT Gateway configuration
  enable_nat_gateway = true
  single_nat_gateway = var.environment == "development"
  one_nat_gateway_per_az = var.environment == "production"

  # DNS configuration
  enable_dns_hostnames = true
  enable_dns_support   = true

  # VPC Flow Logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true

  # Subnet tags for EKS
  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
    "kubernetes.io/cluster/-eks" = "shared"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
    "kubernetes.io/cluster/-eks" = "shared"
  }

  tags = local.common_tags
}

# EKS Cluster
module "eks" {
  source = "./modules/eks"
  
  cluster_name    = "-eks"
  cluster_version = var.kubernetes_version
  
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets
  
  # Node groups configuration
  node_groups = {
    application = {
      instance_types = ["m5.large", "m5.xlarge"]
      capacity_type  = "ON_DEMAND"
      min_size       = 2
      max_size       = 10
      desired_size   = 3
      
      labels = {
        role = "application"
        environment = var.environment
      }
      
      taints = [
        {
          key    = "application"
          value  = "fraud-detection"
          effect = "NO_SCHEDULE"
        }
      ]
    }
    
    ml_workload = {
      instance_types = ["c5n.2xlarge", "c5n.4xlarge"]
      capacity_type  = "SPOT"
      min_size       = 0
      max_size       = 20
      desired_size   = 2
      
      labels = {
        role = "ml-workload"
        environment = var.environment
      }
    }
  }
  
  # Add-ons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
    aws-efs-csi-driver = {
      most_recent = true
    }
  }
  
  tags = local.common_tags
}

# RDS PostgreSQL
module "rds" {
  source = "./modules/rds"
  
  identifier = "-postgres"
  
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = var.rds_instance_class
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type         = "gp3"
  storage_encrypted    = true
  
  db_name  = "fraud_detection"
  username = "fraud_user"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.database_subnets
  
  # High availability
  multi_az               = var.environment == "production"
  backup_retention_period = var.environment == "production" ? 30 : 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  # Performance Insights
  performance_insights_enabled = true
  performance_insights_retention_period = 7
  
  # Monitoring
  monitoring_interval = 60
  
  tags = local.common_tags
}

# ElastiCache Redis
module "redis" {
  source = "./modules/redis"
  
  cluster_id = "-redis"
  
  node_type = var.redis_node_type
  num_cache_nodes = var.environment == "production" ? 3 : 1
  
  engine_version = "7.0"
  port           = 6379
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # High availability
  automatic_failover_enabled = var.environment == "production"
  multi_az_enabled          = var.environment == "production"
  
  # Backup
  snapshot_retention_limit = var.environment == "production" ? 7 : 1
  snapshot_window         = "03:00-05:00"
  
  tags = local.common_tags
}

# S3 Buckets
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "-model-artifacts"
  tags   = local.common_tags
}

resource "aws_s3_bucket" "data_lake" {
  bucket = "-data-lake"
  tags   = local.common_tags
}

resource "aws_s3_bucket" "backups" {
  bucket = "-backups"
  tags   = local.common_tags
}

# S3 Bucket configurations
resource "aws_s3_bucket_encryption_configuration" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  
  rule {
    id     = "delete_old_versions"
    status = "Enabled"
    
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "fraud_detection_api" {
  name              = "/aws/fraud-detection/api"
  retention_in_days = var.environment == "production" ? 30 : 7
  
  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "fraud_detection_ml" {
  name              = "/aws/fraud-detection/ml"
  retention_in_days = var.environment == "production" ? 90 : 14
  
  tags = local.common_tags
}

# IAM Roles for EKS workloads
resource "aws_iam_role" "fraud_detection_pod_role" {
  name = "-pod-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Condition = {
          StringEquals = {
            ":sub": "system:serviceaccount:default:fraud-detection-sa"
            ":aud": "sts.amazonaws.com"
          }
        }
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
      }
    ]
  })
  
  tags = local.common_tags
}

# IAM Policy for S3 access
resource "aws_iam_policy" "s3_access" {
  name        = "-s3-access"
  description = "IAM policy for fraud detection S3 access"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "/*",
          "/*",
          "/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.model_artifacts.arn,
          aws_s3_bucket.data_lake.arn,
          aws_s3_bucket.backups.arn
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "fraud_detection_s3" {
  policy_arn = aws_iam_policy.s3_access.arn
  role       = aws_iam_role.fraud_detection_pod_role.name
}

# Outputs
output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = module.rds.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = module.redis.endpoint
  sensitive   = true
}

output "s3_buckets" {
  description = "S3 bucket names"
  value = {
    model_artifacts = aws_s3_bucket.model_artifacts.bucket
    data_lake       = aws_s3_bucket.data_lake.bucket
    backups         = aws_s3_bucket.backups.bucket
  }
}
