"""
Fraud Detection API Routes
Core endpoints for real-time fraud detection and risk assessment
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional
import time
import uuid
import logging

from schemas.fraud import (
    FraudDetectionRequest,
    FraudDetectionResponse,
    BatchFraudRequest,
    BatchFraudResponse,
    RiskScoreRequest,
    RiskScoreResponse
)
from services.fraud_service import FraudDetectionService
from services.model_service import ModelService
from services.cache_service import CacheService
from core.database import get_database
from utils.auth import get_current_user, require_permission
from utils.rate_limit import rate_limit
from utils.monitoring import track_api_call, track_performance

logger = logging.getLogger(__name__)

router = APIRouter()

# Service dependencies
fraud_service = FraudDetectionService()
model_service = ModelService()
cache_service = CacheService()


@router.post("/detect", response_model=FraudDetectionResponse)
@rate_limit(limit="500/minute")
@track_api_call("fraud_detect")
@track_performance()
async def detect_fraud(
    request: FraudDetectionRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    database = Depends(get_database)
):
    """
    Real-time fraud detection for individual transactions
    
    Analyzes transaction data and returns:
    - Fraud score (0-1)
    - Risk level (low/medium/high/critical)
    - Decision recommendation
    - Explanation of risk factors
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Processing fraud detection request: {request_id}")
        
        # Check cache first for duplicate transactions
        cache_key = f"fraud_score:{request.transaction_id}"
        cached_result = await cache_service.get(cache_key)
        
        if cached_result:
            logger.info(f"Returning cached result for transaction: {request.transaction_id}")
            cached_result["metadata"]["request_id"] = request_id
            cached_result["metadata"]["processing_time_ms"] = int((time.time() - start_time) * 1000)
            cached_result["metadata"]["from_cache"] = True
            return cached_result
        
        # Perform fraud detection
        fraud_result = await fraud_service.detect_fraud(
            transaction_data=request.dict(),
            user_id=current_user.id if current_user else None
        )
        
        # Prepare response
        response = FraudDetectionResponse(
            status="success",
            data={
                "fraud_score": fraud_result.fraud_score,
                "risk_level": fraud_result.risk_level,
                "decision": fraud_result.decision,
                "confidence": fraud_result.confidence,
                "explanation": {
                    "top_risk_factors": fraud_result.risk_factors,
                    "feature_importance": fraud_result.feature_importance,
                    "model_version": fraud_result.model_version
                },
                "recommended_actions": fraud_result.recommended_actions
            },
            metadata={
                "request_id": request_id,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "model_version": fraud_result.model_version,
                "timestamp": int(time.time()),
                "from_cache": False
            }
        )
        
        # Cache result for 5 minutes
        await cache_service.set(cache_key, response.dict(), ttl=300)
        
        # Store prediction in database (background task)
        background_tasks.add_task(
            fraud_service.store_prediction,
            transaction_id=request.transaction_id,
            prediction_result=fraud_result,
            request_metadata={"user_id": current_user.id if current_user else None}
        )
        
        # Send real-time alert if high risk
        if fraud_result.risk_level in ["high", "critical"]:
            background_tasks.add_task(
                fraud_service.send_fraud_alert,
                transaction_data=request.dict(),
                fraud_result=fraud_result
            )
        
        logger.info(f"Fraud detection completed: {request_id}, Score: {fraud_result.fraud_score}")
        return response
        
    except Exception as e:
        logger.error(f"Fraud detection failed: {request_id}, Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Fraud detection failed",
                "message": str(e),
                "request_id": request_id
            }
        )


@router.post("/batch", response_model=BatchFraudResponse)
@rate_limit(limit="50/minute")
@require_permission("fraud:batch_detect")
@track_api_call("fraud_batch_detect")
async def batch_detect_fraud(
    request: BatchFraudRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    database = Depends(get_database)
):
    """
    Batch fraud detection for multiple transactions
    
    Processes multiple transactions efficiently and returns results for each
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Processing batch fraud detection: {request_id}, Count: {len(request.transactions)}")
        
        # Validate batch size
        if len(request.transactions) > 1000:
            raise HTTPException(
                status_code=400,
                detail="Batch size cannot exceed 1000 transactions"
            )
        
        # Process transactions in batch
        batch_results = await fraud_service.batch_detect_fraud(
            transactions=request.transactions,
            options=request.options or {},
            user_id=current_user.id if current_user else None
        )
        
        response = BatchFraudResponse(
            status="success",
            data={
                "results": batch_results.results,
                "summary": {
                    "total_processed": len(request.transactions),
                    "fraud_detected": sum(1 for r in batch_results.results if r.risk_level in ["high", "critical"]),
                    "average_score": sum(r.fraud_score for r in batch_results.results) / len(batch_results.results),
                    "processing_time_ms": int((time.time() - start_time) * 1000)
                }
            },
            metadata={
                "request_id": request_id,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "timestamp": int(time.time())
            }
        )
        
        # Store batch predictions (background task)
        background_tasks.add_task(
            fraud_service.store_batch_predictions,
            batch_results=batch_results,
            request_metadata={"user_id": current_user.id if current_user else None}
        )
        
        logger.info(f"Batch fraud detection completed: {request_id}")
        return response
        
    except Exception as e:
        logger.error(f"Batch fraud detection failed: {request_id}, Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Batch fraud detection failed",
                "message": str(e),
                "request_id": request_id
            }
        )


@router.post("/risk-score", response_model=RiskScoreResponse)
@rate_limit(limit="200/minute")
@track_api_call("risk_score")
async def calculate_risk_score(
    request: RiskScoreRequest,
    current_user = Depends(get_current_user)
):
    """
    Calculate risk score for a user or merchant
    
    Returns comprehensive risk assessment based on historical data
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Calculating risk score: {request_id}, Type: {request.entity_type}")
        
        risk_result = await fraud_service.calculate_risk_score(
            entity_type=request.entity_type,
            entity_id=request.entity_id,
            lookback_days=request.lookback_days,
            include_features=request.include_features
        )
        
        response = RiskScoreResponse(
            status="success",
            data={
                "entity_type": request.entity_type,
                "entity_id": request.entity_id,
                "risk_score": risk_result.risk_score,
                "risk_level": risk_result.risk_level,
                "confidence": risk_result.confidence,
                "factors": risk_result.risk_factors,
                "historical_data": risk_result.historical_summary,
                "recommendations": risk_result.recommendations
            },
            metadata={
                "request_id": request_id,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "timestamp": int(time.time())
            }
        )
        
        logger.info(f"Risk score calculated: {request_id}, Score: {risk_result.risk_score}")
        return response
        
    except Exception as e:
        logger.error(f"Risk score calculation failed: {request_id}, Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Risk score calculation failed",
                "message": str(e),
                "request_id": request_id
            }
        )


@router.get("/transaction/{transaction_id}")
@require_permission("fraud:view_details")
async def get_transaction_details(
    transaction_id: str,
    current_user = Depends(get_current_user),
    database = Depends(get_database)
):
    """
    Get detailed fraud analysis for a specific transaction
    """
    try:
        details = await fraud_service.get_transaction_details(transaction_id)
        
        if not details:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        return {
            "status": "success",
            "data": details,
            "metadata": {
                "timestamp": int(time.time())
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get transaction details: {transaction_id}, Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve transaction details: {str(e)}"
        )


@router.post("/explain/{transaction_id}")
@require_permission("fraud:explain")
async def explain_fraud_decision(
    transaction_id: str,
    current_user = Depends(get_current_user)
):
    """
    Get detailed explanation of fraud detection decision
    """
    try:
        explanation = await fraud_service.explain_decision(transaction_id)
        
        if not explanation:
            raise HTTPException(status_code=404, detail="Transaction explanation not found")
        
        return {
            "status": "success",
            "data": {
                "transaction_id": transaction_id,
                "explanation": explanation.explanation,
                "feature_importance": explanation.feature_importance,
                "decision_path": explanation.decision_path,
                "model_info": explanation.model_info
            },
            "metadata": {
                "timestamp": int(time.time())
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to explain decision: {transaction_id}, Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate explanation: {str(e)}"
        )
