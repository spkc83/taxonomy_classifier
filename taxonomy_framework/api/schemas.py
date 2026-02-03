from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Text to classify")
    taxonomy_id: Optional[str] = Field(None, description="ID of taxonomy to use")
    options: Optional[Dict[str, Any]] = Field(None, description="Classification options")


class ClassifyResponse(BaseModel):
    predicted_category: str = Field(..., description="Predicted category name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    path: List[str] = Field(default_factory=list, description="Hierarchy path")
    alternatives: List[Dict[str, Any]] = Field(
        default_factory=list, description="Alternative predictions"
    )


class BatchClassifyRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of texts to classify")
    taxonomy_id: Optional[str] = Field(None, description="ID of taxonomy to use")
    options: Optional[Dict[str, Any]] = Field(None, description="Classification options")


class BatchClassifyResponse(BaseModel):
    results: List[ClassifyResponse] = Field(..., description="Classification results")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


class TaxonomyNode(BaseModel):
    name: str = Field(..., description="Category name")
    description: Optional[str] = Field(None, description="Category description")
    children: List["TaxonomyNode"] = Field(default_factory=list, description="Child categories")


class TaxonomyResponse(BaseModel):
    id: str = Field(..., description="Taxonomy ID")
    name: str = Field(..., description="Taxonomy name")
    root: TaxonomyNode = Field(..., description="Root node of taxonomy tree")
    total_nodes: int = Field(..., description="Total number of nodes in taxonomy")
