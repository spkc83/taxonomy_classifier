"""Classification endpoints for the Taxonomy Classifier API."""

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from taxonomy_framework.api.dependencies import get_llm_client, get_settings
from taxonomy_framework.api.schemas import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    ClassifyRequest,
    ClassifyResponse,
    TaxonomyNode,
    TaxonomyResponse,
)
from taxonomy_framework.config.settings import Settings

router = APIRouter(prefix="/classify", tags=["classification"])


@router.post("", response_model=ClassifyResponse)
async def classify_text(
    request: ClassifyRequest,
    settings: Settings = Depends(get_settings),
    llm_client: Optional[Any] = Depends(get_llm_client),
) -> ClassifyResponse:
    """
    Classify a single text into taxonomy categories.

    Returns a placeholder response until LLM integration is complete.
    """
    # Placeholder response - real LLM integration comes later
    return ClassifyResponse(
        predicted_category="placeholder_category",
        confidence=0.85,
        path=["Root", "Level1", "placeholder_category"],
        alternatives=[
            {"category": "alternative_1", "confidence": 0.10},
            {"category": "alternative_2", "confidence": 0.05},
        ],
    )


@router.post("/batch", response_model=BatchClassifyResponse)
async def classify_batch(
    request: BatchClassifyRequest,
    settings: Settings = Depends(get_settings),
    llm_client: Optional[Any] = Depends(get_llm_client),
) -> BatchClassifyResponse:
    """
    Classify multiple texts in batch.

    Returns placeholder responses until LLM integration is complete.
    """
    if not request.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one text is required for batch classification",
        )

    # Generate placeholder responses for each text
    results = [
        ClassifyResponse(
            predicted_category=f"placeholder_category_{i}",
            confidence=0.85,
            path=["Root", "Level1", f"placeholder_category_{i}"],
            alternatives=[
                {"category": "alternative_1", "confidence": 0.10},
                {"category": "alternative_2", "confidence": 0.05},
            ],
        )
        for i, _ in enumerate(request.texts)
    ]

    return BatchClassifyResponse(results=results)


# Separate router for taxonomy endpoint (no /classify prefix)
taxonomy_router = APIRouter(tags=["taxonomy"])


@taxonomy_router.get("/taxonomy", response_model=TaxonomyResponse)
async def get_taxonomy(
    settings: Settings = Depends(get_settings),
) -> TaxonomyResponse:
    """
    Get the current taxonomy structure.

    Returns a placeholder taxonomy until real taxonomy loading is implemented.
    """
    # Placeholder taxonomy structure
    placeholder_root = TaxonomyNode(
        name="Root",
        description="Root category",
        children=[
            TaxonomyNode(
                name="Category_A",
                description="First top-level category",
                children=[
                    TaxonomyNode(name="Subcategory_A1", description="Subcategory A1"),
                    TaxonomyNode(name="Subcategory_A2", description="Subcategory A2"),
                ],
            ),
            TaxonomyNode(
                name="Category_B",
                description="Second top-level category",
                children=[
                    TaxonomyNode(name="Subcategory_B1", description="Subcategory B1"),
                ],
            ),
        ],
    )

    return TaxonomyResponse(
        id="default",
        name="Default Taxonomy",
        root=placeholder_root,
        total_nodes=6,  # Root + 2 top-level + 3 subcategories
    )
